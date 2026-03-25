import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import trimesh
from tqdm import tqdm

try:
    from pysdf import SDF
except ImportError:
    SDF = None


DEFAULT_MESH_DIR = Path("/home/kemove/devdata1/ljr/dataset/Tallinn/meshes")
DEFAULT_SPLIT_DIR = Path("/home/kemove/devdata1/ljr/dataset/Tallinn/split")
DEFAULT_OUTPUT_DIR = Path("/home/kemove/devdata1/ljr/dataset/Tallinn/michelangelo_npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Tallinn OBJ meshes into Michelangelo training samples."
    )
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--surface-samples", type=int, default=100000)
    parser.add_argument("--volume-samples", type=int, default=100000)
    parser.add_argument("--near-samples", type=int, default=100000)
    parser.add_argument("--query-bounds", type=float, default=1.05)
    parser.add_argument(
        "--near-sigmas",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.06),
        help="Gaussian std values used to perturb near-surface samples.",
    )
    parser.add_argument(
        "--near-sigma-probs",
        type=float,
        nargs="+",
        default=(0.7, 0.2, 0.1),
        help="Sampling probabilities for near-surface sigma values.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_split(split_path: Path) -> List[str]:
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("empty scene")
        loaded = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded)!r}")

    mesh = loaded.copy()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh has no vertices or faces after cleanup")

    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    loc = vertices.mean(axis=0).astype(np.float32)
    centered = vertices - loc
    scale = float(np.abs(centered).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid normalization scale")

    normalized_vertices = centered / scale
    normalized_mesh = trimesh.Trimesh(
        vertices=normalized_vertices.astype(np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    normalized_mesh.remove_unreferenced_vertices()
    return normalized_mesh, loc, scale


class OccupancyOracle:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.sdf = None
        self.inside_is_negative = True

        if SDF is not None:
            self.sdf = SDF(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int32),
            )
            self.inside_is_negative = self._infer_sdf_sign()

    def _infer_sdf_sign(self) -> bool:
        if self.sdf is None or not self.mesh.is_watertight:
            return True

        rng = np.random.default_rng(0)
        probe = rng.uniform(-1.0, 1.0, size=(2048, 3)).astype(np.float64)
        try:
            contains = self.mesh.contains(probe)
        except Exception:
            return True

        signed = np.asarray(self.sdf(probe), dtype=np.float64)
        neg_score = np.mean((signed <= 0) == contains)
        pos_score = np.mean((signed >= 0) == contains)
        return bool(neg_score >= pos_score)

    def query(self, points: np.ndarray) -> np.ndarray:
        points64 = np.asarray(points, dtype=np.float64)

        if self.sdf is not None:
            signed = np.asarray(self.sdf(points64), dtype=np.float64)
            inside = signed <= 0 if self.inside_is_negative else signed >= 0
            return inside.astype(np.uint8)

        try:
            inside = self.mesh.contains(points64)
            return np.asarray(inside, dtype=np.uint8)
        except Exception as exc:
            raise RuntimeError(
                "Occupancy query failed. Install `pysdf` or ensure trimesh contains() works."
            ) from exc


def sample_surface(mesh: trimesh.Trimesh, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    face_normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)

    # Michelangelo can select either watertight normals or normals; duplicate the
    # same face normals into both slots so the sample stays format-compatible.
    return np.concatenate([points, face_normals, face_normals], axis=1).astype(np.float16)


def sample_volume(
    oracle: OccupancyOracle,
    n_samples: int,
    bounds: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    points = rng.uniform(-bounds, bounds, size=(n_samples, 3)).astype(np.float32)
    labels = oracle.query(points)
    return points.astype(np.float16), labels.astype(np.uint8)


def sample_near_surface(
    mesh: trimesh.Trimesh,
    oracle: OccupancyOracle,
    n_samples: int,
    sigmas: Sequence[float],
    sigma_probs: Sequence[float],
    bounds: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)

    sigma_choices = rng.choice(np.asarray(sigmas, dtype=np.float32), size=n_samples, p=sigma_probs)
    offsets = rng.normal(loc=0.0, scale=sigma_choices, size=n_samples).astype(np.float32)
    near_points = points + normals * offsets[:, None]
    near_points = np.clip(near_points, -bounds, bounds)
    labels = oracle.query(near_points)
    return near_points.astype(np.float16), labels.astype(np.uint8)


def save_sample(
    output_path: Path,
    surface: np.ndarray,
    vol_points: np.ndarray,
    vol_label: np.ndarray,
    near_points: np.ndarray,
    near_label: np.ndarray,
    loc: np.ndarray,
    scale: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        surface=surface,
        vol_points=vol_points,
        vol_label=vol_label,
        near_points=near_points,
        near_label=near_label,
        loc=loc.astype(np.float32),
        scale=np.float32(scale),
    )


def process_one(task: Dict[str, object]) -> Tuple[str, bool, str]:
    sample_id = str(task["sample_id"])
    mesh_path = Path(task["mesh_path"])
    output_path = Path(task["output_path"])
    overwrite = bool(task["overwrite"])
    if output_path.exists() and not overwrite:
        return sample_id, True, "skipped_existing"

    rng = np.random.default_rng(int(task["seed"]))
    try:
        mesh = load_mesh(mesh_path)
        mesh, loc, scale = normalize_mesh(mesh)
        oracle = OccupancyOracle(mesh)

        surface = sample_surface(mesh, int(task["surface_samples"]), rng)
        vol_points, vol_label = sample_volume(
            oracle,
            int(task["volume_samples"]),
            float(task["query_bounds"]),
            rng,
        )
        near_points, near_label = sample_near_surface(
            mesh,
            oracle,
            int(task["near_samples"]),
            task["near_sigmas"],
            task["near_sigma_probs"],
            float(task["query_bounds"]),
            rng,
        )
        save_sample(
            output_path,
            surface,
            vol_points,
            vol_label,
            near_points,
            near_label,
            loc,
            scale,
        )
        return sample_id, True, "ok"
    except Exception:
        return sample_id, False, traceback.format_exc()


def build_tasks(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, List[str]]]:
    split_names = ("train", "val", "test")
    split_map: Dict[str, List[str]] = {}
    tasks: List[Dict[str, object]] = []

    near_probs = np.asarray(args.near_sigma_probs, dtype=np.float64)
    if len(args.near_sigmas) != len(near_probs):
        raise ValueError("--near-sigmas and --near-sigma-probs must have the same length")
    if np.any(near_probs < 0) or near_probs.sum() <= 0:
        raise ValueError("--near-sigma-probs must be non-negative and sum to a positive value")
    near_probs = near_probs / near_probs.sum()

    seen_ids = set()
    for split_name in split_names:
        split_ids = read_split(args.split_dir / f"{split_name}.txt")
        split_map[split_name] = split_ids
        for sample_id in split_ids:
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            mesh_path = args.mesh_dir / f"{sample_id}.obj"
            output_path = args.output_dir / "samples" / f"{sample_id}.npz"
            tasks.append(
                {
                    "sample_id": sample_id,
                    "mesh_path": str(mesh_path),
                    "output_path": str(output_path),
                    "surface_samples": args.surface_samples,
                    "volume_samples": args.volume_samples,
                    "near_samples": args.near_samples,
                    "query_bounds": args.query_bounds,
                    "near_sigmas": tuple(float(v) for v in args.near_sigmas),
                    "near_sigma_probs": near_probs.tolist(),
                    "seed": args.seed + len(tasks),
                    "overwrite": args.overwrite,
                }
            )
    return tasks, split_map


def write_split_manifests(
    output_dir: Path, split_map: Dict[str, List[str]], valid_ids: Sequence[str]
) -> None:
    split_out_dir = output_dir / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    valid_id_set = set(valid_ids)
    for split_name, sample_ids in split_map.items():
        out_path = split_out_dir / f"{split_name}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for sample_id in sample_ids:
                if sample_id in valid_id_set:
                    f.write(f"{sample_id}\n")


def write_summary(output_dir: Path, stats: Dict[str, int]) -> None:
    summary_path = output_dir / "preprocess_summary.txt"
    lines = [f"{key}: {value}" for key, value in stats.items()]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tasks(tasks: List[Dict[str, object]], jobs: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    successes: List[str] = []
    failures: List[Tuple[str, str]] = []

    if jobs <= 1:
        iterator: Iterable[Tuple[str, bool, str]] = (
            process_one(task) for task in tqdm(tasks, desc="Preprocessing", dynamic_ncols=True)
        )
        for mesh_path, ok, info in iterator:
            if ok:
                successes.append(mesh_path)
            else:
                failures.append((mesh_path, info))
        return successes, failures

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_map = {executor.submit(process_one, task): task for task in tasks}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Preprocessing", dynamic_ncols=True):
            mesh_path, ok, info = future.result()
            if ok:
                successes.append(mesh_path)
            else:
                failures.append((mesh_path, info))
    return successes, failures


def write_failures(output_dir: Path, failures: List[Tuple[str, str]]) -> None:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failure_path = logs_dir / "failed.txt"
    with failure_path.open("w", encoding="utf-8") as f:
        for mesh_path, error_text in failures:
            f.write(f"[{mesh_path}]\n{error_text}\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks, split_map = build_tasks(args)
    successes, failures = run_tasks(tasks, args.jobs)
    write_split_manifests(args.output_dir, split_map, successes)
    write_failures(args.output_dir, failures)
    write_summary(
        args.output_dir,
        {
            "total_tasks": len(tasks),
            "successful": len(successes),
            "failed": len(failures),
            "surface_samples": args.surface_samples,
            "volume_samples": args.volume_samples,
            "near_samples": args.near_samples,
        },
    )

    if failures:
        raise SystemExit(
            f"Finished with {len(failures)} failures. See {args.output_dir / 'logs' / 'failed.txt'}."
        )

    print(f"Finished preprocessing {len(successes)} meshes into {args.output_dir}.")


if __name__ == "__main__":
    main()
