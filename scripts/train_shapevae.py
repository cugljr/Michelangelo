import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from michelangelo.utils.misc import get_config_from_file, instantiate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Michelangelo shape VAE on Tallinn NPZ samples.")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Override trainer accelerator, e.g. gpu, cpu, auto.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Override trainer devices. Examples: 0, 1, 0,1, or auto.",
    )
    return parser.parse_args()


def parse_devices_arg(devices_arg: str):
    value = devices_arg.strip()
    if value.lower() == "auto":
        return "auto"
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if value.isdigit():
        return [int(value)]
    return value


def build_loggers(log_dir: Path, experiment_name: str, trainer_cfg):
    loggers = [
        CSVLogger(save_dir=str(log_dir), name=experiment_name),
        TensorBoardLogger(save_dir=str(log_dir), name=experiment_name),
    ]

    wandb_cfg = trainer_cfg.get("wandb")
    if wandb_cfg and wandb_cfg.get("enabled", False):
        loggers.append(
            WandbLogger(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("name", experiment_name),
                save_dir=str(log_dir),
                offline=wandb_cfg.get("offline", False),
                tags=wandb_cfg.get("tags"),
                notes=wandb_cfg.get("notes"),
            )
        )

    return loggers


def main() -> None:
    args = parse_args()
    config = get_config_from_file(args.config_path)

    pl.seed_everything(args.seed, workers=True)

    model = instantiate_from_config(config.model)
    data = instantiate_from_config(config.data)
    trainer_cfg = config.trainer
    wandb_cfg = trainer_cfg.get("wandb")
    if wandb_cfg is not None:
        wandb_cfg["config_path"] = args.config_path

    accelerator = args.accelerator if args.accelerator is not None else trainer_cfg.get("accelerator", "auto")
    devices = parse_devices_arg(args.devices) if args.devices is not None else trainer_cfg.get("devices", "auto")

    experiment_name = trainer_cfg.get("experiment_name", Path(args.config_path).stem)
    output_dir = Path(trainer_cfg.get("output_dir", "runs")) / experiment_name
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}",
            monitor=trainer_cfg.get("monitor", "val/total_loss"),
            mode=trainer_cfg.get("monitor_mode", "min"),
            save_top_k=trainer_cfg.get("save_top_k", 3),
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        logger=build_loggers(log_dir, experiment_name, trainer_cfg),
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", 32),
        max_epochs=trainer_cfg.get("max_epochs", 100),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 20),
        val_check_interval=trainer_cfg.get("val_check_interval", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 0.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
    )

    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
