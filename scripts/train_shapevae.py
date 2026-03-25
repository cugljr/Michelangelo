import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from michelangelo.utils.misc import get_config_from_file, instantiate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Michelangelo shape VAE on Tallinn NPZ samples.")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_loggers(log_dir: Path, experiment_name: str):
    loggers = [
        CSVLogger(save_dir=str(log_dir), name=experiment_name),
        TensorBoardLogger(save_dir=str(log_dir), name=experiment_name),
    ]
    return loggers


def main() -> None:
    args = parse_args()
    config = get_config_from_file(args.config_path)

    pl.seed_everything(args.seed, workers=True)

    model = instantiate_from_config(config.model)
    data = instantiate_from_config(config.data)
    trainer_cfg = config.trainer

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
        logger=build_loggers(log_dir, experiment_name),
        callbacks=callbacks,
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
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
