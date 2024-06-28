#!/usr/bin/env python3
# main.py

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any

import pytorch_lightning as pl
import torch
from lightning import LightningModule
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader

from src.engine.augmentation import apply_augmentations
from src.engine.config import BenchmarkConfig
from src.engine.data_utils import get_dataset_loader
from src.engine.metrics import calculate_uncertainty_metrics, calculate_auc_roc, calculate_log_loss, \
    calculate_mutual_information, calculate_predictive_entropy, calculate_diversity_metrics, calculate_bleu_score, \
    calculate_rouge_scores, calculate_perplexity, calculate_accuracy, calculate_precision_recall_f1

console = Console()

# Configure loguru for advanced logging
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="🚀 Advanced Benchmarking Tool for Uncertain Transformers")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--mode", choices=["train", "finetune", "benchmark"], default="benchmark",
                        help="Operation mode")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results and visualizations")
    parser.add_argument("--dataset", choices=["qa", "coding", "generalization", "medical", "misc"], default="misc",
                        help="Dataset to use")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory to cache datasets")
    return parser


def setup_environment(config: BenchmarkConfig):
    pl.seed_everything(config.seed)
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_trainer(config: BenchmarkConfig) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
        EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, mode="min"),
        LearningRateMonitor(logging_interval="step")
    ]

    logger = TensorBoardLogger("logs", name="uncertain_transformer")

    return pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        precision=16 if config.use_mixed_precision else 32,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        progress_bar_refresh_rate=0,  # Disable default progress bar
    )


def run_training(model: pl.LightningModule, data_module: pl.LightningDataModule, trainer: pl.Trainer):
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training", total=trainer.max_epochs)
        for epoch in range(trainer.max_epochs):
            trainer.fit(model, data_module)
            progress.update(task, advance=1)


def display_results(results: Dict[str, Any]):
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for category, metrics in results.items():
        table.add_row(category.capitalize(), "")
        for metric, value in metrics.items():
            table.add_row(f"  {metric}", f"{value:.4f}" if isinstance(value, float) else str(value))

    console.print(table)


def save_results(results: Dict[str, Any], output_dir: str):
    output_path = Path(output_dir) / "results.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    console.print(f"Results saved to: [bold]{output_path}[/bold]")


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    rprint(Panel.fit("🚀 [bold cyan]Advanced Benchmarking Tool for Uncertain Transformers[/bold cyan] 🚀"))

    config = BenchmarkConfig.from_json(args.config)
    setup_environment(config)

    console.print(f"⚙️  Loaded configuration from: [bold]{args.config}[/bold]")

    # Set up data module with caching
    data_module = get_dataset_loader(args.dataset, config, cache_dir=args.cache_dir)


def setup_training(config: BenchmarkConfig, model: LightningModule) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        LearningRateMonitor(logging_interval="step")
    ]

    logger = TensorBoardLogger("logs", name="uncertain_transformer")

    console.print("[bold green]✨ All operations completed successfully! ✨[/bold green]")

    return pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gpus=torch.cuda.device_count(),
        precision="16-mixed" if config.use_mixed_precision else 32,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
    )



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during execution:")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)
