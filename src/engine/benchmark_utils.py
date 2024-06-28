# benchmark_utils.py
import time
import traceback
from typing import Dict, Any, List

import torch
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from torch.utils.data import DataLoader

from src import Panel, Table
from src import console
from src.engine.augmentation import apply_augmentations
from src.engine.config import BenchmarkConfig
from src.engine.metrics import calculate_bleu_score, calculate_rouge_scores, calculate_diversity_metrics


def run_benchmark(model, tokenizer, data_module, config: BenchmarkConfig) -> Dict[str, Any]:
    results = {}

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        for task in config.benchmark_tasks:
            task_id = progress.add_task(f"Running {task}", total=100)
            logger.info(f"Starting benchmark for task: {task}")

            if task == "language_modeling":
                results[task] = benchmark_language_modeling(model, data_module.test_dataloader(), progress, task_id)
            elif task == "text_generation":
                results[task] = benchmark_text_generation(model, tokenizer, data_module.test_dataloader(), config,
                                                          progress, task_id)

            progress.update(task_id, completed=100)

    return results


def benchmark_language_modeling(model, dataloader, progress, task_id) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(0)
            total_tokens += attention_mask.sum().item()

            progress.update(task_id, advance=100 / len(dataloader))

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return {"perplexity": perplexity}


def generate_ood_data(config: BenchmarkConfig):
    # This is a placeholder. In a real scenario, you'd generate or load OOD data.
    # For now, we'll just return a small random dataset
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=128, vocab_size=50257):
            self.data = torch.randint(0, vocab_size, (size, seq_len))
            self.attention_mask = torch.ones_like(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {
                "input_ids": self.data[idx],
                "attention_mask": self.attention_mask[idx]
            }

    dataset = RandomDataset()
    return DataLoader(dataset, batch_size=config.batch_size)


def compute_metrics(train_losses: List[float], val_losses: List[float]) -> Dict[str, float]:
    return {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": min(val_losses),
        "train_loss_reduction": train_losses[0] - train_losses[-1],
        "val_loss_reduction": val_losses[0] - val_losses[-1],
    }


def display_task_results(task_name, results):
    table = Table(title=f"{task_name} Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for metric, value in results.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))

    console.print(table)


def benchmark_text_generation(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating text generation capabilities...[/cyan]")
    model.eval()
    generated_texts = []
    reference_texts = []

    for batch in data_module.test_dataloader():
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=config.max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        generated_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        reference_texts.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))

        progress.update(task_id, advance=100 / len(data_module.test_dataloader()))

    bleu_score = calculate_bleu_score(reference_texts, generated_texts)
    rouge_scores = calculate_rouge_scores(reference_texts, generated_texts)
    diversity_metrics = calculate_diversity_metrics(generated_texts)

    return {
        "bleu": bleu_score,
        **rouge_scores,
        **diversity_metrics
    }


def benchmark_augmentation_impact(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating impact of data augmentation...[/cyan]")
    original_perplexity = benchmark_language_modeling(model, tokenizer, data_module, config)[
        "perplexity"]

    # Apply augmentation to the test dataset
    augmented_dataset = apply_augmentations(data_module.test_dataset, config)
    augmented_dataloader = DataLoader(augmented_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    augmented_perplexity = \
        benchmark_language_modeling(model, tokenizer, augmented_dataloader, config)["perplexity"]

    return {
        "original_perplexity": original_perplexity,
        "augmented_perplexity": augmented_perplexity,
        "perplexity_change": augmented_perplexity - original_perplexity
    }


def benchmark_diversity(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating output diversity...[/cyan]")
    model.eval()
    generated_texts = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=config.max_length,
                num_return_sequences=5,  # Generate multiple sequences for diversity
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

            generated_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

            progress.update(task_id, advance=100 / len(data_module.test_dataloader()))

    return calculate_diversity_metrics(generated_texts)


def run_comprehensive_benchmark(model, tokenizer, data_module, config):
    results = {}
    tasks = [
        ("Language Modeling", benchmark_language_modeling),
        ("Text Generation", benchmark_text_generation),
        ("Augmentation Impact", benchmark_augmentation_impact),
        ("Diversity", benchmark_diversity)
    ]

    console.print(Panel.fit("🚀 [bold cyan]Starting Comprehensive Benchmark[/bold cyan] 🚀"))

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
    ) as progress:
        overall_task = progress.add_task("[yellow]Overall Progress", total=len(tasks))

        for task_name, task_func in tasks:
            task_progress = progress.add_task(f"[cyan]{task_name}", total=100)
            try:
                console.print(f"\n[bold green]Starting {task_name}...[/bold green]")
                start_time = time.time()

                # Run the benchmark task
                task_results = task_func(model, tokenizer, data_module, config, progress, task_progress)

                end_time = time.time()
                duration = end_time - start_time

                results[task_name] = task_results
                console.print(f"[bold green]{task_name} completed in {duration:.2f} seconds.[/bold green]")

                # Display task-specific results
                display_task_results(task_name, task_results)

            except Exception as e:
                console.print(f"[bold red]Error in {task_name}:[/bold red]")
                console.print(Panel(str(e), title="Error Details", border_style="red"))
                console.print(traceback.format_exc())
                console.print(f"[yellow]Skipping {task_name} due to error. Continuing with next task...[/yellow]")
                results[task_name] = {"error": str(e)}

            progress.update(overall_task, advance=1)
            progress.update(task_progress, completed=100)

    console.print(Panel.fit("[bold green]Comprehensive Benchmark Completed![/bold green]"))
    return results
