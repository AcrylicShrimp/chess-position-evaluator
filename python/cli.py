"""
Unified CLI for Chess Position Evaluator.

Usage:
    cpe train <experiment-name> --epochs N --steps N --batch N --lr F --wd F [--resume]
    cpe analyze-rank <model-name>
    cpe analyze-material-labels
    cpe analyze-material-signal
    cpe eval-dataset <model-name>
    cpe eval <model-name>
    cpe battle <model-name>
    cpe export-onnx <model-name>
"""

from dotenv import find_dotenv, load_dotenv
import os
from pathlib import Path
import re
import typer

app = typer.Typer(
    name="cpe",
    help="Chess Position Evaluator CLI",
    add_completion=False,
)

dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path)

EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@app.command()
def train(
    experiment_name: str = typer.Argument(
        ..., help="Experiment name for checkpoints (pattern: [A-Za-z0-9_-]+)"
    ),
    epochs: int = typer.Option(..., help="Number of epochs"),
    steps: int = typer.Option(..., help="Steps per epoch"),
    batch: int = typer.Option(..., help="Batch size"),
    lr: float = typer.Option(..., help="Learning rate"),
    wd: float = typer.Option(..., help="Weight decay"),
    model_variant: str = typer.Option(
        "stacked-edge-gate-ffn",
        help=(
            "Model variant: stacked-edge-gate-ffn, one-layer-edge-gate, "
            "no-attention, or parallel-cnn-attn-fuse"
        ),
    ),
    scheduler: str = typer.Option(
        "warm-restart",
        help="Scheduler: warm-restart or warmup-cosine",
    ),
    t0: int = typer.Option(
        10,
        min=1,
        help="Scheduler T_0 for warm-restart",
    ),
    t_mult: int = typer.Option(
        2,
        min=1,
        help="Scheduler T_mult for warm-restart",
    ),
    eta_min: float = typer.Option(
        1e-6,
        min=0.0,
        help="Scheduler minimum learning rate",
    ),
    warmup_epochs: int = typer.Option(
        0,
        min=0,
        help="Linear warmup epochs for warmup-cosine",
    ),
    warmup_start_factor: float = typer.Option(
        0.1,
        help="Initial LR factor for warmup-cosine; must be in (0, 1]",
    ),
    grad_clip: float = typer.Option(
        1.0, help="Max grad norm for clipping (passed to clip_grad_norm_)"
    ),
    train_workers: int = typer.Option(
        4,
        min=0,
        help="Number of worker processes for the training dataloader",
    ),
    val_workers: int = typer.Option(
        2,
        min=0,
        help="Number of worker processes for the validation dataloader",
    ),
    upload_checkpoints: bool = typer.Option(
        True,
        "--upload-checkpoints/--no-upload-checkpoints",
        help="Upload checkpoints to Weights & Biases artifacts",
    ),
    resume: bool = typer.Option(False, help="Resume from existing checkpoint"),
):
    """Train the model with explicit hyperparameters and WandB logging."""
    # Validate experiment name
    if not EXPERIMENT_NAME_PATTERN.match(experiment_name):
        print(f"Error: Invalid experiment name '{experiment_name}'")
        print("Must match pattern: [A-Za-z0-9_-]+")
        raise typer.Exit(1)

    supported_model_variants = {
        "stacked-edge-gate-ffn",
        "one-layer-edge-gate",
        "no-attention",
        "parallel-cnn-attn-fuse",
    }
    if model_variant not in supported_model_variants:
        allowed = ", ".join(sorted(supported_model_variants))
        print(f"Error: Unsupported model variant '{model_variant}'")
        print(f"Expected one of: {allowed}")
        raise typer.Exit(1)

    # Check WANDB_API_KEY
    if not os.environ.get("WANDB_API_KEY"):
        print("Error: WANDB_API_KEY env var is required")
        raise typer.Exit(1)

    supported_schedulers = {"warm-restart", "warmup-cosine"}
    if scheduler not in supported_schedulers:
        allowed = ", ".join(sorted(supported_schedulers))
        print(f"Error: Unsupported scheduler '{scheduler}'")
        print(f"Expected one of: {allowed}")
        raise typer.Exit(1)

    if scheduler == "warmup-cosine" and warmup_epochs <= 0:
        print("Error: --warmup-epochs must be greater than zero for warmup-cosine")
        raise typer.Exit(1)

    if not 0.0 < warmup_start_factor <= 1.0:
        print("Error: --warmup-start-factor must be in the range (0, 1]")
        raise typer.Exit(1)

    from libs.paths import checkpoint_path

    # Check checkpoint exists
    model_checkpoint_path = checkpoint_path(experiment_name)
    if model_checkpoint_path.exists() and not resume:
        print(f"Error: {model_checkpoint_path} already exists")
        print("Use --resume to continue training")
        raise typer.Exit(1)

    # Import here to avoid slow startup for other commands
    from train.entry import run_training

    run_training(
        experiment_name=experiment_name,
        model_variant=model_variant,
        epochs=epochs,
        steps_per_epoch=steps,
        batch_size=batch,
        lr=lr,
        wd=wd,
        scheduler=scheduler,
        t0=t0,
        t_mult=t_mult,
        eta_min=eta_min,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        grad_clip=grad_clip,
        resume=resume,
        train_workers=train_workers,
        val_workers=val_workers,
        upload_checkpoints=upload_checkpoints,
    )


@app.command("analyze-rank")
def analyze_rank(
    model_name: str = typer.Argument(...,
                                     help="Model name (without .pth extension)"),
):
    """Analyze activation rank to check model capacity."""
    from analyze_rank import run_analyze_rank

    run_analyze_rank(model_name)


@app.command("analyze-material-labels")
def analyze_material_labels(
    split: str = typer.Option(
        "validation",
        help="Dataset split: train, validation, or test",
    ),
    dataset: Path | None = typer.Option(None, help="Override dataset path"),
    rows: int | None = typer.Option(
        None,
        min=1,
        help="Analyze the first N rows. Mutually exclusive with --full.",
    ),
    full: bool = typer.Option(False, help="Analyze the full dataset split"),
    batch: int = typer.Option(8192, min=1, help="Analysis batch size"),
    seed: int = typer.Option(0, help="Recorded sampling seed"),
    workers: int = typer.Option(
        0,
        min=0,
        help="Number of worker processes for dataset loading",
    ),
    output: Path | None = typer.Option(None, help="Report output path"),
):
    """Analyze dataset labels against the fixed material-score prior."""
    from analyze_material_label_calibration import run_material_label_analysis

    try:
        run_material_label_analysis(
            split=split,
            dataset_path=dataset,
            rows=rows,
            full=full,
            batch_size=batch,
            seed=seed,
            workers=workers,
            output_path=output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise typer.Exit(1) from exc


@app.command("analyze-material-signal")
def analyze_material_signal(
    split: str = typer.Option(
        "all",
        help="Staging split: train, validation, test, or all",
    ),
    staging: Path | None = typer.Option(
        None, help="Override staging DuckDB path"),
    rows: int | None = typer.Option(
        None,
        min=1,
        help="Analyze the first N staging rows in the split. Mutually exclusive with --full.",
    ),
    full: bool = typer.Option(False, help="Analyze the full staging split"),
    batch: int = typer.Option(20_000, min=1, help="DuckDB fetch batch size"),
    output: Path | None = typer.Option(None, help="Report output path"),
):
    """Analyze source/staging material difference against engine cp targets."""
    from analyze_material_signal import run_material_signal_analysis

    try:
        run_material_signal_analysis(
            split=split,
            staging_path=staging,
            rows=rows,
            full=full,
            batch_size=batch,
            output_path=output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise typer.Exit(1) from exc


@app.command()
def eval(
    model_name: str = typer.Argument(...,
                                     help="Model name (without .pth extension)"),
):
    """Interactive FEN evaluation."""
    from eval import run_eval

    run_eval(model_name)


@app.command("eval-dataset")
def eval_dataset(
    model_name: str = typer.Argument(...,
                                     help="Model name (without .pth extension)"),
    split: str = typer.Option(
        "validation",
        help="Dataset split: train, validation, or test",
    ),
    dataset: Path | None = typer.Option(None, help="Override dataset path"),
    rows: int | None = typer.Option(
        None,
        min=1,
        help="Evaluate the first N rows. Mutually exclusive with --full.",
    ),
    full: bool = typer.Option(False, help="Evaluate the full dataset split"),
    batch: int = typer.Option(4096, min=1, help="Evaluation batch size"),
    seed: int = typer.Option(0, help="Recorded sampling seed"),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or mps"),
    model_variant: str | None = typer.Option(
        None,
        help="Override checkpoint model variant",
    ),
    output: Path | None = typer.Option(None, help="Report output path"),
):
    """Evaluate a checkpoint against a processed dataset split."""
    from eval_dataset import run_eval_dataset

    try:
        run_eval_dataset(
            model_name=model_name,
            split=split,
            dataset_path=dataset,
            rows=rows,
            full=full,
            batch_size=batch,
            seed=seed,
            device_name=device,
            model_variant=model_variant,
            output_path=output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise typer.Exit(1) from exc


@app.command()
def battle(
    model_name: str = typer.Argument(
        ...,
        help="Model name without .pth extension.",
    ),
):
    """Play against the AI."""
    from battle.entry import run_battle

    run_battle(model_name)


@app.command("export-onnx")
def export_onnx(
    model_name: str = typer.Argument(...,
                                     help="Model name (without .pth extension)"),
):
    """Export model to ONNX format."""
    from export_onnx import run_export_onnx

    run_export_onnx(model_name)


if __name__ == "__main__":
    app()
