"""
Unified CLI for Chess Position Evaluator.

Usage:
    cpe train <experiment-name> --epochs N --steps N --batch N --lr F --wd F --patience N --factor F [--resume]
    cpe analyze-rank <model-name>
    cpe eval <model-name>
    cpe export-onnx <model-name>
"""

from dotenv import find_dotenv, load_dotenv
import os
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
    t0: int = typer.Option(
        10,
        min=1,
        help="Scheduler T_0 (epochs before first restart) for CosineAnnealingWarmRestarts",
    ),
    t_mult: float = typer.Option(
        2.0,
        min=1.0,
        help="Scheduler T_mult (restart period multiplier) for CosineAnnealingWarmRestarts",
    ),
    eta_min: float = typer.Option(
        1e-6,
        min=0.0,
        help="Scheduler minimum learning rate for CosineAnnealingWarmRestarts",
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

    # Check WANDB_API_KEY
    if not os.environ.get("WANDB_API_KEY"):
        print("Error: WANDB_API_KEY env var is required")
        raise typer.Exit(1)

    # Check checkpoint exists
    checkpoint_path = f"models/checkpoints/{experiment_name}.pth"
    if os.path.exists(checkpoint_path) and not resume:
        print(f"Error: {checkpoint_path} already exists")
        print("Use --resume to continue training")
        raise typer.Exit(1)

    # Import here to avoid slow startup for other commands
    from train.entry import run_training

    run_training(
        experiment_name=experiment_name,
        epochs=epochs,
        steps_per_epoch=steps,
        batch_size=batch,
        lr=lr,
        wd=wd,
        t0=t0,
        t_mult=t_mult,
        eta_min=eta_min,
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


@app.command()
def eval(
    model_name: str = typer.Argument(...,
                                     help="Model name (without .pth extension)"),
):
    """Interactive FEN evaluation."""
    from eval import run_eval

    run_eval(model_name)


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
