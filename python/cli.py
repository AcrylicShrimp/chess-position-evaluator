"""
Unified CLI for Chess Position Evaluator.

Usage:
    cpe train <experiment-name> --epochs N --steps N --batch N --lr F --wd F --patience N --factor F [--resume]
    cpe analyze-rank <model-name>
    cpe eval <model-name>
    cpe export-onnx <model-name>
"""

import os
import re
import typer

app = typer.Typer(
    name="cpe",
    help="Chess Position Evaluator CLI",
    add_completion=False,
)

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
    patience: int = typer.Option(
        ..., help="Scheduler patience (epochs before LR reduction)"
    ),
    factor: float = typer.Option(
        ..., help="Scheduler factor (LR multiplier on plateau)"
    ),
    grad_clip: float = typer.Option(
        1.0, help="Max grad norm for clipping (passed to clip_grad_norm_)"
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
        patience=patience,
        factor=factor,
        grad_clip=grad_clip,
        resume=resume,
    )


@app.command("analyze-rank")
def analyze_rank(
    model_name: str = typer.Argument(..., help="Model name (without .pth extension)"),
):
    """Analyze activation rank to check model capacity."""
    from analyze_rank import run_analyze_rank

    run_analyze_rank(model_name)


@app.command()
def eval(
    model_name: str = typer.Argument(..., help="Model name (without .pth extension)"),
):
    """Interactive FEN evaluation."""
    from eval import run_eval

    run_eval(model_name)


@app.command("export-onnx")
def export_onnx(
    model_name: str = typer.Argument(..., help="Model name (without .pth extension)"),
):
    """Export model to ONNX format."""
    from export_onnx import run_export_onnx

    run_export_onnx(model_name)


if __name__ == "__main__":
    app()
