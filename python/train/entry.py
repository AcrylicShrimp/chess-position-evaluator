import torch

from libs.dataset import ChessEvaluationDataset
from libs.model import EvalOnlyModel
from train.trainer import Trainer


def worker_init_fn(_: int):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.open_file()


def run_training(
    experiment_name: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    wd: float,
    patience: int,
    factor: float,
    grad_clip: float,
    resume: bool,
):
    """Run training with the given hyperparameters."""
    print("=== Train Evaluation ===")
    print(f"[✓] Using torch version: {torch.__version__}")

    train_data_path = "train.chesseval"
    validation_data_path = "validation.chesseval"
    checkpoint_path = f"models/checkpoints/{experiment_name}.pth"
    best_checkpoint_path = f"models/checkpoints/{experiment_name}-best.pth"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[✓] Using device: {device}")
    print(f"[✓] Experiment: {experiment_name}")
    print(
        f"[✓] Hyperparameters: epochs={epochs}, steps={steps_per_epoch}, batch={batch_size}"
    )
    print(f"[✓] Optimizer: lr={lr}, wd={wd}")
    print(f"[✓] Scheduler: patience={patience}, factor={factor}")
    print(f"[✓] Grad clip: {grad_clip}")

    model = EvalOnlyModel()
    trainer = Trainer(
        model=model,
        device=device,
        experiment_name=experiment_name,
        lr=lr,
        wd=wd,
        patience=patience,
        factor=factor,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        grad_clip=grad_clip,
    )

    if resume:
        trainer.load_checkpoint(checkpoint_path)

    train_data = ChessEvaluationDataset(train_data_path)
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    validation_data = ChessEvaluationDataset(validation_data_path)
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    print(f"[✓] Data loaded from {train_data_path} ({len(train_data)} rows)")
    print(f"[✓] Data loaded from {validation_data_path} ({len(validation_data)} rows)")

    trainer.train(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    print("[✓] Training completed")
