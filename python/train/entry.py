import torch

from libs.dataset import ChessEvaluationDataset
from libs.model import ValueOnlyModel, model_variant_from_checkpoint
from libs.paths import TRAIN_DATA_PATH, VALIDATION_DATA_PATH, checkpoint_path
from train.schedulers import SCHEDULER_WARM_RESTART, SCHEDULER_WARMUP_COSINE
from train.trainer import Trainer


def worker_init_fn(_: int):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.open_file()


def _resolve_model_variant_for_training(
    model_checkpoint_path,
    requested_model_variant: str,
    resume: bool,
) -> str:
    if not resume or not model_checkpoint_path.exists():
        return requested_model_variant

    checkpoint = torch.load(
        model_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_model_variant = model_variant_from_checkpoint(checkpoint)
    if checkpoint_model_variant != requested_model_variant:
        print(
            "[✓] Resume checkpoint model variant: "
            f"{checkpoint_model_variant} "
            f"(overrides requested {requested_model_variant})"
        )
    return checkpoint_model_variant


def run_training(
    experiment_name: str,
    model_variant: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    wd: float,
    scheduler: str,
    t0: int,
    t_mult: int,
    eta_min: float,
    warmup_epochs: int,
    warmup_start_factor: float,
    grad_clip: float,
    resume: bool,
    train_workers: int = 4,
    val_workers: int = 2,
    upload_checkpoints: bool = True,
):
    """Run training with the given hyperparameters."""
    print("=== Train Evaluation ===")
    print(f"[✓] Using torch version: {torch.__version__}")

    train_data_path = TRAIN_DATA_PATH
    validation_data_path = VALIDATION_DATA_PATH
    model_checkpoint_path = checkpoint_path(experiment_name)
    best_checkpoint_path = checkpoint_path(f"{experiment_name}-best")
    model_variant = _resolve_model_variant_for_training(
        model_checkpoint_path,
        model_variant,
        resume,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[✓] Using device: {device}")
    print(f"[✓] Experiment: {experiment_name}")
    print(f"[✓] Model variant: {model_variant}")
    print(
        f"[✓] Hyperparameters: epochs={epochs}, steps={steps_per_epoch}, batch={batch_size}"
    )
    print(f"[✓] Optimizer: lr={lr}, wd={wd}")
    if scheduler == SCHEDULER_WARM_RESTART:
        print(
            f"[✓] Scheduler: warm-restart (T0={t0}, T_mult={t_mult}, eta_min={eta_min})"
        )
    elif scheduler == SCHEDULER_WARMUP_COSINE:
        print(
            "[✓] Scheduler: warmup-cosine "
            f"(warmup_epochs={warmup_epochs}, "
            f"warmup_start_factor={warmup_start_factor}, eta_min={eta_min})"
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}")
    print(f"[✓] Grad clip: {grad_clip}")
    print(
        f"[✓] DataLoader workers: train={train_workers}, val={val_workers} | Upload checkpoints: {upload_checkpoints}"
    )

    model = ValueOnlyModel(model_variant=model_variant)
    trainer = Trainer(
        model=model,
        device=device,
        experiment_name=experiment_name,
        model_variant=model_variant,
        lr=lr,
        wd=wd,
        scheduler_name=scheduler,
        t0=t0,
        t_mult=t_mult,
        eta_min=eta_min,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        grad_clip=grad_clip,
        upload_checkpoints=upload_checkpoints,
    )

    if resume:
        trainer.load_checkpoint(model_checkpoint_path)

    train_data = ChessEvaluationDataset(str(train_data_path))
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=train_workers > 0,
        worker_init_fn=worker_init_fn if train_workers > 0 else None,
    )

    validation_data = ChessEvaluationDataset(str(validation_data_path))
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
        worker_init_fn=worker_init_fn if val_workers > 0 else None,
    )

    print(f"[✓] Data loaded from {train_data_path} ({len(train_data)} rows)")
    print(f"[✓] Data loaded from {validation_data_path} ({len(validation_data)} rows)")

    trainer.train(
        checkpoint_path=model_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    print("[✓] Training completed")
