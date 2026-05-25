import math

import torch


SCHEDULER_WARM_RESTART = "warm-restart"
SCHEDULER_WARMUP_COSINE = "warmup-cosine"
SUPPORTED_SCHEDULERS = (SCHEDULER_WARM_RESTART, SCHEDULER_WARMUP_COSINE)


class EpochWarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        eta_min: float,
        warmup_start_factor: float,
    ):
        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be greater than or equal to zero")
        if warmup_epochs > total_epochs:
            raise ValueError("warmup_epochs cannot exceed total_epochs")
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in the range (0, 1]")
        if eta_min < 0.0:
            raise ValueError("eta_min must be greater than or equal to zero")

        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = 0

        for base_lr in self.base_lrs:
            if eta_min > base_lr:
                raise ValueError("eta_min cannot exceed an optimizer base learning rate")

        self._set_lrs(self._compute_lrs(self.last_epoch))

    def state_dict(self) -> dict[str, object]:
        return {
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "eta_min": self.eta_min,
            "warmup_start_factor": self.warmup_start_factor,
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.total_epochs = int(state_dict["total_epochs"])
        self.warmup_epochs = int(state_dict["warmup_epochs"])
        self.eta_min = float(state_dict["eta_min"])
        self.warmup_start_factor = float(state_dict["warmup_start_factor"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.last_epoch = int(state_dict["last_epoch"])
        self._set_lrs(self._compute_lrs(self.last_epoch))

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self._set_lrs(self._compute_lrs(self.last_epoch))

    def _compute_lrs(self, epoch: int) -> list[float]:
        epoch = min(max(epoch, 0), self.total_epochs - 1)

        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return [self._warmup_lr(base_lr, epoch) for base_lr in self.base_lrs]

        return [self._cosine_lr(base_lr, epoch) for base_lr in self.base_lrs]

    def _warmup_lr(self, base_lr: float, epoch: int) -> float:
        if self.warmup_epochs == 1:
            return base_lr

        progress = epoch / (self.warmup_epochs - 1)
        factor = self.warmup_start_factor + (
            1.0 - self.warmup_start_factor
        ) * progress
        return base_lr * factor

    def _cosine_lr(self, base_lr: float, epoch: int) -> float:
        if self.total_epochs == 1:
            return base_lr

        if self.warmup_epochs == 0:
            progress = epoch / (self.total_epochs - 1)
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs
            if decay_epochs <= 0:
                return base_lr
            progress = (epoch - self.warmup_epochs + 1) / decay_epochs

        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eta_min + (base_lr - self.eta_min) * cosine

    def _set_lrs(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            group["lr"] = lr


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    *,
    t0: int,
    t_mult: int,
    eta_min: float,
    epochs: int,
    warmup_epochs: int,
    warmup_start_factor: float,
):
    if scheduler_name == SCHEDULER_WARM_RESTART:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
        )

    if scheduler_name == SCHEDULER_WARMUP_COSINE:
        return EpochWarmupCosineScheduler(
            optimizer=optimizer,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            eta_min=eta_min,
            warmup_start_factor=warmup_start_factor,
        )

    allowed = ", ".join(SUPPORTED_SCHEDULERS)
    raise ValueError(f"Unsupported scheduler {scheduler_name!r}; expected one of: {allowed}")
