import itertools
import os
import signal

import torch
import wandb
from tqdm import tqdm

from libs.model import EvalOnlyModel


class Trainer:
    def __init__(
        self,
        model: EvalOnlyModel,
        device: torch.device,
        experiment_name: str,
        lr: float,
        wd: float,
        t0: int,
        t_mult: int,
        eta_min: float,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        grad_clip: float,
        upload_checkpoints: bool,
    ):
        self.model = model
        self.device = device
        self.experiment_name = experiment_name
        self.upload_checkpoints = upload_checkpoints
        self.enable_autocast = device.type != "cpu"
        self.autocast_dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported(False)
            else torch.float16 if device.type != "cpu" else None
        )
        self.enable_grad_scaler = (
            device.type == "cuda" and not torch.cuda.is_bf16_supported(False)
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
        )
        self.grad_scaler = torch.amp.GradScaler(
            enabled=self.enable_grad_scaler)

        self.model.to(self.device)

        if self.device.type == "cuda":
            self.model.compile(mode="reduce-overhead")
            print("[✓] Model compiled (cuda)")
        else:
            print("[!] Model will not be compiled; no cuda device found")

        self.should_stop = False
        signal.signal(signal.SIGINT, self.signal_handler)

        self.best_validation_loss = float("inf")
        self.epoch = 0

        # Initialize WandB
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT"),
            name=experiment_name,
            config={
                "lr": lr,
                "wd": wd,
                "scheduler": "CosineAnnealingWarmRestarts",
                "scheduler_t0": t0,
                "scheduler_t_mult": t_mult,
                "scheduler_eta_min": eta_min,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "batch_size": batch_size,
                "grad_clip": grad_clip,
            },
        )
        self.grad_clip = grad_clip

    def signal_handler(self, signum, frame):
        self.should_stop = True

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            return

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.grad_scaler.load_state_dict(checkpoint["grad_scaler"])

        if "best_validation_loss" in checkpoint:
            self.best_validation_loss = checkpoint["best_validation_loss"]

        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]

        print(f"[✓] Loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path: str, epoch: int):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "grad_scaler": self.grad_scaler.state_dict(),
                "best_validation_loss": self.best_validation_loss,
                "epoch": epoch,
            },
            checkpoint_path,
        )

    def _upload_checkpoint(self, checkpoint_path: str, aliases: list[str]):
        if not self.upload_checkpoints:
            return

        if not os.path.exists(checkpoint_path):
            return

        artifact = wandb.Artifact(
            name=f"{self.experiment_name}-checkpoint",
            type="model",
            description="Training checkpoint",
        )
        artifact.add_file(
            checkpoint_path, name=os.path.basename(checkpoint_path))
        wandb.log_artifact(artifact, aliases=aliases)

    def train(
        self,
        checkpoint_path: str,
        best_checkpoint_path: str,
        train_data_loader: torch.utils.data.DataLoader,
        validation_data_loader: torch.utils.data.DataLoader,
        epochs: int,
        steps_per_epoch: int,
    ):
        torch.set_float32_matmul_precision("medium")

        try:
            for epoch in range(self.epoch, epochs):
                self.model.train()

                loss_acc = 0.0
                pbar: tqdm[tuple[torch.Tensor, torch.Tensor]] = tqdm(
                    itertools.islice(train_data_loader, steps_per_epoch),
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    unit="step",
                )

                for step, (input, label) in enumerate(pbar):
                    input = input.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.enable_autocast,
                        dtype=self.autocast_dtype,
                    ):
                        output = self.model(input)
                        loss = compute_loss(output, label)

                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip
                    )

                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                    loss = loss.item()
                    loss_acc += loss
                    avg_loss = loss_acc / (step + 1)

                    global_step = epoch * steps_per_epoch + step
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/avg_loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                    pbar.set_postfix(loss=avg_loss)

                    if self.should_stop:
                        self.save_checkpoint(checkpoint_path, epoch)
                        del train_data_loader._iterator
                        del validation_data_loader._iterator
                        wandb.finish()
                        print("[!] Stopping training (Ctrl+C pressed)")
                        os._exit(0)

                self.save_checkpoint(checkpoint_path, epoch)
                print(
                    f"[✓] Epoch {epoch + 1} completed — Final Loss: {avg_loss:.4f}")

                self.model.eval()

                validation_loss_acc = 0.0
                validation_steps = max(steps_per_epoch // 10, 1)

                for step, (input, label) in enumerate(
                    itertools.islice(validation_data_loader, validation_steps)
                ):
                    with torch.no_grad():
                        input = input.to(self.device)
                        label = label.to(self.device)

                        output = self.model(input)
                        loss = compute_loss(output, label)

                        loss = loss.item()
                        validation_loss_acc += loss

                avg_validation_loss = validation_loss_acc / validation_steps

                wandb.log(
                    {
                        "val/loss": avg_validation_loss,
                        "val/best_loss": self.best_validation_loss,
                    },
                    step=(epoch + 1) * steps_per_epoch,
                )

                print(f"[✓] Validation Loss: {avg_validation_loss:.4f}")

                # Step cosine scheduler per epoch
                self.scheduler.step(epoch + 1)

                if avg_validation_loss < self.best_validation_loss:
                    self.best_validation_loss = avg_validation_loss
                    self.save_checkpoint(best_checkpoint_path, epoch + 1)
                    self._upload_checkpoint(
                        best_checkpoint_path,
                        aliases=[
                            "best",
                            f"epoch-{epoch + 1}",
                        ],
                    )
                    print(
                        f"[🎉] New best validation loss: {self.best_validation_loss:.4f} (saved to {best_checkpoint_path})"
                    )
        finally:
            wandb.finish()


def compute_loss(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(output.float(), label.float())
