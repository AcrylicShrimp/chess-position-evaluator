import itertools
import os
import signal
import torch
from model import Model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: Model,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.enable_amp = device.type != "cpu"

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.grad_scaler = torch.amp.GradScaler()

        self.model.to(self.device)

        if self.device.type == "cuda":
            self.model.compile(mode="reduce-overhead")
            print(f"[✓] Model compiled (cuda)")
        else:
            print("[!] Model will not be compiled; no cuda device found")

        self.should_stop = False
        signal.signal(signal.SIGINT, self.signal_handler)

        self.best_validation_loss = float("inf")
        self.epoch = 0

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

    def train(
        self,
        checkpoint_path: str,
        best_checkpoint_path: str,
        train_data_loader: torch.utils.data.DataLoader,
        validation_data_loader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        epochs: int,
        steps_per_epoch: int = 512,
    ):
        torch.set_float32_matmul_precision("high")

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
                    enabled=self.enable_amp,
                    dtype=torch.bfloat16 if self.device.type == "cuda" else None,
                ):
                    output = self.model(input)
                    loss = compute_loss(output, label)

                if self.enable_amp:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                loss = loss.item()
                loss_acc += loss
                avg_loss = loss_acc / (step + 1)

                global_step = epoch * steps_per_epoch + step
                writer.add_scalar("Loss/train", loss, global_step)

                pbar.set_postfix(loss=avg_loss)

                if self.should_stop:
                    self.save_checkpoint(checkpoint_path, epoch)
                    del train_data_loader._iterator
                    del validation_data_loader._iterator
                    writer.flush()
                    print("[!] Stopping training (Ctrl+C pressed)")
                    os._exit(0)

            self.save_checkpoint(
                checkpoint_path,
                epoch,
            )
            print(f"[✓] Epoch {epoch + 1} completed — Final Loss: {avg_loss:.4f}")

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

                    global_step = epoch * validation_steps + step
                    writer.add_scalar("Loss/validation", loss, global_step)

            avg_validation_loss = validation_loss_acc / validation_steps

            print(f"[✓] Validation Loss: {avg_validation_loss:.4f}")

            self.scheduler.step(avg_validation_loss)

            if avg_validation_loss < self.best_validation_loss:
                self.best_validation_loss = avg_validation_loss
                self.save_checkpoint(best_checkpoint_path, epoch + 1)
                print(
                    f"[🎉] New best validation loss: {self.best_validation_loss:.4f} (saved to {best_checkpoint_path})"
                )

                writer.add_scalar(
                    "Best Validation Loss",
                    self.best_validation_loss,
                    (epoch + 1) * steps_per_epoch,
                )

            writer.flush()


def compute_loss(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(output, label)
