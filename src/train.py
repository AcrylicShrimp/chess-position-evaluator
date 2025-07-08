import itertools
import os
import signal
import torch
from livelossplot import PlotLosses
from model import Model
from tqdm import tqdm


class Trainer:
    def __init__(self, model: Model, device: torch.device):
        self.model = model
        self.device = device
        self.enable_amp = device.type != "cpu"

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
        self.grad_scaler = torch.amp.GradScaler()

        self.model.to(self.device)

        if self.device.type == "cuda":
            self.model.compile()
            print(f"[✓] Model compiled (cuda)")
        else:
            print("[!] Model will not be compiled; no cuda device found")

        self.should_stop = False
        signal.signal(signal.SIGINT, self.signal_handler)

        self.best_validation_loss = float("inf")

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

        print(f"[✓] Loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "grad_scaler": self.grad_scaler.state_dict(),
                "best_validation_loss": self.best_validation_loss,
            },
            checkpoint_path,
        )

    def train(
        self,
        checkpoint_path: str,
        best_checkpoint_path: str,
        train_data_loader: torch.utils.data.DataLoader,
        validation_data_loader: torch.utils.data.DataLoader,
        epochs: int,
        steps_per_epoch: int = 512,
        loss_plot: PlotLosses | None = None,
    ):
        torch.set_float32_matmul_precision("high")

        for epoch in range(epochs):
            self.model.train()

            pbar: tqdm[tuple[torch.Tensor, torch.Tensor]] = tqdm(
                itertools.islice(train_data_loader, steps_per_epoch),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="step",
            )
            loss_acc = 0.0
            cp_loss_acc = 0.0
            mate_loss_acc = 0.0

            for step, (input, label) in enumerate(pbar):
                input = input.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type, enabled=self.enable_amp
                ):
                    output = self.model(input)
                    loss, cp_loss, mate_loss = compute_loss(output, label)

                if self.enable_amp:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                loss_acc += loss.item()
                cp_loss_acc += cp_loss.item()
                mate_loss_acc += mate_loss.item()

                avg_loss = loss_acc / (step + 1)
                avg_cp_loss = cp_loss_acc / (step + 1)
                avg_mate_loss = mate_loss_acc / (step + 1)

                pbar.set_postfix(
                    loss=avg_loss,
                    cp_loss=avg_cp_loss,
                    mate_loss=avg_mate_loss,
                )

                if self.should_stop:
                    self.save_checkpoint(checkpoint_path)
                    del train_data_loader._iterator
                    del validation_data_loader._iterator
                    print("[!] Stopping training (Ctrl+C pressed)")
                    exit(0)

            self.save_checkpoint(checkpoint_path)
            print(f"[✓] Epoch {epoch + 1} completed — Final Loss: {avg_loss:.4f}")

            if loss_plot is not None:
                loss_plot.update(
                    {
                        "loss": avg_loss,
                        "cp_loss": avg_cp_loss,
                        "mate_loss": avg_mate_loss,
                    }
                )
                loss_plot.send()

            self.scheduler.step()
            self.model.eval()

            validation_loss_acc = 0.0
            validation_cp_loss_acc = 0.0
            validation_mate_loss_acc = 0.0
            validation_steps = max(steps_per_epoch // 10, 1)

            for input, label in itertools.islice(
                validation_data_loader, validation_steps
            ):
                with torch.no_grad():
                    input = input.to(self.device)
                    label = label.to(self.device)

                    output = self.model(input)
                    loss, cp_loss, mate_loss = compute_loss(output, label)

                    validation_loss_acc += loss.item()
                    validation_cp_loss_acc += cp_loss.item()
                    validation_mate_loss_acc += mate_loss.item()

            avg_validation_loss = validation_loss_acc / validation_steps
            avg_validation_cp_loss = validation_cp_loss_acc / validation_steps
            avg_validation_mate_loss = validation_mate_loss_acc / validation_steps

            print(
                f"[✓] Validation Loss: {avg_validation_loss:.4f} — CP Loss: {avg_validation_cp_loss:.4f} — Mate Loss: {avg_validation_mate_loss:.4f}"
            )

            if avg_validation_loss < self.best_validation_loss:
                self.best_validation_loss = avg_validation_loss
                self.save_checkpoint(best_checkpoint_path)
                print(
                    f"[🎉] New best validation loss: {self.best_validation_loss:.4f} (saved to {best_checkpoint_path})"
                )


def compute_loss(
    output: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_output = output[:, 0]
    mate_output = output[:, 1:]

    cp_label = labels[:, 0]
    mate_label = labels[:, 1].long()

    cp_loss = torch.nn.functional.mse_loss(cp_output, cp_label)
    mate_loss = torch.nn.functional.cross_entropy(mate_output, mate_label)
    loss = cp_loss + 0.5 * mate_loss

    return loss, cp_loss, mate_loss
