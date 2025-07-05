import itertools
import torch
from livelossplot import PlotLosses
from model import Model
from tqdm import tqdm


def train(
    model: Model,
    data_loader: torch.utils.data.DataLoader,
    checkpoint_path: str,
    epochs: int,
    steps_per_epoch: int = 512,
    loss_plot: PlotLosses | None = None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[✓] Using device: {device}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    grad_scaler = torch.amp.GradScaler()
    enable_amp = device.type != "cpu"

    for epoch in range(epochs):
        pbar = tqdm(
            itertools.islice(data_loader, steps_per_epoch),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="step",
        )
        loss_acc = 0.0
        cp_loss_acc = 0.0
        mate_loss_acc = 0.0

        for step, (input, label) in enumerate(pbar):
            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=enable_amp):
                output = model(input)
                loss, cp_loss, mate_loss = compute_loss(output, label)

            if enable_amp:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

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

        torch.save(model.state_dict(), checkpoint_path)
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
