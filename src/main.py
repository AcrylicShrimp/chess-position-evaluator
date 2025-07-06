import os
import torch
from model import Model
from chess_dataset import ChessDataset, worker_init_fn
from train import train


def main():
    train_data_path = "train.chesseval"
    checkpoint_path = "model.pth"
    batch_size = 128
    epochs = 10
    steps_per_epoch = 512

    model = Model()

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"[✓] Loaded model from {checkpoint_path}")

    train_data = ChessDataset(
        train_data_path,
    )
    data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    print(f"[✓] Data loaded from {train_data_path}")

    while True:
        train(
            model,
            data_loader,
            checkpoint_path,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
