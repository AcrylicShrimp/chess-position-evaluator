import os
import torch
from model import Model
from chess_dataset import ChessDataset, worker_init_fn
from train import Trainer


def main():
    train_data_path = "train.chesseval"
    validation_data_path = "validation.chesseval"
    checkpoint_path = "model.pth"
    best_checkpoint_path = "best_model.pth"
    batch_size = 512
    epochs = 100
    steps_per_epoch = 4096

    if os.environ.get("CHECKPOINT_PATH"):
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if os.environ.get("BEST_CHECKPOINT_PATH"):
        best_checkpoint_path = os.environ.get("BEST_CHECKPOINT_PATH")

    model = Model()
    trainer = Trainer(model)
    trainer.load_checkpoint(checkpoint_path)

    train_data = ChessDataset(
        train_data_path,
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    validation_data = ChessDataset(
        validation_data_path,
    )
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

    while True:
        trainer.train(
            checkpoint_path,
            best_checkpoint_path,
            train_data_loader,
            validation_data_loader,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
