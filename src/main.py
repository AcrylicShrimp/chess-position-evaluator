import torch
from model import Model
from chess_dataset import ChessDataset, worker_init_fn
from train import Trainer


def main():
    train_data_path = "train.chesseval"
    checkpoint_path = "model.pth"
    batch_size = 128
    epochs = 100
    steps_per_epoch = 1024

    model = Model()
    trainer = Trainer(model)
    trainer.load_checkpoint(checkpoint_path)

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
    print(f"[✓] Data loaded from {train_data_path} ({len(train_data)} rows)")

    while True:
        trainer.train(
            checkpoint_path,
            data_loader,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
