import os
import torch
from chess_dataset import ChessDataset, worker_init_fn
from model import Model
from train import Trainer
from torch.utils.tensorboard import SummaryWriter


def main():
    print("=== Train Evaluation ===")
    print(f"[✓] Using torch version: {torch.__version__}")

    train_data_path = "train.chesseval"
    validation_data_path = "validation.chesseval"
    checkpoint_path = "model.pth"
    best_checkpoint_path = "model-best.pth"
    tensorboard_path = "tensorboard/chess-ai"
    batch_size = 4096
    epochs = 100000
    steps_per_epoch = 4096

    if os.environ.get("CHECKPOINT_PATH"):
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if os.environ.get("BEST_CHECKPOINT_PATH"):
        best_checkpoint_path = os.environ.get("BEST_CHECKPOINT_PATH")
    if os.environ.get("TENSORBOARD_PATH"):
        tensorboard_path = os.environ.get("TENSORBOARD_PATH")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[✓] Using device: {device}")

    model = Model()
    trainer = Trainer(model, device)
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

    with SummaryWriter(tensorboard_path) as writer:
        print(f"[✓] Tensorboard writer initialized at {tensorboard_path}")

        trainer.train(
            checkpoint_path,
            best_checkpoint_path,
            train_data_loader,
            validation_data_loader,
            writer,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

    print(f"[✓] Training completed")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
