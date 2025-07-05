import torch
from livelossplot import PlotLosses
from model import Model
from prepare_train_data import TrainData
from train import train


def main():
    model = Model()
    loss_plot = PlotLosses(figsize=(10, 5))

    while True:
        print(f"[✓] Loading data...")
        train_data = TrainData(
            "lichess_db_eval.parquet",
            percentage=0.001,
        )

        train(
            model,
            train_data,
            "model.pth",
            epochs=10,
            steps_per_epoch=512,
            batch_size=128,
            loss_plot=loss_plot,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
