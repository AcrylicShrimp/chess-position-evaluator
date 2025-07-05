import torch
from livelossplot import PlotLosses
from model import Model
from prepare_train_data import TrainData, worker_init_fn
from train import train


def main():
    model = Model()
    loss_plot = PlotLosses(figsize=(10, 5))

    while True:
        print(f"[✓] Loading data...")
        train_data = TrainData(
            "lichess_db_eval.duckdb",
        )
        data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=128,
            pin_memory=True,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )

        train(
            model,
            data_loader,
            "model.pth",
            epochs=10,
            steps_per_epoch=512,
            loss_plot=loss_plot,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
