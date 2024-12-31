#!/usr/bin/env python

"""Train the NNUE weights."""

import torch
import pandas as pd
import numpy as np
import logging
import argparse

from torch.utils.data import Dataset, DataLoader
from torch import nn
from pathlib import Path
from scipy.optimize import curve_fit


logging.basicConfig(level=logging.INFO)


################################
################################
## CLI argument parsing
################################
################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "datafile",
    type=Path,
    help="Path to load training data from (expects .tsv.gz format).",
    default=Path("combined_training.tsv.gz"),
)
parser.add_argument(
    "--save",
    type=Path,
    help="Path to save trained model to.",
    default=Path("trained_weights.pth"),
)
parser.add_argument(
    "--load",
    type=Path,
    help="Path to load trained model from.",
    default=Path("trained_weights.pth"),
)


################################
################################
## Data loading / parsing
################################
################################


LAMBDA = 0.4
"""
Interpolation coefficient between expected win probability, and real win probability.

0 discards the engine label completely, and 1 discards the real game result completely.
"""


def convert_str_to_ndarray(x: str):
    """Convert board feature string to numpy ndarray."""
    arr = np.empty(INPUT_SIZE)
    for i, c in enumerate(x):
        match c:
            case "1":
                arr[i] = 1
            case "0":
                arr[i] = 0
            case _:
                raise ValueError(f"Invalid character in board features '{c}'.")
    return arr


class ChessPositionDataset(Dataset):
    def __init__(self, data_file: Path):
        self.data = pd.read_csv(data_file, delimiter="\t")
        self.data.columns = ["fen", "board_features", "centipawns", "game_result"]

        # convert from (-1, 0, 1) to WDL-space (0, 0.5, 1)
        self.data["game_result"] = (self.data["game_result"] + 1) / 2

        # convert features to tensors
        self.data["board_features"] = self.data["board_features"].apply(
            convert_str_to_ndarray
        )

        # tune sigmoid
        self.k = tune_sigmoid(self.data["centipawns"], self.data["game_result"])

        # interpolate engine analysis and real result in WDL-space
        self.data["expected_result"] = (LAMBDA) * np_sigmoid(
            self.data["centipawns"], self.k
        ) + (1 - LAMBDA) * self.data["game_result"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def plot_sigmoid(self):
        """Display the curve that correlates centipawns to win-draw-loss."""
        import matplotlib.pyplot as plt

        plt.plot(
            self.data["centipawns"], self.data["game_result"], "o", label="Real result"
        )
        plt.plot(
            self.data["centipawns"],
            self.data["expected_result"],
            "o",
            label="Interpolated result",
        )

        x = np.linspace(min(self.data["centipawns"]), max(self.data["centipawns"]))
        y = np_sigmoid(x, self.k)
        plt.plot(x, y, label="Sigmoid")

        plt.legend()
        plt.xlabel("Centipawn evaluation")
        plt.ylabel("Win-Draw-Loss evaluation")

        plt.show()


def collate_chess_positions(data):
    """Combine multiple examples within a batch."""
    return pd.concat(data)


################################
## sigmoid parameter tuning
################################


def np_sigmoid(x, k):
    return 1 / (1 + np.exp(-x / k))


def tune_sigmoid(cp, wdl) -> np.double:
    """
    Fit a sigmoid to correlate centipawns to win-draw-loss.

    Torch's sigmoid is

        sigma(x) = 1/(1 + exp(-x))

    and we will simply add a parameter K:

        sigma(x) = 1/(1 + exp(-x/K)).

    This is the only parameter needed, since sigma(0) = 0.5 (no advantage is a draw),
    and the range is [0, 1], which is what we need for win-draw-loss.

    Arguments
    ---------
    cp: Column with centipawn evaluations.
    wdl: Column with actual game result (WDL [0, 1]).

    Returns
    -------
    Parameter K of the sigmoid.
    """

    popt, pcov = curve_fit(np_sigmoid, cp, wdl, [100], method="dogbox")
    return popt[0]


################################
################################
## neural net architecture
################################
################################

INPUT_SIZE = 2 * 6 * 64  # 768
"""Board feature input size."""

HIDDEN_SIZE = 3
"""Hidden layer size."""


class NNUE(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.k = k
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1, dtype=torch.double),
        )

    def forward(self, x):
        logit = self.linear_relu_stack(x)
        # i _think_ these logits can be interpreted as centipawns?

        # return WDL space probability
        return torch.sigmoid(logit / self.k)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

################################
################################
## neural net training
################################
################################

LEARN_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 5

def get_x_y_from_batch(batch):
    """Returns training input (X) and label (Y) for a given batch."""
    x = torch.from_numpy(np.stack(batch["board_features"].values))
    y = torch.from_numpy(np.stack(batch["expected_result"].values)).unsqueeze(-1)

    return x, y

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        x, y = get_x_y_from_batch(batch)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_set_size = len(dataloader.dataset)

        if batch_idx % 10 == 0:
            loss = loss.item()
            current = batch_idx * BATCH_SIZE + len(x)
            print(f"loss: {loss:>7f} [{current:>5d} / {train_set_size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    n_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = get_x_y_from_batch(batch)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()

    test_loss /= n_batches
    print(f"avg loss: {test_loss:>5f}\n")


def train(
    full_dataset: ChessPositionDataset,
    model: NNUE,
    save_path: Path | None,
    load_path: Path | None,
):
    """
    Train the model's parameters.
    """

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [0.8, 0.2]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_chess_positions,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_chess_positions,
    )

    logging.info("Using device %s for training", device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    if load_path and load_path.is_file():
        checkpoint = torch.load(load_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch_idx in range(EPOCHS):
        print(f"\nEPOCH {epoch_idx + 1} / {EPOCHS}\n---------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    if save_path:
        torch.save(
            dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            ),
            save_path,
        )


################################
## main entry point
################################

if __name__ == "__main__":
    args = parser.parse_args()

    full_dataset = ChessPositionDataset(args.datafile)
    model = NNUE(full_dataset.k)

    train(full_dataset, model, args.save, args.load)
