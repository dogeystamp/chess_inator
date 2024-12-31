#!/usr/bin/env python

"""Train the NNUE weights."""

import torch
import pandas as pd
import numpy as np
import logging

from torch.utils.data import Dataset, DataLoader
from torch import nn
from pathlib import Path
from scipy.optimize import curve_fit


logging.basicConfig(level=logging.INFO)


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


class ChessPositionDataset(Dataset):
    def __init__(self, data_file: Path):
        self.data = pd.read_csv(data_file, delimiter="\t")
        self.data.columns = ["fen", "board_features", "centipawns", "game_result"]

        # convert from (-1, 0, 1) to WDL-space (0, 0.5, 1)
        self.data["game_result"] = (self.data["game_result"] + 1) / 2

        # convert features to tensors
        self.data["board_features"] = self.data["board_features"].apply(
            lambda x: torch.as_tensor([1 if c == "1" else 0 for c in x])
        )

        # tune sigmoid
        self.k = tune_sigmoid(self.data["centipawns"], self.data["game_result"])

        # interpolate engine analysis and real result in WDL-space
        self.data["expected_result"] = (LAMBDA) * np_sigmoid(self.data["centipawns"], self.k) + (1 - LAMBDA) * self.data["game_result"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def plot_sigmoid(self):
        """Display the curve that correlates centipawns to win-draw-loss."""
        import matplotlib.pyplot as plt

        plt.plot(self.data["centipawns"], self.data["game_result"], "o", label="Real result")
        plt.plot(self.data["centipawns"], self.data["expected_result"], "o", label="Interpolated result")

        x = np.linspace(min(self.data["centipawns"]), max(self.data["centipawns"]))
        y = np_sigmoid(x, self.k)
        plt.plot(x, y, label="Sigmoid")

        plt.legend()
        plt.xlabel("Centipawn evaluation")
        plt.ylabel("Win-Draw-Loss evaluation")

        plt.show()


################################
## sigmoid parameter tuning
################################


def np_sigmoid(x, k):
    return 1 / (1 + np.exp(-x/k))


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
    def __init__(self) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        logit = self.linear_relu_stack(x)
        # i _think_ these logits can be interpreted as centipawns?

        # anyways, return WDL space probability
        return torch.sigmoid(logit)

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

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        x = batch["board_features"]
        y = batch["expected_result"]
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_set_size = len(dataloader) 

        if batch % 100 == 0:
            loss = loss.item()
            current = batch_idx * BATCH_SIZE + len(batch)
            print(f"loss: {loss:>7f} [{current:>5d} / [{train_set_size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    n_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["board_features"]
            y = batch["expected_result"]
            pred = model(x)
            test_loss += loss_fn(pred, y).item()

    test_loss /= n_batches
    print(f"avg loss: {test_loss:>5f}\n")

def main():
    full_dataset = ChessPositionDataset(Path("combined_training.tsv.gz"))

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [0.8, 0.2]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    logging.info("Using device %s for training", device)

    model = NNUE()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    for epoch_idx in range(EPOCHS):
        print(f"\nEPOCH {epoch_idx + 1} / {EPOCHS}\n---------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)


if __name__ == "__main__":
    main()
