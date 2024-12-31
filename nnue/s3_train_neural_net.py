#!/usr/bin/env python

"""Train the NNUE weights."""

import torch
import pandas as pd
import numpy as np
import logging

from torch.utils.data import Dataset, DataLoader
from torch import nn
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import curve_fit


logging.basicConfig(level=logging.INFO)


################################
################################
## Data loading / parsing
################################
################################


class ChessPositionDataset(Dataset):
    def __init__(self, data_file: Path):
        self.data = pd.read_csv(data_file, delimiter="\t")
        self.data.columns = ["fen", "board_features", "centipawns", "game_result"]

        # convert from (-1, 0, 1) to WDL-space (0, 0.5, 1)
        self.data["game_result"] = (self.data["game_result"] + 1) / 2

        # tune sigmoid
        self.k = tune_sigmoid(self.data["centipawns"], self.data["game_result"])

        # convert features to tensors
        self.data["board_features"] = self.data["board_features"].apply(
            lambda x: torch.as_tensor([1 if c == "1" else 0 for c in x])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def plot_sigmoid(self):
        """Display the curve that correlates centipawns to win-draw-loss."""
        import matplotlib.pyplot as plt
        plt.plot(self.data["centipawns"], self.data["game_result"], "o")
        x = np.linspace(min(self.data["centipawns"]), max(self.data["centipawns"]))

        def np_sigmoid(x, k):
            return 1 / (1 + np.exp(-x/k))

        y = np_sigmoid(x, self.k)
        plt.plot(x, y)


################################
## sigmoid parameter tuning
################################


def sigmoid_series(x, k):
    """
    Sigmoid for series values.

    Prefer `torch.sigmoid` for tensor values.
    """
    return torch.sigmoid(torch.tensor(x / k))


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

    popt, pcov = curve_fit(sigmoid_series, cp, wdl, [100], method="dogbox")
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

    def forward(self, x: pd.DataFrame):
        features = x["board_features"]
        output = self.linear_relu_stack(features)
        return output


if __name__ == "__main__":
    full_dataset = ChessPositionDataset(Path("combined_training.tsv.gz"))

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [0.8, 0.2]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    logging.info("Using device %s for training", device)
