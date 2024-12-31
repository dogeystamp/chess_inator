#!/usr/bin/env python

"""Train the NNUE weights."""

import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass


################################
################################
## Data loading / parsing
################################
################################


@dataclass
class Position:
    """Single board position."""

    fen: str
    """Normal board representation."""

    board: torch.Tensor
    """Multi-hot board representation."""

    cp_eval: np.double
    """Centipawn evaluation (white perspective)."""

    result: np.double
    """
    Game result.

    - -1: black win
    - 0: draw
    - 1: white win
    """

    expected_points: np.double
    """
    Points expected to be gained for white from the game, based on centipawn evaluation.

    - 0: black win
    - 0.5: draw
    - 1: white win
    """


def sigmoid(x):
    """Calculate sigmoid of `x`, using scaling constant `K`."""
    K = 150
    return 1 / (1 + np.exp(-K * x / 400))


class ChessPositionDataset(Dataset):
    def __init__(self, data_file: Path):
        self.data = pd.read_csv(data_file, delimiter="\t")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        eval = np.double(row.iloc[2])
        result=row.iloc[3]

        actual_points = (result + 1) / 2

        return Position(
            fen=row.iloc[0],
            board=torch.as_tensor([1 if c == "1" else 0 for c in row.iloc[1]]),
            cp_eval=eval,
            result=result,
            expected_points=(sigmoid(eval/100) + actual_points)/2,
        )

if __name__ == "__main__":
    full_dataset = ChessPositionDataset(Path("combined_training.tsv.gz"))

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
