#!/usr/bin/env python

# This file is part of chess_inator.
# chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
#
# chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.
#
# Copyright © 2024 dogeystamp <dogeystamp@disroot.org>

"""
Train the NNUE weights.

Due to memory concerns, the training data is split into individual datasets
(each has `BIG_BATCH_SIZE` rows). Each of these datasets has its train/test,
and each dataset will be used for `EPOCHS` epochs.
"""

import itertools
import math
from typing import Iterable
import torch
import pandas as pd
import numpy as np
import logging
import argparse
import csv
import gzip

from torch.utils.data import Dataset, DataLoader
from torch import nn
from pathlib import Path
from scipy.optimize import curve_fit
from tqdm import tqdm
import gc


logging.basicConfig(level=logging.INFO)
pd.options.mode.copy_on_write = True

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

NUM_WORKERS = 0
"""Separate workers to use for loading training data."""

################################
## hyperparameters
################################

LAMBDA = 0.92
"""
(Default) interpolation coefficient between expected win probability, and real win probability.

To set this, use the `--lambda` command-line flag.

0 discards the engine label completely, and 1 discards the real game result
completely. Using a naive material-counting engine in the analysis, this default
value should work. For a smarter engine, use a higher value, like 0.97, so that
it can go through reinforcement learning based on its existing knowledge.
"""

LEARN_RATE = 1e-3
BATCH_SIZE = 16384
BIG_BATCH_SIZE = 2**22

EPOCHS = 7

# neural net architecture

HIDDEN_SIZE = 320
"""Hidden layer size."""

INPUT_SIZE = 2 * 6 * 64  # 768
"""Board feature input size."""

ARCHITECTURE = f"A07_CReLU_{INPUT_SIZE}_N_1_K"
"""
Unique ID / version for this architecture.

This serves as a compatibility indicator for the Rust engine code.
"""

ARCHITECTURE_SPECIFIC = f"A07_CReLU_{INPUT_SIZE}_{HIDDEN_SIZE}_1_K"
"""
Architecture string, including specifics like the hidden layer size.

This helps the Python training code determine compatibility of weight files.
"""

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
    default=Path(f"weights_{ARCHITECTURE_SPECIFIC}.pth"),
)
parser.add_argument(
    "--load",
    type=Path,
    help="Path to load trained model from.",
    default=Path(f"weights_{ARCHITECTURE_SPECIFIC}.pth"),
)
parser.add_argument(
    "--force-load",
    action="store_true",
    help="Load models even if their architecture information is incompatible.",
)
parser.add_argument(
    "--force-epochs",
    action="store_true",
    help="Sets the epoch counter on a loaded checkpoint to 0. Useful for fine-tuning using existing weights.",
)
parser.add_argument(
    "--log",
    type=Path,
    help="Path to log (as .csv) the results of the loss function.",
)
parser.add_argument(
    "--lambda",
    type=float,
    help="Interpolation coefficient. 0.0 uses the engine label, and 1.0 uses the game result as a label.",
)


################################
################################
## Data loading / parsing
################################
################################


class StrToTensor(nn.Module):
    """
    Multi-hot bitstring to sparse tensor conversion.

    Takes in a string comprised of '0' and '1', and returns the indices where the string is '1'.
    """

    def forward(self, x: str):
        with torch.no_grad():
            arr = np.frombuffer(bytearray(x, "utf-8"), np.int8) - ord("0")
            arr.setflags(write=True)

            # u8 can't fit indices up to 768
            # by default, a wasteful int64 might be used
            indices = np.astype(np.nonzero(arr)[0], np.uint16)

            return indices


class MvToDevice(nn.Module):
    """Moves a tensor to the GPU (or whatever device training is done on)."""

    def forward(self, x: torch.Tensor):
        return x.to(device=device)


class FromNumpy(nn.Module):
    def forward(self, x: np.ndarray):
        return torch.from_numpy(x)


class ConvertSparse(nn.Module):
    """
    Convert our binary sparse tensor to a Torch dense tensor.

    The reason why we can't directly use the Torch sparse type is that it stores redundant values,
    while we know that in our tensors all values are '1'.
    """

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            return (
                torch.sparse_coo_tensor(
                    x.unsqueeze(0),
                    torch.ones(x.shape[0], device=device),
                    device=device,
                    size=[INPUT_SIZE],
                )
                .to_dense()
                .to(dtype=torch.double)
            )


str_to_tensor = StrToTensor()
convert_sparse = ConvertSparse()


def count_total_rows(data_file: Path):
    """Count total rows in a .tsv.gz file."""
    with gzip.open(data_file, "rb") as f:
        t = tqdm(unit=" rows", desc="COUNT ROWS", delay=0.5, unit_scale=True)

        BUF_SIZE = 1024 * 1024
        buf = f.read(BUF_SIZE)
        while buf:
            t.update(buf.count(b"\n"))
            buf = f.read(BUF_SIZE)
        return t.n


class ChessPositionDataset(Dataset):
    """
    Single big batch.

    Raises `pandas.errors.EmptyDataError` if there is no data at this batch.
    """

    def __chunked_reader(self) -> pd.DataFrame:
        """
        Read data in chunks, and incrementally compress it in memory.

        Important steps done are:
        - using a sparse board feature representation
        - downsizing to smaller numeric types (e.g. int32, float16)

        This function automatically reads `BIG_BATCH_SIZE` rows,
        starting from the `cur_big_batch`.

        Returns
        -------
        Pandas dataframe, or raises `pandas.errors.EmptyDataError` if
        there is no more data.
        """

        output = []

        CHUNK_SIZE = 65536

        for chunk in tqdm(
            pd.read_csv(
                self.data_file,
                delimiter="\t",
                chunksize=CHUNK_SIZE,
                skiprows=self.cur_big_batch * BIG_BATCH_SIZE,
                nrows=BIG_BATCH_SIZE,
            ),
            desc="READ BIG BATCH",
            total=math.ceil(BIG_BATCH_SIZE / CHUNK_SIZE),
            unit="chunk",
            postfix=dict(chk_sz=CHUNK_SIZE),
        ):
            chunk: pd.DataFrame

            chunk.columns = ["fen", "board_features", "centipawns", "game_result"]

            # save on memory
            del chunk["fen"]

            chunk["centipawns"] = chunk["centipawns"].astype(np.int32, copy=False)

            # convert from (-1, 0, 1) to WDL-space (0, 0.5, 1)
            chunk["game_result"] = ((chunk["game_result"] + 1) / 2).astype(
                np.float32, copy=False
            )

            # convert features
            chunk["board_features"] = chunk["board_features"].apply(str_to_tensor)

            output += chunk.to_dict(orient="records")

            # https://stackoverflow.com/a/49144260
            # magical incantation to summon the garbage collector
            del chunk
            gc.collect()
            chunk = pd.DataFrame()  # noqa: F841

        df = pd.DataFrame(output)
        return df

    def __init__(self, data_file: Path, cur_big_batch: int):
        self.total_rows = 0
        self.data_file = data_file

        self.cur_big_batch = cur_big_batch
        """Index of the big batch to load."""

        self.data = self.__chunked_reader()

        # tune sigmoid
        # self.k = tune_sigmoid(self.data["centipawns"], self.data["game_result"])
        self.k = np.double(400.0)

        tqdm.pandas(desc="CONVERT NP -> TORCH")
        self.data["board_features"] = self.data["board_features"].progress_apply(
            FromNumpy()
        )

        if device != "cpu":
            tqdm.pandas(desc="SENDING TO DEVICE")
            self.data["board_features"] = self.data["board_features"].progress_apply(
                MvToDevice()
            )

        # interpolate engine analysis and real result in WDL-space
        self.data["expected_result"] = (LAMBDA) * np_sigmoid(
            self.data["centipawns"], self.k
        ) + (1 - LAMBDA) * self.data["game_result"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            convert_sparse(row["board_features"]),
            torch.tensor(
                row["expected_result"], dtype=torch.double, device=device
            ).unsqueeze(-1),
        )

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

        min_x = min(self.data["centipawns"])
        max_x = max(self.data["centipawns"])

        x = np.linspace(min_x, max_x)
        y = np_sigmoid(x, self.k)
        plt.plot(x, y, label="Sigmoid")

        plt.plot([0, 0], [0, 1], linestyle="dashed", color="gray")
        plt.plot([min_x, max_x], [0.5, 0.5], linestyle="dashed", color="gray")

        plt.legend()
        plt.grid()
        plt.title(f"WDL-CP sigmoid (k={self.k:.2f}, n={len(self.data)})")
        plt.xlabel("Centipawn evaluation")
        plt.ylabel("Win-Draw-Loss evaluation")

        plt.show()


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


class CReLU(nn.Module):
    """Clamped ReLU."""

    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 1)


class NNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k: np.double | None = None
        self.arch = ARCHITECTURE
        self.arch_specific = ARCHITECTURE_SPECIFIC
        self.l1_size = HIDDEN_SIZE

        with torch.no_grad():
            # initialize model to a simple piece value evaluation
            l1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, dtype=torch.double, device=device)
            out = nn.Linear(HIDDEN_SIZE, 1, dtype=torch.double, device=device)

            l1_params = list(l1.parameters())
            out_params = list(out.parameters())

            def pc_value(sign, pc_idx):
                return torch.flatten(
                    torch.tensor(
                        [
                            [
                                [
                                    m * pc_val / np.double(10) + torch.rand(1) * 0.001
                                    for _ in range(64)
                                ]
                                if i == pc_idx
                                else [0.001 * torch.rand(1) for _ in range(64)]
                                for i, pc_val in enumerate((5, 3, 3, 0.001, 9, 1))
                            ]
                            for m in sign
                        ],
                        dtype=torch.double,
                        device=device,
                    )
                )

            # by default the output layer shouldn't have big parameters
            out_params[0].data *= 0.001
            out_params[1].data *= 0.001

            # white pieces
            for i in range(6):
                l1_params[0].data[i] = pc_value((1.0, 0.001), i)
                # weight assumes k = 400. should be 1000 / k.
                out_params[0].data[0][i] = torch.tensor(
                    2.5, dtype=torch.double, device=device
                )
            # black pieces
            for i in range(6):
                l1_params[0].data[i + 6] = pc_value((0.001, 1.0), i)
                out_params[0].data[0][i + 6] = torch.tensor(
                    -2.5, dtype=torch.double, device=device
                )
            # bias
            for i in range(12):
                l1_params[1].data[i] = torch.randn(1, device=device) * 0.001

        self.linear_relu_stack = nn.Sequential(l1, CReLU(), out)

    def forward(self, x):
        logit = self.linear_relu_stack(x)
        # this logit, times k, gives a "centipawn" evaluation

        # return WDL space probability
        return torch.sigmoid(logit)


################################
################################
## neural net training
################################
################################


def train_loop(dataloader, model, loss_fn, optimizer) -> np.double:
    """
    Train model for one epoch.

    Returns
    -------
    Average train loss for the epoch.
    """

    model.train()

    avg_loss = np.double(0)

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc="TRAIN LOOP",
            unit="batch",
            postfix=dict(batch_sz=BATCH_SIZE),
        )
    ):
        x, y = batch
        pred = model(x)
        loss = loss_fn(pred, y)
        avg_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 32 == 0:
            loss = loss.item()

    avg_loss /= len(dataloader)
    return avg_loss


def test_loop(dataloader, model, loss_fn) -> np.double:
    """
    Test the model after one epoch.

    Returns
    -------
    Average test/validation loss.
    """
    model.eval()
    n_batches = len(dataloader)
    test_loss = np.double(0)

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=" TEST LOOP",
            unit="batch",
            postfix=dict(batch_sz=BATCH_SIZE),
        ):
            x, y = batch
            pred = model(x)
            test_loss += loss_fn(pred, y).item()

    test_loss /= n_batches
    return test_loss


class BigBatchLoader:
    """
    Load big batches from the .tsv.gz.

    Arguments
    ---------
    start: Start at this big batch index.
    """

    def __init__(self, data_file: Path, start: int):
        self.total_rows = count_total_rows(data_file)
        self.data_file = data_file
        self.start = start

    def big_batches(self) -> Iterable[tuple[int, ChessPositionDataset]]:
        try:
            for i in itertools.count(start=self.start):
                yield i, ChessPositionDataset(self.data_file, i)
        except pd.errors.EmptyDataError:
            pass

    def __len__(self) -> int:
        return math.ceil(self.total_rows / BIG_BATCH_SIZE)


def train(
    model: NNUE,
    save_path: Path | None = None,
    load_path: Path | None = None,
    log_path: Path | None = None,
):
    """
    Train the model's parameters.
    """

    logging.info("Using device '%s' for training.", device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=1.0, total_iters=8
    )

    big_epoch_start, epoch_start = 0, 0

    if load_path and load_path.is_file():
        logging.info("Loading saved model from '%s'...", load_path)
        saved_epoch = load_model(load_path, model, optimizer, scheduler)
        if saved_epoch is not None:
            last_big_epoch, last_epoch = divmod(saved_epoch, EPOCHS)
            big_epoch_start, epoch_start = divmod(saved_epoch+1, EPOCHS)
            logging.info(
                "Last session ended with big batch %d, epoch %d.",
                last_big_epoch + 1,
                last_epoch + 1,
            )

    big_batch_loader = BigBatchLoader(args.datafile, big_epoch_start)
    for big_batch_idx, big_batch in big_batch_loader.big_batches():
        logging.info("Loaded big batch dataset (%d rows).", len(big_batch))
        model.k = big_batch.k

        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = torch.utils.data.random_split(
            big_batch, [0.97, 0.03], generator=generator
        )

        train_dataloader = DataLoader(
            train_dataset,
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        for epoch_idx in range(epoch_start, EPOCHS):
            print(
                f"\nBIG BATCH {big_batch_idx + 1}/{len(big_batch_loader)}, EPOCH {epoch_idx + 1} / {EPOCHS}\n---------------------------------"
            )
            train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loss = test_loop(test_dataloader, model, loss_fn)

            scheduler.step()

            total_epoch = big_batch_idx * EPOCHS + epoch_idx

            print(f"\navg TRAIN loss: {train_loss:>5f}")
            print(f"avg  TEST loss: {test_loss:>5f}\n")
            if save_path:
                save_model(
                    save_path,
                    model,
                    optimizer,
                    scheduler,
                    total_epoch,
                )
                logging.info("Saved progress to '%s'.", save_path)
            if log_path:
                if not log_path.exists():
                    with log_path.open("w") as f:
                        writer = csv.writer(f)
                        writer.writerow(["epoch", "train_loss", "test_loss"])
                with log_path.open("a") as f:
                    writer = csv.writer(f)
                    writer.writerow([total_epoch, train_loss, test_loss])

        epoch_start = 0


def visualize_train_log(log_path: Path = Path("log_training.csv")):
    """Visualize the training loss log."""

    import pandas as pd
    import matplotlib.pyplot as plt

    df: pd.DataFrame = pd.read_csv(log_path)

    end_epoch = max(df["epoch"])
    df = df.set_index("epoch")

    print(df)

    max_v = max(df["test_loss"].values.max(), df["train_loss"].values.max()) # type: ignore

    for i in range(0, end_epoch + 1, EPOCHS):
        plt.plot([i, i], [0, max_v], linestyle="dashed", color="gray")

    plt.plot(df.index, df["train_loss"], label="Train loss")
    plt.plot(df.index, df["test_loss"], label="Test loss")
    plt.ylabel("Loss")
    plt.title("Training progress")
    plt.grid()
    plt.legend()
    plt.show()


################################
################################
## saving/loading models
################################
################################


def load_model(
    load_path: Path,
    model: NNUE,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> int | None:
    """
    Load a model checkpoint.

    Returns
    -------
    Epoch number, if saved.
    """

    checkpoint = torch.load(load_path, weights_only=True, map_location=device)
    if arch := checkpoint.get("arch"):
        if arch != model.arch:
            if "args" in globals() and args.force_load:
                logging.warning(
                    "Force-loading arch '%s', but was expecting '%s'.", arch, model.arch
                )
            else:
                raise ValueError(
                    f"Tried to load from arch '{arch}', but was expecting '{model.arch}'. There is a version mismatch."
                )
    if arch_s := checkpoint.get("arch_specific"):
        if arch_s != model.arch_specific:
            if "args" in globals() and args.force_load:
                logging.warning(
                    "Force-loading specific arch '%s', but was expecting '%s'.",
                    arch_s,
                    model.arch_specific,
                )
            else:
                raise ValueError(
                    f"Tried to load from specific arch '{arch_s}', but was expecting '{model.arch_specific}'. There is a version mismatch."
                )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.k = model.k or checkpoint["k"].detach().numpy().item()
    if not model.k:
        raise ValueError("Missing the sigmoid K parameter.")
    if optimizer:
        if state := checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(state)
    if scheduler:
        if state := checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(state)
    model.l1_size = checkpoint.get("l1_size") or model.l1_size
    if "args" in globals() and args.force_epochs:
        return None
    else:
        return checkpoint.get("epoch")


def save_model(
    save_path: Path,
    model: NNUE,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int | None,
):
    """Save a model as a checkpoint."""

    optim_state = optimizer.state_dict() if optimizer else None
    scheduler_state = scheduler.state_dict() if scheduler else None

    torch.save(
        dict(
            model_state_dict=model.state_dict(),
            scheduler_state_dict=scheduler_state,
            optimizer_state_dict=optim_state,
            epoch=epoch,
            k=torch.as_tensor(model.k),
            arch=model.arch,
            arch_specific=model.arch_specific,
            l1_size=HIDDEN_SIZE,
        ),
        save_path,
    )


################################
## main entry point
################################

if __name__ == "__main__":
    args = parser.parse_args()

    model = NNUE()
    model.to(device)
    train(model, args.save, args.load, args.log)
