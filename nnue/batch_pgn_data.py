#!/usr/bin/env python

"""
Batch PGN data into files, since the training data pipeline can't resume processing within a single file.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "chess",
# ]
# ///

from typing import Iterator
import chess.pgn
import argparse
import itertools

from pathlib import Path

"""Games to include per file in output."""

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", type=Path)
parser.add_argument("--batch-size", type=int, help="Number of games to save in each output file.", default=8)
parser.add_argument("--output-folder", type=Path, help="Folder to save batched games in.", default=Path("batches"))
args = parser.parse_args()

def generate_games_in_file(path: Path) -> Iterator[chess.pgn.Game]:
    """Read games from a single PGN file."""
    with open(path) as f:
        while game := chess.pgn.read_game(f):
            game.headers["PGNPath"] = str(path)
            yield game

def generate_games() -> Iterator[chess.pgn.Game]:
    """Read games from all files."""
    for path in args.files:
        yield from generate_games_in_file(path)

def batch_games():
    """Write games in batches."""
    output_folder: Path = args.output_folder
    output_folder.mkdir(exist_ok=True)
    for idx, batch in enumerate(itertools.batched(generate_games(), args.batch_size)):
        with (output_folder / f"batch{idx:04}.pgn").open("w") as f:
            for game in batch:
                f.write(str(game) + "\n")

batch_games()
