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

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", type=Path)
parser.add_argument("--batch-size", type=int, help="Number of games to save in each output file. Set this to two to four times the amount of concurrent workers used in the processing step.", default=8)
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
        with (output_folder / f"batch{idx}.pgn").open("w") as f:
            print(f"Writing batch {idx}...")
            for game in batch:
                f.write(str(game) + "\n\n")

batch_games()
