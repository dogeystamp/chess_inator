#!/usr/bin/env python

"""
Processes PGN game data into a tsv format suitable for training.
Inputs from stdin, outputs to stdout.

Output columns:
- FEN (for reference)
- ALL 768-bit binary string representing the position
- Evaluation (centipawns) from white perspective
- Result of the game (-1, 0, 1)

This script depends on the `chess` package.
Install it, or run this script using `pipx run process_pgn_data.py`.
The script also depends on the chess_inator engine for analysis and filtering.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "chess",
# ]
# ///

import argparse
from asyncio import Queue, TaskGroup, create_task, run, sleep
import logging
import datetime
import multiprocessing
import csv

import chess
import chess.engine
from typing import AsyncIterator, Literal
from chess import pgn
from sys import stdin, stdout
from pathlib import Path

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--engine",
    help="Set the file path of the chess_inator engine used to analyze the positions.",
    type=Path,
)
parser.add_argument(
    "--max-workers",
    help="Max concurrent workers to analyse games with (limit this to your hardware thread count).",
    default=min(4, multiprocessing.cpu_count()),
    type=int,
)
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)


"""Skip these many plies from the start (avoid training on opening)."""
SKIP_PLIES: int = 20

"""Time limit in seconds for each position to be analyzed."""
TIME_LIMIT: float = 5


output_queue: Queue[tuple[str, str, int, Literal[-1, 0, 1]]] = Queue()


async def load_games():
    """Load a PGN file and divide up the games for the workers to process."""
    while game := pgn.read_game(stdin):
        yield game


async def worker(game_generator: AsyncIterator[pgn.Game]) -> None:
    """
    Single worker that analyzes whole games.

    Code pattern taken from https://stackoverflow.com/a/54975674.

    Puts rows of output into a global queue.
    """
    transport, engine = await chess.engine.popen_uci(args.engine)
    await engine.configure(dict(NNUETrainInfo="true"))

    async for game in game_generator:
        wdl: int | None = None

        match game.headers["Result"]:
            case "1-0":
                wdl = 1
            case "0-1":
                wdl = -1
            case "1/2-1/2":
                wdl = 0
            case other_result:
                logging.error("invalid 'Result' header: '%s'", other_result)
                continue

        board = game.board()

        skipped = 0

        logging.info("Processing game %s, %s (%s) between %s as White and %s as Black.", game.headers["Event"], game.headers["Site"], game.headers["Date"], game.headers["White"], game.headers["Black"])

        for move in game.mainline_moves():
            board.push(move)
            if skipped < SKIP_PLIES:
                skipped += 1
                continue
            result = await engine.play(
                board,
                chess.engine.Limit(time=TIME_LIMIT),
                info=chess.engine.INFO_ALL,
                game=game,
            )

            info_str = result.info.get("string")
            if not info_str:
                raise RuntimeError("Could not analyze position with engine.")
            (name, quiet, eval_abs, tensor) = info_str.split()
            if not name == "NNUETrainInfo":
                raise RuntimeError(f"Unexpected output from engine: {info_str}")

            if quiet == "non-quiet":
                logging.debug("discarded as non-quiet: '%s'", board.fen())
                continue
            elif quiet != "quiet":
                raise RuntimeError(f"Unexpected output from engine: {info_str}")

            await output_queue.put((board.fen(), tensor, int(eval_abs), wdl))


async def analyse_games():
    """Task that manages reading PGNs and analyzing them."""
    games_generator = load_games()

    async with TaskGroup() as tg:
        worker_count: int = min(args.max_workers, multiprocessing.cpu_count())
        logging.info("Using %d concurrent worker tasks.", worker_count)
        for i in range(worker_count):
            tg.create_task(worker(games_generator))


completed = 0
start_time = datetime.datetime.now()


async def output_rows():
    """TSV writer task."""

    writer = csv.writer(stdout, delimiter="\t")
    while True:
        row = await output_queue.get()
        writer.writerow(row)
        stdout.flush()
        output_queue.task_done()
        global completed
        completed += 1


async def status_logger():
    """Periodically print status."""
    while True:
        await sleep(5)
        logging.info(
            "Completed %d rows in %f seconds.",
            completed,
            (datetime.datetime.now() - start_time).total_seconds(),
        )


async def main():
    analyse_task = create_task(analyse_games())
    output_task = create_task(output_rows())
    status_task = create_task(status_logger())

    await analyse_task
    output_task.cancel()
    status_task.cancel()


run(main())
