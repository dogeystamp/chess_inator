#!/usr/bin/env python

"""
Processes PGN game data into a tsv format suitable for training.

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
import gzip
import csv

import chess
import chess.engine
from typing import AsyncIterator, Literal
from chess import pgn
from pathlib import Path

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    "--log",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    default="INFO",
    help="Sets log level.",
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
parser.add_argument(
    "--preserve-partial",
    action="store_true",
    help="Keep output files that have not been fully written. These files may confuse this script when resuming operations.",
)
parser.add_argument("files", nargs="+", type=Path)
args = parser.parse_args()


logging.basicConfig(level=getattr(logging, str.upper(args.log)))


"""Skip these many plies from the start (avoid training on opening)."""
SKIP_PLIES: int = 20

"""Time limit in seconds for each position to be analyzed."""
TIME_LIMIT: float = 3


output_queue: Queue[tuple[str, str, int, Literal[-1, 0, 1]]] = Queue()


# stats for progress
completed = 0
discarded = 0
current_outp: Path | None = None
start_time = datetime.datetime.now()


async def load_games(file: Path):
    """Load a PGN file and divide up the games for the workers to process."""
    with open(file) as f:
        while game := pgn.read_game(f):
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

        logging.info(
            "Processing game %s, %s (%s) between %s as White and %s as Black.",
            game.headers["Event"],
            game.headers["Site"],
            game.headers["Date"],
            game.headers["White"],
            game.headers["Black"],
        )

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
                global discarded
                discarded += 1
                logging.debug("discarded as non-quiet: '%s'", board.fen())
                continue
            elif quiet != "quiet":
                raise RuntimeError(f"Unexpected output from engine: {info_str}")

            await output_queue.put((board.fen(), tensor, int(eval_abs), wdl))

    await engine.quit()


async def analyse_games(file: Path):
    """Task that manages reading PGNs and analyzing them."""
    games_generator = load_games(file)

    async with TaskGroup() as tg:
        worker_count: int = min(args.max_workers, multiprocessing.cpu_count())
        logging.info("Using %d concurrent worker tasks.", worker_count)
        for i in range(worker_count):
            tg.create_task(worker(games_generator))


async def output_rows(outp_file: Path):
    """TSV writer task."""

    with gzip.open(outp_file, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        while True:
            row = await output_queue.get()
            writer.writerow(row)
            output_queue.task_done()
            global completed
            completed += 1


async def status_logger():
    """Periodically print status."""
    while True:
        await sleep(5)
        logging.info(
            "Completed %d rows in %f seconds. Discarded %d non-quiet positions.",
            completed,
            (datetime.datetime.now() - start_time).total_seconds(),
            discarded,
        )


async def main():
    status_task = create_task(status_logger())

    outp_dir = Path("train_data")
    outp_dir.mkdir(exist_ok=True)

    any_file = False
    skipped = False

    for file in args.files:
        file: Path

        outp_file = outp_dir / file.with_suffix(".tsv.gz").name

        if outp_file.exists():
            skipped = True
            continue

        any_file = True

        if skipped:
            logging.info("Resuming at file '%s'.", file)
            skipped = False
        else:
            logging.info("Reading file '%s'.", file)

        global current_outp
        current_outp = outp_file

        output_task = create_task(output_rows(outp_file))
        analyse_task = create_task(analyse_games(file))
        await analyse_task
        output_task.cancel()

    if not any_file:
        logging.warning("Nothing to do. All input files have outputs already.")

    status_task.cancel()


try:
    run(main())
except KeyboardInterrupt:
    logging.critical("shutting down.")
    if current_outp and not args.preserve_partial:
        logging.critical("discarding partial output file %s", current_outp)
        current_outp.unlink()
