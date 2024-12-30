# NNUE training tools

Python training pipeline for the evaluation neural network.
See the docstring in `src/nnue.rs` for information about the architecture of the NNUE.
The network is trained on both self-play games, and its games on Lichess.
Both of these sources provide games in PGN format.

This folder includes the following scripts:
- `batch_pgn_data.py`: Combine and convert big PGN files into small chunked files.
- `process_pgn_data.py`: Convert PGN data into a format suitable for training.

Example training pipeline:
```bash
# chunk all the PGN files in `games/`. outputs by default to `batches/batch%d.pgn`.
./batch_pgn_data.py games/*.pgn

# analyze batches 0 to 20 to turn them into training data. outputs by default to train_data/batch%d.tsv.gz.
# set max-workers to the number of hardware threads / cores you have.
./process_pgn_data.py --engine ../target/release/chess_inator --max-workers 8 batches/batch{0..20}.pgn
```
