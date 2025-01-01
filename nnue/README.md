# NNUE training tools

Python training pipeline for the evaluation neural network.
See the docstring in `src/nnue.rs` for information about the architecture of the NNUE.
The network is trained on both self-play games, and its games on Lichess.
Both of these sources provide games in PGN format.

Required packages:
- pandas
- numpy
- torch
- scipy
- (optional) matplotlib

This folder includes the following scripts:
- `s1_batch_pgn_data.py`: Combine and convert big PGN files into small chunked files.
    - The batches in this context are different from batches in gradient descent.
    - Batches should ideally be a few times bigger than the number of workers
      you have. Big batches have less overhead, but you lose more progress when
      interrupting the processing step.
- `s2_process_pgn_data.py`: Convert PGN data into a format suitable for training.
- `s3_train_neural_net.py`: Train neural network weights based on the data.
    - This will output a PyTorch `.pth` weights file, which preserves the
      training state. However, it can not be read in the engine itself.
- `s4_weights_to_bin.py`: Convert the `.pth` weights into a `.bin` file.
    - This `.bin` format is readable by the engine.

All above scripts have options. To view the options, run the script with the
`--help` flag.

Example training pipeline:
```bash
# chunk all the PGN files in `games/`. outputs by default to `batches/batch%d.pgn`.
./s1_batch_pgn_data.py --batch-size 32 games/*.pgn

# analyze batches to turn them into training data. outputs by default to
# train_data/batch%d.tsv.gz. set max-workers to the number of hardware threads /
# cores you have. this is the longest part. you may interrupt and resume this
# process, at the cost of possibly discarding a partially done  batch.
./s2_process_pgn_data.py --engine ../target/release/chess_inator --max-workers 8 batches/batch*.pgn

# combine all processed data into a single training set file.
zcat train_data/*.tsv.gz | gzip > combined_training.tsv.gz

# optimize a neural network, saving weights (by default) to `weights.pth`.
# this process may be interrupted and resumed, at the cost of losing an
# unfinished epoch.
./s3_train_neural_net.py combined_training.tsv.gz --log log_training.csv

# convert the finished weights to `.bin` format
./s4_weights_to_bin.py weights.pth
cp weights.bin ../src/
```
