# NNUE training tools

Python training pipeline for the evaluation neural network.
See the docstring in `src/nnue.rs` for information about the architecture of the NNUE,
as well as the top-level README.

## pipeline

Required packages:
- pandas
- numpy
- torch
- chess
- tqdm
- (optional) scipy
    - For tuning the sigmoid parameters
- (optional) matplotlib
    - For visualizations and graphing

This folder includes the following scripts:
- `s1_batch_pgn_data.py`: Combine and convert big PGN files into small chunked files.
    - The batches in this context are different from batches in gradient descent.
    - Batches should be many many many times bigger than the number of workers
      you have. Big batches have less overhead, but you lose more progress when
      interrupting the processing step.
    - _This script does not work for huge datasets; UNIX tools like `csplit` work better._
- `s2_process_pgn_data.py`: Analyze PGN data and convert it to a format suitable for training.
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
./s1_batch_pgn_data.py --batch-size 16384 games/*.pgn

# analyze batches to turn them into training data. outputs by default to
# train_data/batch%d.tsv.gz. set max-workers to the number of hardware threads /
# cores you have. you may interrupt and resume this process, at the cost of possibly
# discarding a partially done batch.
./s2_process_pgn_data.py --engine ../target/release/chess_inator --max-workers 8 batches/batch*.pgn
# make a separate test set batch (optional, highly recommended)
./s2_process_pgn_data.py --engine ../target/release/chess_inator --max-workers 8 test_dataset.pgn

# combine all processed data into a single training set file.
# gzip files can be concatenated directly like this.
cat train_data/*.tsv.gz > combined_training.tsv.gz

# optimize a neural network, saving weights (by default) to `weights[...].pth`.
# this process may be interrupted and resumed, at the cost of losing an
# unfinished epoch.
#
# the --test-data flag is optional, but without a separate test dataset there
# probably will be leakage (positions from the same game might be in both
# testing and training data sets)
./s3_train_neural_net.py combined_training.tsv.gz --log log_training.csv --test-data train_data/test_dataset.tsv.gz

# convert the finished weights to `.bin` format
./s4_weights_to_bin.py weights.pth
cp -f weights.bin ../src/
```
