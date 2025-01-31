# chess-inator

A chess engine built from scratch, powered by a neural network.

This engine is trained on master level games from Lichess. Notably,
chess-inator does not use analysis from existing engines like Stockfish; it
learns entirely on its own, scoring positions with prior versions of itself.

The engine is trained with little pre-existing knowledge of chess.
Specifically, chess-inator started off knowing:

- The rules of chess
- The traditional piece values (pawn = 1, bishop, knight = 3, rook = 6, queen = 9)

See the "training process" section below for more information.

To play against chess-inator, see its [Lichess](https://lichess.org/@/chess_inator_bot) page.
Note that it may be offline for long periods of time, since I do not
permanently run the engine on a server. Alternatively, run it locally, as
described in the "development instructions" section.

## features

These are some technical details about the features implemented in the engine.

- Mailbox and redundant bitboard representation
    - Naive pseudo-legal move generation
- Make/unmake
- Negamax search
    - Alpha-beta pruning
    - Principal Variation Search (PVS)
    - Killer move heuristic
- NNUE evaluation
- UCI compatibility
- Iterative deepening
    - Time management (soft, hard limit)
- Transposition table (Zobrist hashing)
    - Age/depth replacement scheme
- Quiescence search
- Check extension

At runtime, chess-inator has zero dependencies other than the Rust standard library.

## neural network architecture

chess-inator has a relatively simple neural network architecture, consisting of
the following:

- "ALL" input feature, multi-hot tensor (768 neurons)
    - Each combination of piece (6), color (2), and square (64) has a neuron.
- Hidden layer / accumulator (N neurons)
    - This layer is fully connected to the last.
    - Clipped ReLU activation (i.e. clamp between 0, 1)
- Output neuron (1 neuron)
    - Sigmoid to get a "WDL space" evaluation (0 is a black win, 1 is a white win, 0.5 is a draw).
    - A scaling factor is applied so that the logits (raw values before the
      sigmoid) correspond roughly to centipawns.

This architecture is known as an NNUE (efficiently updateable neural network),
since we only have to store the accumulator values, and every move made or
unmade can incrementally update these values (i.e. we don't have to do a
complete forward pass for every move).

For efficiency reasons, the network is also quantized to `int16` after
training. This provides a considerable speed-up compared to `float64`, which is
used during training.

## training process / history

The engine's neural network is trained on chess positions labelled with:

- board state
- a centipawn evaluation given by a prior version of the engine
- the real outcome (win/draw/loss) of the game

The real outcome is interpolated with the engine evaluation to give an
"expected evaluation" for the position, and that the engine trains on.
By labelling the position with the real game outcome, the engine gets
feedback on positions that are good and bad.

Here is a log of the machine learning used to train the engine, and the
source of the data being used. See the [NNUE readme](./nnue/README.md) for
technical details about the training pipeline.

- **Generation 1** (branch `hce`): based on a naive hand-crafted evaluation. It
  implements material counting, as well as a very simple (i.e. I punched in
  numbers myself) piece-square table evaluation.
- **Generation 2:** (tag `nnue2`) is neural-network based, and is trained on
  positions from the [Lichess elite database](https://database.nikonoel.fr/),
  October 2024. These positions were scored using gen 1's evaluation.
- **Generation 3:** (tag `nnue3-192`) increases the hidden layer size from 16
  to 192 neurons. It is trained on Lichess elite database positions (September
  2024), scored by gen 2's evaluation.
- **Generation 4:** (tag `nnue4-320`) has 320 hidden layer neurons, and is
  trained from gen 3's evaluation of around 18 million Lichess elite
  database positions (June & July 2024).

## development instructions

The following are instructions to run the engine locally (on a development
device). The engine **does not implement a GUI**, you need separate software
for that. For instance, try [CuteChess](https://github.com/cutechess/cutechess)
or [Arena](http://www.playwitharena.de/).

For development purposes, [fast-chess](https://github.com/disservin/fastchess),
a CLI interface, is used for running tournaments against different versions of
the engine. See `contrib/fast-chess-tag.sh` for help using it.

(For neural net weights) [Set up `git-lfs`.](https://graphite.dev/guides/how-to-use-git-large-file-storage-lfs)

Clone the repo:

    git clone https://github.com/dogeystamp/chess_inator

To run the engine (in UCI mode):

    cargo run --release

Quick unit tests:

    cargo test

Longer duration, more rigorous tests:

    cargo test --release

Flamegraph (on perft):

    export CARGO_PROFILE_RELEASE_DEBUG true
    cargo flamegraph --test perft
