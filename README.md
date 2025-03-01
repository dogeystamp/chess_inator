# chess-inator

A chess engine built from scratch, powered by a neural network.

This engine is trained on games played with prior versions of itself, and
master level games from Lichess. Notably, chess-inator does not learn using
analysis from existing engines like Stockfish; it learns entirely on its own,
scoring positions with prior versions of itself.

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

- NNUE evaluation
- Mailbox and redundant bitboard representation
    - Simple pseudo-legal move generation
- Make/unmake
- Negamax search
    - Alpha-beta pruning
    - Principal Variation Search (PVS)
    - Killer move heuristic
    - Null-move pruning
    - Late move reductions
    - Quiescence search
    - Check extension
- Iterative deepening
    - Time management (soft, hard limit)
- Transposition table (Zobrist hashed)
    - Best move
    - Evaluation
    - Static evaluation
    - Depth
- UCI compatibility

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

## acknowledgements

This project would not have been possible without the following:

- [Chess Programming Wiki](https://www.chessprogramming.org/Main_Page): the source of a lot of algorithms used in chess-inator
- [fastchess](https://github.com/Disservin/fastchess): the main tool used to test this engine
- [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/): used in the training pipeline
- [Stockfish opening books](https://github.com/official-stockfish/books/): used in testing and training
- [Stockfish NNUE docs](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md): very helpful in understanding NNUE
- [Bullet NNUE docs](https://github.com/jw1912/bullet/blob/main/docs/1-basics.md): helpful in understanding NNUE
- [PyTorch](https://pytorch.org/): network training framework
- [Rust](https://www.rust-lang.org/): great language

## training process / history

The engine's neural network is trained on chess positions labelled with:

- board state
- a centipawn evaluation given by a prior version of the engine
- the real outcome (win/draw/loss) of the game

The real outcome is interpolated with the engine evaluation to give an
"expected evaluation" for the position, on which the engine trains.
By labelling the position with the real game outcome, the engine gets
feedback on positions that are good and bad.

<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Tag</th>
            <th>Description</th>
            <th>Notes</th>
        </tr>
    </thead>
    <tbody>
<tr>
<td>Generation 1</td>
<td>

`hce`

</td>
<td>

Hand-crafted evaluation. Has material counting and very simple (i.e. I punched in numbers)
piece-square table evaluation.

</td>
<td>No data available.</td>
</tr>



<tr>
<td>Generation 2</td>
<td>

`nnue2`

</td>
<td>

First neural network. Trained on 
the [Lichess elite database](https://database.nikonoel.fr/),
October 2024. Positions were scored using gen 1's evaluation.

</td>
<td>No data available.</td>
</tr>


<tr>
<td>Generation 3</td>
<td>

`nnue3-192`

</td>
<td>

Hidden layer size increased from 16 to 192 neurons.
Trained on Lichess elite database, September 2024,
using gen 2's evaluation.

</td>
<td>No data available.</td>
</tr>


<tr>
<td>Generation 4</td>
<td>

`nnue4-320`

</td>
<td>

Hidden layer size increased from to 320 neurons.
Trained on Lichess elite database, June & July 2024,
using gen 3's evaluation. Used around 18 million
positions for training.

</td>
<td>

```
vs c_i pvs12-5 (06d195b)
nElo: 56.67 +/- 25.56
Wins: 371, Losses: 260, Draws: 79
```

</td>
</tr>


<tr>
<td>Generation 5</td>
<td>

`nnue05a-320`

</td>
<td>

Fine-tuned gen 4 on 3 million self-play positions.

</td>
<td>

```
vs c_i hash-non-two (2c4a38f)
nElo: 34.67 +/- 19.56
Wins: 574, Losses: 463, Draws: 175
```

</td>
</tr>


<tr>
<td>Generation 6</td>
<td>

`nnue06a-320`

</td>
<td>

Fine-tuned gen 5 on 2 million self-play positions.

</td>
<td>

```
vs c_i check-handling3 (ef178a3)
nElo: 32.75 +/- 18.97
Wins: 596, Losses: 486, Draws: 206
```

</td>
</tr>


<tr>
<td>Generation 7b</td>
<td>

`nnue07b-512`

</td>
<td>

Network hidden layer increased from 320 to 512 neurons; gen 6's trained weights
were transferred to the new net. Trained on almost 5 million self-play positions.

Training seemed to stall and always overfit without the hidden layer size increase,
which is why 7a was discarded.

</td>
<td>

```
vs c_i avoid-rep-new (16aebf8)
nElo: 51.63 +/- 24.45
Wins: 371, Losses: 269, Draws: 136
```

</td>
</tr>


<tr>
<td>Generation 7c</td>
<td>

`nnue07c-512`

</td>
<td>

Identical to gen 7b (i.e. trained on the same data starting from the same gen 6
net), but trained with vertical mirroring as a data augmentation strategy.

Discarded in favour of 7d (vertical mirroring isn't as useful since perspective
nets are very common.)

</td>
<td>

```
vs c_i nnue07d-512 (a26a5d7)
nElo: -39.38 +/- 27.04
Wins: 226, Losses: 288, Draws: 120
```

</td>
</tr>


<tr>
<td>Generation 7d</td>
<td>

`nnue07d-512`

</td>
<td>

Another sibling of 7b, but trained with horizontal mirroring as a data
augmentation strategy.

</td>
<td>

```
vs c_i nnue07b-512 (e10a0c8)
nElo: 40.44 +/- 21.38
Wins: 475, Losses: 373, Draws: 166
```

</td>
</tr>


<tr>
<td>Generation 8a</td>
<td>

`nnue08a-512`

</td>
<td>

Fine-tuned gen 7d on ~2.4M self-play positions. Does not seem to gain much in
terms of Elo, possibly because of using a dataset half the size as gen 7.

</td>
<td>

```
vs c_i eval-saturating (9b3074e)
nElo: 22.46 +/- 15.13
Wins: 867, Losses: 753, Draws: 406
```

</td>
</tr>


<tr>
<td>Generation 8b</td>
<td>

N/A

</td>
<td>

Trained like 8a, but with 768 neurons. This overfitted faster than 8a, so it
has been discarded.

</td>
<td>

No data.

</td>
</tr>

    </tbody>
</table>
