# chess-inator

A chess engine.

Features:
- Negamax search
- Alpha-beta pruning
- NNUE (neural network) based evaluation
- UCI compatibility
- Iterative deepening
    - Time management
- Transposition table (Zobrist hashing)
- Quiescence search

## instructions

To run the engine (in UCI mode):

    cargo run --release

Quick unit tests:

    cargo test

Longer duration, more rigorous tests:

    cargo test --release

Flamegraph (on perft):

    export CARGO_PROFILE_RELEASE_DEBUG true
    cargo flamegraph --test perft
