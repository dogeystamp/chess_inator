# chess-inator

A chess engine.

Features:
- Negamax search
- Alpha-beta pruning
- Piece-square tables
    - Tapered midgame-endgame evaluation
- UCI compatibility
- Iterative deepening

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
