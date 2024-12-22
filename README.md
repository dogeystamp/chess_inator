Branch: small transposition table entries

Seemingly, having smaller transposition table entries could increase performance.
But actually, looking at the length of the PV is misleading because transposition table hits may cut the PV short.



# chess-inator

A chess engine.

Features:
- Negamax search
- Alpha-beta pruning
- Piece-square tables
    - Tapered midgame-endgame evaluation
- UCI compatibility
- Iterative deepening
- Transposition table (Zobrist hashing)
    - Currently only stores best move.

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
