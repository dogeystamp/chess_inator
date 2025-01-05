/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Perft verification on known positions.
//!
//! See https://www.chessprogramming.org/Perft

use chess_inator::fen::{FromFen, START_POSITION};
use chess_inator::movegen::perft;
use chess_inator::Board;

#[test]
fn test_perft() {
    // https://www.chessprogramming.org/Perft_Results
    let test_cases = [
        (
            // fen
            START_POSITION,
            // expected perft values
            vec![1, 20, 400, 8_902, 197_281, 4_865_609, 119_060_324],
            // limit depth when not under `cargo test --release` (unoptimized build too slow)
            3,
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            vec![1, 48, 2_039, 97_862, 4_085_603],
            2,
        ),
        (
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            vec![1, 14, 191, 2_812, 43_238, 674_624, 11_030_083],
            3,
        ),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            vec![1, 6, 264, 9467, 422_333, 15_833_292],
            2,
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            vec![1, 44, 1_486, 62_379, 2_103_487, 89_941_194],
            2,
        ),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            vec![1, 46, 2_079, 89_890, 3_894_594],
            2,
        ),
    ];
    for (fen, expected_values, _debug_limit_depth) in test_cases {
        let mut pos = Board::from_fen(fen).unwrap();

        for (depth, expected) in expected_values.iter().enumerate() {
            eprintln!("running perft depth {depth} on position '{fen}'");
            #[cfg(debug_assertions)]
            {
                if depth > _debug_limit_depth {
                    break;
                }
            }
            assert_eq!(perft(depth, &mut pos), *expected,);
        }
    }
}
