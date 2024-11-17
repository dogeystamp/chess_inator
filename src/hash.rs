/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Zobrist hash implementation.

use crate::random::{random_arr_2d_64, random_arr_64};
use crate::{
    Board, CastleRights, ColPiece, Color, Square, BOARD_WIDTH, N_COLORS, N_PIECES, N_SQUARES,
};

const PIECE_KEYS: [[[u64; N_SQUARES]; N_PIECES]; N_COLORS] =
    [random_arr_2d_64(11), random_arr_2d_64(22)];

// 4 bits in castle perms -> 16 keys
const CASTLE_KEYS: [u64; 16] = random_arr_64(33);

// ep can be specified by the file
const EP_KEYS: [u64; BOARD_WIDTH] = random_arr_64(44);

// current turn
const COL_KEY: [u64; N_COLORS] = random_arr_64(55);

/// Zobrist hash state.
///
/// This is not synced to board state, so ensure that all changes made are reflected in the hash
/// too.
#[derive(PartialEq, Eq, Clone, Copy, Default, Debug)]
pub(crate) struct Zobrist {
    hash: u64,
}

impl Zobrist {
    /// Toggle a piece.
    pub(crate) fn toggle_pc(&mut self, pc: &ColPiece, sq: &Square) {
        let key = PIECE_KEYS[pc.col as usize][pc.pc as usize][usize::from(sq.0)];
        self.hash ^= key;
    }

    /// Toggle an en-passant target square (only square file is used).
    pub(crate) fn toggle_ep(&mut self, sq: Option<Square>) {
        if let Some(sq) = sq {
            let (_r, c) = sq.to_row_col();
            self.hash ^= EP_KEYS[c];
        }
    }

    /// Toggle castle rights key.
    pub(crate) fn toggle_castle(&mut self, castle: &CastleRights) {
        let bits = ((0x1) & castle.0[0].k as u8)
            | ((0x2) & castle.0[0].q as u8)
            | (0x4) & castle.0[1].k as u8
            | (0x8) & castle.0[1].q as u8;

        self.hash ^= CASTLE_KEYS[bits as usize];
    }

    /// Toggle player to move.
    pub(crate) fn toggle_turn(&mut self, turn: Color) {
        self.hash ^= COL_KEY[turn as usize];
    }

    /// Toggle all of castling rights, en passant and player to move.
    ///
    /// This is done because it's simpler to do this every time at the start and end of a
    /// move/unmove rather than keep track of when castling and ep square and whatever rights
    /// change. Piece moves, unlike this information, have a centralized implementation.
    pub(crate) fn toggle_board_info(pos: &mut Board) {
        pos.zobrist.toggle_ep(pos.ep_square);
        pos.zobrist.toggle_castle(&pos.castle);
        pos.zobrist.toggle_turn(pos.turn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::FromFen;
    use crate::movegen::{FromUCIAlgebraic, Move};

    /// Zobrist hashes of the same positions should be the same. (basic sanity test)
    #[test]
    fn test_zobrist_equality() {
        let test_cases = [
            (
                "4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1",
                "4k2r/8/8/8/8/8/8/2KR4 b k - 1 1",
                "e1c1",
            ),
            (
                "4k2r/8/8/8/8/8/8/R3K3 b Qk - 0 1",
                "5rk1/8/8/8/8/8/8/R3K3 w Q - 1 2",
                "e8g8",
            ),
            (
                "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1",
                "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                "e2e4",
            ),
            (
                "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                "4k3/8/8/8/8/4p3/8/4K3 w - - 0 2",
                "d4e3",
            ),
        ];
        for (pos1_fen, pos2_fen, mv_uci) in test_cases {
            eprintln!("tc: {}", pos1_fen);
            let mut pos1 = Board::from_fen(pos1_fen).unwrap();
            let hash1_orig = pos1.zobrist;
            eprintln!("refreshing board 2 '{}'", pos2_fen);
            let pos2 = Board::from_fen(pos2_fen).unwrap();
            eprintln!("making mv {}", mv_uci);
            let mv = Move::from_uci_algebraic(mv_uci).unwrap();
            let anti_mv = mv.make(&mut pos1);
            assert_eq!(pos1.zobrist, pos2.zobrist);
            anti_mv.unmake(&mut pos1);
            assert_eq!(pos1.zobrist, hash1_orig);
        }
    }
}
