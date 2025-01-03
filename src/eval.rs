/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Static position evaluation.

use crate::prelude::*;
use core::cmp::max;

/// Signed centipawn type.
///
/// Positive is good for White, negative good for Black.
pub type EvalInt = i32;

pub trait Eval {
    /// Evaluate a position and assign it a score.
    ///
    /// Negative for Black advantage and positive for White.
    fn eval(&self) -> EvalInt;
}

pub trait EvalSEE {
    /// Evaluate the outcome of an exchange at a square (static exchange evaluation).
    ///
    /// # Arguments
    ///
    /// * dest: Square where the exchange happens.
    /// * first_move_side: Side to move first in the exchange.
    ///
    /// This function may panic if a piece already at the destination is the same color as the side
    /// to move.
    ///
    /// # Returns
    ///
    /// Expected gain from this exchange.
    fn eval_see(&self, dest: Square, first_move_side: Color) -> EvalInt;
}

impl Eval for Board {
    fn eval(&self) -> EvalInt {
        self.nnue.output()
    }
}

impl EvalSEE for Board {
    fn eval_see(&self, dest: Square, first_mv_side: Color) -> EvalInt {
        let attackers = self.gen_attackers(dest, false, None);

        // indexed by the Piece enum order
        let mut atk_qty = [[0u8; N_PIECES]; N_COLORS];

        // counting sort
        for (attacker, _src) in attackers {
            atk_qty[attacker.col as usize][attacker.pc as usize] += 1;
        }

        let dest_pc = self.get_piece(dest);

        // it doesn't make sense if the piece already on the square is first to move
        debug_assert!(!dest_pc.is_some_and(|pc| pc.col == first_mv_side));

        // Simulate the exchange.
        //
        // Returns the expected gain for the side in the exchange.
        //
        // TODO: promotions aren't accounted for.
        fn sim_exchange(
            side: Color,
            dest_pc: Option<ColPiece>,
            atk_qty: &mut [[u8; N_PIECES]; N_COLORS],
        ) -> EvalInt {
            use Piece::*;
            let val_idxs = [Pawn, Knight, Bishop, Rook, Queen, King];

            let mut ptr = 0;
            let mut eval = 0;

            // while the count of this piece is zero, move to the next piece
            while atk_qty[side as usize][val_idxs[ptr] as usize] == 0 {
                ptr += 1;
                if ptr == N_PIECES {
                    return eval;
                }
            }
            let cur_pc = val_idxs[ptr];
            let pc_ptr = cur_pc as usize;

            debug_assert!(atk_qty[side as usize][pc_ptr] > 0);
            atk_qty[side as usize][pc_ptr] -= 1;

            if let Some(dest_pc) = dest_pc {
                eval += dest_pc.pc.value();
                // this player may either give up now, or capture. pick the best (max score).
                // anything the other player gains is taken from us, hence the minus.
                eval = max(0, eval - sim_exchange(side.flip(), Some(dest_pc), atk_qty))
            }

            eval
        }

        sim_exchange(first_mv_side, dest_pc, &mut atk_qty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::FromFen;

    /// Sanity check.
    #[test]
    fn test_eval() {
        let board1 = Board::from_fen("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1").unwrap();
        let eval1 = board1.eval();
        let board2 = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/8/4K3 w kq - 0 1").unwrap();
        let eval2 = board2.eval();

        assert!(eval1 > 0, "got eval {eval1}");
        assert!(eval2 < 0, "got eval {eval2}");
    }

    /// Static exchange evaluation tests.
    #[test]
    fn test_see_eval() {
        // set side to move appropriately in the fen
        //
        // otherwise the exchange doesn't work
        use Piece::*;
        let test_cases = [
            (
                // fen
                "8/4n3/8/2qRr3/8/4N3/8/8 b - - 0 1",
                // square where exchange happens
                "d5",
                // expected (signed) value gain of exchange
                Rook.value(),
            ),
            ("8/8/4b3/2kq4/2PKP3/8/8/8 w - - 0 1", "d5", Queen.value()),
            (
                "r3k2r/1pq2pbp/6p1/p2Qpb2/1N6/2P3P1/PB2PPBP/R3K2R w KQkq - 0 14",
                "e5",
                0,
            ),
            (
                "r3k2r/1pq2pbp/6p1/p3p3/P5b1/2P3P1/1BN1PPBP/R2QK2R b KQkq - 0 14",
                "e2",
                0,
            ),
        ];

        for (fen, dest, expected) in test_cases {
            let board = Board::from_fen(fen).unwrap();
            let dest: Square = dest.parse().unwrap();
            let res = board.eval_see(dest, board.turn);
            assert_eq!(res, expected, "failed {}", fen);
        }
    }
}
