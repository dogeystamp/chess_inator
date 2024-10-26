/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Position evaluation.

use crate::{Board, Color, N_PIECES};

/// Signed centipawn type.
///
/// Positive is good for White, negative good for Black.
type EvalInt = i16;

pub trait Eval {
    /// Evaluate a position and assign it a score.
    fn eval(&self) -> EvalInt;
}

impl Eval for Board {
    fn eval(&self) -> EvalInt {
        use crate::Piece::*;
        let mut score: EvalInt = 0;

        // scores in centipawns for each piece
        let material_score: [EvalInt; N_PIECES] = [
            500,   // rook
            300,   // bishop
            300,   // knight
            20000, // king
            900,   // queen
            100,   // pawn
        ];

        for pc in [Rook, Queen, Pawn, Knight, Bishop, King] {
            let tally_white = self.pl(Color::White).board(pc).0.count_ones();
            let tally_black = self.pl(Color::Black).board(pc).0.count_ones();
            let tally =
                EvalInt::try_from(tally_white).unwrap() - EvalInt::try_from(tally_black).unwrap();

            score += material_score[pc as usize] * tally;
        }

        score
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
}
