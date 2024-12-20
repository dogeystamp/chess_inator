/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Position evaluation.

use crate::{Board, Color, Piece, Square, N_COLORS, N_PIECES, N_SQUARES};
use core::cmp::{max, min};
use core::ops::Index;

/// Signed centipawn type.
///
/// Positive is good for White, negative good for Black.
pub type EvalInt = i16;

pub trait Eval {
    /// Evaluate a position and assign it a score.
    ///
    /// Negative for Black advantage and positive for White.
    fn eval(&self) -> EvalInt;
}

pub(crate) mod eval_score {
    //! Opaque "score" counters to be used in the board.

    use super::{EvalInt, PST_ENDGAME, PST_MIDGAME};
    use crate::{ColPiece, Square};

    /// Internal score-keeping for a board.
    ///
    /// This is kept in order to efficiently update evaluation with moves.
    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Debug)]
    pub struct EvalScores {
        /// Middle-game perspective evaluation of this board.
        pub midgame: EvalScore,
        /// End-game perspective evaluation of this board.
        pub endgame: EvalScore,
        /// Non-pawn/king piece count, used to determine when the endgame has begun.
        pub min_maj_pieces: u8,
    }

    impl EvalScores {
        /// Add/remove the value of a piece based on the PST.
        ///
        /// Use +1 as sign to add, -1 to delete.
        fn change_piece(&mut self, pc: &ColPiece, sq: &Square, sign: i8) {
            assert!(sign == 1 || sign == -1);
            let tables = [
                (&mut self.midgame, &PST_MIDGAME),
                (&mut self.endgame, &PST_ENDGAME),
            ];
            for (phase, pst) in tables {
                phase.score += pst[pc.pc][pc.col][*sq] * EvalInt::from(pc.col.sign() * sign);
            }

            use crate::Piece::*;
            if matches!(pc.pc, Rook | Queen | Knight | Bishop) {
                match sign {
                    -1 => self.min_maj_pieces -= 1,
                    1 => self.min_maj_pieces += 1,
                    _ => panic!(),
                }
            }
        }

        /// Remove the value of a piece on a square.
        pub fn del_piece(&mut self, pc: &ColPiece, sq: &Square) {
            self.change_piece(pc, sq, -1);
        }

        /// Add the value of a piece on a square.
        pub fn add_piece(&mut self, pc: &ColPiece, sq: &Square) {
            self.change_piece(pc, sq, 1);
        }
    }

    /// Score from a given perspective (e.g. midgame, endgame).
    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Debug)]
    pub struct EvalScore {
        /// Signed score.
        pub score: EvalInt,
    }
}

/// The main piece-square-table (PST) type that assigns scores to pieces on given squares.
///
/// This is the main source of positional knowledge, as well as the ability to count material.
pub struct Pst([PstPiece; N_PIECES]);
/// A PST for a specific piece.
type PstPiece = [PstSide; N_COLORS];
/// A PST for a given piece, of a given color.
type PstSide = [EvalInt; N_SQUARES];

impl Index<Piece> for Pst {
    type Output = PstPiece;

    fn index(&self, index: Piece) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl Index<Color> for PstPiece {
    type Output = PstSide;

    fn index(&self, index: Color) -> &Self::Output {
        &self[index as usize]
    }
}

impl Index<Square> for PstSide {
    type Output = EvalInt;

    fn index(&self, index: Square) -> &Self::Output {
        &self[usize::from(index)]
    }
}

#[rustfmt::skip]
const PERSPECTIVE_WHITE: [usize; N_SQUARES] = [
    56, 57, 58, 59, 60, 61, 62, 63,
    48, 49, 50, 51, 52, 53, 54, 55,
    40, 41, 42, 43, 44, 45, 46, 47,
    32, 33, 34, 35, 36, 37, 38, 39,
    24, 25, 26, 27, 28, 29, 30, 31,
    16, 17, 18, 19, 20, 21, 22, 23,
     8,  9, 10, 11, 12, 13, 14, 15,
     0,  1,  2,  3,  4,  5,  6,  7,
];

/// This perspective is also horizontally reversed so the king is on the right side.
#[rustfmt::skip]
const PERSPECTIVE_BLACK: [usize; N_SQUARES] = [
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
];

/// Helper to have the right board perspective in the source code.
///
/// In the source code, a1 will be at the bottom left, while h8 will be at the top right,
/// corresponding to how humans usually see the board. This means that a8 is index 0, and h1 is
/// index 63. This function shifts it so that a1 is 0, and h8 is 63, as in our implementation.
///
/// # Arguments
/// * pst: Square values in centipawns.
/// * base_val: The base value of the piece, which is added to every square.
const fn pst_perspective(
    pst: PstSide,
    base_val: EvalInt,
    perspective: [usize; N_SQUARES],
) -> PstSide {
    let mut ret = pst;
    let mut i = 0;
    while i < N_SQUARES {
        let j = perspective[i];
        ret[i] = pst[j] + base_val;
        i += 1;
    }
    ret
}

/// Construct PSTs for a single piece, from white's perspective.
const fn make_pst(val: PstSide, base_val: EvalInt) -> PstPiece {
    [
        pst_perspective(val, base_val, PERSPECTIVE_WHITE),
        pst_perspective(val, base_val, PERSPECTIVE_BLACK),
    ]
}

/// Middle-game PSTs.
#[rustfmt::skip]
pub const PST_MIDGAME: Pst = Pst([
    // rook
    make_pst([
        1,   3,   2,   1,   4,   3,   2,   1, // 8
       20,  20,  20,  20,  20,  20,  20,  20, // 7
        1,   2,   3,   1,   2,   1,   2,   1, // 6
       -1,  -2,   1,   2,   1,  -1,   1,  -1, // 5
       -1,  -1,   2,  -1,   1,  -1,   2,   1, // 4
        2,   1,   1,   0,   0,   0,   0,   0, // 3
       -5,   0,   0,   0,   0,   0,   0,  -5, // 2
       -5,  -3,   0,   0,   0,   2,  -3,  -5, // 1
    //  a    b    c    d    e    f    g    h
    ], 500),

    // bishop
    make_pst([
        0,   0,   0,   0,   0,   0,   0,   0, // 8
        0,   0,   0,   0,   0,   0,   0,   0, // 7
        0,   0,   0,   0,   0,   0,   0,   0, // 6
        0, -20,  10,  10,  10,  10, -20,   0, // 5
        0,   0,  10,  10,  10,  10,   0,   0, // 4
        0,   0,   0,   0,   0,   0,   0,   0, // 3
        0,   0,   0,   0,   0,   0,   0,   0, // 2
        0,   0, -10,   0,   0, -10,   0,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 300),

    // knight
    make_pst([
     -100,-100,-100,-100,-100,-100,-100,-100, // 8
     -100,   0,   0,   0,   0,   0,   0,-100, // 7
     -100,   1,   0,   0,   0,   0,   0,-100, // 6
     -100,   2,   0,  40,  40,   0,   0,-100, // 5
     -100,   0,   1,  40,  40,   0,   0,-100, // 4
     -100,   2,  30,   0,   0,  20,   0,-100, // 3
     -100,   1,   0,   0,   0,   0,   0,-100, // 2
     -100,-100,-100,-100,-100,-100,-100,-100, // 1
    //  a    b    c    d    e    f    g    h
    ], 300),

    // king
    make_pst([
        0,   0,   0,   0,   0,   0,   0,   0, // 8
        0,   0,   0,   0,   0,   0,   0,   0, // 7
        0,   0,   0,   0,   0,   0,   0,   0, // 6
        0,   0,   0,   0,   0,   0,   0,   0, // 5
        0,   0,   0,   0,   0,   0,   0,   0, // 4
        0,   0,   0,   0,   0,   0,   0,   0, // 3
      -70, -70, -70, -70, -70, -70, -70, -70, // 2
        0,   0,  10,   0,   0,   0,  20,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 20_000),

    // queen
    make_pst([
      -50, -50, -50, -50, -50, -50, -50, -50, // 8
      -50, -50, -50, -50, -50, -50, -50, -50, // 7
      -50, -50, -50, -50, -50, -50, -50, -50, // 6
      -50, -50, -50, -50, -50, -50, -50, -50, // 5
      -50, -50, -50, -50, -50, -50, -50, -50, // 4
      -50, -50, -50, -50, -50, -50, -50, -50, // 3
        0,   0,   0,   0,   0,   0,   0,   0, // 2
        0,   0,   0,   0,   0,   0,   0,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 900),

    // pawn
    make_pst([
        0,   0,   0,   0,   0,   0,   0,   0, // 8
       19,  19,  19,  19,  19,  19,  19,  19, // 7
        8,   8,   8,   8,   8,   8,   8,   8, // 6
        7,   7,   7,   8,   8,   7,   7,   7, // 5
        2,   6,   6,  20,  60, -20, -20, -20, // 4
        2,   2,   2,   2, -20, -20, -20, -20, // 3
        0,   0,   0,   0,   0,   0,   0,   0, // 2
        0,   0,   0,   0,   0,   0,   0,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 100),
]);

#[rustfmt::skip]
pub const PST_ENDGAME: Pst = Pst([
    // rook
    make_pst([
        0,   0,   0,   0,   0,   0,   0,   0, // 8
        0,   0,   0,   0,   0,   0,   0,   0, // 7
        1,   2,   3,   1,   2,   1,   2,   1, // 6
        1,   2,   1,   2,   1,   1,   1,   1, // 5
        1,   1,   2,   1,   1,   1,   2,   1, // 4
        2,   1,   1,   0,   0,   0,   0,   0, // 3
        0,   0,   0,   0,   0,   0,   0,   0, // 2
        0,   0,   0,   0,   0,   0,   0,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 500),

    // bishop
    make_pst([
        0,   0,   0,   0,   0,   0,   0,   0, // 8
        0,   0,   0,   0,   0,   0,   0,   0, // 7
        0,   0,   0,   0,   0,   0,   0,   0, // 6
        0,   0,   0,   0,   0,   0,   0,   0, // 5
        0,   0,   0,   0,   0,   0,   0,   0, // 4
        0,   0,   0,   0,   0,   0,   0,   0, // 3
        0,   0,   0,   0,   0,   0,   0,   0, // 2
        0,   0,   0,   0,   0,   0,   0,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 300),

    // knight
    make_pst([
     -100,-100,-100,-100,-100,-100,-100,-100, // 8
     -100,   0,   0,   0,   0,   0,   0,-100, // 7
     -100,   0,   0,   0,   0,   0,   0,-100, // 6
     -100,   0,   0,   0,   0,   0,   0,-100, // 5
     -100,   0,   0,   0,   0,   0,   0,-100, // 4
     -100,   0,   0,   0,   0,   0,   0,-100, // 3
     -100,   0,   0,   0,   0,   0,   0,-100, // 2
     -100,-100,-100,-100,-100,-100,-100,-100, // 1
    //  a    b    c    d    e    f    g    h
    ], 300),

    // king
    make_pst([
     -100,-100, -90, -70, -70, -90,-100,-100, // 8
     -100, -20, -15, -13, -13, -15, -20,-100, // 7
      -90, -15,  -5,  -5,  -5,  -5, -15, -90, // 6
      -70, -13,  -5,   4,   4,  -5, -13, -70, // 5
      -70, -13,  -5,   4,   4,  -5, -13, -70, // 4
      -90, -15,  -5,  -5,  -5,  -5, -15, -90, // 3
     -100, -20, -15, -13, -13, -15, -20,-100, // 2
     -100,-100, -90, -70, -70, -90,-100,-100, // 1
    //  a    b    c    d    e    f    g    h
    ], 20_000),

    // queen
    make_pst([
        0,   0,   0,   0,   0,   0,   0,   0, // 8
        0,   0,   0,   0,   0,   0,   0,   0, // 7
        0,   0,   0,   0,   0,   0,   0,   0, // 6
        0,   0,   0,   0,   0,   0,   0,   0, // 5
        0,   0,   0,   0,   0,   0,   0,   0, // 4
        0,   0,   0,   0,   0,   0,   0,   0, // 3
        0,   0,   0,   0,   0,   0,   0,   0, // 2
        0,   0,   0,   0,   0,   0,   0,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 900),

    // pawn
    make_pst([
       10,  10,  10,  10,  10,  10,  10,  10, // 8
      139, 139, 139, 139, 139, 139, 139, 139, // 7
      138, 138, 138, 138, 138, 138, 138, 108, // 6
       67,  67,  67,  68,  68,  67,  67,  67, // 5
       36,  36,  36,  36,  36,  36,  36,  36, // 4
       32,  32,  32,  34,  34,  30,  32,  30, // 3
        0,   0,   0,   0,   0,   0,   0,   0, // 2
        0,   0,   0,   0,   0,   0,   0,   0, // 1
    //  a    b    c    d    e    f    g    h
    ], 100),
]);

/// Centipawn, signed, eval metrics.
pub struct EvalMetrics {
    pub pst_eval: i32,
    /// Manhattan distance between kings.
    pub king_distance: usize,
    pub king_distance_eval: i32,
    pub total_eval: i32,
    /// game phase, from 14 (start) to 0 (end endgame).
    pub phase: i32,
}

pub fn eval_metrics(board: &Board) -> EvalMetrics {
    // we'll define endgame as the moment when there are 7 non pawn/king pieces left on the
    // board in total.
    //
    // `phase` will range from 14 (game start) to 7 (endgame) to 0 (end end game).
    let phase = i32::from(board.eval.min_maj_pieces);

    // piece square tables
    let pst_eval = i32::from(board.eval.midgame.score) * max(7, phase - 7) / 7
        + i32::from(board.eval.endgame.score) * min(14 - phase, 7) / 7;

    // Signed factor of the color with the material advantage.
    let advantage = pst_eval / 10;

    let kings1 = board[board.turn][Piece::King].into_iter();
    let kings2 = board[board.turn.flip()][Piece::King].into_iter();
    let king_distance = kings1
        .zip(kings2)
        .next()
        .map_or(0, |(k1, k2)| k1.manhattan(k2));

    // attempt to minimize king distance for checkmates
    let king_distance_eval =
        -advantage * i32::try_from(king_distance).unwrap() * max(7 - phase, 0) / 100;

    let eval = pst_eval + king_distance_eval;

    EvalMetrics {
        pst_eval,
        king_distance,
        king_distance_eval,
        total_eval: eval,
        phase,
    }
}

impl Eval for Board {
    fn eval(&self) -> EvalInt {
        let res = eval_metrics(self);

        res.total_eval.try_into().unwrap()
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

        assert!(eval1 > 0, "got eval {eval1} ({:?})", board1.eval);
        assert!(eval2 < 0, "got eval {eval2} ({:?})", board2.eval);
    }
}
