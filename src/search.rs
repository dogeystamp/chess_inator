/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Game-tree search.

use crate::eval::{Eval, EvalInt};
use crate::movegen::{Move, MoveGen, MoveGenType};
use crate::Board;
use std::cmp::max;

/// Search the game tree to find the absolute (positive good) eval for the current player.
fn minmax(board: &mut Board, depth: usize) -> EvalInt {
    if depth == 0 {
        let eval = board.eval();
        match board.turn {
            crate::Color::White => return eval,
            crate::Color::Black => return -eval,
        }
    }

    let mvs: Vec<_> = board.gen_moves(MoveGenType::Legal).into_iter().collect();

    let mut abs_best = EvalInt::MIN;

    for mv in mvs {
        let anti_mv = mv.make(board);
        abs_best = max(abs_best, -minmax(board, depth - 1));
        anti_mv.unmake(board);
    }

    abs_best
}

/// Find the best move for a position (internal interface).
fn search(board: &mut Board) -> Option<Move> {
    const DEPTH: usize = 4;
    let mvs: Vec<_> = board.gen_moves(MoveGenType::Legal).into_iter().collect();

    // absolute eval value
    let mut best_eval = EvalInt::MIN;
    let mut best_mv: Option<Move> = None;

    for mv in mvs {
        let anti_mv = mv.make(board);
        let abs_eval = -minmax(board, DEPTH);
        if abs_eval > best_eval {
            best_eval = abs_eval;
            best_mv = Some(mv);
        }
        anti_mv.unmake(board);
    }

    best_mv
}

/// Find the best move.
pub fn best_move(board: &mut Board) -> Option<Move> {
    search(board)
}
