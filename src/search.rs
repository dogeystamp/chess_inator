/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Game-tree search.

use crate::eval::{Eval, EvalInt};
use crate::movegen::{Move, MoveGen, ToUCIAlgebraic};
use crate::Board;
use std::cmp::max;

// min can't be represented as positive
const EVAL_WORST: EvalInt = -(EvalInt::MAX);
const EVAL_BEST: EvalInt = EvalInt::MAX;

#[cfg(test)]
mod test_eval_int {
    use super::*;

    #[test]
    fn test_eval_worst_best_symm() {
        // int limits will bite you if you don't test this
        assert_eq!(EVAL_WORST, -EVAL_BEST);
        assert_eq!(-EVAL_WORST, EVAL_BEST);
    }
}

/// Search the game tree to find the absolute (positive good) eval for the current player.
///
/// # Arguments
///
/// * board: board position to analyze.
/// * depth: how deep to analyze the game tree.
/// * alpha: best score (absolute, from current player perspective) guaranteed for current player.
/// * beta: best score (absolute, from current player perspective) guaranteed for other player.
fn minmax(board: &mut Board, depth: usize, alpha: Option<EvalInt>, beta: Option<EvalInt>) -> EvalInt {
    // default to worst, then gradually improve
    let mut alpha = alpha.unwrap_or(EVAL_WORST);
    // our best is their worst
    let beta = beta.unwrap_or(EVAL_BEST);

    if depth == 0 {
        let eval = board.eval();
        match board.turn {
            crate::Color::White => return eval,
            crate::Color::Black => return -eval,
        }
    }

    let mvs: Vec<_> = board.gen_moves().into_iter().collect();

    let mut abs_best = EVAL_WORST;

    if mvs.is_empty() {
        if board.is_check(board.turn) {
            return EVAL_WORST;
        } else {
            // stalemate
            return 0;
        }
    }

    for mv in mvs {
        let anti_mv = mv.make(board);
        let abs_score = -minmax(board, depth - 1, Some(-beta), Some(-alpha));
        abs_best = max(abs_best, abs_score);
        alpha = max(alpha, abs_best);
        anti_mv.unmake(board);
        if alpha >= beta  {
            // alpha-beta prune.
            //
            // Beta represents the best eval that the other player can get in sibling branches
            // (different moves in the parent node). Alpha >= beta means the eval here is _worse_
            // for the other player, so they will never make the move that leads into this branch.
            // Therefore, we stop evaluating this branch at all.
            break;
        }
    }

    abs_best
}

/// Find the best move for a position (internal interface).
fn search(board: &mut Board) -> Option<Move> {
    const DEPTH: usize = 4;
    let mvs: Vec<_> = board.gen_moves().into_iter().collect();

    // absolute eval value
    let mut best_eval = EVAL_WORST;
    let mut best_mv: Option<Move> = None;

    for mv in mvs {
        let anti_mv = mv.make(board);
        let abs_eval = -minmax(board, DEPTH, None, None);
        if abs_eval >= best_eval {
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
