/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Game-tree search.

use crate::eval::{Eval, EvalInt};
use crate::movegen::{Move, MoveGen};
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

/// Eval in the context of search.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum SearchEval {
    /// Mate in |n| - 1 half moves, negative for own mate.
    Checkmate(i8),
    /// Centipawn score.
    Centipawns(EvalInt),
}

impl SearchEval {
    /// Flip side, and increment the "mate in n" counter.
    fn increment(self) -> Self {
        match self {
            SearchEval::Checkmate(n) => {
                debug_assert_ne!(n, 0);
                if n < 0 {
                    Self::Checkmate(-(n - 1))
                } else {
                    Self::Checkmate(-(n + 1))
                }
            }
            SearchEval::Centipawns(eval) => Self::Centipawns(-eval),
        }
    }
}

impl From<SearchEval> for EvalInt {
    fn from(value: SearchEval) -> Self {
        match value {
            SearchEval::Checkmate(n) => {
                debug_assert_ne!(n, 0);
                if n < 0 {
                    EVAL_WORST - EvalInt::from(n)
                } else {
                    EVAL_BEST - EvalInt::from(n)
                }
            }
            SearchEval::Centipawns(eval) => eval,
        }
    }
}

impl Ord for SearchEval {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let e1 = EvalInt::from(*self);
        let e2 = EvalInt::from(*other);
        e1.cmp(&e2)
    }
}

impl PartialOrd for SearchEval {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Search the game tree to find the absolute (positive good) move and corresponding eval for the
/// current player.
///
/// # Arguments
///
/// * board: board position to analyze.
/// * depth: how deep to analyze the game tree.
/// * alpha: best score (absolute, from current player perspective) guaranteed for current player.
/// * beta: best score (absolute, from current player perspective) guaranteed for other player.
///
/// # Returns
///
/// The best line (in reverse move order), and its corresponding absolute eval for the current player.
fn minmax(
    board: &mut Board,
    depth: usize,
    alpha: Option<EvalInt>,
    beta: Option<EvalInt>,
) -> (Vec<Move>, SearchEval) {
    // default to worst, then gradually improve
    let mut alpha = alpha.unwrap_or(EVAL_WORST);
    // our best is their worst
    let beta = beta.unwrap_or(EVAL_BEST);

    if depth == 0 {
        let eval = board.eval();
        return (
            Vec::new(),
            SearchEval::Centipawns(eval * EvalInt::from(board.turn.sign())),
        );
    }

    let mvs: Vec<_> = board.gen_moves().into_iter().collect();

    let mut abs_best = SearchEval::Centipawns(EVAL_WORST);
    let mut best_move: Option<Move> = None;
    let mut best_continuation: Vec<Move> = Vec::new();

    if mvs.is_empty() {
        if board.is_check(board.turn) {
            return (Vec::new(), SearchEval::Checkmate(-1));
        } else {
            // stalemate
            return (Vec::new(), SearchEval::Centipawns(0));
        }
    }

    for mv in mvs {
        let anti_mv = mv.make(board);
        let (continuation, score) = minmax(board, depth - 1, Some(-beta), Some(-alpha));
        let abs_score = score.increment();
        if abs_score > abs_best {
            abs_best = abs_score;
            best_move = Some(mv);
            best_continuation = continuation;
        }
        alpha = max(alpha, abs_best.into());
        anti_mv.unmake(board);
        if alpha >= beta {
            // alpha-beta prune.
            //
            // Beta represents the best eval that the other player can get in sibling branches
            // (different moves in the parent node). Alpha > beta means the eval here is _worse_
            // for the other player, so they will never make the move that leads into this branch.
            // Therefore, we stop evaluating this branch at all.
            break;
        }
    }

    if let Some(mv) = best_move {
        best_continuation.push(mv);
    }

    (best_continuation, abs_best)
}

/// Find the best line (in reverse order) and its evaluation.
pub fn best_line(board: &mut Board) -> (Vec<Move>, SearchEval) {
    let (line, eval) = minmax(board, 5, None, None);
    (line, eval)
}

/// Find the best move.
pub fn best_move(board: &mut Board) -> Option<Move> {
    let (line, _eval) = best_line(board);
    line.last().copied()
}
