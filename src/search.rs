/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Game-tree search.

use crate::eval::{Eval, EvalInt};
use crate::hash::ZobristTable;
use crate::movegen::{Move, MoveGen};
use crate::{Board, Piece};
use std::cmp::max;
use std::sync::mpsc;

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

/// Configuration for the gametree search.
#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    /// Enable alpha-beta pruning.
    alpha_beta_on: bool,
    /// Limit regular search depth
    depth: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            alpha_beta_on: true,
            // try to make this even to be more conservative and avoid horizon problem
            depth: 10,
        }
    }
}

/// If a move is a capture, return which piece is capturing what.
fn move_get_capture(board: &mut Board, mv: &Move) -> Option<(Piece, Piece)> {
    // TODO: en passant
    board
        .get_piece(mv.dest)
        .map(|cap_pc| (board.get_piece(mv.src).unwrap().into(), cap_pc.into()))
}

/// Least valuable victim, most valuable attacker heuristic for captures.
fn lvv_mva_eval(src_pc: Piece, cap_pc: Piece) -> EvalInt {
    let pc_values = [500, 300, 300, 20000, 900, 100];
    pc_values[cap_pc as usize] - pc_values[src_pc as usize]
}

/// Assign a priority to a move based on how promising it is.
fn move_priority(board: &mut Board, mv: &Move) -> EvalInt {
    // move eval
    let mut eval: EvalInt = 0;
    if let Some((src_pc, cap_pc)) = move_get_capture(board, mv) {
        // least valuable victim, most valuable attacker
        eval += lvv_mva_eval(src_pc, cap_pc)
    }

    eval
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
    config: &SearchConfig,
    depth: usize,
    alpha: Option<EvalInt>,
    beta: Option<EvalInt>,
    cache: &mut TranspositionTableOpt,
) -> (Vec<Move>, SearchEval) {
    // default to worst, then gradually improve
    let mut alpha = alpha.unwrap_or(EVAL_WORST);
    // our best is their worst
    let beta = beta.unwrap_or(EVAL_BEST);

    if depth == 0 {
        let eval = board.eval() * EvalInt::from(board.turn.sign());
        return (Vec::new(), SearchEval::Centipawns(eval));
    }

    let mut mvs: Vec<_> = board
        .gen_moves()
        .into_iter()
        .collect::<Vec<_>>()
        .into_iter()
        .map(|mv| (move_priority(board, &mv), mv))
        .collect();

    // get transposition table entry
    if let Some(cache) = cache {
        if let Some(entry) = &cache[board.zobrist] {
            // the entry has a deeper knowledge than we do, so follow its best move exactly instead of
            // just prioritizing what it thinks is best
            if entry.depth >= depth {
                // we don't save PV line in transposition table, so no information on that
                return (vec![entry.best_move], entry.eval)
            }
            mvs.push((EVAL_BEST, entry.best_move));
        }
    }

    // sort moves by decreasing priority
    mvs.sort_unstable_by_key(|mv| -mv.0);

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

    for (_priority, mv) in mvs {
        let anti_mv = mv.make(board);
        let (continuation, score) =
            minmax(board, config, depth - 1, Some(-beta), Some(-alpha), cache);
        let abs_score = score.increment();
        if abs_score > abs_best {
            abs_best = abs_score;
            best_move = Some(mv);
            best_continuation = continuation;
        }
        alpha = max(alpha, abs_best.into());
        anti_mv.unmake(board);
        if alpha >= beta && config.alpha_beta_on {
            // alpha-beta prune.
            //
            // Beta represents the best eval that the other player can get in sibling branches
            // (different moves in the parent node). Alpha > beta means the eval here is _worse_
            // for the other player, so they will never make the move that leads into this branch.
            // Therefore, we stop evaluating this branch at all.
            break;
        }
    }

    if let Some(best_move) = best_move {
        best_continuation.push(best_move);
        if let Some(cache) = cache {
            cache[board.zobrist] = Some(TranspositionEntry {
                best_move,
                eval: abs_best,
                depth,
            });
        }
    }

    (best_continuation, abs_best)
}

/// Messages from the interface to the search thread.
pub enum InterfaceMsg {
    Stop,
}

type InterfaceRx = mpsc::Receiver<InterfaceMsg>;

#[derive(Clone, Copy, Debug)]
pub struct TranspositionEntry {
    /// best move found last time
    best_move: Move,
    /// last time's eval
    eval: SearchEval,
    /// depth of this entry
    depth: usize,
}

pub type TranspositionTable = ZobristTable<TranspositionEntry>;
type TranspositionTableOpt = Option<TranspositionTable>;

/// Iteratively deepen search until it is stopped.
fn iter_deep(
    board: &mut Board,
    config: &SearchConfig,
    interface: Option<InterfaceRx>,
    cache: &mut TranspositionTableOpt,
) -> (Vec<Move>, SearchEval) {
    let (mut prev_line, mut prev_eval) = minmax(board, config, 1, None, None, cache);
    for depth in 2..=config.depth {
        let (line, eval) = minmax(board, config, depth, None, None, cache);

        if let Some(ref rx) = interface {
            // don't interrupt a depth 1 search so that there's at least a move to be played
            if depth != 1 {
                match rx.try_recv() {
                    Ok(msg) => match msg {
                        InterfaceMsg::Stop => {
                            if depth & 1 == 1 && (EvalInt::from(eval) - EvalInt::from(prev_eval) > 300) {
                                // be skeptical if we move last and we suddenly earn a lot of
                                // centipawns. this may be a sign of horizon problem
                                return (prev_line, prev_eval)
                            } else {
                                return (line, eval)
                            }
                        },
                    },
                    Err(e) => match e {
                        mpsc::TryRecvError::Empty => {}
                        mpsc::TryRecvError::Disconnected => panic!("interface thread stopped"),
                    },
                }
            }
        } else if depth == config.depth - 1 {
            return (line, eval);
        }
        (prev_line, prev_eval) = (line, eval);
    }
    (prev_line, prev_eval)
}

/// Find the best line (in reverse order) and its evaluation.
pub fn best_line(
    board: &mut Board,
    config: Option<SearchConfig>,
    interface: Option<InterfaceRx>,
    cache: &mut TranspositionTableOpt,
) -> (Vec<Move>, SearchEval) {
    let config = config.unwrap_or_default();
    let (line, eval) = iter_deep(board, &config, interface, cache);
    (line, eval)
}

/// Find the best move.
pub fn best_move(
    board: &mut Board,
    config: Option<SearchConfig>,
    interface: Option<InterfaceRx>,
    cache: &mut TranspositionTableOpt,
) -> Option<Move> {
    let (line, _eval) = best_line(board, Some(config.unwrap_or_default()), interface, cache);
    line.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{FromFen, ToFen};
    use crate::movegen::ToUCIAlgebraic;

    /// Theoretically, alpha-beta pruning should not affect the result of minmax.
    #[test]
    fn alpha_beta_same_result() {
        let test_cases = [
            "r2q1rk1/1bp1pp1p/p2p2p1/1p1P2P1/2n1P3/3Q1P2/PbPBN2P/3RKB1R b K - 5 15",
            "r1b1k2r/p1qpppbp/1p4pn/2B3N1/1PP1P3/2P5/P4PPP/RN1QR1K1 w kq - 0 14",
        ];
        for fen in test_cases {
            let mut board = Board::from_fen(fen).unwrap();
            let mv_no_prune = best_move(
                &mut board,
                Some(SearchConfig {
                    alpha_beta_on: false,
                    depth: 3,
                }),
                None,
                &mut None,
            )
            .unwrap();

            assert_eq!(board.to_fen(), fen);

            let mv_with_prune = best_move(
                &mut board,
                Some(SearchConfig {
                    alpha_beta_on: true,
                    depth: 3,
                }),
                None,
                &mut None,
            )
            .unwrap();

            assert_eq!(board.to_fen(), fen);

            println!(
                "without ab prune got {}, otherwise {}, fen {}",
                mv_no_prune.to_uci_algebraic(),
                mv_with_prune.to_uci_algebraic(),
                fen
            );

            assert_eq!(mv_no_prune, mv_with_prune);
        }
    }
}
