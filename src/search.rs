/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Game-tree search.

use crate::coordination::MsgToEngine;
use crate::eval::{Eval, EvalInt};
use crate::hash::ZobristTable;
use crate::movegen::{Move, MoveGen};
use crate::{Board, Piece};
use std::cmp::max;
use std::sync::mpsc;
use std::time::Instant;

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
    /// Search was hard-stopped.
    Stopped,
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
            SearchEval::Stopped => SearchEval::Stopped,
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
            SearchEval::Stopped => 0,
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
    pub alpha_beta_on: bool,
    /// Limit regular search depth
    pub depth: usize,
    /// Enable transposition table.
    pub enable_trans_table: bool,
    /// Transposition table size (2^n where this is n)
    pub transposition_size: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            alpha_beta_on: true,
            // try to make this even to be more conservative and avoid horizon problem
            depth: 10,
            enable_trans_table: true,
            transposition_size: 24,
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
    state: &mut EngineState,
    depth: usize,
    alpha: Option<EvalInt>,
    beta: Option<EvalInt>,
) -> (Vec<Move>, SearchEval) {
    if false {
        if state.node_count % 2048 == 1 {
            // respect the hard stop if given
            match state.rx_engine.try_recv() {
                Ok(msg) => match msg {
                    MsgToEngine::Go(_) => panic!("received go while thinking"),
                    MsgToEngine::Stop => return (Vec::new(), SearchEval::Stopped),
                    MsgToEngine::NewGame => panic!("received newgame while thinking"),
                },
                Err(e) => match e {
                    mpsc::TryRecvError::Empty => {}
                    mpsc::TryRecvError::Disconnected => panic!("thread Main stopped"),
                },
            }

            if let Some(hard) = state.time_lims.hard {
                if Instant::now() > hard {
                    return (Vec::new(), SearchEval::Stopped);
                }
            }
        }
    }

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
    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache[board.zobrist] {
            // the entry has a deeper knowledge than we do, so follow its best move exactly instead of
            // just prioritizing what it thinks is best
            if entry.depth >= depth {
                // we don't save PV line in transposition table, so no information on that
                return (vec![entry.best_move], entry.eval);
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
        let (continuation, score) = minmax(board, state, depth - 1, Some(-beta), Some(-alpha));

        // propagate hard stops
        if matches!(score, SearchEval::Stopped) {
            return (Vec::new(), SearchEval::Stopped);
        }

        let abs_score = score.increment();
        if abs_score > abs_best {
            abs_best = abs_score;
            best_move = Some(mv);
            best_continuation = continuation;
        }
        alpha = max(alpha, abs_best.into());
        anti_mv.unmake(board);
        if alpha >= beta && state.config.alpha_beta_on {
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
        if state.config.enable_trans_table {
            state.cache[board.zobrist] = Some(TranspositionEntry {
                best_move,
                eval: abs_best,
                depth,
            });
        }
    }

    state.node_count += 1;

    (best_continuation, abs_best)
}

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

/// Iteratively deepen search until it is stopped.
fn iter_deep(board: &mut Board, state: &mut EngineState) -> (Vec<Move>, SearchEval) {
    // keep two previous lines (in case current one is halted)
    // 1 is the most recent
    let (mut line1, mut eval1) = minmax(board, state, 1, None, None);
    let (mut line2, mut eval2) = (line1.clone(), eval1);

    macro_rules! ret_best {
        ($depth: expr) => {
            if $depth & 1 == 1 && (EvalInt::from(eval1) - EvalInt::from(eval2) > 300) {
                // be skeptical if we move last and we suddenly earn a lot of
                // centipawns. this may be a sign of horizon problem
                return (line2, eval2);
            } else {
                return (line1, eval1);
            }
        };
    }

    for depth in 2..=state.config.depth {
        let (line, eval) = minmax(board, state, depth, None, None);
        if matches!(eval, SearchEval::Stopped) {
            ret_best!(depth - 1)
        } else {
            (line2, eval2) = (line1, eval1);
            (line1, eval1) = (line, eval);
        }

        if let Some(soft_lim) = state.time_lims.soft {
            if Instant::now() > soft_lim {
                ret_best!(depth)
            }
        }
    }
    (line1, eval1)
}

/// Deadlines for the engine to think of a move.
#[derive(Default)]
pub struct TimeLimits {
    /// The engine must respect this time limit. It will abort if this deadline is passed.
    pub hard: Option<Instant>,
    pub soft: Option<Instant>,
}

/// Helper type to avoid retyping the same arguments into every function prototype.
///
/// This should be owned outside the actual thinking part so that the engine can remember state
/// between moves.
pub struct EngineState {
    pub config: SearchConfig,
    /// Main -> Engine channel receiver
    pub rx_engine: mpsc::Receiver<MsgToEngine>,
    pub cache: TranspositionTable,
    /// Nodes traversed (i.e. number of times minmax called)
    node_count: usize,
    pub time_lims: TimeLimits,
}

impl EngineState {
    pub fn new(
        config: SearchConfig,
        interface: mpsc::Receiver<MsgToEngine>,
        cache: TranspositionTable,
        time_lims: TimeLimits,
    ) -> Self {
        Self {
            config,
            rx_engine: interface,
            cache,
            node_count: 0,
            time_lims,
        }
    }
}

/// Find the best line (in reverse order) and its evaluation.
pub fn best_line(board: &mut Board, engine_state: &mut EngineState) -> (Vec<Move>, SearchEval) {
    let (line, eval) = iter_deep(board, engine_state);
    (line, eval)
}

/// Find the best move.
pub fn best_move(board: &mut Board, engine_state: &mut EngineState) -> Option<Move> {
    let (line, _eval) = best_line(board, engine_state);
    line.last().copied()
}
