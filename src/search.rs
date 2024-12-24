/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Game-tree search.

use crate::hash::ZobristTable;
use crate::prelude::*;
use std::cmp::{max, min};
use std::sync::mpsc;
use std::time::{Duration, Instant};

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
    /// Centipawn score (exact).
    Exact(EvalInt),
    /// Centipawn score (lower bound).
    Lower(EvalInt),
    /// Centipawn score (upper bound).
    Upper(EvalInt),
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
            SearchEval::Exact(eval) => Self::Exact(-eval),
            SearchEval::Lower(eval) => Self::Upper(-eval),
            SearchEval::Upper(eval) => Self::Lower(-eval),
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
            SearchEval::Exact(eval) => eval,
            SearchEval::Lower(eval) => eval,
            SearchEval::Upper(eval) => eval,
            SearchEval::Stopped => panic!("Attempted to evaluate a halted search"),
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
    /// Limit quiescence search depth
    pub qdepth: usize,
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
            qdepth: 2,
            enable_trans_table: true,
            transposition_size: 24,
        }
    }
}

/// Least valuable victim, most valuable attacker heuristic for captures.
fn lvv_mva_eval(src_pc: Piece, cap_pc: Piece) -> EvalInt {
    let pc_values = [500, 300, 300, 20000, 900, 100];
    pc_values[cap_pc as usize] - pc_values[src_pc as usize]
}

/// Assign a priority to a move based on how promising it is.
fn move_priority(board: &mut Board, mv: &Move, state: &mut EngineState) -> EvalInt {
    // move eval
    let mut eval: EvalInt = 0;
    let src_pc = board.get_piece(mv.src).unwrap();
    let anti_mv = mv.make(board);

    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache[board.zobrist] {
            eval = entry.eval.into();
        }
    } else if let Some(cap_pc) = anti_mv.cap {
        // least valuable victim, most valuable attacker
        eval += lvv_mva_eval(src_pc.into(), cap_pc)
    }

    anti_mv.unmake(board);

    eval
}

/// State specifically for a minmax call.
struct MinmaxState {
    /// how many plies left to search in this call
    depth: usize,
    /// best score (absolute, from current player perspective) guaranteed for current player.
    alpha: Option<EvalInt>,
    /// best score (absolute, from current player perspective) guaranteed for other player.
    beta: Option<EvalInt>,
    /// quiescence search flag
    quiesce: bool,
}

/// Search the game tree to find the absolute (positive good) move and corresponding eval for the
/// current player.
///
/// This also integrates quiescence search, which looks for a calm (quiescent) position where
/// there are no recaptures.
///
/// # Arguments
///
/// * board: board position to analyze.
/// * depth: how deep to analyze the game tree.
///
/// # Returns
///
/// The best line (in reverse move order), and its corresponding absolute eval for the current player.
fn minmax(board: &mut Board, state: &mut EngineState, mm: MinmaxState) -> (Vec<Move>, SearchEval) {
    // these operations are relatively expensive, so only run them occasionally
    if state.node_count % (1 << 16) == 0 {
        // respect the hard stop if given
        match state.rx_engine.try_recv() {
            Ok(msg) => match msg {
                MsgToEngine::Go(_) => panic!("received go while thinking"),
                MsgToEngine::Stop => {
                    return (Vec::new(), SearchEval::Stopped);
                }
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

    if mm.depth == 0 {
        if mm.quiesce || board.recap_sq.is_none() {
            // if we're done with quiescence, static eval.
            // if there is no capture, skip straight to static eval.
            let eval = board.eval() * EvalInt::from(board.turn.sign());
            return (Vec::new(), SearchEval::Exact(eval));
        } else {
            return minmax(
                board,
                state,
                MinmaxState {
                    depth: state.config.qdepth,
                    alpha: mm.alpha,
                    beta: mm.beta,
                    quiesce: true,
                },
            );
        }
    }

    // default to worst, then gradually improve
    let mut alpha = mm.alpha.unwrap_or(EVAL_WORST);
    // our best is their worst
    let beta = mm.beta.unwrap_or(EVAL_BEST);

    let mvs = if mm.quiesce {
        board.gen_captures().into_iter().collect::<Vec<_>>()
    } else {
        board.gen_moves().into_iter().collect::<Vec<_>>()
    };
    let mut mvs: Vec<_> = mvs
        .into_iter()
        .map(|mv| (move_priority(board, &mv, state), mv))
        .collect();

    // get transposition table entry
    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache[board.zobrist] {
            if entry.is_qsearch == mm.quiesce && entry.depth >= mm.depth {
                if let SearchEval::Exact(_) | SearchEval::Upper(_) = entry.eval {
                    // no point looking for a better move
                    return (vec![entry.best_move], entry.eval);
                }
            }
            mvs.push((EVAL_BEST, entry.best_move));
        }
    }

    // sort moves by decreasing priority
    mvs.sort_unstable_by_key(|mv| -mv.0);

    let mut abs_best = SearchEval::Exact(EVAL_WORST);
    let mut best_move: Option<Move> = None;
    let mut best_continuation: Vec<Move> = Vec::new();

    let n_non_qmoves = mvs.len();

    // determine moves that are allowed in quiescence
    if mm.quiesce {
        // use static exchange evaluation to prune moves
        mvs.retain(|(_priority, mv): &(EvalInt, Move)| -> bool {
            let see = board.eval_see(mv.dest, board.turn);

            see > 0
        });
    }

    if n_non_qmoves == 0 {
        let is_in_check = board.is_check(board.turn);

        if is_in_check {
            return (Vec::new(), SearchEval::Checkmate(-1));
        } else {
            // stalemate
            return (Vec::new(), SearchEval::Exact(0));
        }
    } else if mvs.is_empty() {
        // pruned all the moves due to quiescence
        let eval = board.eval() * EvalInt::from(board.turn.sign());
        return (Vec::new(), SearchEval::Exact(eval));
    }

    for (_priority, mv) in mvs {
        let anti_mv = mv.make(board);
        let (continuation, score) = minmax(
            board,
            state,
            MinmaxState {
                depth: mm.depth - 1,
                alpha: Some(-beta),
                beta: Some(-alpha),
                quiesce: mm.quiesce,
            },
        );

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
            if let SearchEval::Upper(eval) | SearchEval::Exact(eval) = abs_best {
                abs_best = SearchEval::Lower(eval);
            }
            break;
        }
    }

    if let Some(best_move) = best_move {
        best_continuation.push(best_move);
        if state.config.enable_trans_table {
            state.cache[board.zobrist] = Some(TranspositionEntry {
                best_move,
                eval: abs_best,
                depth: mm.depth,
                is_qsearch: mm.quiesce,
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
    /// is this score within the context of quiescence
    is_qsearch: bool,
}

pub type TranspositionTable = ZobristTable<TranspositionEntry>;

/// Iteratively deepen search until it is stopped.
fn iter_deep(board: &mut Board, state: &mut EngineState) -> (Vec<Move>, SearchEval) {
    let (mut prev_line, mut prev_eval) = minmax(
        board,
        state,
        MinmaxState {
            depth: 1,
            alpha: None,
            beta: None,
            quiesce: false,
        },
    );

    for depth in 2..=state.config.depth {
        let (line, eval) = minmax(
            board,
            state,
            MinmaxState {
                depth,
                alpha: None,
                beta: None,
                quiesce: false,
            },
        );

        if matches!(eval, SearchEval::Stopped) {
            return (prev_line, prev_eval);
        } else {
            if let Some(soft_lim) = state.time_lims.soft {
                if Instant::now() > soft_lim {
                    return (line, eval);
                }
            }
            (prev_line, prev_eval) = (line, eval);
        }
    }
    (prev_line, prev_eval)
}

/// Deadlines for the engine to think of a move.
#[derive(Default)]
pub struct TimeLimits {
    /// The engine must respect this time limit. It will abort if this deadline is passed.
    pub hard: Option<Instant>,
    pub soft: Option<Instant>,
}

impl TimeLimits {
    /// Make time limits based on wtime, btime (but color-independent).
    ///
    /// Also takes in eval metrics, for instance to avoid wasting too much time in the opening.
    pub fn from_ourtime_theirtime(
        ourtime_ms: u64,
        _theirtime_ms: u64,
        eval: EvalMetrics,
    ) -> TimeLimits {
        // hard timeout (max)
        let mut hard_ms = 100_000;
        // soft timeout (max)
        let mut soft_ms = 1_200;

        // if we have more than 5 minutes, and we're out of the opening, we can afford to think longer
        if ourtime_ms > 300_000 && eval.phase <= 13 {
            soft_ms = 4_500
        }

        let factor = if ourtime_ms > 5_000 { 10 } else { 40 };
        hard_ms = min(ourtime_ms / factor, hard_ms);
        soft_ms = min(ourtime_ms / 50, soft_ms);

        let hard_limit = Instant::now() + Duration::from_millis(hard_ms);
        let soft_limit = Instant::now() + Duration::from_millis(soft_ms);

        TimeLimits {
            hard: Some(hard_limit),
            soft: Some(soft_limit),
        }
    }
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
    pub node_count: usize,
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

    /// Wipe state between different games.
    ///
    /// Configuration is preserved.
    pub fn wipe_state(&mut self) {
        self.cache = TranspositionTable::new(self.config.transposition_size);
        self.node_count = 0;
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
