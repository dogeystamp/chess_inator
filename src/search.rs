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
use crate::util::arrayvec::ArrayVec;
use std::cmp::min;
use std::sync::mpsc;
use std::time::{Duration, Instant};

// a bit less than int max, as a safety margin
const EVAL_BEST: EvalInt = EvalInt::MAX - 3;
const EVAL_WORST: EvalInt = -(EVAL_BEST);

/// Maximum number of plies to search per search.
pub const MAX_PLY: usize = 128;

/// Number of moves to keep in the killer moves table
const KILLER_TABLE_MOVES: usize = 2;

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

    /// Tells the engine to never stop thinking, because it is currently pondering.
    ///
    /// Not to be confused with `pondering_enabled`, which tells the engine that it may ponder,
    /// sooner or later in this game.
    pub pondering: bool,

    /// A hint to the engine that it will ponder in this game, so that it may adjust its time
    /// management.
    ///
    /// Not to be confused with `pondering`, which means the engine is doing the action of
    /// pondering.
    pub pondering_enabled: bool,

    /// Parameter (centipawns) that sets how confident the engine is.
    ///
    /// Positive means avoid draws, and try to win instead.
    ///
    /// An alternative interpretation of this: the contempt factor is the negative of the value
    /// assigned to a draw.
    pub contempt: EvalInt,
    /// Enable transposition table.
    pub enable_trans_table: bool,
    /// Transposition table size (in MiB)
    pub transposition_size: usize,
    /// Print machine-readable information about the position during NNUE training data generation.
    pub nnue_train_info: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            alpha_beta_on: true,
            depth: 16,
            qdepth: 6,
            contempt: 0,
            enable_trans_table: true,
            transposition_size: 16,
            nnue_train_info: false,
            pondering: false,
            pondering_enabled: false,
        }
    }
}

/// Least valuable victim, most valuable attacker heuristic for captures.
fn lvv_mva_eval(src_pc: Piece, cap_pc: Piece) -> EvalInt {
    let pc_values = [500, 300, 300, 20000, 900, 100];
    pc_values[cap_pc as usize] - pc_values[src_pc as usize]
}

/// Assign a priority to a move based on how promising it is.
fn move_priority(
    board: &mut Board,
    mm: &MinmaxState,
    mv: &Move,
    state: &mut EngineState,
) -> EvalInt {
    // move eval
    let mut eval: EvalInt = 0;
    let src_pc = board.get_piece(mv.src).unwrap();
    let anti_mv = mv.make(board);

    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache[board.zobrist] {
            eval = entry.eval.into();
        } else if state.killer_table.probe(mv, mm.plies) {
            eval = 10000;
        }
    } else if let Some(cap_pc) = anti_mv.cap {
        // least valuable victim, most valuable attacker
        eval += lvv_mva_eval(src_pc.into(), cap_pc)
    }

    // TODO: use lvv mva when there is no transposition entry

    anti_mv.unmake(board);

    eval
}

/// PVS search node type
#[derive(Clone, Copy, Debug, PartialEq)]
enum NodeType {
    /// Node is part of the main line (principal variation)
    PV,
    /// Node is outside the main line
    NonPV,
}

/// State specifically for a minmax call.
struct MinmaxState {
    /// how many plies deep this call will search
    depth: usize,
    /// how many plies have been searched so far before this call
    plies: usize,
    /// best score (absolute, from current player perspective) guaranteed for current player.
    alpha: Option<EvalInt>,
    /// best score (absolute, from current player perspective) guaranteed for other player.
    beta: Option<EvalInt>,
    /// quiescence search flag
    quiesce: bool,
    /// flag to prevent consecutive null moves
    allow_null_mv: bool,

    node_type: NodeType,
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
/// The best move, and its corresponding absolute eval for the current player.
fn minmax(
    board: &mut Board,
    state: &mut EngineState,
    mm: MinmaxState,
) -> (Option<Move>, SearchEval) {
    // occasionally check if we should stop the engine
    let interrupt_cycle = match state.interrupts {
        InterruptMode::Normal => Some(1 << 16),
        InterruptMode::MustComplete => None,
    };

    if let Some(interrupt_cycle) = interrupt_cycle {
        if state.node_count % (interrupt_cycle) == 0 {
            match state.rx_engine.try_recv() {
                Ok(msg) => match msg {
                    MsgToEngine::Go(_) => panic!("received go while thinking"),
                    MsgToEngine::Configure(cfg) => state.config = cfg,
                    // respect the hard stop if given
                    MsgToEngine::Stop => {
                        return (None, SearchEval::Stopped);
                    }
                    MsgToEngine::NewGame => panic!("received newgame while thinking"),
                },
                Err(e) => match e {
                    mpsc::TryRecvError::Empty => {}
                    mpsc::TryRecvError::Disconnected => panic!("thread Main stopped"),
                },
            }

            if !state.config.pondering {
                if let Some(hard) = state.time_lims.hard {
                    if Instant::now() > hard {
                        return (None, SearchEval::Stopped);
                    }
                }
            }
        }
    }

    let is_repetition_draw = board.is_repetition();
    let is_in_check = board.is_check(board.turn);

    let do_extension = is_in_check;

    // positive here since we're looking from the opposite perspective.
    // if white caused a draw, then we'd be black here.
    // therefore, white would see a negative value for the draw.
    let contempt = state.config.contempt;

    // quiescence stand-pat score (only calculated if needed).
    // this is where static eval goes.
    let mut board_eval: Option<EvalInt> = None;

    if mm.quiesce {
        board_eval = if is_repetition_draw {
            Some(contempt)
        } else {
            Some(board.eval() * EvalInt::from(board.turn.sign()))
        }
    }

    if mm.depth == 0 {
        if mm.quiesce {
            // we hit the limit on quiescence depth
            return (None, SearchEval::Exact(board_eval.unwrap()));
        } else {
            // enter quiescence search
            return minmax(
                board,
                state,
                MinmaxState {
                    depth: state.config.qdepth,
                    alpha: mm.alpha,
                    beta: mm.beta,
                    plies: mm.plies + 1,
                    quiesce: true,
                    allow_null_mv: true,
                    node_type: mm.node_type,
                },
            );
        }
    }

    #[derive(Debug)]
    enum MoveGenerator {
        /// Use heavily pruned search to generate moves leading to a quiet position.
        Quiescence,
        /// Generate all legal moves.
        Normal,
        /// Only evaluate a single move.
        None,
    }
    let mut move_generator = if mm.quiesce && !is_in_check {
        // if in check, we need to find all possible evasions
        MoveGenerator::Quiescence
    } else {
        MoveGenerator::Normal
    };

    let mut trans_table_move: Option<Move> = None;

    // get transposition table entry
    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache[board.zobrist] {
            trans_table_move = Some(entry.best_move);
            if usize::from(entry.depth) >= mm.depth
                && entry.is_qsearch == mm.quiesce
                && mm.node_type == entry.node_type
            {
                if let SearchEval::Exact(_) | SearchEval::Upper(_) = entry.eval {
                    // at this point, we could just return the best move + eval given, but this
                    // bypasses the draw by repetition checks in `minmax`. so just don't generate
                    // any other moves than the best move.
                    move_generator = MoveGenerator::None;
                }
            }
        }
    }

    let is_pv = matches!(mm.node_type, NodeType::PV);

    // true in a pv node, until we find a move that raises alpha. then, it becomes false.
    let mut is_next_pv = is_pv;

    // default to worst, then gradually improve
    let mut alpha = mm.alpha.unwrap_or(EVAL_WORST);
    // our best is their worst
    let beta = mm.beta.unwrap_or(EVAL_BEST);

    // R parameter
    const NULL_MOVE_REDUCTION: usize = 4;

    // conditions to perform null move pruning:
    let do_null_move = mm.allow_null_mv
        // prevent going to negative depth
        && mm.depth > NULL_MOVE_REDUCTION
        // quiescence is already reduced, so don't reduce it further
        && !mm.quiesce
        // zugzwang happens mostly during king-pawn endgames. zugzwang is when passing our turn
        // would be a disadvantage for the opponent, thus we can't prune that using null moves.
        && board.info.n_min_maj_pcs > 0
        // null move while in check leads to the king getting captured; no need to verify that
        && !is_in_check;

    // doing nothing is generally very good for the opponent.
    // if we do a null move, and the opponent can't beat their current best score (beta),
    // then we consider that our best real move is not worse.
    if do_null_move {
        let anti_mv = board.make_null_move();
        let (_, score) = minmax(
            board,
            state,
            MinmaxState {
                depth: mm.depth - NULL_MOVE_REDUCTION - 1,
                // null window around beta: our opponent tries to beat their current best
                alpha: Some(-beta),
                beta: Some(-beta + 1),
                plies: mm.plies + 1,
                quiesce: mm.quiesce,
                allow_null_mv: false,
                node_type: if is_next_pv {
                    NodeType::PV
                } else {
                    NodeType::NonPV
                },
            },
        );
        board.unmake_null_move(anti_mv);

        // propagate hard stops
        if matches!(score, SearchEval::Stopped) {
            return (None, SearchEval::Stopped);
        }

        let abs_score = score.increment();
        let score_int = EvalInt::from(abs_score);

        if score_int >= beta {
            // beta cutoff
            // TODO: possibly do transposition table logic too
            return (None, abs_score);
        }

        if score_int > alpha {
            // raise alpha
            alpha = score_int;
        }
    }

    let mvs = match move_generator {
        MoveGenerator::Quiescence => board.gen_captures(),
        MoveGenerator::Normal => board.gen_moves(),
        MoveGenerator::None => MoveList::new(),
    };

    let mut mvs: ArrayVec<{ crate::movegen::MAX_MOVES }, _> = mvs
        .into_iter()
        .map(|mv| (move_priority(board, &mm, &mv, state), mv))
        .collect();

    if let Some(trans_table_move) = trans_table_move {
        mvs.push((EVAL_BEST, trans_table_move))
    }

    // sort moves by decreasing priority
    mvs.sort_unstable_by_key(|mv| -mv.0);

    let mut abs_best = SearchEval::Exact(EVAL_WORST);

    if mm.quiesce && !is_in_check {
        // stand pat
        // (when in check, we don't have the option to "do nothing")
        abs_best = SearchEval::Exact(board_eval.unwrap());
    }

    let mut best_move: Option<Move> = None;

    // determine moves that are allowed in quiescence
    if mm.quiesce {
        // use static exchange evaluation to prune moves
        mvs.retain(|(_priority, mv): &(EvalInt, Move)| -> bool {
            let see = board.eval_see(mv.dest, board.turn);

            see >= 0
        });
    }

    if mvs.is_empty() {
        if mm.quiesce && !is_in_check {
            // use stand pat
            return (None, SearchEval::Exact(board_eval.unwrap()));
        }

        if is_in_check {
            return (None, SearchEval::Checkmate(-1));
        } else {
            // stalemate
            return (None, SearchEval::Exact(0));
        }
    }

    for (_priority, mv) in mvs {
        let anti_mv = mv.make(board);

        // only use null window when we have move ordering through the transposition table
        let do_null_window = !is_next_pv && trans_table_move.is_some() && mm.depth > 2;

        let new_depth = mm.depth - if do_extension { 0 } else { 1 };

        let (_, mut score) = minmax(
            board,
            state,
            MinmaxState {
                depth: new_depth,
                alpha: Some(if do_null_window { -(alpha + 1) } else { -beta }),
                beta: Some(-alpha),
                plies: mm.plies + 1,
                quiesce: mm.quiesce,
                // if we're doing null window, we really just want to prune the move
                allow_null_mv: do_null_window,
                node_type: if is_next_pv {
                    NodeType::PV
                } else {
                    NodeType::NonPV
                },
            },
        );

        if do_null_window {
            let abs_score = score.increment();
            if let SearchEval::Lower(cp) = abs_score {
                if alpha <= cp && cp <= beta {
                    // alpha means this move was better than expected, so re-search with full window
                    // if it's above beta then don't even bother re-searching, it causes a cutoff
                    (_, score) = minmax(
                        board,
                        state,
                        MinmaxState {
                            depth: new_depth,
                            alpha: Some(-beta),
                            beta: Some(-alpha),
                            plies: mm.plies + 1,
                            quiesce: mm.quiesce,
                            allow_null_mv: true,
                            node_type: NodeType::PV,
                        },
                    );
                }
            }
        }

        anti_mv.unmake(board);

        // propagate hard stops
        if matches!(score, SearchEval::Stopped) {
            return (None, SearchEval::Stopped);
        }

        let abs_score = score.increment();
        if abs_score > abs_best {
            abs_best = abs_score;
            best_move = Some(mv);
        }
        if EvalInt::from(abs_best) > alpha {
            alpha = abs_best.into();
            is_next_pv = false;
        }
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
            state.killer_table.write_mv(mv, mm.plies);
            break;
        }
    }

    if is_repetition_draw {
        abs_best = SearchEval::Exact(contempt);
    }

    if let Some(best_move) = best_move {
        if state.config.enable_trans_table {
            state.cache.save_entry(
                board.zobrist,
                TranspositionEntry {
                    best_move,
                    eval: abs_best,
                    depth: u8::try_from(mm.depth).unwrap(),
                    is_qsearch: mm.quiesce,
                    // `as u8` will wrap around to 0, but that's accounted for
                    age: board.plies as u8,
                    node_type: mm.node_type,
                },
            );
        }
    }

    state.node_count += 1;
    (best_move, abs_best)
}

#[derive(Clone, Copy, Debug)]
pub struct TranspositionEntry {
    /// best move found last time
    best_move: Move,
    /// last time's eval
    eval: SearchEval,
    /// depth of this entry
    depth: u8,
    /// is this score within the context of quiescence
    is_qsearch: bool,
    /// half move number when this entry was saved
    age: u8,
    node_type: NodeType,
}

impl crate::hash::TableReplacement for TranspositionEntry {
    fn replaces(&self, other: &Self) -> bool {
        if self.depth >= other.depth {
            return true;
        }

        let age_diff = self.age.wrapping_sub(other.age);

        age_diff >= (other.depth - self.depth)
    }
}

#[cfg(test)]
mod replacement_test {
    use super::*;

    #[test]
    fn ordering() {
        let e1 = TranspositionEntry {
            best_move: Move {
                src: Square(0),
                dest: Square(0),
                move_type: crate::movegen::MoveType::Normal,
            },
            eval: SearchEval::Exact(0),
            depth: 2,
            is_qsearch: false,
            age: 0,
            node_type: NodeType::PV,
        };
        let e2 = TranspositionEntry {
            age: 253,
            depth: 2,
            ..e1
        };
        use crate::hash::TableReplacement;
        assert!(e1.replaces(&e2));

        let e2_again = TranspositionEntry {
            age: 255,
            depth: 4,
            ..e1
        };

        assert!(!e1.replaces(&e2_again));

        let e2_again_pt_ii = TranspositionEntry {
            age: 254,
            depth: 4,
            ..e1
        };

        assert!(e1.replaces(&e2_again_pt_ii));

        let e3 = TranspositionEntry {
            age: 0,
            depth: 2,
            ..e1
        };
        assert!(e3.replaces(&e1));

        let e4 = TranspositionEntry {
            age: 2,
            depth: 2,
            ..e1
        };
        assert!(e4.replaces(&e1));

        let e5 = TranspositionEntry {
            age: 2,
            depth: 1,
            ..e1
        };
        assert!(e5.replaces(&e1));

        let e6 = TranspositionEntry {
            age: 0,
            depth: 1,
            ..e1
        };
        assert!(!e6.replaces(&e1));
    }
}

pub type TranspositionTable = ZobristTable<TranspositionEntry>;

/// Iteratively deepen search until it is stopped.
fn iter_deep(board: &mut Board, state: &mut EngineState) -> (Option<Move>, SearchEval) {
    // wipe the table
    state.killer_table = KillerMoves::new();

    state.interrupts = InterruptMode::MustComplete;

    let (mut prev_move, mut prev_eval) = minmax(
        board,
        state,
        MinmaxState {
            depth: 1,
            alpha: None,
            beta: None,
            plies: 0,
            quiesce: false,
            allow_null_mv: false,
            node_type: NodeType::PV,
        },
    );

    state.interrupts = InterruptMode::Normal;

    let max_depth = if state.config.pondering {
        // i'm just going to hope it doesn't reach this depth
        240
    } else {
        state.config.depth
    };

    for depth in 2..=max_depth {
        let (mv, eval) = minmax(
            board,
            state,
            MinmaxState {
                depth,
                alpha: None,
                beta: None,
                plies: 0,
                quiesce: false,
                allow_null_mv: false,
                node_type: NodeType::PV,
            },
        );

        if matches!(eval, SearchEval::Stopped) {
            return (prev_move, prev_eval);
        } else {
            if !state.config.pondering {
                if let Some(soft_lim) = state.time_lims.soft {
                    if Instant::now() > soft_lim {
                        return (mv, eval);
                    }
                }
            }
            (prev_move, prev_eval) = (mv, eval);
        }
    }
    (prev_move, prev_eval)
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
    /// Takes in a board object to change this based on game phase.
    pub fn from_ourtime_theirtime(ourtime_ms: u64, _theirtime_ms: u64, board: &Board) -> Self {
        // hard timeout (max)
        let mut hard_ms = 100_000;
        // soft timeout (default max)
        let mut soft_ms = 1_500;

        // in some situations we can think longer
        if board.plies >= 12 {
            soft_ms = if ourtime_ms > 300_000 {
                3_000
            } else if ourtime_ms > 600_000 {
                10_000
            } else if ourtime_ms > 1_200_000 {
                20_000
            } else {
                soft_ms
            }
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

    /// Make time limit based on an exact hard limit.
    pub fn from_movetime(movetime_ms: u64) -> Self {
        let hard_limit = Instant::now() + Duration::from_millis(movetime_ms);

        TimeLimits {
            hard: Some(hard_limit),
            soft: None,
        }
    }
}

/// How often to check for interrupts because of time or a `stop` command.
#[derive(Default)]
pub enum InterruptMode {
    /// Disable all interrupts.
    MustComplete,
    /// Checks for interrupts.
    #[default]
    Normal,
}

/// Killer move heuristic data.
///
/// `N` is the amount of moves to store per ply.
pub struct KillerMoves<const N: usize> {
    mvs: [[Option<Move>; N]; MAX_PLY],
}

impl<const N: usize> KillerMoves<N> {
    pub fn new() -> Self {
        KillerMoves {
            mvs: [[None; N]; MAX_PLY],
        }
    }

    /// Check if this move is a killer move for this ply.
    fn probe(&self, mv: &Move, ply: usize) -> bool {
        for k_mv in self.mvs[ply].into_iter().flatten() {
            if k_mv == *mv {
                return true;
            }
        }
        false
    }

    /// Insert a move into the killer move table.
    /// Does not duplicate moves.
    fn write_mv(&mut self, mv: Move, ply: usize) {
        // offset moves to make space (possibly overwrite later moves)
        if let Some(existing_mv) = self.mvs[ply][0] {
            if existing_mv == mv {
                return;
            }
        }
        for i in 1..N {
            self.mvs[ply][i] = self.mvs[ply][i - 1];
        }
        self.mvs[ply][0] = Some(mv);
    }
}

impl<const N: usize> Default for KillerMoves<N> {
    fn default() -> Self {
        Self::new()
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
    /// Sets how often Engine checks for Main thread interrupts
    pub interrupts: InterruptMode,
    /// Killer move table
    pub killer_table: KillerMoves<KILLER_TABLE_MOVES>,
}

impl EngineState {
    pub fn new(
        config: SearchConfig,
        interface: mpsc::Receiver<MsgToEngine>,
        cache: TranspositionTable,
        time_lims: TimeLimits,
        interrupts: InterruptMode,
    ) -> Self {
        Self {
            config,
            rx_engine: interface,
            cache,
            node_count: 0,
            time_lims,
            interrupts,
            killer_table: Default::default(),
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

pub const MAX_PV: usize = 32;
pub type PVStack = ArrayVec<MAX_PV, Move>;

/// Find a series of best moves from the transposition table.
///
/// # Arguments
///
/// * `board`: Position to find best moves from.
/// * `stack`:
pub fn probe_pv(board: &mut Board, state: &mut EngineState, stack: &mut PVStack) {
    if stack.is_full() {
        // maximum attained
        return;
    }

    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache[board.zobrist] {
            let mv = entry.best_move;
            stack.push(mv);
            let anti_mv = mv.make(board);
            probe_pv(board, state, stack);
            anti_mv.unmake(board);
        }
    }
}

/// Find the best line and its evaluation.
pub fn best_line(board: &mut Board, state: &mut EngineState) -> (PVStack, SearchEval) {
    let (best_mv, eval) = iter_deep(board, state);

    let mut best_line = ArrayVec::<MAX_PV, Move>::new();
    if let Some(best_mv) = best_mv {
        best_line.push(best_mv);
        let anti_mv = best_mv.make(board);
        probe_pv(board, state, &mut best_line);
        anti_mv.unmake(board);
    }

    (best_line, eval)
}

/// Find the best move.
pub fn best_move(board: &mut Board, engine_state: &mut EngineState) -> Option<Move> {
    let (best_mv, _eval) = iter_deep(board, engine_state);
    best_mv
}

/// Utility for NNUE training set generation to determine if a position is quiet or not.
///
/// Our definition of "quiet" is that there are no checks, and the static and quiescence search
/// evaluations are similar. (See https://arxiv.org/html/2412.17948v1.)
///
/// It is the caller's responsibility to get the search evaluation and pass it to this function.
pub fn is_quiescent_position(board: &Board, eval: SearchEval) -> bool {
    // max centipawn value difference to call "similar"
    const THRESHOLD: EvalInt = 120;

    if board.is_check(board.turn) {
        return false;
    }

    if matches!(eval, SearchEval::Checkmate(_)) {
        return false;
    }

    // white perspective
    let abs_eval = EvalInt::from(eval) * EvalInt::from(board.turn.sign());

    (board.eval() - EvalInt::from(abs_eval)).abs() <= THRESHOLD.abs()
}

#[cfg(test)]
#[cfg(not(debug_assertions))]
mod tests {
    use super::*;

    /// Test that running minmax does not alter the board.
    #[test]
    fn test_board_same() {
        let (_tx, rx) = mpsc::channel();
        let cache = TranspositionTable::new(1);
        let mut engine_state = EngineState::new(
            SearchConfig {
                depth: 3,
                ..Default::default()
            },
            rx,
            cache,
            TimeLimits::from_movetime(20),
            InterruptMode::MustComplete,
        );
        let mut board =
            Board::from_fen("2rq1rk1/pp1bbppp/3p4/4p1B1/2B1P1n1/1PN5/P1PQ1PPP/R3K2R w KQ - 1 14")
                .unwrap();
        let orig_board = board;
        let (_line, _eval) = best_line(&mut board, &mut engine_state);
        assert_eq!(
            board,
            orig_board,
            "failed eq: '{}' vs '{}'",
            orig_board.to_fen(),
            board.to_fen()
        )
    }
}
