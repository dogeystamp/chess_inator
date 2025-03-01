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

/// Depth equivalent to one ply (because fractional plies exist)
pub const ONE_PLY: usize = 1;

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

/// Evaluation of a position.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Score {
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

impl Score {
    /// Flip side, and increment the "mate in n" counter.
    fn increment(self) -> Self {
        match self {
            Score::Checkmate(n) => {
                debug_assert_ne!(n, 0);
                if n < 0 {
                    Self::Checkmate(-(n - 1))
                } else {
                    Self::Checkmate(-(n + 1))
                }
            }
            Score::Exact(eval) => Self::Exact(-eval),
            Score::Lower(eval) => Self::Upper(-eval),
            Score::Upper(eval) => Self::Lower(-eval),
            Score::Stopped => Score::Stopped,
        }
    }
}

impl From<Score> for EvalInt {
    fn from(value: Score) -> Self {
        match value {
            Score::Checkmate(n) => {
                debug_assert_ne!(n, 0);
                if n < 0 {
                    EVAL_WORST - EvalInt::from(n)
                } else {
                    EVAL_BEST - EvalInt::from(n)
                }
            }
            Score::Exact(eval) => eval,
            Score::Lower(eval) => eval,
            Score::Upper(eval) => eval,
            Score::Stopped => panic!("Attempted to evaluate a halted search"),
        }
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let e1 = EvalInt::from(*self);
        let e2 = EvalInt::from(*other);
        e1.cmp(&e2)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration for the gametree search.
#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    /// Enable alpha-beta pruning.
    pub alpha_beta_on: bool,
    /// Limit regular search depth. Measured in fractional ply; multiply by plies by [`ONE_PLY`].
    pub depth: usize,
    /// Limit quiescence search depth. Measured in fractional ply; multiply by plies by [`ONE_PLY`].
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
            depth: 64 * ONE_PLY,
            qdepth: 4 * ONE_PLY,
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
const fn lvv_mva_eval(src_pc: Piece, cap_pc: Piece) -> EvalInt {
    cap_pc.value() - src_pc.value()
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

    let mut is_mate_score = false;

    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache.get(board.zobrist) {
            eval = EvalInt::from(entry.eval) / 100;
            if let Score::Checkmate(_) = entry.eval {
                is_mate_score = true;
            }
        }
    }

    if !is_mate_score {
        if state.killer_table.probe(mv, mm.plies) {
            eval = eval.saturating_add(800);
        } else if let Some(cap_pc) = anti_mv.cap {
            // least valuable victim, most valuable attacker
            eval = eval.saturating_add(lvv_mva_eval(src_pc.into(), cap_pc) / 10);

            if let Some(recap_sq) = board.recap_sq {
                if recap_sq == mv.dest {
                    eval = eval.saturating_add(90);
                }
            }
        }
    }

    anti_mv.unmake(board);

    eval
}

/// PVS search node type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeType {
    /// Node is part of the main line (principal variation)
    PV,
    /// Node is outside the main line
    NonPV,
}

/// State specifically for a minmax call.
struct MinmaxState {
    /// how many deep this call will search (measured in fractions of ply; one real ply is [ONE_PLY] here)
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
fn minmax(board: &mut Board, state: &mut EngineState, mm: MinmaxState) -> (Option<Move>, Score) {
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
                        return (None, Score::Stopped);
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
                        return (None, Score::Stopped);
                    }
                }
            }
        }
    }

    let is_repetition_draw = board.is_repetition();
    let is_in_check = board.is_check(board.turn);

    // default to worst, then gradually improve
    let mut alpha = mm.alpha.unwrap_or(EVAL_WORST);
    // our best is their worst
    let beta = mm.beta.unwrap_or(EVAL_BEST);

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
    let mut trans_table_static_eval: Option<EvalInt> = None;

    // get transposition table entry
    if state.config.enable_trans_table {
        if let Some(entry) = &state.cache.get(board.zobrist) {
            trans_table_move = Some(entry.best_move);
            trans_table_static_eval = entry.static_eval;
            if usize::from(entry.depth) >= mm.depth
                && entry.is_qsearch == mm.quiesce
                && mm.node_type == entry.node_type
            {
                if let Score::Exact(_) | Score::Upper(_) = entry.eval {
                    // at this point, we could just return the best move + eval given, but this
                    // bypasses the draw by repetition checks in `minmax`. so just don't generate
                    // any other moves than the best move.
                    move_generator = MoveGenerator::None;
                }

                if let Score::Lower(eval) = entry.eval {
                    if eval > beta && mm.beta.is_some() {
                        // cutoff
                        return (None, entry.eval);
                    }
                }
            }
        }
    }

    let do_extension = is_in_check;
    // conditions to perform null move pruning:
    let do_null_move = mm.allow_null_mv
        // prevent going to negative depth
        && mm.depth > NULL_MOVE_REDUCTION + ONE_PLY
        // quiescence is already reduced, so don't reduce it further
        && !mm.quiesce
        // zugzwang happens mostly during king-pawn endgames. zugzwang is when passing our turn
        // would be a disadvantage for the opponent, thus we can't prune that using null moves.
        && board.info.n_min_maj_pcs > 0
        // null move while in check leads to the king getting captured; no need to verify that
        && !is_in_check
        // draws should not be played out
        && !is_repetition_draw;

    // positive here since we're looking from the opposite perspective.
    // if white caused a draw, then we'd be black here.
    // therefore, white would see a negative value for the draw.
    let contempt = state.config.contempt;

    // static evaluation from our perspective
    let board_eval = if is_repetition_draw {
        Some(contempt)
    } else if mm.quiesce || do_null_move {
        // we only need static eval for quiescence (stand pat, leaf evals) and for null move prune
        // conditions
        trans_table_static_eval.or(Some(board.eval() * EvalInt::from(board.turn.sign())))
    } else {
        None
    };

    if mm.depth == 0 {
        if mm.quiesce {
            // we hit the limit on quiescence depth
            return (None, Score::Exact(board_eval.unwrap()));
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

    let is_pv = matches!(mm.node_type, NodeType::PV);

    // true in a pv node, until we find a move that raises alpha. then, it becomes false.
    let mut is_next_pv = is_pv;

    // R parameter
    const NULL_MOVE_REDUCTION: usize = 2 * ONE_PLY;
    // if our current board is already worse than beta, then null move will often not prune
    let do_null_move = do_null_move && board_eval.unwrap() >= EvalInt::from(beta);

    // doing nothing is generally very good for the opponent.
    // if we do a null move, and the opponent can't beat their current best score (beta),
    // then we consider that our best real move is not worse.
    if do_null_move {
        let anti_mv = board.make_null_move();
        let (_, score) = minmax(
            board,
            state,
            MinmaxState {
                depth: mm.depth - NULL_MOVE_REDUCTION - ONE_PLY,
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
        if matches!(score, Score::Stopped) {
            return (None, Score::Stopped);
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
        .filter(|mv| {
            if let Some(hash_mv) = trans_table_move {
                // we're going to directly push the transposition table move later,
                // so don't include it in the normal list
                *mv != hash_mv
            } else {
                true
            }
        })
        .map(|mv| (move_priority(board, &mm, &mv, state), mv))
        .collect();

    if let Some(trans_table_move) = trans_table_move {
        mvs.push((EVAL_BEST, trans_table_move))
    }

    // sort moves by decreasing priority
    mvs.sort_unstable_by_key(|mv| -mv.0);

    let mut abs_best = Score::Exact(EVAL_WORST);

    if mm.quiesce && !is_in_check {
        // stand pat
        // (when in check, we don't have the option to "do nothing")
        abs_best = Score::Exact(board_eval.unwrap());
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
        if mm.quiesce {
            return if !is_in_check {
                // use stand pat
                (None, Score::Exact(board_eval.unwrap()))
            } else {
                // return very bad score, but not checkmate, because checkmate is handled specially
                (None, Score::Exact(board_eval.unwrap().saturating_sub(2000)))
            };
        }

        if is_in_check {
            return (None, Score::Checkmate(-1));
        } else {
            // stalemate
            return (None, Score::Exact(0));
        }
    }

    for (move_idx, (_priority, mv)) in mvs.iter().enumerate() {
        let anti_mv = mv.make(board);

        let mut reduction = 0;

        // after how many moves does late move reduction kick in
        const LMR_THRESH: usize = 4;
        // how much to reduce by in LMR
        const LMR_R: usize = ONE_PLY;

        let do_late_move_reduction = !is_pv
            && move_idx >= LMR_THRESH
            // quiet moves only
            && anti_mv.cap.is_none()
            // do not reduce extended moves
            && !do_extension
            && mm.depth > 2 * ONE_PLY
            // only reduce when we have move ordering via transposition table
            && trans_table_move.is_some();

        if do_late_move_reduction {
            reduction += LMR_R
        }

        let new_depth = mm
            .depth
            .saturating_sub(if do_extension { 0 } else { ONE_PLY });

        let reduced_depth = new_depth.saturating_sub(reduction);

        // only use null window when we have move ordering through the transposition table
        let do_null_window = !is_next_pv && trans_table_move.is_some() && mm.depth > 2;

        let (_, mut score) = minmax(
            board,
            state,
            MinmaxState {
                depth: reduced_depth,
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
        score = score.increment();

        if reduction > 0 {
            if let Score::Lower(cp) = score {
                if alpha <= cp {
                    // a reduced move is not supposed to good; if we are better than alpha,
                    // we should probably re-search to get an accurate evaluation of the move
                    (_, score) = minmax(
                        board,
                        state,
                        MinmaxState {
                            depth: new_depth,
                            // still null-window
                            alpha: Some(if do_null_window { -(alpha + 1) } else { -beta }),
                            beta: Some(-alpha),
                            plies: mm.plies + 1,
                            quiesce: mm.quiesce,
                            allow_null_mv: true,
                            node_type: NodeType::PV,
                        },
                    );
                    score = score.increment();
                }
            }
        }

        if do_null_window {
            if let Score::Lower(cp) = score {
                if alpha <= cp && cp <= beta {
                    // at this point, either:
                    // - there was no reduction, and we did null window, and it failed high
                    // - there was a reduction, and it failed high, and the null-window re-search failed high as well

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
                    score = score.increment();
                }
            }
        }

        anti_mv.unmake(board);

        // propagate hard stops
        if matches!(score, Score::Stopped) {
            return (None, Score::Stopped);
        }

        if score > abs_best {
            abs_best = score;
            best_move = Some(*mv);
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
            if let Score::Upper(eval) | Score::Exact(eval) = abs_best {
                abs_best = Score::Lower(eval);
            }
            state.killer_table.write_mv(mv, mm.plies);
            break;
        }
    }

    if is_repetition_draw {
        abs_best = Score::Exact(contempt);
    }

    if let Some(best_move) = best_move {
        if state.config.enable_trans_table {
            state.cache.write(
                board.zobrist,
                TTableEntry {
                    best_move,
                    eval: abs_best,
                    depth: u8::try_from(mm.depth).unwrap(),
                    is_qsearch: mm.quiesce,
                    node_type: mm.node_type,
                    static_eval: board_eval,
                },
            );
        }
    }

    state.node_count += 1;
    (best_move, abs_best)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TTableEntry {
    /// best move found last time
    best_move: Move,
    /// last time's eval
    eval: Score,
    /// depth of this entry
    depth: u8,
    /// is this score within the context of quiescence
    is_qsearch: bool,
    node_type: NodeType,
    /// Static evaluation of the board, if calculated.
    static_eval: Option<EvalInt>,
}

/// [`TTableEntry`], but with less memory usage
#[derive(Clone, Copy, Debug)]
pub struct PackedTTableEntry {
    eval: EvalInt,
    static_eval: EvalInt,
    best_move: u16,
    flags: std::num::NonZeroU8,
    depth: u8,
}

/// Helper to accurately determine the size of the transposition table entries in the hash table.
struct _HashRecord(crate::hash::Zobrist, Option<PackedTTableEntry>);

impl crate::hash::PackUnpack<PackedTTableEntry> for TTableEntry {
    fn pack(self) -> PackedTTableEntry {
        // best_move, 16 bits:
        //  0 --  5   source square
        //  6 -- 11   dest square
        //    12      is promotion move?
        // 13 -- 14   promote to what piece?
        //    15      RESERVED

        #[allow(clippy::assertions_on_constants)]
        {
            debug_assert!(
                N_SQUARES <= 64,
                "Can't use efficient move packing when N_SQUARES = {N_SQUARES}."
            );
        }

        let is_prom = match self.best_move.move_type {
            crate::movegen::MoveType::Promotion(_) => true,
            crate::movegen::MoveType::Normal => false,
        } as u16;

        let prom_pc = match self.best_move.move_type {
            crate::movegen::MoveType::Promotion(promote_piece) => promote_piece as u16,
            crate::movegen::MoveType::Normal => 0,
        };

        debug_assert!(is_prom <= 0b1, "ran out of bits for is_prom {is_prom}");
        debug_assert!(prom_pc <= 0b11, "ran out of bits for prom_pc {prom_pc}");
        debug_assert!(
            self.best_move.dest.0 <= 0b111_111,
            "ran out of bits for dest"
        );
        debug_assert!(self.best_move.src.0 <= 0b111_111, "ran out of bits for src");

        let src = self.best_move.src.0 as u16;
        let dest = (self.best_move.dest.0 as u16) << 6;
        let is_prom = is_prom << 12;
        let prom_pc = prom_pc << 13;

        debug_assert_eq!(
            src & dest & is_prom & prom_pc,
            0,
            "Invalid move packing on {self:?}"
        );

        // flags: 8 bits
        // 0    is there a static eval?
        // 1    is this a qsearch score?
        // 2    is this a PV node?
        // 3    is this an exact score?
        // 4    if not exact, is this an upper bound?
        // 5    is this a mate score?
        // 6    always one (A `None` option sets this bit to zero.)
        // 7    RESERVED

        let flags = self.static_eval.is_some() as u8
            | (self.is_qsearch as u8) << 1
            | match self.node_type {
                NodeType::PV => 1,
                NodeType::NonPV => 0,
            } << 2
            | match self.eval {
                Score::Checkmate(_) => 0b100,
                Score::Exact(_) => 0b001,
                Score::Lower(_) => 0b000,
                Score::Upper(_) => 0b010,
                Score::Stopped => panic!("attempted to pack Stopped score"),
            } << 3
            | 1 << 6;

        let flags = unsafe {
            // SAFETY: we just set bit 6 to be always one, so the value must be non-zero
            std::num::NonZeroU8::new_unchecked(flags)
        };

        // eval will be the moves till checkmate if it's a mate score,
        // or the eval otherwise.
        let eval = match self.eval {
            Score::Checkmate(m) => m as i16,
            Score::Exact(s) | Score::Lower(s) | Score::Upper(s) => s,
            Score::Stopped => panic!("attempted to pack Stopped score"),
        };

        PackedTTableEntry {
            static_eval: self.static_eval.unwrap_or(0),
            eval,
            best_move: src | dest | is_prom | prom_pc,
            flags,
            depth: self.depth,
        }
    }

    fn unpack(p: &PackedTTableEntry) -> Self {
        let is_prom = p.best_move & (1 << 12) != 0;
        let prom_pc = crate::movegen::PromotePiece::try_from((p.best_move & (0b11 << 13)) >> 13)
            .expect("Invalid packed move.");
        let src = Square((p.best_move & 0b111111) as crate::SquareIdx);
        let dest = Square(((p.best_move & (0b111111 << 6)) >> 6) as crate::SquareIdx);

        let best_move = Move {
            src,
            dest,
            move_type: if is_prom {
                crate::movegen::MoveType::Promotion(prom_pc)
            } else {
                crate::movegen::MoveType::Normal
            },
        };

        let flags = p.flags.get();

        let have_static = flags & 1 != 0;
        let is_qsearch = flags & (1 << 1) != 0;
        let is_pv = flags & (1 << 2) != 0;
        let is_exact = flags & (1 << 3) != 0;
        let is_upper = flags & (1 << 4) != 0;
        let is_mate = flags & (1 << 5) != 0;

        let eval = if is_mate {
            Score::Checkmate(p.eval as i8)
        } else if is_exact {
            Score::Exact(p.eval)
        } else if is_upper {
            Score::Upper(p.eval)
        } else {
            Score::Lower(p.eval)
        };

        let node_type = if is_pv { NodeType::PV } else { NodeType::NonPV };

        let static_eval = if have_static {
            Some(p.static_eval)
        } else {
            None
        };

        Self {
            best_move,
            eval,
            depth: p.depth,
            is_qsearch,
            node_type,
            static_eval,
        }
    }
}

// opt in to always-replace
impl crate::hash::TableReplacement for TTableEntry {}

pub type TranspositionTable = ZobristTable<PackedTTableEntry, TTableEntry>;

/// Result of [`iter_deep`].
struct IterDeepResult {
    /// Best move in this position.
    ///
    /// May be `None` since sometimes there are no legal moves,
    /// or the search was halted.
    best_mv: Option<Move>,

    /// The evaluation of the position.
    eval: Score,

    /// Nominal depth searched, measured in fractional plies.
    ///
    /// Divide by [`ONE_PLY`] to get depth in plies.
    depth: usize,
}

impl IterDeepResult {
    /// Instantiate this struct from the result of [`minmax`].
    fn from_minmax_ret(depth: usize, ret: (Option<Move>, Score)) -> Self {
        Self {
            best_mv: ret.0,
            eval: ret.1,
            depth,
        }
    }
}

/// Iteratively deepen search until it is stopped.
fn iter_deep(board: &mut Board, state: &mut EngineState) -> IterDeepResult {
    // wipe the table
    state.killer_table = KillerMoves::new();

    state.interrupts = InterruptMode::MustComplete;

    let mut prev = IterDeepResult::from_minmax_ret(
        1,
        minmax(
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
        ),
    );

    state.interrupts = InterruptMode::Normal;

    let max_depth = if state.config.pondering {
        // i'm just going to hope it doesn't reach this depth
        240
    } else {
        state.config.depth
    };

    for depth in 2..=max_depth {
        let cur = IterDeepResult::from_minmax_ret(
            depth,
            minmax(
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
            ),
        );

        if matches!(cur.eval, Score::Stopped) {
            return prev;
        } else {
            if !state.config.pondering {
                if let Some(soft_lim) = state.time_lims.soft {
                    if Instant::now() > soft_lim {
                        return cur;
                    }
                }

                if let Score::Checkmate(_) = cur.eval {
                    // no point looking further after we get a mate score
                    return cur;
                }
            }
            prev = cur;
        }
    }
    prev
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
    fn write_mv(&mut self, mv: &Move, ply: usize) {
        // offset moves to make space (possibly overwrite later moves)
        if let Some(existing_mv) = self.mvs[ply][0] {
            if existing_mv == *mv {
                return;
            }
        }
        for i in 1..N {
            self.mvs[ply][i] = self.mvs[ply][i - 1];
        }
        self.mvs[ply][0] = Some(*mv);
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
        if let Some(entry) = &state.cache.get(board.zobrist) {
            let mv = entry.best_move;
            stack.push(mv);
            let anti_mv = mv.make(board);
            probe_pv(board, state, stack);
            anti_mv.unmake(board);
        }
    }
}

/// Result of [`search`].
pub struct SearchResult {
    /// Best continuation stack (last element is the current best move).
    pub pv: PVStack,
    /// Evaluation of the position.
    pub eval: Score,

    /// The (nominal) depth searched.
    ///
    /// This does not include quiescence depth (more depth), and it does not include pruning (less
    /// depth).
    pub depth: usize,
}

/// Search for the best move, and return all information possible.
pub fn search(board: &mut Board, state: &mut EngineState) -> SearchResult {
    let res = iter_deep(board, state);

    let mut best_line = PVStack::new();
    if let Some(best_mv) = res.best_mv {
        best_line.push(best_mv);
        let anti_mv = best_mv.make(board);
        probe_pv(board, state, &mut best_line);
        anti_mv.unmake(board);
    }

    SearchResult {
        pv: best_line,
        eval: res.eval,
        depth: res.depth / ONE_PLY,
    }
}

/// Find the best move.
pub fn best_move(board: &mut Board, engine_state: &mut EngineState) -> Option<Move> {
    let res = iter_deep(board, engine_state);
    res.best_mv
}

/// Find the best continuation.
pub fn best_line(board: &mut Board, engine_state: &mut EngineState) -> (PVStack, Score) {
    let res = search(board, engine_state);
    (res.pv, res.eval)
}

/// Utility for NNUE training set generation to determine if a position is quiet or not.
///
/// Our definition of "quiet" is that there are no checks, and the static and quiescence search
/// evaluations are similar. (See https://arxiv.org/html/2412.17948v1.)
///
/// It is the caller's responsibility to get the search evaluation and pass it to this function.
pub fn is_quiescent_position(board: &mut Board, eval: Score) -> bool {
    // max centipawn value difference to call "similar"
    const THRESHOLD: EvalInt = 120;

    if board.is_check(board.turn) {
        return false;
    }

    if matches!(eval, Score::Checkmate(_)) {
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
        let orig_board = board.clone();
        let _result = best_line(&mut board, &mut engine_state);
        assert_eq!(
            board,
            orig_board,
            "failed eq: '{}' vs '{}'",
            orig_board.to_fen(),
            board.to_fen()
        )
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use crate::hash::PackUnpack;

    #[test]
    fn test_ttable_pack_unpack() {
        let test_cases = [
            TTableEntry {
                best_move: Move::from_uci_algebraic("e2e4").unwrap(),
                eval: Score::Exact(3),
                depth: 2,
                is_qsearch: false,
                node_type: NodeType::PV,
                static_eval: Some(2),
            },
            TTableEntry {
                best_move: Move::from_uci_algebraic("e7e8q").unwrap(),
                eval: Score::Lower(3),
                depth: 0,
                is_qsearch: true,
                node_type: NodeType::NonPV,
                static_eval: None,
            },
            TTableEntry {
                best_move: Move::from_uci_algebraic("e7e8b").unwrap(),
                eval: Score::Upper(-3),
                depth: 5,
                is_qsearch: true,
                node_type: NodeType::NonPV,
                static_eval: Some(-3),
            },
            TTableEntry {
                best_move: Move::from_uci_algebraic("e7e8r").unwrap(),
                eval: Score::Checkmate(-3),
                depth: 5,
                is_qsearch: true,
                node_type: NodeType::NonPV,
                static_eval: Some(-3),
            },
            TTableEntry {
                best_move: Move::from_uci_algebraic("e7e8n").unwrap(),
                eval: Score::Checkmate(3),
                depth: 5,
                is_qsearch: true,
                node_type: NodeType::NonPV,
                static_eval: Some(-3),
            },
        ];

        for tc in test_cases {
            let packed: PackedTTableEntry = tc.pack();
            let unpacked = TTableEntry::unpack(&packed);
            assert_eq!(unpacked, tc);
        }
    }
}
