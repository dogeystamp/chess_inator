/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Move generation.

use crate::fen::ToFen;
use crate::hash::{Zobrist, ZobristTable};
use crate::util::arrayvec::ArrayVec;
use crate::{
    Board, CastleRights, ColPiece, Color, Piece, Square, SquareError, BOARD_HEIGHT, BOARD_WIDTH,
    N_SQUARES,
};
use std::ops::Not;

/// Max moves that can be stored per position.
pub const MAX_MOVES: usize = 256;

/// Internal type alias to help switch vector types easier
pub type MoveList = ArrayVec<MAX_MOVES, Move>;

/// Piece enum specifically for promotions.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PromotePiece {
    Rook,
    Bishop,
    Knight,
    Queen,
}

impl From<PromotePiece> for Piece {
    fn from(value: PromotePiece) -> Self {
        match value {
            PromotePiece::Rook => Piece::Rook,
            PromotePiece::Bishop => Piece::Bishop,
            PromotePiece::Knight => Piece::Knight,
            PromotePiece::Queen => Piece::Queen,
        }
    }
}

impl From<PromotePiece> for char {
    fn from(value: PromotePiece) -> Self {
        Piece::from(value).into()
    }
}

pub struct NonPromotePiece;

impl TryFrom<Piece> for PromotePiece {
    type Error = NonPromotePiece;

    fn try_from(value: Piece) -> Result<Self, Self::Error> {
        match value {
            Piece::Rook => Ok(PromotePiece::Rook),
            Piece::Bishop => Ok(PromotePiece::Bishop),
            Piece::Knight => Ok(PromotePiece::Knight),
            Piece::Queen => Ok(PromotePiece::Queen),
            Piece::King => Err(NonPromotePiece),
            Piece::Pawn => Err(NonPromotePiece),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AntiMoveType {
    Normal,
    /// En passant.
    EnPassant {
        cap: Square,
    },
    /// Pawn promotion.
    Promotion,
    /// King-rook castle. The king is the one considered to move.
    Castle {
        rook_src: Square,
        rook_dest: Square,
    },
}

/// Information used to reverse (unmake) a move.
#[derive(Debug, Clone, Copy)]
pub struct AntiMove {
    dest: Square,
    src: Square,
    /// Captured piece, always assumed to be of enemy color.
    pub(crate) cap: Option<Piece>,
    move_type: AntiMoveType,
    /// Half-move counter prior to this move
    half_moves: usize,
    /// Castling rights prior to this move.
    castle: CastleRights,
    /// En passant target square prior to this move.
    ep_square: Option<Square>,
}

impl AntiMove {
    /// Undo the move.
    pub fn unmake(self, pos: &mut Board) {
        self.unmake_general(pos, true)
    }

    /// Undo a `make_no_update` move.
    pub(crate) fn unmake_no_update(self, pos: &mut Board) {
        self.unmake_general(pos, false)
    }

    /// Undo the move (internal API).
    fn unmake_general(self, pos: &mut Board, update_metrics: bool) {
        if update_metrics {
            Zobrist::toggle_board_info(pos);
        }

        pos.move_piece(self.dest, self.src, update_metrics);
        pos.irreversible_half = self.half_moves;
        pos.castle = self.castle;
        pos.ep_square = self.ep_square;

        /// Restore captured piece at a given square.
        macro_rules! cap_sq {
            ($sq: expr) => {
                if let Some(cap_pc) = self.cap {
                    pos.set_piece(
                        $sq,
                        ColPiece {
                            pc: cap_pc,
                            col: pos.turn.flip(),
                        },
                        update_metrics,
                    );
                }
            };
        }

        pos.turn = pos.turn.flip();
        pos.plies -= 1;

        match self.move_type {
            AntiMoveType::Normal => {
                cap_sq!(self.dest)
            }
            AntiMoveType::EnPassant { cap } => {
                cap_sq!(cap);
            }
            AntiMoveType::Promotion => {
                cap_sq!(self.dest);
                pos.set_piece(
                    self.src,
                    ColPiece {
                        pc: Piece::Pawn,
                        col: pos.turn,
                    },
                    update_metrics,
                );
            }
            AntiMoveType::Castle {
                rook_src,
                rook_dest,
            } => {
                pos.move_piece(rook_dest, rook_src, update_metrics);
            }
        }

        if update_metrics {
            Zobrist::toggle_board_info(pos);
            pos.nnue.unmake();
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub enum MoveType {
    /// Pawn promotes to another piece.
    Promotion(PromotePiece),
    /// Capture, or push move. Includes castling and en-passant too.
    Normal,
}
/// Pseudo-legal move.
///
/// No checking is done when constructing this.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct Move {
    pub src: Square,
    pub dest: Square,
    pub move_type: MoveType,
}

impl Move {
    /// Apply move to a position.
    pub fn make(self, pos: &mut Board) -> AntiMove {
        self.make_general(pos, true)
    }

    /// Apply a move without incremental updates of evaluation state.
    pub(crate) fn make_no_update(self, pos: &mut Board) -> AntiMove {
        self.make_general(pos, false)
    }

    /// Apply move to a position.
    ///
    /// * `update_metrics`, if set false, will disable incremental updates of Zobrist hashes, and
    ///   NNUE state.
    fn make_general(self, pos: &mut Board, update_metrics: bool) -> AntiMove {
        let mut anti_move = AntiMove {
            dest: self.dest,
            src: self.src,
            cap: None,
            move_type: AntiMoveType::Normal,
            half_moves: pos.irreversible_half,
            castle: pos.castle,
            ep_square: pos.ep_square,
        };

        if update_metrics {
            // undo hashes (we will update them at the end of this function)
            Zobrist::toggle_board_info(pos);
            pos.nnue.start_delta();
        }

        // reset en passant
        let ep_square = pos.ep_square;
        pos.ep_square = None;

        pos.plies += 1;

        /// Get the piece at the source square.
        macro_rules! pc_src {
            ($data: ident) => {
                pos.get_piece($data.src)
                    .expect("Move source should have a piece")
            };
        }
        /// Perform sanity checks.
        macro_rules! pc_asserts {
            ($pc_src: ident) => {
                debug_assert_eq!(
                    $pc_src.col,
                    pos.turn,
                    "Moving piece on wrong turn. Move {} -> {} on board '{}'",
                    self.src,
                    self.dest,
                    pos.to_fen()
                );
                debug_assert_ne!(self.src, self.dest, "Moving piece to itself.");
            };
        }

        match self.move_type {
            MoveType::Promotion(to_piece) => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src);
                debug_assert_eq!(pc_src.pc, Piece::Pawn);

                pos.irreversible_half = 0;

                anti_move.move_type = AntiMoveType::Promotion;

                pos.del_piece(self.src, update_metrics);
                let cap_pc = pos.set_piece(
                    self.dest,
                    ColPiece {
                        pc: Piece::from(to_piece),
                        col: pc_src.col,
                    },
                    update_metrics,
                );
                anti_move.cap = cap_pc.map(|pc| pc.pc);
            }
            MoveType::Normal => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src);

                let pc_dest: Option<ColPiece> = pos.get_piece(self.dest);
                anti_move.cap = pc_dest.map(|pc| pc.pc);

                let (src_row, src_col) = self.src.to_row_col_signed();
                let (dest_row, dest_col) = self.dest.to_row_col_signed();

                if matches!(pc_src.pc, Piece::Pawn) {
                    // pawn moves are irreversible
                    pos.irreversible_half = 0;

                    // set en-passant target square
                    if src_row.abs_diff(dest_row) == 2 {
                        let ep_col = src_col;
                        debug_assert_eq!(src_col, dest_col);
                        let ep_row = dest_row
                            + match pc_src.col {
                                Color::White => -1,
                                Color::Black => 1,
                            };
                        let ep_targ = Square::from_row_col_signed(ep_row, ep_col)
                            .expect("En-passant target should be valid.");
                        pos.ep_square = Some(ep_targ)
                    } else if pc_dest.is_none() && src_col != dest_col {
                        // we took en passant
                        debug_assert!(src_row.abs_diff(dest_row) == 1);
                        assert_eq!(
                            self.dest,
                            ep_square.expect("ep target should exist if taking ep")
                        );
                        // square to actually capture at
                        let ep_capture = Square::try_from(match pc_src.col {
                            Color::White => usize::from(self.dest.0) - BOARD_WIDTH,
                            Color::Black => usize::from(self.dest.0) + BOARD_WIDTH,
                        })
                        .expect("En-passant capture square should be valid");

                        anti_move.move_type = AntiMoveType::EnPassant { cap: ep_capture };
                        if let Some(pc_cap) = pos.del_piece(ep_capture, update_metrics) {
                            debug_assert_eq!(
                                pc_cap.col,
                                pos.turn.flip(),
                                "attempt to en passant wrong color, pos '{}', move {:?}",
                                pos.to_fen(),
                                self
                            );
                            anti_move.cap = Some(pc_cap.pc);
                        } else {
                            panic!(
                                "En-passant capture square should have piece. Position '{}', move {:?}",
                                pos.to_fen(),
                                self
                            );
                        }
                    }
                } else {
                    pos.irreversible_half += 1;
                }

                if pc_dest.is_some() {
                    // captures are irreversible
                    pos.irreversible_half = 0;
                }

                let castle = &mut pos.castle[pc_src.col];
                if matches!(pc_src.pc, Piece::King) {
                    // forfeit castling rights
                    castle.k = false;
                    castle.q = false;

                    // and maybe perform a castle
                    let horiz_diff = src_col.abs_diff(dest_col);
                    if horiz_diff == 2 {
                        let rook_row = src_row;
                        let rook_src_col = if src_col > dest_col {
                            0
                        } else {
                            isize::try_from(BOARD_WIDTH).unwrap() - 1
                        };
                        let rook_dest_col = if src_col > dest_col {
                            dest_col + 1
                        } else {
                            dest_col - 1
                        };
                        let rook_src = Square::from_row_col_signed(rook_row, rook_src_col)
                            .expect("rook castling src square should be valid");
                        let rook_dest = Square::from_row_col_signed(rook_row, rook_dest_col)
                            .expect("rook castling dest square should be valid");
                        debug_assert!(pos.get_piece(rook_src).is_some(), "rook castling src square has no rook (move: {rook_src} -> {rook_dest})");
                        anti_move.move_type = AntiMoveType::Castle {
                            rook_src,
                            rook_dest,
                        };
                        pos.move_piece(rook_src, rook_dest, update_metrics);
                    }
                    debug_assert!(
                        (0..=2).contains(&horiz_diff),
                        "king moved horizontally {} squares",
                        horiz_diff
                    );
                } else if matches!(pc_src.pc, Piece::Rook) {
                    // forfeit castling rights
                    match pc_src.col {
                        Color::White => {
                            if self.src == Square(0) {
                                castle.q = false;
                            } else if self.src == Square::try_from(BOARD_WIDTH - 1).unwrap() {
                                castle.k = false;
                            };
                        }
                        Color::Black => {
                            if self.src
                                == Square::try_from((BOARD_HEIGHT - 1) * BOARD_WIDTH).unwrap()
                            {
                                castle.q = false;
                            } else if self.src == Square::try_from(N_SQUARES - 1).unwrap() {
                                castle.k = false;
                            };
                        }
                    }
                }

                pos.move_piece(self.src, self.dest, update_metrics);
            }
        };

        pos.turn = pos.turn.flip();

        if update_metrics {
            // redo hashes (we undid them at the start of this function)
            Zobrist::toggle_board_info(pos);
            pos.nnue.commit_delta();
        }

        anti_move
    }
}

/// Convert from UCI long algebraic move notation.
pub trait FromUCIAlgebraic {
    type Error;
    fn from_uci_algebraic(value: &str) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

/// Convert to UCI long algebraic move notation.
pub trait ToUCIAlgebraic {
    fn to_uci_algebraic(&self) -> String;
}

#[derive(Debug)]
pub enum MoveAlgebraicError {
    /// String is invalid length; refuse to parse
    InvalidLength(usize),
    /// Invalid character at given index.
    InvalidCharacter(usize),
    /// Can't promote to a given piece (char at given index).
    InvalidPromotePiece(usize),
    /// Could not parse square string at a certain index.
    SquareError(usize, SquareError),
}

impl FromUCIAlgebraic for Move {
    type Error = MoveAlgebraicError;

    fn from_uci_algebraic(value: &str) -> Result<Self, Self::Error> {
        let value_len = value.len();
        if !(4..=5).contains(&value_len) {
            return Err(MoveAlgebraicError::InvalidLength(value_len));
        }

        let src_sq = match value[0..=1].parse::<Square>() {
            Ok(sq) => sq,
            Err(e) => {
                return Err(MoveAlgebraicError::SquareError(0, e));
            }
        };

        let dest_sq = match value[2..=3].parse::<Square>() {
            Ok(sq) => sq,
            Err(e) => {
                return Err(MoveAlgebraicError::SquareError(0, e));
            }
        };

        let mut move_type = MoveType::Normal;

        if value_len == 5 {
            let promote_char = value.as_bytes()[4] as char;

            let err = Err(MoveAlgebraicError::InvalidCharacter(4));
            let pc = Piece::try_from(promote_char).or(err)?;

            let err = Err(MoveAlgebraicError::InvalidPromotePiece(4));
            move_type = MoveType::Promotion(PromotePiece::try_from(pc).or(err)?);
        }

        Ok(Move {
            src: src_sq,
            dest: dest_sq,
            move_type,
        })
    }
}

impl ToUCIAlgebraic for Move {
    fn to_uci_algebraic(&self) -> String {
        let prom_str = match self.move_type {
            MoveType::Promotion(promote_piece) => char::from(promote_piece).to_string(),
            _ => "".to_string(),
        };

        format!("{}{}{}", self.src, self.dest, prom_str)
    }
}

/// Lookup tables for move generation.
mod lookup_tables {
    use super::{DIRS_DIAG, DIRS_KNIGHT, DIRS_STAR, DIRS_STRAIGHT};
    use crate::prelude::*;
    use crate::{Bitboard, SquareIdx};

    /// Trace a ray between two squares (excluding endpoints).
    const fn iter_ray(
        mut r1: isize,
        mut c1: isize,
        r2: isize,
        c2: isize,
        delta_r: isize,
        delta_c: isize,
    ) -> Bitboard {
        let mut ret = Bitboard(0);
        loop {
            r1 += delta_r;
            c1 += delta_c;
            if r1 == r2 && c1 == c2 {
                break;
            }
            if let Ok(sq) = Square::from_row_col_signed(r1, c1) {
                ret.on_sq(sq)
            } else {
                panic!("Ray went out of bounds.")
            }
        }
        ret
    }

    /// Generate table for rays between two squares.
    const fn gen_rays() -> [[Bitboard; N_SQUARES]; N_SQUARES] {
        let mut ret: [[Bitboard; N_SQUARES]; N_SQUARES] = [[Bitboard(0); N_SQUARES]; N_SQUARES];
        let mut i = 0;
        while i < N_SQUARES {
            let (r1, c1) = Square(i as SquareIdx).to_row_col_signed();
            let mut j = 0;
            while j < N_SQUARES {
                if i == j {
                    j += 1;
                    continue;
                }
                if i > j {
                    ret[i][j] = ret[j][i];
                }

                let (r2, c2) = Square(j as SquareIdx).to_row_col_signed();

                const fn sign(x: isize) -> isize {
                    if x < 0 {
                        -1
                    } else if x == 0 {
                        0
                    } else {
                        1
                    }
                }

                if r1 == r2 || c1 == c2 || r1.abs_diff(r2) == c1.abs_diff(c2) {
                    ret[i][j] = iter_ray(r1, c1, r2, c2, sign(r2 - r1), sign(c2 - c1));
                } else {
                    // two squares have no line (straight or diagonal) between them
                }

                j += 1;
            }
            i += 1;
        }
        ret
    }

    /// Generate knight move lookup table.
    const fn gen_knight() -> [Bitboard; N_SQUARES] {
        let mut ret: [Bitboard; N_SQUARES] = [Bitboard(0); N_SQUARES];
        let mut i = 0;
        while i < N_SQUARES {
            let (r, c) = Square(i as SquareIdx).to_row_col_signed();
            let mut j = 0;
            while j < DIRS_KNIGHT.len() {
                let delta = DIRS_KNIGHT[j];
                let (nr, nc) = (r + delta.0, c + delta.1);
                if let Ok(dest_sq) = Square::from_row_col_signed(nr, nc) {
                    ret[i].on_sq(dest_sq);
                }
                j += 1;
            }
            i += 1;
        }
        ret
    }

    /// Generate slider piece (queen, bishop, rook, king) lookup table.
    const fn gen_slider<const N: usize>(
        dirs: [(isize, isize); N],
        keep_going: bool,
    ) -> [Bitboard; N_SQUARES] {
        let mut ret: [Bitboard; N_SQUARES] = [Bitboard(0); N_SQUARES];
        let mut i = 0;
        while i < N_SQUARES {
            let (r, c) = Square(i as SquareIdx).to_row_col_signed();
            let mut j = 0;
            while j < dirs.len() {
                let delta = dirs[j];
                let (mut nr, mut nc) = (r, c);
                loop {
                    (nr, nc) = (nr + delta.0, nc + delta.1);
                    if let Ok(dest_sq) = Square::from_row_col_signed(nr, nc) {
                        ret[i].on_sq(dest_sq);
                    } else {
                        break;
                    }
                    if !keep_going {
                        break;
                    }
                }
                j += 1;
            }
            i += 1;
        }
        ret
    }

    pub const RAYS: [[Bitboard; N_SQUARES]; N_SQUARES] = gen_rays();

    pub const MOVE_TABLES: [[Bitboard; N_SQUARES]; N_PIECES - 1] = [
        // rook
        gen_slider(DIRS_STRAIGHT, true),
        // bishop
        gen_slider(DIRS_DIAG, true),
        // knight
        gen_knight(),
        // king
        gen_slider(DIRS_STAR, false),
        // queen
        gen_slider(DIRS_STAR, true),
        // pawn is handled separately because it moves weird
    ];
}

pub trait GenAttackers {
    /// Generate attackers/attacks for a given square.
    ///
    /// # Arguments
    ///
    /// * `dest`: Square that is attacked.
    /// * `single`: Exit early if any attack is found.
    /// * `filter_color`: Matches only attackers of this color, if given.
    fn gen_attackers(
        &self,
        dest: Square,
        single: bool,
        filter_color: Option<Color>,
    ) -> ArrayVec<MAX_MOVES, (ColPiece, Move)>;
}

impl GenAttackers for Board {
    fn gen_attackers(
        &self,
        dest: Square,
        single: bool,
        filter_color: Option<Color>,
    ) -> ArrayVec<MAX_MOVES, (ColPiece, Move)> {
        let mut ret: ArrayVec<MAX_MOVES, (ColPiece, Move)> = ArrayVec::new();

        /// Filter attackers and add them to the return vector.
        ///
        /// Returns true if attacker was added.
        fn push_ans(
            pc: ColPiece,
            sq: Square,
            dest: Square,
            ret: &mut ArrayVec<MAX_MOVES, (ColPiece, Move)>,
            filter_color: Option<Color>,
        ) -> bool {
            if let Some(filter_color) = filter_color {
                if filter_color != pc.col {
                    return false;
                }
            }
            let (r, _c) = dest.to_row_col();
            let is_promotion = matches!(pc.pc, Piece::Pawn) && r == Board::last_rank(pc.col);

            if is_promotion {
                use PromotePiece::*;
                for prom_pc in [Queen, Knight, Rook, Bishop] {
                    ret.push((
                        pc,
                        Move {
                            src: sq,
                            dest,
                            move_type: MoveType::Promotion(prom_pc),
                        },
                    ));
                }
            } else {
                ret.push((
                    pc,
                    Move {
                        src: sq,
                        dest,
                        move_type: MoveType::Normal,
                    },
                ));
            }

            true
        }

        /// Check each square one-by-one for an attacker piece.
        /// TODO: lump this into detect_attacker properly
        macro_rules! find_pawns {
            ($dirs: ident, $pc: pat, $color: pat, $keep_going: expr) => {
                for dir in $dirs.into_iter() {
                    let (mut r, mut c) = dest.to_row_col_signed();
                    loop {
                        let (nr, nc) = (r + dir.0, c + dir.1);
                        if let Ok(sq) = Square::from_row_col_signed(nr, nc) {
                            if let Some(pc) = self.get_piece(sq) {
                                if matches!(pc.pc, $pc) && matches!(pc.col, $color) {
                                    let added = push_ans(pc, sq, dest, &mut ret, filter_color);
                                    if single && added {
                                        return ret;
                                    }
                                }
                                break;
                            }
                        } else {
                            break;
                        }
                        if (!($keep_going)) {
                            break;
                        }
                        r = nr;
                        c = nc;
                    }
                }
            };
        }

        /// Find attackers with line of sight.
        ///
        /// Does not work with pawns.
        fn detect_attacker(
            board: &Board,
            dest: Square,
            pc: Piece,
            ret: &mut ArrayVec<MAX_MOVES, (ColPiece, Move)>,
            filter_color: Option<Color>,
            use_line_of_sight: bool,
            single: bool,
        ) {
            for col in [Color::White, Color::Black] {
                if filter_color.is_none_or(|x| x == col) {
                    let attackers =
                        board[col][pc] & lookup_tables::MOVE_TABLES[pc as usize][usize::from(dest)];
                    for src in attackers {
                        if use_line_of_sight
                            && (lookup_tables::RAYS[usize::from(src)][usize::from(dest)]
                                & board.occupancy)
                                .is_empty()
                                .not()
                        {
                            // no line of sight; not an attacker
                            continue;
                        }
                        ret.push((
                            ColPiece { pc, col },
                            Move {
                                src,
                                dest,
                                move_type: MoveType::Normal,
                            },
                        ));
                        if single {
                            return;
                        }
                    }
                }
            }
        }

        // inverted because our perspective is from the attacked square
        let dirs_white_pawn = [(-1, 1), (-1, -1)];
        let dirs_black_pawn = [(1, 1), (1, -1)];

        use Piece::*;

        macro_rules! detect {
            ($pc: ident, $line_of_sight: expr) => {
                detect_attacker(
                    self,
                    dest,
                    $pc,
                    &mut ret,
                    filter_color,
                    $line_of_sight,
                    single,
                );
                if single && !ret.is_empty() {
                    return ret;
                }
            };
        }

        detect!(Knight, false);
        detect!(Queen, true);
        detect!(Bishop, true);
        detect!(Rook, true);

        // this shouldn't happen in legal chess but we're using this function in a pseudo-legal
        // move gen context
        detect!(King, false);

        if filter_color.is_none_or(|c| matches!(c, Color::Black)) {
            find_pawns!(dirs_black_pawn, Pawn, Color::Black, false);
        }
        if filter_color.is_none_or(|c| matches!(c, Color::White)) {
            find_pawns!(dirs_white_pawn, Pawn, Color::White, false);
        }

        ret
    }
}

/// Options for movegen.
pub struct MoveGenConfig {
    /// Restricts movegen to only output capture moves.
    ///
    /// This is more efficient than filtering captures after generating moves.
    captures_only: bool,
    legality: MoveGenType,
}

impl Default for MoveGenConfig {
    fn default() -> Self {
        MoveGenConfig {
            captures_only: false,
            legality: MoveGenType::Legal,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MoveGenType {
    /// Legal move generation.
    Legal,
    /// Allow capturing friendly pieces, moving into check, but not castling through check.
    _Pseudo,
}

/// Internal movegen interface with more options
trait MoveGenInternal {
    fn gen_moves_general(&mut self, config: MoveGenConfig) -> MoveList;
}

pub trait MoveGen {
    /// Legal move generation.
    fn gen_moves(&mut self) -> MoveList;

    /// Pseudo-legal move generation (see `MoveGenType::_Pseudo` for more information).
    fn gen_pseudo(&mut self) -> MoveList;

    /// Legal capture generation.
    fn gen_captures(&mut self) -> MoveList;
}

impl<T: MoveGenInternal> MoveGen for T {
    fn gen_moves(&mut self) -> MoveList {
        self.gen_moves_general(MoveGenConfig::default())
    }

    fn gen_pseudo(&mut self) -> MoveList {
        let config = MoveGenConfig {
            legality: MoveGenType::_Pseudo,
            ..Default::default()
        };
        self.gen_moves_general(config)
    }

    fn gen_captures(&mut self) -> MoveList {
        let config = MoveGenConfig {
            captures_only: true,
            ..Default::default()
        };
        self.gen_moves_general(config)
    }
}

pub const DIRS_STRAIGHT: [(isize, isize); 4] = [(0, 1), (1, 0), (-1, 0), (0, -1)];
pub const DIRS_DIAG: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
pub const DIRS_STAR: [(isize, isize); 8] = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1),
];
pub const DIRS_KNIGHT: [(isize, isize); 8] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
];
enum SliderDirection {
    /// Rook movement
    Straight,
    /// Bishop movement
    Diagonal,
    /// Queen/king movement
    Star,
}
/// Generate slider moves for a given square.
///
/// # Arguments
///
/// * `board`: Board to generate moves with.
/// * `src`: Square on which the slider piece is on.
/// * `move_list`: Vector to append generated moves to.
/// * `slide_type`: Directions the piece is allowed to go in.
/// * `keep_going`: Allow sliding more than one square (true for everything except king).
fn move_slider(
    board: &Board,
    src: Square,
    move_list: &mut MoveList,
    slide_type: SliderDirection,
    keep_going: bool,
    config: &MoveGenConfig,
) {
    let dirs = match slide_type {
        SliderDirection::Straight => DIRS_STRAIGHT.iter(),
        SliderDirection::Diagonal => DIRS_DIAG.iter(),
        SliderDirection::Star => DIRS_STAR.iter(),
    };

    for dir in dirs {
        let (mut r, mut c) = src.to_row_col_signed();
        loop {
            // increment
            let nr = r + dir.0;
            let nc = c + dir.1;

            if let Ok(dest) = Square::from_row_col_signed(nr, nc) {
                r = nr;
                c = nc;

                let obstructed = board.get_piece(dest).is_some();

                let mut gen_move = true;

                if config.captures_only && !obstructed {
                    gen_move = false;
                }

                if gen_move {
                    move_list.push(Move {
                        src,
                        dest,
                        move_type: MoveType::Normal,
                    });
                }

                if obstructed {
                    break;
                }
            } else {
                break;
            }

            if !keep_going {
                break;
            }
        }
    }
}

fn is_legal(board: &mut Board, mv: Move) -> bool {
    // mut required for check checking
    // disallow friendly fire
    let src_pc = board
        .get_piece(mv.src)
        .expect("move source should have piece");
    if let Some(dest_pc) = board.get_piece(mv.dest) {
        if dest_pc.col == src_pc.col {
            return false;
        }
    }

    // disallow moving into check
    let anti_move = mv.make_no_update(board);
    let is_check = board.is_check(board.turn.flip());
    anti_move.unmake_no_update(board);
    if is_check {
        return false;
    }

    true
}

impl MoveGenInternal for Board {
    fn gen_moves_general(&mut self, config: MoveGenConfig) -> MoveList {
        let mut ret = MoveList::new();
        let pl = self[self.turn];
        macro_rules! squares {
            ($pc: ident) => {
                pl[Piece::$pc].into_iter()
            };
        }

        for sq in squares!(Rook) {
            move_slider(self, sq, &mut ret, SliderDirection::Straight, true, &config);
        }
        for sq in squares!(Bishop) {
            move_slider(self, sq, &mut ret, SliderDirection::Diagonal, true, &config);
        }
        for sq in squares!(Queen) {
            move_slider(self, sq, &mut ret, SliderDirection::Star, true, &config);
        }
        for src in squares!(King) {
            move_slider(self, src, &mut ret, SliderDirection::Star, false, &config);

            if config.captures_only {
                // castling can't capture
                continue;
            }

            let (r, c) = src.to_row_col_signed();
            let rights = self.castle[self.turn];
            let castle_sides = [(rights.k, 2, BOARD_WIDTH as isize - 1), (rights.q, -2, 0)];
            for (is_allowed, move_offset, endpoint) in castle_sides {
                if !is_allowed {
                    continue;
                }

                let (rook_r, rook_c) = (r, endpoint);
                let rook_sq = Square::from_row_col_signed(rook_r, rook_c).unwrap();
                let rook_exists = self
                    .get_piece(rook_sq)
                    .map_or(false, |pc| pc.pc == Piece::Rook);
                if !rook_exists {
                    continue;
                }

                let path_range = if c < endpoint {
                    (c + 1)..endpoint
                } else {
                    (endpoint + 1)..c
                };

                debug_assert_ne!(
                    path_range.len(),
                    0,
                    "c {:?}, endpoint {:?}, range {:?}",
                    c,
                    endpoint,
                    path_range
                );

                let mut path_squares =
                    path_range.map(|nc| Square::from_row_col_signed(r, nc).unwrap());
                debug_assert_ne!(path_squares.len(), 0);

                // find first blocking piece
                let is_path_blocked = path_squares.find_map(|sq| self.get_piece(sq)).is_some();
                if is_path_blocked {
                    continue;
                }

                let nc: isize = c + move_offset;
                let dest = Square::from_row_col_signed(r, nc)
                    .expect("Castle destination square should be valid");

                debug_assert!(c.abs_diff(nc) == 2);

                // ensure the path is not being attacked (castle through check)
                let check_range = if c < nc { c..=nc } else { nc..=c };
                debug_assert!(!check_range.is_empty());
                let is_any_checked = check_range
                    .map(|nc| Square::from_row_col_signed(r, nc).unwrap())
                    .map(|dest| {
                        let cap_pc = self.move_piece(src, dest, false);

                        let ret = self.is_check(self.turn);

                        let orig_pc = self.set_square(dest, cap_pc, false);
                        self.set_square(src, orig_pc, false);
                        ret
                    })
                    .any(|x| x);
                if is_any_checked {
                    continue;
                }

                ret.push(Move {
                    src,
                    dest,
                    move_type: MoveType::Normal,
                })
            }
        }
        for src in squares!(Pawn) {
            let (r, c) = src.to_row_col_signed();

            let last_row = isize::try_from(Board::last_rank(self.turn)).unwrap();
            let ep_row = isize::try_from(Board::ep_rank(self.turn)).unwrap();

            let nr = r + isize::from(self.turn.sign());
            let is_promotion = nr == last_row;

            macro_rules! push_moves {
                ($src: ident, $dest: ident) => {
                    if is_promotion {
                        use PromotePiece::*;
                        for prom_pc in [Queen, Knight, Rook, Bishop] {
                            ret.push(Move {
                                $src,
                                $dest,
                                move_type: MoveType::Promotion(prom_pc),
                            });
                        }
                    } else {
                        ret.push(Move {
                            $src,
                            $dest,
                            move_type: MoveType::Normal,
                        });
                    }
                };
            }

            // capture
            for horiz in [-1, 1] {
                let nc = c + horiz;
                let dest = match Square::from_row_col_signed(nr, nc) {
                    Ok(sq) => sq,
                    Err(_) => continue,
                };
                if self.get_piece(dest).is_some() || (r == ep_row && self.ep_square == Some(dest)) {
                    push_moves!(src, dest);
                }
            }

            if config.captures_only {
                continue;
            }

            // single push
            let nc = c;
            let dest = match Square::from_row_col_signed(nr, nc) {
                Ok(sq) => sq,
                Err(_) => continue,
            };

            if self.get_piece(dest).is_none() {
                push_moves!(src, dest);

                // double push
                if r == match self.turn {
                    Color::White => 1,
                    Color::Black => isize::try_from(BOARD_HEIGHT).unwrap() - 2,
                } {
                    let nr = (r)
                        + match self.turn {
                            Color::White => 2,
                            Color::Black => -2,
                        };
                    let nc = c;
                    let dest = Square::from_row_col_signed(nr, nc)
                        .expect("Pawn double push should have valid destination");
                    if self.get_piece(dest).is_none() {
                        push_moves!(src, dest);
                    }
                }
            }
        }
        for src in squares!(Knight) {
            let (r, c) = src.to_row_col_signed();

            for dir in DIRS_KNIGHT {
                let nr = r + dir.0;
                let nc = c + dir.1;
                if let Ok(dest) = Square::from_row_col_signed(nr, nc) {
                    if config.captures_only && self.get_piece(dest).is_none() {
                        continue;
                    }
                    ret.push(Move {
                        src,
                        dest,
                        move_type: MoveType::Normal,
                    })
                }
            }
        }
        ret.retain(move |mv| match config.legality {
            MoveGenType::Legal => is_legal(self, *mv),
            MoveGenType::_Pseudo => true,
        });
        ret
    }
}

fn perft_internal(
    depth: usize,
    pos: &mut Board,
    cache: &mut ZobristTable<(usize, usize)>,
) -> usize {
    if let Some((ans, cache_depth)) = cache[pos.zobrist] {
        if depth == cache_depth {
            return ans;
        }
    }

    if depth == 0 {
        return 1;
    };

    let mut ans = 0;

    let moves: MoveList = pos.gen_moves();
    for mv in moves {
        let anti_move = mv.make(pos);
        ans += perft_internal(depth - 1, pos, cache);
        anti_move.unmake(pos);
    }

    cache[pos.zobrist] = Some((ans, depth));
    ans
}

/// How many nodes at depth N can be reached from this position.
pub fn perft(depth: usize, pos: &mut Board) -> usize {
    let mut cache = ZobristTable::new(23);
    perft_internal(depth, pos, &mut cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{FromFen, ToFen, START_POSITION};

    #[test]
    /// Ensure that bitboard properly reflects captures.
    fn test_bitboard_capture() {
        let mut pos = Board::from_fen("8/8/8/8/8/8/r7/R7 w - - 0 1").unwrap();
        let mv = Move::from_uci_algebraic("a1a2").unwrap();
        let _anti_move = mv.make(&mut pos);

        use std::collections::hash_set::HashSet;
        use Piece::*;
        for pc in [Rook, Bishop, Knight, Queen, King, Pawn] {
            let white: HashSet<_> = pos[Color::White][pc].into_iter().collect();
            let black: HashSet<_> = pos[Color::Black][pc].into_iter().collect();
            let intersect = white.intersection(&black).collect::<Vec<_>>();
            assert!(
                intersect.is_empty(),
                "Bitboard in illegal state: {pc:?} collides at {}",
                intersect[0]
            );
        }
    }

    /// Helper to produce test cases.
    fn decondense_moves(test_case: (&str, Vec<(&str, Vec<&str>, MoveType)>)) -> (Board, Vec<Move>) {
        let (fen, expected) = test_case;
        let board = Board::from_fen(fen).unwrap();

        let mut expected_moves = expected
            .iter()
            .map(|(src, dests, move_type)| {
                let src = src.parse::<Square>().unwrap();
                let dests = dests
                    .iter()
                    .map(|x| x.parse::<Square>())
                    .map(|x| x.unwrap());
                dests.map(move |dest| Move {
                    src,
                    dest,
                    move_type: *move_type,
                })
            })
            .flatten()
            .collect::<Vec<Move>>();

        expected_moves.sort_unstable();
        (board, expected_moves)
    }

    /// Generate new test cases by flipping colors on existing ones.
    fn flip_test_case(board: Board, moves: &Vec<Move>) -> (Board, Vec<Move>) {
        let mut move_vec = moves
            .iter()
            .map(|mv| Move {
                src: mv.src.mirror_vert(),
                dest: mv.dest.mirror_vert(),
                move_type: mv.move_type,
            })
            .collect::<Vec<Move>>();
        move_vec.sort_unstable();
        (board.flip_colors(), move_vec)
    }

    /// Test movegen through contrived positions.
    #[test]
    fn test_movegen() {
        let test_cases = [
            // rook test
            (
                // start position
                "8/8/8/8/8/8/8/R7 w - - 0 1",
                // expected moves
                vec![(
                    // source piece
                    "a1",
                    // destination squares
                    vec![
                        "a2", "a3", "a4", "a5", "a6", "a7", "a8", "b1", "c1", "d1", "e1", "f1",
                        "g1", "h1",
                    ],
                    MoveType::Normal,
                )],
            ),
            // white castle test (blocked)
            (
                "8/8/8/8/8/8/r6r/R3Kn1R w KQ - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    // NOTE: pseudo-legal f1
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                    // NOTE: pseudo-legal d2, e2, f2, f1
                    (
                        "e1",
                        vec!["c1", "d1", "f1", "d2", "e2", "f2"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // white castle test (blocked again)
            (
                "8/8/8/8/8/8/r6r/R3K1nR w KQ - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    ("h1", vec!["g1", "h2"], MoveType::Normal),
                    // NOTE: pseudo-legal d2, e2, f2, f1
                    (
                        "e1",
                        vec!["c1", "d1", "f1", "d2", "e2", "f2"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // white castle test (no rights, blocked)
            (
                "8/8/8/8/8/8/r6r/R3Kn1R w K - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    // NOTE: pseudo-legal f1
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                    // NOTE: pseudo-legal d2, e2, f2, f1
                    ("e1", vec!["d1", "f1", "d2", "e2", "f2"], MoveType::Normal),
                ],
            ),
            // white castle test
            (
                "8/8/8/8/8/8/r6r/R3K2R w KQ - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    // NOTE: pseudo-legal e1
                    ("h1", vec!["g1", "f1", "e1", "h2"], MoveType::Normal),
                    // NOTE: pseudo-legal d2, e2, f2
                    (
                        "e1",
                        vec!["g1", "c1", "d1", "f1", "d2", "e2", "f2"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // black castle test
            (
                "r3k2r/R6R/8/8/8/8/8/8 b kq - 0 1",
                vec![
                    // NOTE: pseudo-legal e8
                    ("a8", vec!["b8", "a7", "c8", "d8", "e8"], MoveType::Normal),
                    // NOTE: pseudo-legal e8
                    ("h8", vec!["g8", "f8", "e8", "h7"], MoveType::Normal),
                    // NOTE: pseudo-legal d7, e7, f7
                    (
                        "e8",
                        vec!["g8", "c8", "d8", "f8", "d7", "e7", "f7"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // horse test
            (
                "8/2r1r3/1n3q2/3N4/8/2p5/8/8 w - - 0 1",
                vec![(
                    "d5",
                    vec!["f6", "e7", "c7", "b6", "b4", "c3", "e3", "f4"],
                    MoveType::Normal,
                )],
            ),
            // horse (blocked by boundary)
            (
                "8/8/8/8/8/6n1/5n2/7N w - - 0 1",
                vec![("h1", vec!["f2", "g3"], MoveType::Normal)],
            ),
            // white pawn promotion
            (
                "q1q5/1P6/8/8/8/8/8/8 w - - 0 1",
                vec![
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Knight)),
                ],
            ),
            // black pawn promotion
            (
                "8/8/8/8/8/8/1p6/Q1Q5 b - - 0 1",
                vec![
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Knight)),
                ],
            ),
            // white pawn push/capture
            (
                "8/8/8/8/8/p1p5/1P6/8 w - - 0 1",
                vec![("b2", vec!["a3", "c3", "b3", "b4"], MoveType::Normal)],
            ),
            // white pawn en passant
            (
                "8/8/4p3/3pP3/8/8/8/8 w - d6 0 1",
                vec![("e5", vec!["d6"], MoveType::Normal)],
            ),
            // white pawn blocked
            ("8/8/8/8/8/1p6/1P6/8 w - - 0 1", vec![]),
            // white pawn blocked (partially)
            (
                "8/8/8/8/1p6/8/1P6/8 w - - 0 1",
                vec![("b2", vec!["b3"], MoveType::Normal)],
            ),
            // black pawn push/capture
            (
                "8/1p6/P1P5/8/8/8/8/8 b - - 0 1",
                vec![("b7", vec!["a6", "c6", "b6", "b5"], MoveType::Normal)],
            ),
            // black pawn en passant
            (
                "8/8/8/8/pP6/P7/8/8 b - b3 0 1",
                vec![("a4", vec!["b3"], MoveType::Normal)],
            ),
            // black pawn blocked
            ("8/1p6/1P6/8/8/8/8/8 b - - 0 1", vec![]),
            // king against the boundary
            (
                "3K4/4p3/1q3p2/4p3/1p1r4/8/8/8 w - - 0 1",
                vec![("d8", vec!["c8", "c7", "d7", "e7", "e8"], MoveType::Normal)],
            ),
            // king test
            (
                "8/4p3/1q1K1p2/4p3/1p1r4/8/8/8 w - - 0 1",
                vec![(
                    "d6",
                    vec!["c7", "c6", "c5", "d7", "d5", "e7", "e6", "e5"],
                    MoveType::Normal,
                )],
            ),
            // queen test
            (
                "8/4p3/1q1Q1p2/4p3/1p1r4/8/8/8 w - - 0 1",
                vec![(
                    "d6",
                    vec![
                        "d5", "d4", "d7", "d8", "e7", "c5", "b4", "e6", "f6", "c6", "b6", "e5",
                        "c7", "b8",
                    ],
                    MoveType::Normal,
                )],
            ),
            // rook test (again)
            (
                "8/1p6/8/1R2p3/1p6/8/8/8 w - - 0 1",
                vec![(
                    "b5",
                    vec!["b6", "b7", "b4", "a5", "c5", "d5", "e5"],
                    MoveType::Normal,
                )],
            ),
            // bishop test
            (
                "8/4p3/3B4/4p3/1p6/8/8/8 w - - 0 1",
                vec![(
                    "d6",
                    vec!["e5", "e7", "c5", "b4", "c7", "b8"],
                    MoveType::Normal,
                )],
            ),
            // black test
            (
                "8/3b4/2R1R3/1Q6/1RqRrR2/1QQ5/2R1R3/k7 b - - 0 1",
                vec![
                    ("a1", vec!["a2", "b2", "b1"], MoveType::Normal),
                    (
                        "c4",
                        vec![
                            "b3", "b4", "b5", "c3", "c5", "c6", "d3", "e2", "d4", "d5", "e6",
                        ],
                        MoveType::Normal,
                    ),
                    (
                        "e4",
                        vec!["d4", "f4", "e3", "e2", "e5", "e6"],
                        MoveType::Normal,
                    ),
                    ("d7", vec!["c6", "e6", "c8", "e8"], MoveType::Normal),
                ],
            ),
        ];

        let test_cases = test_cases.map(decondense_moves);

        let augmented_test_cases = test_cases.clone().map(|tc| flip_test_case(tc.0, &tc.1));
        let all_cases = [augmented_test_cases, test_cases].concat();

        for (mut board, expected_moves) in all_cases {
            let mut moves: Vec<Move> = board.gen_pseudo().into_iter().collect();
            moves.sort_unstable();
            let moves = moves;

            assert_eq!(moves, expected_moves, "failed tc {}", board.to_fen());
        }
    }

    /// Test check checker.
    #[test]
    fn test_is_check() {
        let check_cases = [
            "3r4/8/8/3K4/8/8/8/8 b - - 0 1",
            "8/8/8/3K3r/8/8/8/8 b - - 0 1",
            "8/8/8/3K4/8/8/8/3r4 b - - 0 1",
            "8/8/8/r2K4/8/8/8/8 b - - 0 1",
            "1b6/8/8/3K4/1r6/8/8/k6b b - - 0 1",
            "1b6/8/4p3/3K4/1r6/8/8/k5b1 b - - 0 1",
            "1b6/4n3/3p4/3K4/1r6/8/8/k5b1 b - - 0 1",
            "1b6/2n5/3p4/3K4/1r6/8/8/k5b1 b - - 0 1",
            "1b6/8/3p4/3K4/1r3n2/8/8/k5b1 b - - 0 1",
            "1b6/8/3p4/3K4/1r1k4/5n2/8/6b1 b - - 0 1",
            "8/8/8/4b3/r2b4/rnq5/PP6/KRrr4 w - - 0 1",
        ]
        .map(|tc| (tc, true));

        let not_check_cases = [
            "1b6/8/3p4/3K4/1r6/5n2/8/k5b1 b - - 0 1",
            "1bqnb3/3q1n2/3p4/3K4/1r6/2q1qn2/8/k5b1 b - - 0 1",
            "1bqnb1q1/3q1n2/3p4/1qbKp1q1/1r1b4/2q1qn2/8/k5b1 b - - 0 1",
            "8/8/8/4b3/r2b4/r1q5/PP6/KRrr4 w - - 0 1",
        ]
        .map(|tc| (tc, false));

        let all_cases = check_cases.iter().chain(&not_check_cases);
        for (fen, expected) in all_cases {
            let board = Board::from_fen(fen).unwrap();
            eprintln!(
                "got attackers {:?} for {}",
                board
                    .gen_attackers(
                        board[Color::White][Piece::King].into_iter().next().unwrap(),
                        false,
                        Some(Color::Black)
                    )
                    .into_iter()
                    .collect::<Vec<_>>(),
                fen
            );
            assert_eq!(board.is_check(Color::White), *expected, "failed on {}", fen);

            let board_anti = board.flip_colors();
            assert_eq!(
                board_anti.is_check(Color::Black),
                *expected,
                "failed on anti-version of {} ({})",
                fen,
                board_anti.to_fen()
            );
        }
    }

    /// Test legal movegen through contrived positions.
    #[test]
    fn test_legal_movegen() {
        let test_cases = [
            // rook friendly fire test
            (
                // start position
                "8/8/8/8/8/8/rr6/RRr5 w - - 0 1",
                // expected moves
                vec![
                    (
                        // source piece
                        "a1",
                        // destination squares
                        vec!["a2"],
                        MoveType::Normal,
                    ),
                    (
                        // source piece
                        "b1",
                        // destination squares
                        vec!["b2", "c1"],
                        MoveType::Normal,
                    ),
                ],
            ),
            (
                "rnbqkbnr/p1pppppp/8/1P6/8/8/1PPPPPPP/RNBQKBNR b KQkq - 0 2",
                vec![
                    ("a7", vec!["a6", "a5"], MoveType::Normal),
                    ("c7", vec!["c6", "c5"], MoveType::Normal),
                    ("d7", vec!["d6", "d5"], MoveType::Normal),
                    ("e7", vec!["e6", "e5"], MoveType::Normal),
                    ("f7", vec!["f6", "f5"], MoveType::Normal),
                    ("g7", vec!["g6", "g5"], MoveType::Normal),
                    ("h7", vec!["h6", "h5"], MoveType::Normal),
                    ("g8", vec!["h6", "f6"], MoveType::Normal),
                    ("b8", vec!["c6", "a6"], MoveType::Normal),
                    ("c8", vec!["b7", "a6"], MoveType::Normal),
                ],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/6rr/4K2R w KQ - 0 1",
                vec![
                    ("e1", vec!["d1", "f1"], MoveType::Normal),
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                ],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/5r1r/4K2R w KQ - 0 1",
                vec![
                    ("e1", vec!["d1"], MoveType::Normal),
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                ],
            ),
            // castling while checked
            (
                "8/8/8/8/8/8/rrrrr2r/4K2R w KQ - 0 1",
                vec![("e1", vec!["f1"], MoveType::Normal)],
            ),
            // castling while checked
            (
                "8/8/8/8/8/8/r3rrrr/R3K3 w KQ - 0 1",
                vec![("e1", vec!["d1"], MoveType::Normal)],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/r1r5/R3K3 w KQ - 0 1",
                vec![
                    ("e1", vec!["d1", "f1"], MoveType::Normal),
                    ("a1", vec!["a2", "b1", "c1", "d1"], MoveType::Normal),
                ],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/r2r4/R3K3 w KQ - 0 1",
                vec![
                    ("e1", vec!["f1"], MoveType::Normal),
                    ("a1", vec!["a2", "b1", "c1", "d1"], MoveType::Normal),
                ],
            ),
            // check test
            (
                "1bqnb1q1/3q1n2/q2p4/2bKp1q1/1r1b4/1q3n2/8/k2q2b1 w - - 0 1",
                vec![("d5", vec!["e4"], MoveType::Normal)],
            ),
            // check test
            (
                "1b1nb1q1/q2q1n2/q2p4/2bKp1q1/1r1p4/1q3n2/8/k2q2b1 w - - 0 1",
                vec![("d5", vec!["e4"], MoveType::Normal)],
            ),
        ];

        let test_cases = test_cases.map(|tc| decondense_moves(tc));
        let augmented_test_cases = test_cases.clone().map(|tc| flip_test_case(tc.0, &tc.1));

        let all_cases = [augmented_test_cases, test_cases].concat();

        for (mut board, mut expected_moves) in all_cases {
            eprintln!("on test '{}'", board.to_fen());
            expected_moves.sort_unstable();
            let expected_moves = expected_moves;

            let mut moves: Vec<Move> = board.gen_moves().into_iter().collect();
            moves.sort_unstable();
            let moves = moves;

            assert_eq!(moves, expected_moves);
        }
    }

    /// Test that make move and unmake move work as expected.
    ///
    /// Ensure that:
    /// - En passant target is appropriately set
    /// - Castling rights are respected
    /// - Half-moves since last irreversible move counter is maintained
    #[test]
    fn test_make_unmake() {
        // FENs made with https://lichess.org/analysis
        // En-passant target square is manually added, since Lichess doesn't have it when
        // en-passant is not legal.
        let test_cases = [
            (
                START_POSITION,
                vec![
                    // (src, dest, expected fen)
                    (
                        "e2e4",
                        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                    ),
                    (
                        "e7e5",
                        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                    ),
                    (
                        "g1f3",
                        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
                    ),
                    (
                        "g8f6",
                        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                    ),
                    (
                        "f1c4",
                        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
                    ),
                    (
                        "f8c5",
                        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                    ),
                    (
                        "d1e2",
                        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPPQPPP/RNB1K2R b KQkq - 5 4",
                    ),
                    (
                        "d8e7",
                        "rnb1k2r/ppppqppp/5n2/2b1p3/2B1P3/5N2/PPPPQPPP/RNB1K2R w KQkq - 6 5",
                    ),
                    (
                        "f3e5",
                        "rnb1k2r/ppppqppp/5n2/2b1N3/2B1P3/8/PPPPQPPP/RNB1K2R b KQkq - 0 5",
                    ),
                    (
                        "e7e5",
                        "rnb1k2r/pppp1ppp/5n2/2b1q3/2B1P3/8/PPPPQPPP/RNB1K2R w KQkq - 0 6",
                    ),
                ],
            ),
            // promotion unmake regression test
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBPPP3/q4N2/Pp4PP/R2Q1RK1 b kq - 0 1",
                vec![(
                    "b2a1q",
                    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBPPP3/q4N2/P5PP/q2Q1RK1 w kq - 0 2",
                )],
            ),
            // castling rights test (kings)
            (
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                vec![
                    (
                        "e1e2",
                        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2",
                    ),
                    (
                        "e8e7",
                        "rnbq1bnr/ppppkppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w - - 2 3",
                    ),
                ],
            ),
            // pawn promotion test
            (
                "4k3/6P1/8/8/8/8/1p6/4K3 w - - 0 1",
                vec![
                    ("g7g8n", "4k1N1/8/8/8/8/8/1p6/4K3 b - - 0 1"),
                    ("b2b1q", "4k1N1/8/8/8/8/8/8/1q2K3 w - - 0 2"),
                ],
            ),
            // en passant test
            (
                "k7/4p3/8/3P4/3p4/8/4P3/K7 w - - 0 1",
                vec![
                    ("e2e4", "k7/4p3/8/3P4/3pP3/8/8/K7 b - e3 0 1"),
                    ("d4e3", "k7/4p3/8/3P4/8/4p3/8/K7 w - - 0 2"),
                    ("a1b1", "k7/4p3/8/3P4/8/4p3/8/1K6 b - - 1 2"),
                    ("e7e5", "k7/8/8/3Pp3/8/4p3/8/1K6 w - e6 0 3"),
                    ("d5e6", "k7/8/4P3/8/8/4p3/8/1K6 b - - 0 3"),
                ],
            ),
            // castle test (white kingside, black queenside)
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                vec![
                    ("e1g1", "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1"),
                    ("e8c8", "2kr3r/8/8/8/8/8/8/R4RK1 w - - 2 2"),
                ],
            ),
            // castle test (white queenside, black kingside)
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                vec![
                    ("e1c1", "r3k2r/8/8/8/8/8/8/2KR3R b kq - 1 1"),
                    ("e8g8", "r4rk1/8/8/8/8/8/8/2KR3R w - - 2 2"),
                ],
            ),
        ];

        for (i, test_case) in test_cases.iter().enumerate() {
            let (start_pos, moves) = test_case;

            eprintln!("Starting test case {i}, make move.");
            let mut pos = Board::from_fen(start_pos).unwrap();
            for (move_str, expect_fen) in moves {
                let prior_fen = pos.to_fen();
                let mv = Move::from_uci_algebraic(move_str).unwrap();
                eprintln!("Moving {move_str} on {}", prior_fen);
                let anti_move = mv.make(&mut pos);
                eprintln!("Unmaking {move_str} on {}.", pos.to_fen());
                anti_move.unmake(&mut pos);
                assert_eq!(
                    pos.to_fen(),
                    prior_fen.to_string(),
                    "failed unmake with {anti_move:?}"
                );
                eprintln!("Remaking {move_str}.");
                let _anti_move = mv.make(&mut pos);
                assert_eq!(pos.to_fen(), expect_fen.to_string());
            }
        }
    }

    #[test]
    fn test_uci_move_fmt() {
        let test_cases = ["a1e5", "e7e8q", "e7e8r", "e7e8b", "e7e8n"];
        for tc in test_cases {
            let mv = Move::from_uci_algebraic(tc).unwrap();
            assert_eq!(mv.to_uci_algebraic(), tc);
        }
    }

    #[test]
    fn test_gen_attackers() {
        let test_cases = [(
            // fen
            "3q4/3rn3/3r2b1/3rb3/rnpkbN2/2qKK2r/1nPPpN2/2rr4 w - - 0 1",
            // attacked square
            "d3",
            // expected results
            "c3 c2 e3 b4 d4 e4 f2 f4 c4 b2",
        )];

        for (fen, attacked, expected) in test_cases {
            let mut expected = expected
                .split_whitespace()
                .map(str::parse::<Square>)
                .map(|x| x.expect("test case has invalid square"))
                .collect::<Vec<_>>();
            expected.sort();

            let attacked = attacked.parse::<Square>().unwrap();

            let board = Board::from_fen(fen).unwrap();

            let mut attackers = board
                .gen_attackers(attacked, false, None)
                .into_iter()
                .map(|(_pc, mv)| mv.src)
                .collect::<Vec<_>>();
            attackers.sort();

            assert_eq!(attackers, expected, "failed {}", fen);
        }
    }

    #[test]
    fn test_capture_movegen() {
        let test_cases = [
            (
                // fen
                "8/3q4/5N2/8/8/8/8/3K4 w - - 0 1",
                // expected moves generated
                "f6d7",
            ),
            (
                "8/8/8/3pP3/2K5/8/8/8 w - d6 0 1",
                // holy hell
                "e5d6 c4d5",
            ),
            ("8/2q5/3K4/8/8/8/8/8 w - - 0 1", "d6c7"),
            (
                "2Q5/3r2R1/2B1PN2/8/3K4/8/8/8 w - - 0 1",
                "c6d7 e6d7 c8d7 f6d7 g7d7",
            ),
        ];

        for (fen, expected) in test_cases {
            let mut board = Board::from_fen(fen).unwrap();
            let mut moves = board.gen_captures().into_iter().collect::<Vec<_>>();
            moves.sort();
            let mut expected = expected
                .split_whitespace()
                .map(Move::from_uci_algebraic)
                .map(|x| x.unwrap())
                .collect::<Vec<_>>();
            expected.sort();

            assert_eq!(moves, expected, "failed '{}'", fen);
        }
    }
}
