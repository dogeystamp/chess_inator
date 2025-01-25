/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

#![deny(rust_2018_idioms)]

use std::fmt::Display;
use std::ops::{Index, IndexMut};
use std::str::FromStr;

pub mod coordination;
pub mod eval;
pub mod fen;
mod hash;
pub mod movegen;
pub mod nnue;
pub mod search;
pub mod util;

pub mod prelude;

use crate::fen::{FromFen, ToFen, START_POSITION};
use crate::hash::Zobrist;
use crate::movegen::GenAttackers;
use std::ops;

pub const BOARD_WIDTH: usize = 8;
pub const BOARD_HEIGHT: usize = 8;
pub const N_SQUARES: usize = BOARD_WIDTH * BOARD_HEIGHT;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum Color {
    #[default]
    White = 0,
    Black = 1,
}
pub const N_COLORS: usize = 2;

impl Color {
    /// Return opposite color (does not assign).
    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    pub const fn sign(&self) -> i8 {
        match self {
            Color::White => 1,
            Color::Black => -1,
        }
    }
}

impl From<Color> for char {
    fn from(value: Color) -> Self {
        match value {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Piece {
    Rook,
    Bishop,
    Knight,
    King,
    Queen,
    Pawn,
}
pub const N_PIECES: usize = 6;

impl Piece {
    /// Get a piece's base value.
    pub const fn value(&self) -> crate::eval::EvalInt {
        use Piece::*;
        (match self {
            Rook => 5,
            Bishop => 3,
            Knight => 3,
            King => 200,
            Queen => 9,
            Pawn => 1,
        }) * 100
    }
}

pub struct PieceErr;

/// Color and piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColPiece {
    pub pc: Piece,
    pub col: Color,
}

impl TryFrom<char> for ColPiece {
    type Error = PieceErr;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        let col = if value.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let mut lower = value;
        lower.make_ascii_lowercase();
        Ok(ColPiece {
            pc: Piece::try_from(lower)?,
            col,
        })
    }
}

impl From<ColPiece> for char {
    fn from(value: ColPiece) -> Self {
        let lower = char::from(value.pc);
        match value.col {
            Color::White => lower.to_ascii_uppercase(),
            Color::Black => lower,
        }
    }
}

impl From<ColPiece> for Color {
    fn from(value: ColPiece) -> Self {
        value.col
    }
}

impl From<ColPiece> for Piece {
    fn from(value: ColPiece) -> Self {
        value.pc
    }
}

impl ColPiece {
    /// Convert option of piece to character.
    pub fn opt_to_char(opt: Option<Self>) -> char {
        match opt {
            Some(pc) => pc.into(),
            None => '.',
        }
    }
}

type SquareIdx = u8;

/// Square index newtype.
///
/// A1 is (0, 0) -> 0, A2 is (0, 1) -> 2, and H8 is (7, 7) -> 63.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Square(SquareIdx);

#[derive(Debug)]
pub enum SquareError {
    OutOfBounds,
    InvalidCharacter(char),
}

impl TryFrom<SquareIdx> for Square {
    type Error = SquareError;

    fn try_from(value: SquareIdx) -> Result<Self, Self::Error> {
        Self::const_try_from(value)
    }
}

macro_rules! sq_try_from {
    ($T: ty) => {
        impl TryFrom<$T> for Square {
            type Error = SquareError;

            fn try_from(value: $T) -> Result<Self, Self::Error> {
                #[allow(irrefutable_let_patterns)]
                if let Ok(upper_bound) = <$T>::try_from(N_SQUARES) {
                    if (0..upper_bound).contains(&value) {
                        return Ok(Square(value as SquareIdx));
                    }
                }
                Err(SquareError::OutOfBounds)
            }
        }
    };
}

sq_try_from!(i8);
sq_try_from!(i32);
sq_try_from!(isize);
sq_try_from!(usize);

impl From<Square> for SquareIdx {
    fn from(value: Square) -> Self {
        value.0
    }
}

impl From<Square> for usize {
    fn from(value: Square) -> Self {
        value.0.into()
    }
}

macro_rules! from_row_col_generic {
    ($T: ty, $r: ident, $c: ident) => {
        #[allow(unused_comparisons)]
        if !(0 <= $r && $r < (BOARD_HEIGHT as $T)) || !(0 <= $c && $c < (BOARD_WIDTH as $T)) {
            Err(SquareError::OutOfBounds)
        } else {
            let ret = (BOARD_WIDTH as $T) * $r + $c;
            debug_assert!(ret <= SquareIdx::MAX as $T);
            Square::const_try_from(ret as u8)
        }
    };
}

impl Square {
    pub const fn const_try_from(value: SquareIdx) -> Result<Self, SquareError> {
        const LIMIT: SquareIdx = N_SQUARES as SquareIdx;
        #[allow(unused_comparisons, clippy::absurd_extreme_comparisons)]
        if 0 <= value && value < LIMIT {
            Ok(Square(value))
        } else {
            Err(SquareError::OutOfBounds)
        }
    }

    pub const fn from_row_col(r: usize, c: usize) -> Result<Self, SquareError> {
        //! Get index of square based on row and column.
        from_row_col_generic!(usize, r, c)
    }
    pub const fn from_row_col_signed(r: isize, c: isize) -> Result<Self, SquareError> {
        from_row_col_generic!(isize, r, c)
    }
    pub const fn to_row_col(self) -> (usize, usize) {
        //! Get row, column from index
        let div = self.0 / (BOARD_WIDTH as SquareIdx);
        let rem = self.0 % (BOARD_WIDTH as SquareIdx);
        debug_assert!(div <= 7);
        debug_assert!(rem <= 7);
        // as long as this is true, there probably won't be overflows
        debug_assert!(SquareIdx::MAX as u128 <= usize::MAX as u128);
        (div as usize, rem as usize)
    }
    pub const fn to_row_col_signed(self) -> (isize, isize) {
        //! Get row, column (signed) from index
        let (r, c) = self.to_row_col();
        // as long as this is true, there probably won't be overflows
        debug_assert!(SquareIdx::MAX as i128 <= isize::MAX as i128);
        (r as isize, c as isize)
    }

    /// Vertically mirror a square.
    pub fn mirror_vert(&self) -> Self {
        let (r, c) = self.to_row_col();
        let (nr, nc) = (BOARD_HEIGHT - 1 - r, c);
        Square::from_row_col(nr, nc)
            .unwrap_or_else(|e| panic!("mirrored square should be valid: nr {nr} nc {nc}: {e:?}"))
    }

    /// Manhattan (grid-based) distance with another Square.
    pub const fn manhattan(&self, other: Self) -> usize {
        let (r1, c1) = self.to_row_col();
        let (r2, c2) = other.to_row_col();
        r1.abs_diff(r2) + c1.abs_diff(c2)
    }
}

impl Display for Square {
    /// Convert square to typical human-readable form (e.g. `e4`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        let (row, col) = self.to_row_col();
        let rank = (row + 1).to_string();
        let file = letters[col];
        write!(f, "{}{}", file, rank)
    }
}

impl FromStr for Square {
    type Err = SquareError;

    /// Convert typical human-readable form (e.g. `e4`) to square index.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = s.as_bytes();
        let col = match bytes[0] as char {
            'a' => 0,
            'b' => 1,
            'c' => 2,
            'd' => 3,
            'e' => 4,
            'f' => 5,
            'g' => 6,
            'h' => 7,
            _ => return Err(SquareError::InvalidCharacter(bytes[0] as char)),
        };
        if let Some(row) = (bytes[1] as char).to_digit(10) {
            Square::from_row_col(row as usize - 1, col as usize)
        } else {
            Err(SquareError::InvalidCharacter(bytes[1] as char))
        }
    }
}

impl TryFrom<char> for Piece {
    type Error = PieceErr;

    fn try_from(s: char) -> Result<Self, Self::Error> {
        match s {
            'r' => Ok(Piece::Rook),
            'b' => Ok(Piece::Bishop),
            'n' => Ok(Piece::Knight),
            'k' => Ok(Piece::King),
            'q' => Ok(Piece::Queen),
            'p' => Ok(Piece::Pawn),
            _ => Err(PieceErr),
        }
    }
}

impl From<Piece> for char {
    fn from(value: Piece) -> Self {
        match value {
            Piece::Rook => 'r',
            Piece::Bishop => 'b',
            Piece::Knight => 'n',
            Piece::King => 'k',
            Piece::Queen => 'q',
            Piece::Pawn => 'p',
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitboard(u64);

impl Bitboard {
    pub const fn on_sq(&mut self, idx: Square) {
        //! Set a square on.
        self.0 |= 1 << idx.0;
    }

    pub const fn off_sq(&mut self, idx: Square) {
        //! Set a square off.
        self.0 &= !(1 << idx.0);
    }

    pub const fn get_sq(&self, idx: Square) -> bool {
        //! Read the value at a square.
        (self.0 & 1 << idx.0) == 1
    }

    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl ops::BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl IntoIterator for Bitboard {
    type Item = Square;

    type IntoIter = BitboardIterator;

    fn into_iter(self) -> Self::IntoIter {
        BitboardIterator { remaining: self }
    }
}

pub struct BitboardIterator {
    remaining: Bitboard,
}

impl Iterator for BitboardIterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            None
        } else {
            let next_idx = self.remaining.0.trailing_zeros() as usize;
            let sq = Square(next_idx.try_into().unwrap());
            self.remaining.off_sq(sq);
            Some(sq)
        }
    }
}

/// Array form board.
///
/// Complements bitboards, notably for "what piece is at this square?" queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mailbox([Option<ColPiece>; N_SQUARES]);

impl Default for Mailbox {
    fn default() -> Self {
        Mailbox([None; N_SQUARES])
    }
}

impl Mailbox {
    /// Get mutable reference to square at index.
    fn sq_mut(&mut self, idx: Square) -> &mut Option<ColPiece> {
        &mut self.0[usize::from(idx)]
    }

    /// Get non-mutable reference to square at index.
    fn sq(&self, idx: Square) -> &Option<ColPiece> {
        &self.0[usize::from(idx)]
    }
}

/// Piece bitboards and state for one player.
///
/// Default is all empty.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerBoards {
    /// Bitboards for individual pieces. Piece -> locations.
    bit: [Bitboard; N_PIECES],
}

/// Castling rights for one player
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct CastlePlayer {
    /// Kingside
    k: bool,
    /// Queenside
    q: bool,
}

/// Castling rights for both players
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct CastleRights([CastlePlayer; N_COLORS]);

impl Display for CastleRights {
    /// Convert to FEN castling rights format.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ret = String::with_capacity(4);
        for (val, ch) in [
            (self.0[Color::White as usize].k, 'K'),
            (self.0[Color::White as usize].q, 'Q'),
            (self.0[Color::Black as usize].k, 'k'),
            (self.0[Color::Black as usize].q, 'q'),
        ] {
            if val {
                ret.push(ch)
            }
        }
        if ret.is_empty() {
            ret.push('-')
        }
        write!(f, "{}", ret)
    }
}

/// Ring-buffer pointer that will never point outside the buffer.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct RingPtr<const N: usize>(usize);

impl<const N: usize> From<RingPtr<N>> for usize {
    fn from(value: RingPtr<N>) -> Self {
        debug_assert!((0..N).contains(&value.0));
        value.0
    }
}

impl<const N: usize> std::ops::Add<usize> for RingPtr<N> {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self((self.0 + rhs) % N)
    }
}

impl<const N: usize> std::ops::Sub<usize> for RingPtr<N> {
    type Output = Self;

    fn sub(self, rhs: usize) -> Self::Output {
        Self((self.0 + N - rhs) % N)
    }
}

impl<const N: usize> std::ops::AddAssign<usize> for RingPtr<N> {
    fn add_assign(&mut self, rhs: usize) {
        self.0 = (self.0 + rhs) % N;
    }
}

impl<const N: usize> std::ops::SubAssign<usize> for RingPtr<N> {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 = (self.0 + N - rhs) % N;
    }
}

impl<const N: usize> Default for RingPtr<N> {
    fn default() -> Self {
        Self(0)
    }
}

impl<const N: usize> RingPtr<N> {}

#[cfg(test)]
mod ringptr_tests {
    use super::*;

    /// ring buffer pointer behaviour
    #[test]
    fn test_ringptr() {
        let ptr_start: RingPtr<3> = RingPtr::default();

        let ptr: RingPtr<3> = RingPtr::default() + 3;
        assert_eq!(ptr, ptr_start);

        let ptr2: RingPtr<3> = RingPtr::default() + 2;
        assert_eq!(ptr2, ptr_start - 1);
        assert_eq!(ptr2, ptr_start + 2);
    }
}

/// Ring-buffer of previously seen hashes, used to avoid draw by repetition.
///
/// Only stores at most `HISTORY_SIZE` plies.
#[derive(Clone, Copy, Debug)]
struct BoardHistory {
    hashes: [Zobrist; HISTORY_SIZE],
    /// Index of the start of the history in the buffer
    ptr_start: RingPtr<HISTORY_SIZE>,
    /// Index one-past-the-end of the history in the buffer
    ptr_end: RingPtr<HISTORY_SIZE>,
}

impl Default for BoardHistory {
    fn default() -> Self {
        BoardHistory {
            // rust can't derive this
            hashes: [Zobrist::default(); HISTORY_SIZE],
            ptr_start: Default::default(),
            ptr_end: Default::default(),
        }
    }
}

impl PartialEq for BoardHistory {
    /// Always equal, since comparing two boards with different histories shouldn't matter.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for BoardHistory {}

/// Size in plies of the board history.
///
/// Actual capacity is one less than this.
const HISTORY_SIZE: usize = 100;

impl BoardHistory {
    /// Counts occurences of this hash in the history.
    fn _count(&self, hash: Zobrist) -> usize {
        let mut ans = 0;

        let mut i = self.ptr_start;
        while i != self.ptr_end {
            if self.hashes[usize::from(i)] == hash {
                ans += 1;
            }
            i += 1;
        }

        ans
    }

    /// Find if there are at least `n` matches for a hash in the last `recent` plies.
    fn at_least_in_recent(&self, mut n: usize, recent: usize, hash: Zobrist) -> bool {
        let mut i = self.ptr_end - recent;

        while i != self.ptr_end && n > 0 {
            if self.hashes[usize::from(i)] == hash {
                n -= 1;
            }
            i += 1;
        }

        n == 0
    }

    /// Add (push) hash to history.
    fn push(&mut self, hash: Zobrist) {
        self.hashes[usize::from(self.ptr_end)] = hash;

        self.ptr_end += 1;

        // replace old entries
        if self.ptr_end == self.ptr_start {
            self.ptr_start += 1;
        }
    }
}

/// Information to reverse a null-move.
pub struct AntiNullMove {
    /// En-passant target square prior to the null-move.
    ep_square: Option<Square>,
}

/// Extra information about the board.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BoardInformation {
    /// Number of minor and major pieces, i.e. queens rooks bishops and knights.
    n_min_maj_pcs: u8,
    /// Number of pawns on the board
    n_pawns: u8,
}

impl BoardInformation {
    pub fn add_piece(&mut self, pc: ColPiece, _sq: Square) {
        use Piece::*;
        match pc.pc {
            Queen | Rook | Bishop | Knight => {
                self.n_min_maj_pcs += 1
            }
            Pawn => {
                self.n_pawns += 1
            }
            _ => {}
        }
    }

    pub fn del_piece(&mut self, pc: ColPiece, _sq: Square) {
        use Piece::*;
        match pc.pc {
            Queen | Rook | Bishop | Knight => {
                self.n_min_maj_pcs -= 1
            }
            Pawn => {
                self.n_pawns -= 1
            }
            _ => {}
        }
    }
}

/// Game state, describes a position.
///
/// Default is empty.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    /// Player bitboards
    players: [PlayerBoards; N_COLORS],

    /// Bitboard for all pieces of all color.
    occupancy: Bitboard,

    /// Mailbox (array) board. Location -> piece.
    mail: Mailbox,

    /// En-passant square.
    ///
    /// (If a pawn moves twice, this is one square in front of the start position.)
    ep_square: Option<Square>,

    /// Castling rights
    castle: CastleRights,

    /// Plies since last irreversible (capture, pawn) move
    irreversible_half: usize,

    /// Half-moves in total in this game (one white move, then black move, is two half moves)
    plies: usize,

    /// Whose turn it is
    turn: Color,

    /// Neural network state.
    nnue: nnue::Nnue,

    /// Hash state to incrementally update.
    zobrist: Zobrist,

    /// Last captured square
    recap_sq: Option<Square>,

    /// History of recent hashes to avoid repetition draws.
    history: BoardHistory,

    /// Extra information that is maintained.
    info: BoardInformation,
}

impl Board {
    /// Default chess position.
    pub fn starting_pos() -> Self {
        Board::from_fen(START_POSITION).unwrap()
    }

    /// Save the current position's hash in the history.
    pub fn push_history(&mut self) {
        self.history.push(self.zobrist);
    }

    /// Is this position a draw by three repetitions?
    pub fn is_repetition(&mut self) -> bool {
        self.history
            .at_least_in_recent(2, self.irreversible_half, self.zobrist)
    }

    /// Get iterator over all squares.
    pub fn squares() -> impl Iterator<Item = Square> {
        (0..N_SQUARES).map(Square::try_from).map(|x| x.unwrap())
    }

    /// Get the 8th rank from a given player's perspective.
    ///
    /// Useful for promotions.
    pub const fn last_rank(pl: Color) -> usize {
        match pl {
            Color::White => BOARD_HEIGHT - 1,
            Color::Black => 0,
        }
    }

    /// Get the 5th rank from a given player's perspective.
    ///
    /// Useful for en-passant validity.
    pub const fn ep_rank(pl: Color) -> usize {
        match pl {
            Color::White => BOARD_HEIGHT - 4,
            Color::Black => 3,
        }
    }

    /// Have the current player pass their turn.
    ///
    /// Equivalent to flipping the board's turn.
    pub fn make_null_move(&mut self) -> AntiNullMove {
        self.zobrist.toggle_turn(self.turn);
        self.turn = self.turn.flip();
        self.zobrist.toggle_turn(self.turn);
        let ret = AntiNullMove {
            ep_square: self.ep_square,
        };
        self.zobrist.toggle_ep(self.ep_square);
        self.ep_square = None;
        self.plies += 1;
        ret
    }

    /// Undo a null move (see [`Board.null_move`](Self#method.null_move)).
    pub fn unmake_null_move(&mut self, anti_mv: AntiNullMove) {
        self.zobrist.toggle_turn(self.turn);
        self.turn = self.turn.flip();
        self.plies -= 1;
        self.zobrist.toggle_turn(self.turn);
        self.ep_square = anti_mv.ep_square;
        self.zobrist.toggle_ep(self.ep_square);
    }

    /// Create a new piece in a location, and pop any existing piece in the destination.
    pub fn set_piece(
        &mut self,
        sq: Square,
        pc: ColPiece,
        update_metrics: bool,
    ) -> Option<ColPiece> {
        let dest_pc = self.del_piece(sq, update_metrics);
        self[pc.col][pc.pc].on_sq(sq);
        self.occupancy.on_sq(sq);
        *self.mail.sq_mut(sq) = Some(pc);
        if update_metrics {
            self.nnue.add_piece(pc, sq);
            self.info.add_piece(pc, sq);
            self.zobrist.toggle_pc(&pc, &sq);
        }
        dest_pc
    }

    /// Set the piece (or no piece) in a square, and return ("pop") the existing piece.
    pub fn set_square(
        &mut self,
        idx: Square,
        pc: Option<ColPiece>,
        update_metrics: bool,
    ) -> Option<ColPiece> {
        match pc {
            Some(pc) => self.set_piece(idx, pc, update_metrics),
            None => self.del_piece(idx, update_metrics),
        }
    }

    /// Delete the piece in a location, and return ("pop") that piece.
    pub fn del_piece(&mut self, sq: Square, update_metrics: bool) -> Option<ColPiece> {
        if let Some(pc) = *self.mail.sq_mut(sq) {
            self[pc.col][pc.pc].off_sq(sq);
            self.occupancy.off_sq(sq);
            *self.mail.sq_mut(sq) = None;
            if update_metrics {
                self.nnue.del_piece(pc, sq);
                self.info.del_piece(pc, sq);
                self.zobrist.toggle_pc(&pc, &sq);
            }
            Some(pc)
        } else {
            None
        }
    }

    pub fn move_piece(&mut self, src: Square, dest: Square, update_metrics: bool) {
        let pc = self.del_piece(src, update_metrics).unwrap_or_else(|| {
            panic!(
                "move ({src} -> {dest}) should have piece at source (pos '{}')",
                self.to_fen()
            )
        });
        self.set_piece(dest, pc, update_metrics);
    }

    /// Get the piece at a location.
    pub fn get_piece(&self, idx: Square) -> Option<ColPiece> {
        *self.mail.sq(idx)
    }

    /// Mirrors the position so that black and white are switched.
    ///
    /// Mainly to avoid duplication in tests.
    pub fn flip_colors(&self) -> Self {
        let mut new_board = Self {
            turn: self.turn.flip(),
            irreversible_half: self.irreversible_half,
            plies: self.plies ^ 1,
            ep_square: self.ep_square.map(|sq| sq.mirror_vert()),
            castle: CastleRights(self.castle.0),
            zobrist: Zobrist::default(),
            recap_sq: self.recap_sq.map(|sq| sq.mirror_vert()),
            ..Default::default()
        };

        new_board.castle.0.reverse();
        Zobrist::toggle_board_info(&mut new_board);

        for sq in Board::squares() {
            let opt_pc = self.get_piece(sq.mirror_vert()).map(|pc| ColPiece {
                col: pc.col.flip(),
                pc: pc.pc,
            });
            new_board.set_square(sq, opt_pc, true);
        }
        new_board
    }

    /// Is a given player in check?
    pub fn is_check(&self, pl: Color) -> bool {
        for src in self[pl][Piece::King] {
            if self
                .gen_attackers(src, true, Some(pl.flip()))
                .into_iter()
                .next()
                .is_some()
            {
                return true;
            }
        }
        false
    }

    /// Get the current player to move.
    pub fn get_turn(&self) -> Color {
        self.turn
    }

    /// Maximum amount of moves in the counter to parse before giving up
    const MAX_MOVES: usize = 9_999;
}

impl Index<Color> for Board {
    type Output = PlayerBoards;

    fn index(&self, col: Color) -> &Self::Output {
        &self.players[col as usize]
    }
}

impl IndexMut<Color> for Board {
    fn index_mut(&mut self, col: Color) -> &mut Self::Output {
        &mut self.players[col as usize]
    }
}

impl Index<Color> for CastleRights {
    type Output = CastlePlayer;

    fn index(&self, col: Color) -> &Self::Output {
        &self.0[col as usize]
    }
}

impl IndexMut<Color> for CastleRights {
    fn index_mut(&mut self, col: Color) -> &mut Self::Output {
        &mut self.0[col as usize]
    }
}

impl Index<Piece> for PlayerBoards {
    type Output = Bitboard;

    fn index(&self, pc: Piece) -> &Self::Output {
        &self.bit[pc as usize]
    }
}

impl IndexMut<Piece> for PlayerBoards {
    fn index_mut(&mut self, pc: Piece) -> &mut Self::Output {
        &mut self.bit[pc as usize]
    }
}

impl core::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut str = String::with_capacity(N_SQUARES + BOARD_HEIGHT);
        for row in (0..BOARD_HEIGHT).rev() {
            for col in 0..BOARD_WIDTH {
                let idx = Square::from_row_col(row, col).or(Err(std::fmt::Error))?;
                let pc = self.get_piece(idx);
                str.push(ColPiece::opt_to_char(pc));
            }
            str += "\n";
        }
        write!(f, "{}", str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use fen::FromFen;
    use movegen::{FromUCIAlgebraic, Move, ToUCIAlgebraic};

    #[test]
    /// Ensure that the const `as` conversion for `N_SQUARES` doesn't overflow.
    fn square_limit() {
        assert!(u8::try_from(N_SQUARES).unwrap() == (N_SQUARES as u8));
    }

    #[test]
    fn test_square_casts() {
        let fail_cases = [-1, 64, 0x7FFFFFFF, 257, 256, 128, 65, -3, !0x7FFFFFFF];
        for tc in fail_cases {
            macro_rules! try_type {
                ($T: ty) => {
                    #[allow(irrefutable_let_patterns)]
                    if let Ok(conv) = <$T>::try_from(tc) {
                        assert!(matches!(
                            Square::try_from(conv),
                            Err(SquareError::OutOfBounds)
                        ))
                    }
                };
            }
            try_type!(i32);
            try_type!(i8);
            try_type!(isize);
            try_type!(u8);
        }

        let good_cases = 0..SquareIdx::try_from(N_SQUARES).unwrap();
        for tc in good_cases {
            macro_rules! try_type {
                ($T: ty) => {
                    let conv = <$T>::try_from(tc).unwrap();
                    let res = Square::try_from(conv).unwrap();
                    assert_eq!(res.0, tc);
                };
            }
            try_type!(i32);
            try_type!(i8);
            try_type!(isize);
            try_type!(u8);
        }
    }

    #[test]
    fn test_to_from_algebraic() {
        let test_cases = [("a1", 0), ("a8", 56), ("h1", 7), ("h8", 63)];
        for (sqr, idx) in test_cases {
            assert_eq!(Square::try_from(idx).unwrap().to_string(), sqr);
            assert_eq!(
                sqr.parse::<Square>().unwrap(),
                Square::try_from(idx).unwrap()
            );
        }
    }

    #[test]
    fn test_bitboard_iteration() {
        let indices = [0, 5, 17, 24, 34, 39, 42, 45, 49, 50, 63];

        let mut bitboard = Bitboard::default();

        let squares = indices.map(Square);
        for sq in squares {
            bitboard.on_sq(sq);
        }
        // ensure that iteration does not consume the board
        for _ in 0..=1 {
            for (i, sq) in bitboard.into_iter().enumerate() {
                assert_eq!(squares[i], sq)
            }
        }

        let board = Board::from_fen("8/4p3/1q1Q1p2/4p3/1p1r4/8/8/8 w - - 0 1").unwrap();
        let white_queens = board[Color::White][Piece::Queen]
            .into_iter()
            .collect::<Vec<Square>>();
        assert_eq!(white_queens, vec![Square::from_str("d6").unwrap()])
    }

    #[test]
    fn test_square_mirror() {
        for (sq, expect) in [("a1", "a8"), ("h1", "h8"), ("d4", "d5")] {
            let sq = sq.parse::<Square>().unwrap();
            let expect = expect.parse::<Square>().unwrap();
            assert_eq!(sq.mirror_vert(), expect);
        }
    }

    #[test]
    fn test_flip_colors() {
        let test_cases = [
            (
                "2kqrbnp/8/8/8/8/8/8/2KQRBNP w - - 0 1",
                "2kqrbnp/8/8/8/8/8/8/2KQRBNP b - - 0 1",
            ),
            (
                "2kqrbnp/8/8/8/8/8/6N1/2KQRB1P w - a1 0 1",
                "2kqrb1p/6n1/8/8/8/8/8/2KQRBNP b - a8 0 1",
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
                "r3k2r/8/8/8/8/8/8/R3K2R b Qk - 0 1",
            ),
        ];
        for (tc, expect) in test_cases {
            let tc = Board::from_fen(tc).unwrap();
            let expect = Board::from_fen(expect).unwrap();
            assert_eq!(tc.flip_colors(), expect);
        }
    }

    #[test]
    fn manhattan_distance() {
        let test_cases = [
            ("a3", "a3", 0),
            ("a3", "a4", 1),
            ("a3", "b3", 1),
            ("a3", "b4", 2),
            ("a1", "b8", 8),
        ];

        for (sq_str1, sq_str2, expected) in test_cases {
            let sq1 = Square::from_str(sq_str1).unwrap();
            let sq2 = Square::from_str(sq_str2).unwrap();
            let res = sq1.manhattan(sq2);
            assert_eq!(
                res, expected,
                "failed {sq_str1} and {sq_str2}: got manhattan {}, expected {}",
                res, expected
            );
        }
    }

    #[test]
    fn test_history() {
        let board = Board::starting_pos();

        let mut history = BoardHistory::default();
        for _ in 0..(HISTORY_SIZE + 15) {
            history.push(board.zobrist);
        }

        assert_eq!(history._count(board.zobrist), HISTORY_SIZE - 1);
        assert!(history.at_least_in_recent(1, 1, board.zobrist));
        assert!(history.at_least_in_recent(2, 3, board.zobrist));
        assert!(history.at_least_in_recent(1, 3, board.zobrist));

        let board_empty = Board::default();
        history.push(board_empty.zobrist);

        assert!(!history.at_least_in_recent(1, 1, board.zobrist));
        assert!(history.at_least_in_recent(1, 2, board.zobrist));

        assert_eq!(history._count(board.zobrist), HISTORY_SIZE - 2);
        assert_eq!(history._count(board_empty.zobrist), 1);
        assert!(history.at_least_in_recent(1, 3, board.zobrist));
        assert!(history.at_least_in_recent(1, 20, board_empty.zobrist));
        assert!(history.at_least_in_recent(1, 15, board_empty.zobrist));
        assert!(history.at_least_in_recent(1, 1, board_empty.zobrist));

        for _ in 0..3 {
            history.push(board_empty.zobrist);
        }

        assert_eq!(history._count(board_empty.zobrist), 4);
        assert_eq!(history._count(board.zobrist), HISTORY_SIZE - 5);
    }

    #[test]
    fn repetition_detection() {
        let mut board = Board::starting_pos();

        let mvs = "b1c3 b8c6 c3b1 c6b8"
            .split_whitespace()
            .map(Move::from_uci_algebraic)
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();

        for _ in 0..2 {
            for mv in &mvs {
                board.push_history();
                let _ = mv.make(&mut board);
            }
        }

        // this is the third occurence, but beforehand there are two occurences
        assert_eq!(board.history._count(board.zobrist), 2);
        assert!(board.is_repetition(), "fen: {}", board.to_fen());
    }

    /// engine should take advantage of the three time repetition rule
    #[test]
    fn find_repetition() {
        use eval::EvalInt;
        use search::{best_line, EngineState, SearchConfig, TranspositionTable};
        let mut board = Board::from_fen("qqqp4/pkpp4/8/8/8/8/8/K7 b - - 0 1").unwrap();

        let mvs = "b7b6 a1a2 b6b7 a2a1 b7b6 a1a2 b6b7"
            .split_whitespace()
            .map(Move::from_uci_algebraic)
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();

        let expected_bestmv = Move::from_uci_algebraic("a2a1").unwrap();

        let mut cnt = 0;

        for mv in &mvs {
            board.push_history();
            cnt += 1;
            let _ = mv.make(&mut board);
        }

        eprintln!("board is: '{}'", board.to_fen());
        eprintln!("added {} history entries", cnt);

        let (_tx, rx) = std::sync::mpsc::channel();
        let cache = TranspositionTable::new(1);

        let mut engine_state = EngineState::new(
            SearchConfig {
                depth: 1,
                qdepth: 0,
                enable_trans_table: false,
                ..Default::default()
            },
            rx,
            cache,
            search::TimeLimits::default(),
            search::InterruptMode::MustComplete,
        );

        let (line, eval) = best_line(&mut board, &mut engine_state);
        let best_mv = line.last().unwrap();

        expected_bestmv.make(&mut board);
        eprintln!(
            "after expected mv, board repeated {} times",
            board.history._count(board.zobrist)
        );

        assert_eq!(
            *best_mv,
            expected_bestmv,
            "got {} (eval {:?}) instead of {}",
            best_mv.to_uci_algebraic(),
            eval,
            expected_bestmv.to_uci_algebraic()
        );
        assert!(EvalInt::from(eval) == 0);

        // now ensure that it's completely one-sided without the repetition opportunity
        let mut board = Board::from_fen("qqqp4/pkpp4/8/8/8/8/8/K7 w - - 0 1").unwrap();

        let (_tx, rx) = std::sync::mpsc::channel();
        let cache = TranspositionTable::new(1);
        let mut engine_state = EngineState::new(
            SearchConfig {
                depth: 1,
                qdepth: 0,
                enable_trans_table: false,
                ..Default::default()
            },
            rx,
            cache,
            search::TimeLimits::default(),
            search::InterruptMode::MustComplete,
        );

        let (_line, eval) = best_line(&mut board, &mut engine_state);
        assert!(EvalInt::from(eval) < 0);
    }

    #[test]
    fn null_move() {
        let mut board = Board::starting_pos();
        let anti_mv = board.make_null_move();
        let anti_mv2 = Move::from_uci_algebraic("e7e5").unwrap().make(&mut board);
        anti_mv2.unmake(&mut board);
        board.unmake_null_move(anti_mv);
        assert_eq!(board, Board::starting_pos());
    }

    #[test]
    fn min_maj_pcs() {
        let test_cases = [
            (
                // board
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                // a pseudo-legal move to test on the board
                "e2e4",
                // expected min/maj piece count after the move
                14,
            ),
            (
                "8/8/8/3Q4/3k4/8/8/8 b - - 0 1",
                "d4d5",
                0,
            ),
            (
                "8/8/8/3Q4/3k4/8/8/8 b - - 0 1",
                "d4e3",
                1,
            ),
            (
                "8/8/8/3Q4/3kp3/8/8/8 b - - 0 1",
                "d4e3",
                1,
            ),
            (
                "8/8/8/3Q4/3kp3/8/8/8 b - - 0 1",
                "e4d5",
                0,
            ),
            (
                "8/8/8/3Qq3/3kp3/8/8/8 b - - 0 1",
                "e5d5",
                1,
            ),
            (
                "8/8/8/3Qq3/3kp3/8/8/8 b - - 0 1",
                "e5d6",
                2,
            ),
        ];
        for (fen, mv, expected) in test_cases {
            let mut board = Board::from_fen(fen).unwrap();
            let mv = Move::from_uci_algebraic(mv).unwrap();
            mv.make(&mut board);
            assert_eq!(board.info.n_min_maj_pcs, expected, "failed '{}'", fen)
        }
    }
}
