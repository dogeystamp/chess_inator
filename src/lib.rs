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

pub mod eval;
pub mod fen;
mod hash;
pub mod movegen;
pub mod random;
pub mod search;

use crate::fen::{FromFen, ToFen, START_POSITION};
use crate::hash::Zobrist;
use eval::eval_score::EvalScores;

const BOARD_WIDTH: usize = 8;
const BOARD_HEIGHT: usize = 8;
const N_SQUARES: usize = BOARD_WIDTH * BOARD_HEIGHT;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum Color {
    #[default]
    White = 0,
    Black = 1,
}
const N_COLORS: usize = 2;

impl Color {
    /// Return opposite color (does not assign).
    pub fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    pub fn sign(&self) -> i8 {
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
enum Piece {
    Rook,
    Bishop,
    Knight,
    King,
    Queen,
    Pawn,
}
const N_PIECES: usize = 6;

pub struct PieceErr;

/// Color and piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColPiece {
    pc: Piece,
    col: Color,
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
        if (0..N_SQUARES).contains(&value.into()) {
            Ok(Square(value))
        } else {
            Err(SquareError::OutOfBounds)
        }
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
        if !(0..(BOARD_HEIGHT as $T)).contains(&$r) || !(0..(BOARD_WIDTH as $T)).contains(&$c) {
            Err(SquareError::OutOfBounds)
        } else {
            let ret = (BOARD_WIDTH as $T) * $r + $c;
            ret.try_into()
        }
    };
}

impl Square {
    pub fn from_row_col(r: usize, c: usize) -> Result<Self, SquareError> {
        //! Get index of square based on row and column.
        from_row_col_generic!(usize, r, c)
    }
    pub fn from_row_col_signed(r: isize, c: isize) -> Result<Self, SquareError> {
        from_row_col_generic!(isize, r, c)
    }
    pub fn to_row_col(self) -> (usize, usize) {
        //! Get row, column from index
        let div = usize::from(self.0) / BOARD_WIDTH;
        let rem = usize::from(self.0) % BOARD_WIDTH;
        debug_assert!(div <= 7);
        debug_assert!(rem <= 7);
        (div, rem)
    }
    pub fn to_row_col_signed(self) -> (isize, isize) {
        //! Get row, column (signed) from index
        let (r, c) = self.to_row_col();
        (r.try_into().unwrap(), c.try_into().unwrap())
    }

    /// Vertically mirror a square.
    pub fn mirror_vert(&self) -> Self {
        let (r, c) = self.to_row_col();
        let (nr, nc) = (BOARD_HEIGHT - 1 - r, c);
        Square::from_row_col(nr, nc)
            .unwrap_or_else(|e| panic!("mirrored square should be valid: nr {nr} nc {nc}: {e:?}"))
    }

    /// Manhattan (grid-based) distance with another Square.
    pub fn manhattan(&self, other: Self) -> usize {
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
    pub fn on_sq(&mut self, idx: Square) {
        //! Set a square on.
        self.0 |= 1 << usize::from(idx);
    }

    pub fn off_sq(&mut self, idx: Square) {
        //! Set a square off.
        self.0 &= !(1 << usize::from(idx));
    }

    pub fn get_sq(&self, idx: Square) -> bool {
        //! Read the value at a square.
        (self.0 & 1 << usize::from(idx)) == 1
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
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

/// Immutable game state, unique to a position.
///
/// Default is empty.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    /// Player bitboards
    players: [PlayerBoards; N_COLORS],

    /// Mailbox (array) board. Location -> piece.
    mail: Mailbox,

    /// En-passant square.
    ///
    /// (If a pawn moves twice, this is one square in front of the start position.)
    ep_square: Option<Square>,

    /// Castling rights
    castle: CastleRights,

    /// Plies since last irreversible (capture, pawn) move
    half_moves: usize,

    /// Full move counter (incremented after each black turn)
    full_moves: usize,

    /// Whose turn it is
    turn: Color,

    /// Counters for evaluation.
    eval: EvalScores,

    /// Hash state to incrementally update.
    zobrist: Zobrist,
}

impl Board {
    /// Default chess position.
    pub fn starting_pos() -> Self {
        Board::from_fen(START_POSITION).unwrap()
    }

    /// Get iterator over all squares.
    pub fn squares() -> impl Iterator<Item = Square> {
        (0..N_SQUARES).map(Square::try_from).map(|x| x.unwrap())
    }

    /// Create a new piece in a location, and pop any existing piece in the destination.
    pub fn set_piece(&mut self, sq: Square, pc: ColPiece) -> Option<ColPiece> {
        let dest_pc = self.del_piece(sq);
        let pl = &mut self[pc.col];
        pl[pc.into()].on_sq(sq);
        *self.mail.sq_mut(sq) = Some(pc);
        self.eval.add_piece(&pc, &sq);
        self.zobrist.toggle_pc(&pc, &sq);
        dest_pc
    }

    /// Set the piece (or no piece) in a square, and return ("pop") the existing piece.
    pub fn set_square(&mut self, idx: Square, pc: Option<ColPiece>) -> Option<ColPiece> {
        match pc {
            Some(pc) => self.set_piece(idx, pc),
            None => self.del_piece(idx),
        }
    }

    /// Delete the piece in a location, and return ("pop") that piece.
    pub fn del_piece(&mut self, sq: Square) -> Option<ColPiece> {
        if let Some(pc) = *self.mail.sq_mut(sq) {
            let pl = &mut self[pc.col];
            pl[pc.into()].off_sq(sq);
            *self.mail.sq_mut(sq) = None;
            self.eval.del_piece(&pc, &sq);
            self.zobrist.toggle_pc(&pc, &sq);
            Some(pc)
        } else {
            None
        }
    }

    fn move_piece(&mut self, src: Square, dest: Square) {
        let pc = self.del_piece(src).unwrap_or_else(|| {
            panic!(
                "move ({src} -> {dest}) should have piece at source (pos '{}')",
                self.to_fen()
            )
        });
        self.set_piece(dest, pc);
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
            half_moves: self.half_moves,
            full_moves: self.full_moves,
            players: Default::default(),
            mail: Default::default(),
            ep_square: self.ep_square.map(|sq| sq.mirror_vert()),
            castle: CastleRights(self.castle.0),
            eval: Default::default(),
            zobrist: Zobrist::default(),
        };

        new_board.castle.0.reverse();
        Zobrist::toggle_board_info(&mut new_board);

        for sq in Board::squares() {
            let opt_pc = self.get_piece(sq.mirror_vert()).map(|pc| ColPiece {
                col: pc.col.flip(),
                pc: pc.pc,
            });
            new_board.set_square(sq, opt_pc);
        }
        new_board
    }

    /// Is a given player in check?
    pub fn is_check(&self, pl: Color) -> bool {
        for src in self[pl][Piece::King] {
            macro_rules! detect_checker {
                ($dirs: ident, $pc: pat, $keep_going: expr) => {
                    for dir in $dirs.into_iter() {
                        let (mut r, mut c) = src.to_row_col_signed();
                        loop {
                            let (nr, nc) = (r + dir.0, c + dir.1);
                            if let Ok(sq) = Square::from_row_col_signed(nr, nc) {
                                if let Some(pc) = self.get_piece(sq) {
                                    if matches!(pc.pc, $pc) && pc.col != pl {
                                        return true;
                                    } else {
                                        break;
                                    }
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

            let dirs_white_pawn = [(-1, 1), (-1, -1)];
            let dirs_black_pawn = [(1, 1), (1, -1)];

            use Piece::*;

            use movegen::{DIRS_DIAG, DIRS_KNIGHT, DIRS_STAR, DIRS_STRAIGHT};

            detect_checker!(DIRS_DIAG, Bishop | Queen, true);
            detect_checker!(DIRS_STRAIGHT, Rook | Queen, true);
            detect_checker!(DIRS_STAR, King, false);
            detect_checker!(DIRS_KNIGHT, Knight, false);
            match pl {
                Color::White => detect_checker!(dirs_black_pawn, Pawn, false),
                Color::Black => detect_checker!(dirs_white_pawn, Pawn, false),
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
}
