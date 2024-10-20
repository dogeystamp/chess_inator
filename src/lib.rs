#![deny(rust_2018_idioms)]

use std::fmt::Display;
use std::str::FromStr;

pub mod fen;
pub mod movegen;

use fen::FromFen;

const BOARD_WIDTH: usize = 8;
const BOARD_HEIGHT: usize = 8;
const N_SQUARES: usize = BOARD_WIDTH * BOARD_HEIGHT;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
enum Color {
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

struct PieceErr;

/// Color and piece.
#[derive(Debug, Clone, Copy)]
struct ColPiece {
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

/// Square index newtype.
///
/// A1 is (0, 0) -> 0, A2 is (0, 1) -> 2, and H8 is (7, 7) -> 63.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Square(usize);

#[derive(Debug)]
pub enum SquareError {
    OutOfBounds,
    InvalidCharacter(char),
}

impl TryFrom<usize> for Square {
    type Error = SquareError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if (0..N_SQUARES).contains(&value) {
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
                if let Ok(upper_bound) = <$T>::try_from(N_SQUARES) {
                    if (0..upper_bound).contains(&value) {
                        return Ok(Square(value as usize));
                    }
                }
                Err(SquareError::OutOfBounds)
            }
        }
    };
}

sq_try_from!(i32);
sq_try_from!(isize);
sq_try_from!(i8);

impl From<Square> for usize {
    fn from(value: Square) -> Self {
        value.0
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
    fn from_row_col(r: usize, c: usize) -> Result<Self, SquareError> {
        //! Get index of square based on row and column.
        from_row_col_generic!(usize, r, c)
    }
    fn from_row_col_signed(r: isize, c: isize) -> Result<Self, SquareError> {
        from_row_col_generic!(isize, r, c)
    }
    fn to_row_col(self) -> (usize, usize) {
        //! Get row, column from index
        let div = self.0 / BOARD_WIDTH;
        let rem = self.0 % BOARD_WIDTH;
        assert!(div <= 7);
        assert!(rem <= 7);
        (div, rem)
    }
    fn to_row_col_signed(self) -> (isize, isize) {
        //! Get row, column (signed) from index
        let (r, c) = self.to_row_col();
        (r.try_into().unwrap(), c.try_into().unwrap())
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

#[derive(Default, Debug, Clone, Copy)]
struct Bitboard(u64);

impl Bitboard {
    pub fn on_idx(&mut self, idx: Square) {
        //! Set the square at an index to on.
        self.0 |= 1 << usize::from(idx);
    }

    pub fn off_idx(&mut self, idx: Square) {
        //! Set the square at an index to off.
        self.0 &= !(1 << usize::from(idx));
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

struct BitboardIterator {
    remaining: Bitboard,
}

impl Iterator for BitboardIterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            None
        } else {
            let next_idx = self.remaining.0.trailing_zeros() as usize;
            let sq = Square(next_idx);
            self.remaining.off_idx(sq);
            Some(sq)
        }
    }
}

/// Array form board.
///
/// Complements bitboards, notably for "what piece is at this square?" queries.
#[derive(Debug, Clone, Copy)]
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
#[derive(Default, Debug, Clone, Copy)]
struct Player {
    /// Bitboards for individual pieces. Piece -> locations.
    bit: [Bitboard; N_PIECES],
}

impl Player {
    /// Get board (non-mutable) for a specific piece.
    fn board(&self, pc: Piece) -> &Bitboard {
        &self.bit[pc as usize]
    }

    /// Get board (mutable) for a specific piece.
    fn board_mut(&mut self, pc: Piece) -> &mut Bitboard {
        &mut self.bit[pc as usize]
    }
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

impl ToString for CastleRights {
    /// Convert to FEN castling rights format.
    fn to_string(&self) -> String {
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
        ret
    }
}

/// Immutable game state, unique to a position.
///
/// Default is empty.
#[derive(Debug, Default, Clone, Copy)]
pub struct BoardState {
    /// Player bitboards
    players: [Player; N_COLORS],

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
}

/// Piece missing where there should be one.
#[derive(Debug)]
struct NoPieceError;

impl BoardState {
    /// Get mutable reference to a player.
    fn pl_mut(&mut self, col: Color) -> &mut Player {
        &mut self.players[col as usize]
    }

    /// Get immutable reference to a player.
    fn pl(&self, col: Color) -> &Player {
        &self.players[col as usize]
    }

    /// Get immutable reference to castling rights.
    fn pl_castle(&self, col: Color) -> &CastlePlayer {
        &self.castle.0[col as usize]
    }

    /// Get mutable reference to castling rights.
    fn pl_castle_mut(&mut self, col: Color) -> &mut CastlePlayer {
        &mut self.castle.0[col as usize]
    }

    /// Create a new piece in a location.
    fn set_piece(&mut self, idx: Square, pc: ColPiece) {
        let pl = self.pl_mut(pc.col);
        pl.board_mut(pc.into()).on_idx(idx);
        *self.mail.sq_mut(idx) = Some(pc);
    }

    /// Delete the piece in a location, and return ("pop") that piece.
    ///
    /// Returns an error if there is no piece in the location.
    fn del_piece(&mut self, idx: Square) -> Result<ColPiece, NoPieceError> {
        if let Some(pc) = *self.mail.sq_mut(idx) {
            let pl = self.pl_mut(pc.col);
            pl.board_mut(pc.into()).off_idx(idx);
            *self.mail.sq_mut(idx) = None;
            Ok(pc)
        } else {
            Err(NoPieceError)
        }
    }

    fn move_piece(&mut self, src: Square, dest: Square) {
        let pc = self.del_piece(src).expect("Move source should have piece.");
        self.set_piece(dest, pc);
    }

    /// Get the piece at a location.
    fn get_piece(&self, idx: Square) -> Option<ColPiece> {
        *self.mail.sq(idx)
    }

    /// Maximum amount of moves in the counter to parse before giving up
    const MAX_MOVES: usize = 9_999;
}

impl core::fmt::Display for BoardState {
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

    #[test]
    fn test_square_casts() {
        let fail_cases = [-1, 64, 0x7FFFFFFF, 257, 256, 128, 65, -3, !0x7FFFFFFF];
        for tc in fail_cases {
            macro_rules! try_type {
                ($T: ty) => {
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
            try_type!(usize);
        }

        let good_cases = 0..N_SQUARES;
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
            try_type!(usize);
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
        let indices = [
            0usize, 5usize, 17usize, 24usize, 34usize, 39usize, 42usize, 45usize, 49usize, 50usize,
            63usize,
        ];

        let mut bitboard = Bitboard::default();

        let squares = indices.map(Square);
        for sq in squares {
            bitboard.on_idx(sq);
        }
        // ensure that iteration does not consume the board
        for _ in 0..=1 {
            for (i, sq) in bitboard.into_iter().enumerate() {
                assert_eq!(squares[i], sq)
            }
        }

        let board = BoardState::from_fen("8/4p3/1q1Q1p2/4p3/1p1r4/8/8/8 w - - 0 1").unwrap();
        let white_queens = board
            .pl(Color::White)
            .board(Piece::Queen)
            .into_iter()
            .collect::<Vec<Square>>();
        assert_eq!(white_queens, vec![Square::from_str("d6").unwrap()])
    }
}
