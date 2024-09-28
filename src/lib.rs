#![deny(rust_2018_idioms)]

pub mod fen;

const BOARD_WIDTH: usize = 8;
const BOARD_HEIGHT: usize = 8;
const N_SQUARES: usize = BOARD_WIDTH * BOARD_HEIGHT;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
enum Color {
    #[default]
    White,
    Black,
}
const N_COLORS: usize = 2;

#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Index(usize);

enum IndexError {
    OutOfBounds,
}

impl TryFrom<usize> for Index {
    type Error = IndexError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if (0..N_SQUARES).contains(&value) {
            Ok(Index(value))
        } else {
            Err(IndexError::OutOfBounds)
        }
    }
}
impl From<Index> for usize {
    fn from(value: Index) -> Self {
        value.0
    }
}
impl Index {
    fn from_row_col(r: usize, c: usize) -> Result<Self, IndexError> {
        //! Get index of square based on row and column.
        let ret = BOARD_WIDTH * r + c;
        ret.try_into()
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

#[derive(Default, Debug)]
struct Bitboard(u64);

impl Bitboard {
    pub fn on_idx(&mut self, idx: Index) {
        //! Set the square at an index to on.
        self.0 |= 1 << usize::from(idx);
    }

    pub fn off_idx(&mut self, idx: Index) {
        //! Set the square at an index to off.
        self.0 &= !(1 << usize::from(idx));
    }
}

/// Array form board.
///
/// Complements bitboards, notably for "what piece is at this square?" queries.
#[derive(Debug)]
struct Mailbox([Option<ColPiece>; N_SQUARES]);

impl Default for Mailbox {
    fn default() -> Self {
        Mailbox([None; N_SQUARES])
    }
}

impl Mailbox {
    /// Get mutable reference to square at index.
    fn sq_mut(&mut self, idx: Index) -> &mut Option<ColPiece> {
        &mut self.0[usize::from(idx)]
    }

    /// Get non-mutable reference to square at index.
    fn sq(&self, idx: Index) -> &Option<ColPiece> {
        &self.0[usize::from(idx)]
    }
}

/// Piece bitboards and state for one player.
///
/// Default is all empty.
#[derive(Default, Debug)]
struct Player {
    /// Bitboards for individual pieces. Piece -> locations.
    bit: [Bitboard; N_PIECES],
}

impl Player {
    /// Get board for a specific piece.
    fn board(&mut self, pc: Piece) -> &mut Bitboard {
        &mut self.bit[pc as usize]
    }
}

/// Castling rights for one player
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CastlingRights {
    /// Kingside
    k: bool,
    /// Queenside
    q: bool,
}

/// Game state.
///
/// Default is empty.
#[derive(Debug, Default)]
pub struct Position {
    /// Player bitboards
    players: [Player; N_COLORS],

    /// Mailbox (array) board. Location -> piece.
    mail: Mailbox,

    /// En-passant square.
    ///
    /// (If a pawn moves twice, this is one square in front of the start position.)
    ep_square: Option<Index>,

    /// Castling rights
    castle: [CastlingRights; N_COLORS],

    /// Plies since last irreversible (capture, pawn) move
    half_moves: usize,

    /// Full move counter (incremented after each black turn)
    full_moves: usize,

    /// Whose turn it is
    turn: Color,
}

impl Position {
    /// Get mutable reference to a player.
    fn pl_mut(&mut self, col: Color) -> &mut Player {
        &mut self.players[col as usize]
    }

    /// Create a new piece in a location.
    fn set_piece(&mut self, idx: Index, pc: ColPiece) {
        let pl = self.pl_mut(pc.col);
        pl.board(pc.into()).on_idx(idx);
        *self.mail.sq_mut(idx) = Some(pc);
    }

    /// Delete the piece in a location, if it exists.
    fn del_piece(&mut self, idx: Index) {
        if let Some(pc) = *self.mail.sq_mut(idx) {
            let pl = self.pl_mut(pc.col);
            pl.board(pc.into()).off_idx(idx);
            *self.mail.sq_mut(idx) = None;
        }
    }

    /// Get the piece at a location.
    fn get_piece(&self, idx: Index) -> Option<ColPiece> {
        *self.mail.sq(idx)
    }

    /// Maximum amount of moves in the counter to parse before giving up
    const MAX_MOVES: usize = 9_999;
}

impl core::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut str = String::with_capacity(N_SQUARES + BOARD_HEIGHT);
        for row in (0..BOARD_HEIGHT).rev() {
            for col in 0..BOARD_WIDTH {
                let idx = Index::from_row_col(row, col).or(Err(std::fmt::Error))?;
                let pc = self.get_piece(idx);
                str.push(ColPiece::opt_to_char(pc));
            }
            str += "\n";
        }
        write!(f, "{}", str)
    }
}
