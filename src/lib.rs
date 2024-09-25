#![deny(rust_2018_idioms)]

const BOARD_WIDTH: usize = 8;
const BOARD_HEIGHT: usize = 8;
const N_SQUARES: usize = BOARD_WIDTH * BOARD_HEIGHT;

#[derive(Debug, Copy, Clone, Default)]
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
#[derive(Debug, Clone, Copy)]
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

/// FEN parsing error, with index of issue if applicable.
#[derive(Debug)]
pub enum FenError {
    /// Invalid character.
    BadChar(usize),
    /// There is an extraneous character.
    ExtraChar(usize),
    /// FEN is too short, and missing information.
    MissingFields,
    /// Too many pieces on a single row.
    TooManyPieces(usize),
    /// Too little pieces on a single row.
    NotEnoughPieces(usize),
    /// Too many rows.
    TooManyRows(usize),
    /// Too little rows.
    NotEnoughRows(usize),
    /// Parser refuses to keep parsing move counter because it is too big.
    TooManyMoves,
    /// Error in the parser.
    InternalError(usize),
}

/// Castling rights for one player
#[derive(Debug)]
pub struct CastlingRights {
    /// Kingside
    k: bool,
    /// Queenside
    q: bool,
}

impl Default for CastlingRights {
    fn default() -> Self {
        CastlingRights { k: true, q: true }
    }
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

    /// Castling rights (white)
    white_castle: CastlingRights,
    /// Castling rights (black)
    black_castle: CastlingRights,

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

    pub fn from_fen(fen: String) -> Result<Position, FenError> {
        //! Parse FEN string into position.

        /// Parser state machine.
        #[derive(Clone, Copy)]
        enum FenState {
            /// Parses space characters between arguments, and jumps to next state.
            Space,
            /// Accepts pieces in a row, or a slash, and stores row and column (0-indexed)
            Piece(usize, usize),
            /// Player whose turn it is
            Side,
            /// Castling ability
            Castle,
            /// En passant square, letter part
            EnPassantFile,
            /// En passant square, digit part
            EnPassantRank,
            /// Half-move counter for 50-move draw rule
            HalfMove,
            /// Full-move counter
            FullMove,
        }

        /// Maximum amount of moves in the counter to parse before giving up
        const MAX_MOVES: usize = 999_999;

        let mut pos = Position::default();

        let mut parser_state = FenState::Piece(0, 0);
        let mut next_state = FenState::Space;

        /// Create parse error at a given index
        macro_rules! bad_char {
            ($idx:ident) => {
                Err(FenError::BadChar($idx))
            };
        }

        /// Parse a space character, then jump to the given state
        macro_rules! parse_space_and_goto {
            ($next:expr) => {
                parser_state = FenState::Space;
                next_state = $next;
            };
        }

        for (i, c) in fen.chars().enumerate() {
            match parser_state {
                FenState::Space => {
                    match c {
                        ' ' => {
                            parser_state = next_state;
                        }
                        _ => return bad_char!(i),
                    };
                }

                FenState::Piece(mut row, mut col) => {
                    // FEN stores rows differently from our bitboard
                    let real_row = BOARD_HEIGHT - 1 - row;

                    match c {
                        '/' => {
                            if col < BOARD_WIDTH {
                                return Err(FenError::NotEnoughPieces(i));
                            } else if row >= BOARD_HEIGHT {
                                return Err(FenError::TooManyRows(i));
                            }
                            col = 0;
                            row += 1;
                            parser_state = FenState::Piece(row, col)
                        }
                        pc_char @ ('b'..='r' | 'B'..='R') => {
                            let pc = ColPiece::try_from(pc_char).or(bad_char!(i))?;

                            pos.set_piece(
                                Index::from_row_col(real_row, col)
                                    .or(Err(FenError::InternalError(i)))?,
                                pc,
                            );
                            col += 1;
                            if col > 8 {
                                return Err(FenError::TooManyPieces(i));
                            };
                            parser_state = FenState::Piece(row, col)
                        }
                        number @ '1'..='9' => {
                            if let Some(n) = number.to_digit(10) {
                                col += n as usize;
                                if col > BOARD_WIDTH {
                                    return Err(FenError::TooManyPieces(i));
                                };
                                parser_state = FenState::Piece(row, col);
                            } else {
                                return bad_char!(i);
                            }
                        }
                        ' ' => {
                            if row < BOARD_HEIGHT - 1 {
                                return Err(FenError::NotEnoughRows(i));
                            } else if col < BOARD_WIDTH {
                                return Err(FenError::NotEnoughPieces(i));
                            }
                            parser_state = FenState::Side
                        }
                        _ => return bad_char!(i),
                    };
                }
                FenState::Side => {
                    match c {
                        'w' => pos.turn = Color::White,
                        'b' => pos.turn = Color::Black,
                        _ => return bad_char!(i),
                    }
                    parse_space_and_goto!(FenState::Castle);
                }
                FenState::Castle => match c {
                    'Q' => pos.white_castle.q = true,
                    'q' => pos.black_castle.q = true,
                    'K' => pos.white_castle.k = true,
                    'k' => pos.black_castle.k = true,
                    ' ' => parser_state = FenState::EnPassantRank,
                    '-' => {
                        parse_space_and_goto!(FenState::EnPassantRank);
                    }
                    _ => return bad_char!(i),
                },
                FenState::EnPassantRank => {
                    match c {
                        '-' => {
                            parse_space_and_goto!(FenState::HalfMove);
                        }
                        'a'..='h' => {
                            // TODO: fix this
                            pos.ep_square = Some(Index((c as usize - 'a' as usize) * 8));
                            parser_state = FenState::EnPassantFile;
                        }
                        _ => return bad_char!(i),
                    };
                }
                FenState::EnPassantFile => {
                    if let Some(digit) = c.to_digit(10) {
                        pos.ep_square = Some(Index(
                            usize::from(pos.ep_square.unwrap_or(Index(0))) + digit as usize,
                        ));
                    } else {
                        return bad_char!(i);
                    }
                    parse_space_and_goto!(FenState::HalfMove);
                }
                FenState::HalfMove => {
                    if let Some(digit) = c.to_digit(10) {
                        if pos.half_moves > MAX_MOVES {
                            return Err(FenError::TooManyMoves);
                        }
                        pos.half_moves *= 10;
                        pos.half_moves += digit as usize;
                    } else if c == ' ' {
                        parser_state = FenState::FullMove;
                    } else {
                        return bad_char!(i);
                    }
                }
                FenState::FullMove => {
                    if let Some(digit) = c.to_digit(10) {
                        if pos.half_moves > MAX_MOVES {
                            return Err(FenError::TooManyMoves);
                        }
                        pos.full_moves *= 10;
                        pos.full_moves += digit as usize;
                    } else {
                        return bad_char!(i);
                    }
                }
            }
        }

        // parser is always ready to receive another full move digit,
        // so there is no real "stop" state
        if matches!(parser_state, FenState::FullMove) {
            Ok(pos)
        } else {
            Err(FenError::MissingFields)
        }
    }
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
