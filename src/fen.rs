use crate::{Position, Index, Color, ColPiece};
use crate::{BOARD_WIDTH, BOARD_HEIGHT};

pub trait FromFen {
    type Error;
    fn from_fen(_: String) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

/// FEN parsing error, with index of issue if applicable.
#[derive(Debug)]
pub enum FenError {
    /// Invalid character.
    BadChar(usize),
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

impl FromFen for Position {
    type Error = FenError;
    fn from_fen(fen: String) -> Result<Position, FenError> {
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
                FenState::Castle => {
                    macro_rules! wc {
                        () => {
                            pos.castle[Color::White as usize]
                        };
                    }
                    macro_rules! bc {
                        () => {
                            pos.castle[Color::Black as usize]
                        };
                    }
                    match c {
                        'Q' => wc!().q = true,
                        'q' => bc!().q = true,
                        'K' => wc!().k = true,
                        'k' => bc!().k = true,
                        ' ' => parser_state = FenState::EnPassantRank,
                        '-' => {
                            parse_space_and_goto!(FenState::EnPassantRank);
                        }
                        _ => return bad_char!(i),
                    }
                }
                FenState::EnPassantRank => {
                    match c {
                        '-' => {
                            parse_space_and_goto!(FenState::HalfMove);
                        }
                        'a'..='h' => {
                            pos.ep_square = Some(Index(c as usize - 'a' as usize));
                            parser_state = FenState::EnPassantFile;
                        }
                        _ => return bad_char!(i),
                    };
                }
                FenState::EnPassantFile => {
                    if let Some(digit) = c.to_digit(10) {
                        pos.ep_square = Some(Index(
                            usize::from(pos.ep_square.unwrap_or(Index(0)))
                                + (digit as usize - 1) * 8,
                        ));
                    } else {
                        return bad_char!(i);
                    }
                    parse_space_and_goto!(FenState::HalfMove);
                }
                FenState::HalfMove => {
                    if let Some(digit) = c.to_digit(10) {
                        if pos.half_moves > Position::MAX_MOVES {
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
                        if pos.half_moves > Position::MAX_MOVES {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::N_SQUARES;
    use crate::CastlingRights;

    #[test]
    fn test_fen_pieces() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Position::from_fen(fen.into()).unwrap();
        assert_eq!(
            (0..N_SQUARES)
                .map(Index)
                .map(|i| board.get_piece(i))
                .map(ColPiece::opt_to_char)
                .collect::<String>(),
            "RNBQKBNRPPPP.PPP............P...................pppppppprnbqkbnr"
        );
        assert_eq!(board.ep_square.unwrap(), Index(20));
        assert_eq!(board.turn, Color::Black);
    }

    macro_rules! make_board{
        ($fen_fmt: expr) => {
            Position::from_fen(format!($fen_fmt)).unwrap()
        }
    }

    #[test]
    fn test_fen_ep_square() {
        let test_cases = [("e3", 20), ("h8", 63), ("a8", 56), ("h4", 31), ("a1", 0)];
        for (sqr, idx) in test_cases {
            let board = make_board!("8/8/8/8/8/8/8/8 w - {sqr} 0 0");
            assert_eq!(board.ep_square.unwrap(), Index(idx));
        }

        let board = make_board!("8/8/8/8/8/8/8/8 w - - 0 0");
        assert_eq!(board.ep_square, None);
    }

    #[test]
    fn test_fen_turn() {
        let test_cases = [("w", Color::White), ("b", Color::Black)];
        for (col_char, col) in test_cases {
            let board = make_board!("8/8/8/8/8/8/8/8 {col_char} - - 0 0");
            assert_eq!(board.turn, col);
        }
    }

    #[test]
    fn test_fen_castle_rights() {
        let test_cases = [
            (
                "-",
                [
                    CastlingRights { k: false, q: false },
                    CastlingRights { k: false, q: false },
                ],
            ),
            (
                "k",
                [
                    CastlingRights { k: false, q: false },
                    CastlingRights { k: true, q: false },
                ],
            ),
            (
                "kq",
                [
                    CastlingRights { k: false, q: false },
                    CastlingRights { k: true, q: true },
                ],
            ),
            (
                "qk",
                [
                    CastlingRights { k: false, q: false },
                    CastlingRights { k: true, q: true },
                ],
            ),
            (
                "KQkq",
                [
                    CastlingRights { k: true, q: true },
                    CastlingRights { k: true, q: true },
                ],
            ),
            (
                "KQ",
                [
                    CastlingRights { k: true, q: true },
                    CastlingRights { k: false, q: false },
                ],
            ),
            (
                "QK",
                [
                    CastlingRights { k: true, q: true },
                    CastlingRights { k: false, q: false },
                ],
            ),
        ];
        for (castle_str, castle) in test_cases {
            let board = make_board!("8/8/8/8/8/8/8/8 w {castle_str} - 0 0");
            assert_eq!(board.castle, castle);
        }
    }

    #[test]
    fn test_fen_half_move_counter() {
        for i in 0..=Position::MAX_MOVES {
            let board = make_board!("8/8/8/8/8/8/8/8 w - - {i} 0");
            assert_eq!(board.half_moves, i);
            assert_eq!(board.full_moves, 0);
        }
    }

    #[test]
    fn test_fen_move_counter() {
        for i in 0..=Position::MAX_MOVES {
            let board = make_board!("8/8/8/8/8/8/8/8 w - - 0 {i}");
            assert_eq!(board.half_moves, 0);
            assert_eq!(board.full_moves, i);
        }
    }
}
