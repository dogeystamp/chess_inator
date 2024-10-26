/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

use crate::{Board, ColPiece, Color, Square, SquareIdx, BOARD_HEIGHT, BOARD_WIDTH};

pub const START_POSITION: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub trait FromFen {
    type Error;
    fn from_fen(_: &str) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

pub trait ToFen {
    fn to_fen(&self) -> String;
}

/// FEN parsing error, with index of issue if applicable.
#[derive(Debug)]
pub enum FenError {
    /// Invalid character.
    BadChar(usize, char),
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

impl FromFen for Board {
    type Error = FenError;
    fn from_fen(fen: &str) -> Result<Board, FenError> {
        //! Parse FEN string into position.

        /// Parser state machine.
        #[derive(Debug, Clone, Copy)]
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

        let mut pos = Board::default();

        let mut parser_state = FenState::Piece(0, 0);
        let mut next_state = FenState::Space;

        /// Create parse error at a given index
        macro_rules! bad_char {
            ($idx:ident, $char:ident) => {
                Err(FenError::BadChar($idx, $char))
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
                        _ => return bad_char!(i, c),
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
                        pc_char @ ('a'..='z' | 'A'..='Z') => {
                            let pc = ColPiece::try_from(pc_char).or(bad_char!(i, c))?;

                            if col > 7 {
                                return Err(FenError::TooManyPieces(i));
                            };
                            pos.set_piece(
                                Square::from_row_col(real_row, col)
                                    .or(Err(FenError::InternalError(i)))?,
                                pc,
                            );
                            col += 1;
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
                                return bad_char!(i, c);
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
                        _ => return bad_char!(i, c),
                    };
                }
                FenState::Side => {
                    match c {
                        'w' => pos.turn = Color::White,
                        'b' => pos.turn = Color::Black,
                        _ => return bad_char!(i, c),
                    }
                    parse_space_and_goto!(FenState::Castle);
                }
                FenState::Castle => {
                    macro_rules! wc {
                        () => {
                            pos.castle.0[Color::White as usize]
                        };
                    }
                    macro_rules! bc {
                        () => {
                            pos.castle.0[Color::Black as usize]
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
                        _ => return bad_char!(i, c),
                    }
                }
                FenState::EnPassantRank => {
                    match c {
                        '-' => {
                            parse_space_and_goto!(FenState::HalfMove);
                        }
                        'a'..='h' => {
                            pos.ep_square = Some(Square(c as SquareIdx - b'a'));
                            parser_state = FenState::EnPassantFile;
                        }
                        _ => return bad_char!(i, c),
                    };
                }
                FenState::EnPassantFile => {
                    if let Some(digit) = c.to_digit(10) {
                        pos.ep_square = Some(Square(
                            SquareIdx::from(pos.ep_square.unwrap_or(Square(0)))
                                + (digit as SquareIdx - 1) * 8,
                        ));
                    } else {
                        return bad_char!(i, c);
                    }
                    parse_space_and_goto!(FenState::HalfMove);
                }
                FenState::HalfMove => {
                    if let Some(digit) = c.to_digit(10) {
                        if pos.half_moves > Board::MAX_MOVES {
                            return Err(FenError::TooManyMoves);
                        }
                        pos.half_moves *= 10;
                        pos.half_moves += digit as usize;
                    } else if c == ' ' {
                        parser_state = FenState::FullMove;
                    } else {
                        return bad_char!(i, c);
                    }
                }
                FenState::FullMove => {
                    if let Some(digit) = c.to_digit(10) {
                        if pos.half_moves > Board::MAX_MOVES {
                            return Err(FenError::TooManyMoves);
                        }
                        pos.full_moves *= 10;
                        pos.full_moves += digit as usize;
                    } else {
                        return bad_char!(i, c);
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

impl ToFen for Board {
    fn to_fen(&self) -> String {
        let pieces_str = (0..BOARD_HEIGHT)
            .rev()
            .map(|row| {
                let mut row_str = String::with_capacity(8);
                let mut empty_counter = 0;
                macro_rules! compact_empty {
                    () => {
                        if empty_counter > 0 {
                            row_str.push_str(&empty_counter.to_string());
                        }
                    };
                }

                for col in 0..BOARD_WIDTH {
                    let idx = Square::from_row_col(row, col).unwrap();
                    if let Some(pc) = self.get_piece(idx) {
                        compact_empty!();
                        empty_counter = 0;
                        row_str.push(pc.into())
                    } else {
                        empty_counter += 1;
                    }
                }
                compact_empty!();
                row_str
            })
            .collect::<Vec<String>>()
            .join("/");

        let turn = char::from(self.turn);
        let castle = self.castle.to_string();
        let ep_square = match self.ep_square {
            Some(sqr) => sqr.to_string(),
            None => "-".to_string(),
        };
        let half_move = self.half_moves.to_string();
        let full_move = self.full_moves.to_string();

        format!("{pieces_str} {turn} {castle} {ep_square} {half_move} {full_move}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CastlePlayer, CastleRights, N_SQUARES};

    #[test]
    fn test_fen_pieces() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_fen(fen.into()).unwrap();
        assert_eq!(
            (0..SquareIdx::try_from(N_SQUARES).unwrap())
                .map(Square)
                .map(|i| board.get_piece(i))
                .map(ColPiece::opt_to_char)
                .collect::<String>(),
            "RNBQKBNRPPPP.PPP............P...................pppppppprnbqkbnr"
        );
        assert_eq!(board.ep_square.unwrap(), Square(20));
        assert_eq!(board.turn, Color::Black);
    }

    macro_rules! make_board {
        ($fen_fmt: expr) => {
            Board::from_fen(&format!($fen_fmt)).unwrap()
        };
    }

    #[test]
    fn test_fen_ep_square() {
        let test_cases = [("e3", 20), ("h8", 63), ("a8", 56), ("h4", 31), ("a1", 0)];
        for (sqr, idx) in test_cases {
            let board = make_board!("8/8/8/8/8/8/8/8 w - {sqr} 0 0");
            assert_eq!(board.ep_square.unwrap(), Square(idx));
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
                    CastlePlayer { k: false, q: false },
                    CastlePlayer { k: false, q: false },
                ],
            ),
            (
                "k",
                [
                    CastlePlayer { k: false, q: false },
                    CastlePlayer { k: true, q: false },
                ],
            ),
            (
                "kq",
                [
                    CastlePlayer { k: false, q: false },
                    CastlePlayer { k: true, q: true },
                ],
            ),
            // This is the wrong order, but parsers should be lenient
            (
                "qk",
                [
                    CastlePlayer { k: false, q: false },
                    CastlePlayer { k: true, q: true },
                ],
            ),
            (
                "KQkq",
                [
                    CastlePlayer { k: true, q: true },
                    CastlePlayer { k: true, q: true },
                ],
            ),
            (
                "KQ",
                [
                    CastlePlayer { k: true, q: true },
                    CastlePlayer { k: false, q: false },
                ],
            ),
            (
                "QK",
                [
                    CastlePlayer { k: true, q: true },
                    CastlePlayer { k: false, q: false },
                ],
            ),
        ];
        for (castle_str, castle) in test_cases {
            let board = make_board!("8/8/8/8/8/8/8/8 w {castle_str} - 0 0");
            assert_eq!(board.castle, CastleRights(castle));
        }
    }

    #[test]
    fn test_fen_half_move_counter() {
        for i in 0..=Board::MAX_MOVES {
            let board = make_board!("8/8/8/8/8/8/8/8 w - - {i} 0");
            assert_eq!(board.half_moves, i);
            assert_eq!(board.full_moves, 0);
        }
    }

    #[test]
    fn test_fen_move_counter() {
        for i in 0..=Board::MAX_MOVES {
            let board = make_board!("8/8/8/8/8/8/8/8 w - - 0 {i}");
            assert_eq!(board.half_moves, 0);
            assert_eq!(board.full_moves, i);
        }
    }

    #[test]
    fn test_fen_printing() {
        //! Test that FENs printed are equivalent to the original.

        // FENs sourced from https://gist.github.com/peterellisjones/8c46c28141c162d1d8a0f0badbc9cff9
        let test_cases = [
            "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",
            "r6r/1b2k1bq/8/8/7B/8/8/R3K2R b KQ - 3 2",
            "r1bqkbnr/pppppppp/n7/8/8/P7/1PPPPPPP/RNBQKBNR w KQkq - 2 2",
            "r3k2r/p1pp1pb1/bn2Qnp1/2qPN3/1p2P3/2N5/PPPBBPPP/R3K2R b KQkq - 3 2",
            "2kr3r/p1ppqpb1/bn2Qnp1/3PN3/1p2P3/2N5/PPPBBPPP/R3K2R b KQ - 3 2",
            "rnb2k1r/pp1Pbppp/2p5/q7/2B5/8/PPPQNnPP/RNB1K2R w KQ - 3 9",
            "2r5/3pk3/8/2P5/8/2K5/8/8 w - - 5 4",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
            "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
            "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
            "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
            "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1",
            "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1",
            "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
            "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1",
            "8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1",
            "4k3/1P6/8/8/8/8/K7/8 w - - 0 1",
            "8/P1k5/K7/8/8/8/8/8 w - - 0 1",
            "K1k5/8/P7/8/8/8/8/8 w - - 0 1",
            "8/k1P5/8/1K6/8/8/8/8 w - - 0 1",
            "8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1",
        ];

        for fen1 in test_cases {
            println!("fen1: {fen1:?}");
            let fen2 = Board::from_fen(fen1).unwrap().to_fen();

            assert_eq!(fen1.to_string(), fen2, "FEN not equivalent")
        }
    }
}
