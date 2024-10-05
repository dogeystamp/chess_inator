//! Move generation.

use crate::fen::{FromFen, START_POSITION};
use crate::{
    BoardState, ColPiece, Color, Piece, Square, SquareError, BOARD_HEIGHT, BOARD_WIDTH, N_SQUARES,
};
use std::rc::Rc;

/// Game tree node.
#[derive(Clone, Debug)]
struct Node {
    /// Immutable position data.
    pos: BoardState,
    /// Backlink to previous node.
    prev: Option<Rc<Node>>,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            pos: BoardState::from_fen(START_POSITION.to_string())
                .expect("Starting FEN should be valid"),
            prev: None,
        }
    }
}

/// Piece enum specifically for promotions.
#[derive(Debug, Copy, Clone)]
enum PromotePiece {
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

/// Pseudo-legal move.
///
/// No checking is done when constructing this.
enum MoveType {
    /// Pawn promotes to another piece.
    Promotion(PromotePiece),
    /// Capture, or push move. Includes castling and en-passant too.
    Normal,
}
/// Move data common to all move types.
struct Move {
    src: Square,
    dest: Square,
    move_type: MoveType,
}

impl Move {
    /// Make move and return new position.
    ///
    /// Old position is saved in a backlink.
    pub fn make(self, old_node: Node) -> Node {
        let old_pos = old_node.pos;
        let mut node = Node {
            prev: Some(Rc::new(old_node)),
            pos: old_pos,
        };

        if old_pos.turn == Color::Black {
            node.pos.full_moves += 1;
        }

        /// Get the piece at the source square.
        macro_rules! pc_src {
            ($data: ident) => {
                node.pos
                    .get_piece($data.src)
                    .expect("Move source should have a piece")
            };
        }
        /// Perform sanity checks.
        macro_rules! pc_asserts {
            ($pc_src: ident, $data: ident) => {
                debug_assert_eq!($pc_src.col, node.pos.turn, "Moving piece on wrong turn.");
                debug_assert_ne!($data.src, $data.dest, "Moving piece to itself.");
            };
        }

        match self.move_type {
            MoveType::Promotion(to_piece) => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src, self);
                debug_assert_eq!(pc_src.pc, Piece::Pawn);

                node.pos.del_piece(self.src);
                node.pos.set_piece(
                    self.dest,
                    ColPiece {
                        pc: Piece::from(to_piece),
                        col: pc_src.col,
                    },
                );
            }
            MoveType::Normal => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src, self);
                if matches!(pc_src.pc, Piece::Pawn) {
                    // pawn moves are irreversible
                    node.pos.half_moves = 0;

                    // en-passant
                    if self.src.0 + (BOARD_WIDTH) * 2 == self.dest.0 {
                        node.pos.ep_square = Some(
                            Square::try_from(self.src.0 + BOARD_WIDTH)
                                .expect("En-passant target should be valid."),
                        )
                    } else if self.dest.0 + (BOARD_WIDTH) * 2 == self.src.0 {
                        node.pos.ep_square = Some(
                            Square::try_from(self.src.0 - BOARD_WIDTH)
                                .expect("En-passant target should be valid."),
                        )
                    } else {
                        node.pos.ep_square = None;
                    }
                } else {
                    node.pos.half_moves += 1;
                    node.pos.ep_square = None;
                }

                let castle = &mut node.pos.castle.0[pc_src.col as usize];
                // forfeit castling rights
                if matches!(pc_src.pc, Piece::King) {
                    castle.k = false;
                    castle.q = false;
                } else if matches!(pc_src.pc, Piece::Rook) {
                    match pc_src.col {
                        Color::White => {
                            if self.src == Square(0) {
                                castle.q = false;
                            } else if self.src == Square(BOARD_WIDTH - 1) {
                                castle.k = false;
                            };
                        }
                        Color::Black => {
                            if self.src == Square((BOARD_HEIGHT - 1) * BOARD_WIDTH) {
                                castle.q = false;
                            } else if self.src == Square(N_SQUARES - 1) {
                                castle.k = false;
                            };
                        }
                    }
                }

                if let Some(_pc_dest) = node.pos.get_piece(self.dest) {
                    // captures are irreversible
                    node.pos.half_moves = 0;
                }

                node.pos.del_piece(self.src);
                node.pos.set_piece(self.dest, pc_src);
            }
        }

        node.pos.turn = node.pos.turn.flip();

        node
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
enum MoveAlgebraicError {
    /// String is invalid length; refuse to parse
    InvalidLength(usize),
    /// Invalid character at given index.
    InvalidCharacter(usize),
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

        let src_sq = match Square::from_algebraic(&value[0..=1]) {
            Ok(sq) => sq,
            Err(e) => {
                return Err(MoveAlgebraicError::SquareError(0, e));
            }
        };

        let dest_sq = match Square::from_algebraic(&value[2..=3]) {
            Ok(sq) => sq,
            Err(e) => {
                return Err(MoveAlgebraicError::SquareError(0, e));
            }
        };

        let mut move_type = MoveType::Normal;

        if value_len == 5 {
            let promote_char = value.as_bytes()[4] as char;
            match promote_char {
                'q' => move_type = MoveType::Promotion(PromotePiece::Queen),
                'b' => move_type = MoveType::Promotion(PromotePiece::Bishop),
                'n' => move_type = MoveType::Promotion(PromotePiece::Knight),
                'r' => move_type = MoveType::Promotion(PromotePiece::Rook),
                _ => return Err(MoveAlgebraicError::InvalidCharacter(4)),
            }
        }

        Ok(Move {
            src: src_sq,
            dest: dest_sq,
            move_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{START_POSITION, ToFen};

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
        ];

        for (i, test_case) in test_cases.iter().enumerate() {
            let (start_pos, moves) = test_case;

            // make move
            eprintln!("Starting test case {i}, make move.");
            let mut node = Node {
                pos: BoardState::from_fen(start_pos.to_string()).unwrap(),
                prev: None,
            };
            for (move_str, expect_fen) in moves {
                let mv = Move::from_uci_algebraic(move_str).unwrap();
                eprintln!("Moving {move_str}.");
                node = mv.make(node);
                assert_eq!(node.pos.to_fen(), expect_fen.to_string())
            }

            // unmake move
            eprintln!("Starting test case {i}, unmake move.");
            let mut cur_node = Rc::new(node.clone());
            for (_, expect_fen) in moves.iter().rev().chain([("", *start_pos)].iter()) {
                eprintln!("{}", expect_fen);
                assert_eq!(*cur_node.pos.to_fen(), expect_fen.to_string());
                if *expect_fen != *start_pos {
                    cur_node = cur_node.prev.clone().unwrap();
                }
            }
        }
    }
}
