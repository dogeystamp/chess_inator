//! Move generation.

use crate::fen::{FromFen, ToFen, START_POSITION};
use crate::{BoardState, Color, Square, BOARD_WIDTH};
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

/// Move data common to all move types.
struct MoveData {
    src: Square,
    dest: Square,
}
/// Pseudo-legal move.
///
/// No checking is made to see if the move is actually pseudo-legal.
enum Move {
    /// Pawn promotes to another piece.
    Promotion(MoveData, PromotePiece),
    /// King castles with rook.
    Castle(MoveData),
    /// Capture, or push move.
    Normal(MoveData),
    /// This move is an en-passant capture.
    EnPassant(MoveData),
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
        node.pos.turn = node.pos.turn.flip();
        if node.pos.turn == Color::White {
            node.pos.full_moves += 1;
        }

        match self {
            Move::Promotion(data, piece) => todo!(),
            Move::Castle(data) => todo!(),
            Move::Normal(data) => {
                let pc_src = node.pos.get_piece(data.src).unwrap();
                if matches!(pc_src.pc, crate::Piece::Pawn) {
                    // pawn moves are irreversible
                    node.pos.half_moves = 0;

                    // en-passant
                    if data.src.0 + (BOARD_WIDTH) * 2 == data.dest.0 {
                        node.pos.ep_square = Some(
                            Square::try_from(data.src.0 + BOARD_WIDTH)
                                .expect("En-passant target should be valid."),
                        )
                    } else if data.dest.0 + (BOARD_WIDTH) * 2 == data.src.0 {
                        node.pos.ep_square = Some(
                            Square::try_from(data.src.0 - BOARD_WIDTH)
                                .expect("En-passant target should be valid."),
                        )
                    } else {
                        node.pos.ep_square = None;
                    }
                } else {
                    node.pos.half_moves += 1;
                    node.pos.ep_square = None;
                }

                if let Some(_pc_dest) = node.pos.get_piece(data.dest) {
                    // captures are irreversible
                    node.pos.half_moves = 0;
                }

                node.pos.del_piece(data.src);
                node.pos.set_piece(data.dest, pc_src);
            }
            Move::EnPassant(data) => todo!(),
        }

        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_POSITION;

    /// Test that make move and unmake move for simple piece pushes/captures.
    ///
    /// Also tests en passant target square.
    #[test]
    fn test_normal_move() {
        let start_pos = START_POSITION;
        // (src, dest, expected fen)
        // FENs made with https://lichess.org/analysis
        // En-passant target square is manually added, since Lichess doesn't have it when
        // en-passant is not legal.
        let moves = [
            (
                "e2",
                "e4",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            ),
            (
                "e7",
                "e5",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            ),
            (
                "g1",
                "f3",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            ),
            (
                "g8",
                "f6",
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            ),
            (
                "f1",
                "c4",
                "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            ),
            (
                "f8",
                "c5",
                "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            ),
            (
                "d1",
                "e2",
                "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPPQPPP/RNB1K2R b KQkq - 5 4",
            ),
            (
                "d8",
                "e7",
                "rnb1k2r/ppppqppp/5n2/2b1p3/2B1P3/5N2/PPPPQPPP/RNB1K2R w KQkq - 6 5",
            ),
            (
                "f3",
                "e5",
                "rnb1k2r/ppppqppp/5n2/2b1N3/2B1P3/8/PPPPQPPP/RNB1K2R b KQkq - 0 5",
            ),
            (
                "e7",
                "e5",
                "rnb1k2r/pppp1ppp/5n2/2b1q3/2B1P3/8/PPPPQPPP/RNB1K2R w KQkq - 0 6",
            ),
        ];

        // make move
        let mut node = Node::default();
        for (src, dest, expect_fen) in moves {
            let idx_src = Square::from_algebraic(src.to_string()).unwrap();
            let idx_dest = Square::from_algebraic(dest.to_string()).unwrap();
            let mv = Move::Normal(MoveData {
                src: idx_src,
                dest: idx_dest,
            });
            node = mv.make(node);
            assert_eq!(node.pos.to_fen(), expect_fen.to_string())
        }

        // unmake move
        let mut cur_node = Rc::new(node.clone());
        for (_, _, expect_fen) in moves.iter().rev().chain([("", "", START_POSITION)].iter()) {
            println!("{}", expect_fen);
            assert_eq!(*cur_node.pos.to_fen(), expect_fen.to_string());
            if *expect_fen != START_POSITION {
                cur_node = cur_node.prev.clone().unwrap();
            }
        }
    }
}
