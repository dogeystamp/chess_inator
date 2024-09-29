//! Move generation.

use crate::{Color, Square, BoardState};
use std::rc::Rc;

/// Game tree node.
struct Node {
    /// Immutable position data.
    pos: BoardState,
    /// Backlink to previous node.
    prev: Option<Rc<Node>>,
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
enum Move {
    /// Pawn promotes to another piece.
    Promotion { data: MoveData, piece: PromotePiece },
    /// King castles with rook.
    Castle { data: MoveData },
    /// Capture, or push move.
    Normal { data: MoveData },
    /// This move is an en-passant capture.
    EnPassant { data: MoveData },
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
            Move::Promotion { data, piece } => todo!(),
            Move::Castle { data } => todo!(),
            Move::Normal { data } => {
                let pc = node.pos.get_piece(data.src).unwrap();
                node.pos.del_piece(data.src);
                node.pos.set_piece(data.dest, pc);
            }
            Move::EnPassant { data } => todo!(),
        }

        node
    }
}
