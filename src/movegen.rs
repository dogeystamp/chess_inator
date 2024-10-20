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
            pos: BoardState::from_fen(START_POSITION).expect("Starting FEN should be valid"),
            prev: None,
        }
    }
}

/// Piece enum specifically for promotions.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum MoveType {
    /// Pawn promotes to another piece.
    Promotion(PromotePiece),
    /// Capture, or push move. Includes castling and en-passant too.
    Normal,
}
/// Pseudo-legal move.
///
/// No checking is done when constructing this.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Move {
    src: Square,
    dest: Square,
    move_type: MoveType,
}

impl Move {
    /// Make move and return new position.
    ///
    /// Old position is saved in a backlink.
    /// No checking is done to verify even pseudo-legality of the move.
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

                let _ = node.pos.del_piece(self.src);
                node.pos.set_piece(
                    self.dest,
                    ColPiece {
                        pc: Piece::from(to_piece),
                        col: pc_src.col,
                    },
                )
            }
            MoveType::Normal => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src, self);

                let pc_dest: Option<ColPiece> = node.pos.get_piece(self.dest);

                let (src_row, src_col) = self.src.to_row_col();
                let (dest_row, dest_col) = self.dest.to_row_col();

                if matches!(pc_src.pc, Piece::Pawn) {
                    // pawn moves are irreversible
                    node.pos.half_moves = 0;

                    // set en-passant target square
                    if src_row.abs_diff(dest_row) == 2 {
                        let new_idx = match pc_src.col {
                            Color::White => self.src.0 + BOARD_WIDTH,
                            Color::Black => self.src.0 - BOARD_WIDTH,
                        };
                        node.pos.ep_square = Some(
                            Square::try_from(new_idx).expect("En-passant target should be valid."),
                        )
                    } else {
                        node.pos.ep_square = None;
                        if pc_dest.is_none() && src_col != dest_col {
                            // we took en passant
                            debug_assert!(src_row.abs_diff(dest_row) == 1);
                            debug_assert_eq!(self.dest, old_pos.ep_square.unwrap());
                            // square to actually capture at
                            let ep_capture = Square::try_from(match pc_src.col {
                                Color::White => self.dest.0 - BOARD_WIDTH,
                                Color::Black => self.dest.0 + BOARD_WIDTH,
                            })
                            .expect("En-passant capture square should be valid.");
                            node.pos
                                .del_piece(ep_capture)
                                .expect("En-passant capture square should have piece.");
                        }
                    }
                } else {
                    node.pos.half_moves += 1;
                    node.pos.ep_square = None;
                }

                if pc_dest.is_some() {
                    // captures are irreversible
                    node.pos.half_moves = 0;
                }

                let castle = &mut node.pos.castle.0[pc_src.col as usize];
                if matches!(pc_src.pc, Piece::King) {
                    // forfeit castling rights
                    castle.k = false;
                    castle.q = false;

                    // and maybe perform a castle
                    let horiz_diff = src_col.abs_diff(dest_col);
                    if horiz_diff == 2 {
                        let rook_row = src_row;
                        let rook_src_col = if src_col > dest_col {
                            0
                        } else {
                            BOARD_WIDTH - 1
                        };
                        let rook_dest_col = if src_col > dest_col {
                            dest_col + 1
                        } else {
                            dest_col - 1
                        };
                        let rook_src = Square::from_row_col(rook_row, rook_src_col)
                            .expect("rook castling src square should be valid");
                        let rook_dest = Square::from_row_col(rook_row, rook_dest_col)
                            .expect("rook castling dest square should be valid");
                        node.pos.move_piece(rook_src, rook_dest);
                    }
                    debug_assert!(
                        (0..=2).contains(&horiz_diff),
                        "king moved horizontally {} squares",
                        horiz_diff
                    );
                } else if matches!(pc_src.pc, Piece::Rook) {
                    // forfeit castling rights
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

                node.pos.move_piece(self.src, self.dest);
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
pub enum MoveAlgebraicError {
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

        let src_sq = match value[0..=1].parse::<Square>() {
            Ok(sq) => sq,
            Err(e) => {
                return Err(MoveAlgebraicError::SquareError(0, e));
            }
        };

        let dest_sq = match value[2..=3].parse::<Square>() {
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

/// Pseudo-legal move generation.
///
/// "Pseudo-legal" here means that moving into check is allowed, and capturing friendly pieces is
/// allowed. These will be filtered out in the legal move generation step.
pub trait PseudoMoveGen {
    type MoveIterable;
    fn gen_pseudo_moves(self) -> Self::MoveIterable;
}

enum SliderDirection {
    /// Rook movement
    Straight,
    /// Bishop movement
    Diagonal,
    /// Queen/king movement
    Star,
}
/// Generate slider moves for a given square.
///
/// # Arguments
///
/// * `board`: Board to generate moves with.
/// * `src`: Square on which the slider piece is on.
/// * `move_list`: Vector to append generated moves to.
/// * `slide_type`: Directions the piece is allowed to go in.
/// * `keep_going`: Allow sliding more than one square (true for everything except king).
fn move_slider(
    board: &BoardState,
    src: Square,
    move_list: &mut Vec<Move>,
    slide_type: SliderDirection,
    keep_going: bool,
) {
    let dirs_straight = [(0, 1), (1, 0), (-1, 0), (0, -1)];
    let dirs_diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    let dirs_star = [
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (0, 1),
        (1, 0),
        (-1, 0),
        (0, -1),
    ];

    let dirs = match slide_type {
        SliderDirection::Straight => dirs_straight.iter(),
        SliderDirection::Diagonal => dirs_diag.iter(),
        SliderDirection::Star => dirs_star.iter(),
    };

    for dir in dirs {
        let (mut r, mut c) = src.to_row_col();
        loop {
            // increment
            let nr = r as isize + dir.0;
            let nc = c as isize + dir.1;

            if let Ok(dest) = Square::from_row_col_signed(nr, nc) {
                r = nr as usize;
                c = nc as usize;

                move_list.push(Move {
                    src,
                    dest,
                    move_type: MoveType::Normal,
                });

                // Stop at other pieces.
                if let Some(_cap_pc) = board.get_piece(dest) {
                    break;
                }
            } else {
                break;
            }

            if !keep_going {
                break;
            }
        }
    }
}

impl PseudoMoveGen for BoardState {
    type MoveIterable = Vec<Move>;

    fn gen_pseudo_moves(self) -> Self::MoveIterable {
        let mut ret = Vec::new();
        for pl in self.players {
            for sq in pl.board(Piece::Rook).into_iter() {
                move_slider(&self, sq, &mut ret, SliderDirection::Straight, true);
            }
        }
        ret
    }
}

/// Legal move generation.
pub trait LegalMoveGen {
    type MoveIterable;
    fn gen_moves(self) -> Self::MoveIterable;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{ToFen, START_POSITION};

    /// Test that slider pieces can move and capture.
    #[test]
    fn test_slider_movegen() {
        let test_cases = [(
            // start position
            "8/8/8/8/8/8/8/R7 w - - 0 1",
            // expected moves
            vec![(
                // source piece
                "a1",
                // destination squares
                vec![
                    "a2", "a3", "a4", "a5", "a6", "a7", "a8", "b1", "c1", "d1", "e1", "f1", "g1",
                    "h1",
                ],
            )],
        )];

        for (fen, expected) in test_cases {
            let board = BoardState::from_fen(fen).unwrap();

            let mut moves = board.gen_pseudo_moves();
            moves.sort_unstable();
            let moves = moves;

            let expected_moves = expected.iter().map(|(src, dests)| {});
        }
    }

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
            // pawn promotion test
            (
                "4k3/6P1/8/8/8/8/1p6/4K3 w - - 0 1",
                vec![
                    ("g7g8n", "4k1N1/8/8/8/8/8/1p6/4K3 b - - 0 1"),
                    ("b2b1q", "4k1N1/8/8/8/8/8/8/1q2K3 w - - 0 2"),
                ],
            ),
            // en passant test
            (
                "k7/4p3/8/3P4/3p4/8/4P3/K7 w - - 0 1",
                vec![
                    ("e2e4", "k7/4p3/8/3P4/3pP3/8/8/K7 b - e3 0 1"),
                    ("d4e3", "k7/4p3/8/3P4/8/4p3/8/K7 w - - 0 2"),
                    ("a1b1", "k7/4p3/8/3P4/8/4p3/8/1K6 b - - 1 2"),
                    ("e7e5", "k7/8/8/3Pp3/8/4p3/8/1K6 w - e6 0 3"),
                    ("d5e6", "k7/8/4P3/8/8/4p3/8/1K6 b - - 0 3"),
                ],
            ),
            // castle test (white kingside, black queenside)
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                vec![
                    ("e1g1", "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1"),
                    ("e8c8", "2kr3r/8/8/8/8/8/8/R4RK1 w - - 2 2"),
                ],
            ),
            // castle test (white queenside, black kingside)
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                vec![
                    ("e1c1", "r3k2r/8/8/8/8/8/8/2KR3R b kq - 1 1"),
                    ("e8g8", "r4rk1/8/8/8/8/8/8/2KR3R w - - 2 2"),
                ],
            ),
        ];

        for (i, test_case) in test_cases.iter().enumerate() {
            let (start_pos, moves) = test_case;

            // make move
            eprintln!("Starting test case {i}, make move.");
            let mut node = Node {
                pos: BoardState::from_fen(start_pos).unwrap(),
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
