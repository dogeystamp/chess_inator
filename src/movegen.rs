//! Move generation.

use crate::fen::{FromFen, START_POSITION};
use crate::{
    BoardState, ColPiece, Color, Piece, Square, SquareError, BOARD_HEIGHT, BOARD_WIDTH, N_SQUARES,
};
use std::rc::Rc;

/// Game tree node.
#[derive(Clone, Debug)]
pub struct Node {
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

impl Node {
    /// Undo move.
    ///
    /// Intended usage is to always keep an Rc to the current node, and overwrite it with the
    /// result of unmake.
    pub fn unmake(&self) -> Rc<Node> {
        self.prev
            .clone()
            .expect("unmake should not be called on root node")
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
enum MoveType {
    /// Pawn promotes to another piece.
    Promotion(PromotePiece),
    /// Capture, or push move. Includes castling and en-passant too.
    Normal,
}
/// Pseudo-legal move.
///
/// No checking is done when constructing this.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct Move {
    src: Square,
    dest: Square,
    move_type: MoveType,
}

impl Move {
    /// Make move, without setting up the backlink for unmake.
    ///
    /// Call this directly when making new positions that are dead ends (won't be used further).
    fn make_unlinked(self, old_pos: BoardState) -> BoardState {
        let mut new_pos = old_pos;

        if old_pos.turn == Color::Black {
            new_pos.full_moves += 1;
        }

        /// Get the piece at the source square.
        macro_rules! pc_src {
            ($data: ident) => {
                new_pos
                    .get_piece($data.src)
                    .expect("Move source should have a piece")
            };
        }
        /// Perform sanity checks.
        macro_rules! pc_asserts {
            ($pc_src: ident, $data: ident) => {
                debug_assert_eq!($pc_src.col, new_pos.turn, "Moving piece on wrong turn.");
                debug_assert_ne!($data.src, $data.dest, "Moving piece to itself.");
            };
        }

        match self.move_type {
            MoveType::Promotion(to_piece) => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src, self);
                debug_assert_eq!(pc_src.pc, Piece::Pawn);

                let _ = new_pos.del_piece(self.src);
                new_pos.set_piece(
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

                let pc_dest: Option<ColPiece> = new_pos.get_piece(self.dest);

                let (src_row, src_col) = self.src.to_row_col();
                let (dest_row, dest_col) = self.dest.to_row_col();

                if matches!(pc_src.pc, Piece::Pawn) {
                    // pawn moves are irreversible
                    new_pos.half_moves = 0;

                    // set en-passant target square
                    if src_row.abs_diff(dest_row) == 2 {
                        let new_idx = match pc_src.col {
                            Color::White => self.src.0 + BOARD_WIDTH,
                            Color::Black => self.src.0 - BOARD_WIDTH,
                        };
                        new_pos.ep_square = Some(
                            Square::try_from(new_idx).expect("En-passant target should be valid."),
                        )
                    } else {
                        new_pos.ep_square = None;
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
                            new_pos
                                .del_piece(ep_capture)
                                .expect("En-passant capture square should have piece.");
                        }
                    }
                } else {
                    new_pos.half_moves += 1;
                    new_pos.ep_square = None;
                }

                if pc_dest.is_some() {
                    // captures are irreversible
                    new_pos.half_moves = 0;
                }

                let castle = &mut new_pos.pl_castle_mut(pc_src.col);
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
                        new_pos.move_piece(rook_src, rook_dest);
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

                new_pos.move_piece(self.src, self.dest);
            }
        }

        new_pos.turn = new_pos.turn.flip();

        new_pos
    }

    /// Make move and return new position.
    ///
    /// Old position is saved in a backlink.
    /// No checking is done to verify even pseudo-legality of the move.
    pub fn make(self, old_node: Rc<Node>) -> Node {
        let pos = self.make_unlinked(old_node.pos);
        Node {
            prev: Some(old_node),
            pos,
        }
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
    fn gen_pseudo_moves(&self) -> impl IntoIterator<Item = Move>;
}

const DIRS_STRAIGHT: [(isize, isize); 4] = [(0, 1), (1, 0), (-1, 0), (0, -1)];
const DIRS_DIAG: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const DIRS_STAR: [(isize, isize); 8] = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1),
];
const DIRS_KNIGHT: [(isize, isize); 8] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
];
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
    let dirs = match slide_type {
        SliderDirection::Straight => DIRS_STRAIGHT.iter(),
        SliderDirection::Diagonal => DIRS_DIAG.iter(),
        SliderDirection::Star => DIRS_STAR.iter(),
    };

    for dir in dirs {
        let (mut r, mut c) = src.to_row_col_signed();
        loop {
            // increment
            let nr = r + dir.0;
            let nc = c + dir.1;

            if let Ok(dest) = Square::from_row_col_signed(nr, nc) {
                r = nr;
                c = nc;

                move_list.push(Move {
                    src,
                    dest,
                    move_type: MoveType::Normal,
                });

                // stop at other pieces.
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
    fn gen_pseudo_moves(&self) -> impl IntoIterator<Item = Move> {
        let mut ret = Vec::new();
        let pl = self.pl(self.turn);
        macro_rules! squares {
            ($pc: ident) => {
                pl.board(Piece::$pc).into_iter()
            };
        }
        for sq in squares!(Rook) {
            move_slider(self, sq, &mut ret, SliderDirection::Straight, true);
        }
        for sq in squares!(Bishop) {
            move_slider(self, sq, &mut ret, SliderDirection::Diagonal, true);
        }
        for sq in squares!(Queen) {
            move_slider(self, sq, &mut ret, SliderDirection::Star, true);
        }
        for src in squares!(King) {
            move_slider(self, src, &mut ret, SliderDirection::Star, false);
            let (r, c) = src.to_row_col();
            let rights = self.pl_castle(self.turn);
            let castle_sides = [(rights.k, 2, BOARD_WIDTH - 1), (rights.q, -2, 0)];
            for (is_allowed, move_offset, endpoint) in castle_sides {
                if !is_allowed {
                    continue;
                }

                let range = if c < endpoint {
                    (c + 1)..endpoint
                } else {
                    (endpoint + 1)..c
                };

                debug_assert_ne!(
                    range.len(),
                    0,
                    "c {:?}, endpoint {:?}, range {:?}",
                    c,
                    endpoint,
                    range
                );

                let mut range_squares = range.map(|nc| Square::from_row_col(r, nc).unwrap());
                debug_assert_ne!(range_squares.len(), 0);

                // find first blocking piece
                let is_path_blocked = range_squares.find_map(|sq| self.get_piece(sq)).is_some();
                if is_path_blocked {
                    continue;
                }

                let nc: isize = c.try_into().unwrap();
                let dest = Square::from_row_col_signed(r.try_into().unwrap(), nc + move_offset)
                    .expect("Castle destination square should be valid");
                ret.push(Move {
                    src,
                    dest,
                    move_type: MoveType::Normal,
                })
            }
        }
        for src in squares!(Pawn) {
            let (r, c) = src.to_row_col_signed();

            let last_row = match self.turn {
                Color::White => isize::try_from(BOARD_HEIGHT).unwrap() - 1,
                Color::Black => 0,
            };

            let nr = (r)
                + match self.turn {
                    Color::White => 1,
                    Color::Black => -1,
                };
            let is_promotion = nr == last_row;

            macro_rules! push_moves {
                ($src: ident, $dest: ident) => {
                    if is_promotion {
                        use PromotePiece::*;
                        for prom_pc in [Queen, Knight, Rook, Bishop] {
                            ret.push(Move {
                                $src,
                                $dest,
                                move_type: MoveType::Promotion(prom_pc),
                            });
                        }
                    } else {
                        ret.push(Move {
                            $src,
                            $dest,
                            move_type: MoveType::Normal,
                        });
                    }
                };
            }

            // capture
            for horiz in [-1, 1] {
                let nc = c + horiz;
                let dest = match Square::from_row_col_signed(nr, nc) {
                    Ok(sq) => sq,
                    Err(_) => continue,
                };
                if self.get_piece(dest).is_some() || self.ep_square == Some(dest) {
                    push_moves!(src, dest);
                }
            }

            // single push
            let nc = c;
            let dest = match Square::from_row_col_signed(nr, nc) {
                Ok(sq) => sq,
                Err(_) => continue,
            };

            if self.get_piece(dest).is_none() {
                push_moves!(src, dest);

                // double push
                if r == match self.turn {
                    Color::White => 1,
                    Color::Black => isize::try_from(BOARD_HEIGHT).unwrap() - 2,
                } {
                    let nr = (r)
                        + match self.turn {
                            Color::White => 2,
                            Color::Black => -2,
                        };
                    let nc = c;
                    let dest = Square::from_row_col_signed(nr, nc)
                        .expect("Pawn double push should have valid destination");
                    if self.get_piece(dest).is_none() {
                        push_moves!(src, dest);
                    }
                }
            }
        }
        for src in squares!(Knight) {
            let (r, c) = src.to_row_col_signed();

            for dir in DIRS_KNIGHT {
                let nr = r + dir.0;
                let nc = c + dir.1;
                if let Ok(dest) = Square::from_row_col_signed(nr, nc) {
                    ret.push(Move {
                        src,
                        dest,
                        move_type: MoveType::Normal,
                    })
                }
            }
        }
        ret
    }
}

/// Legal move generation.
pub trait LegalMoveGen {
    fn gen_moves(&self) -> impl IntoIterator<Item = Move>;
}

/// Is a given player in check?
fn is_check(board: &BoardState, pl: Color) -> bool {
    for src in board.pl(pl).board(Piece::King).into_iter() {
        macro_rules! detect_checker {
            ($dirs: ident, $pc: ident, $keep_going: expr) => {
                for dir in $dirs.into_iter() {
                    let (mut r, mut c) = src.to_row_col_signed();
                    loop {
                        let (nr, nc) = (r + dir.0, c + dir.1);
                        if let Ok(sq) = Square::from_row_col_signed(nr, nc) {
                            if let Some(pc) = board.get_piece(sq) {
                                if pc.pc == Piece::$pc && pc.col != pl {
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

        detect_checker!(DIRS_STAR, Queen, true);
        detect_checker!(DIRS_DIAG, Bishop, true);
        detect_checker!(DIRS_STRAIGHT, Rook, true);
        detect_checker!(DIRS_STAR, King, false);
        detect_checker!(DIRS_KNIGHT, Knight, false);
        match pl {
            Color::White => detect_checker!(dirs_black_pawn, Pawn, false),
            Color::Black => detect_checker!(dirs_white_pawn, Pawn, false),
        }
    }
    false
}

impl LegalMoveGen for Node {
    fn gen_moves(&self) -> impl IntoIterator<Item = Move> {
        self.pos
            .gen_pseudo_moves()
            .into_iter()
            .filter(|mv| {
                // disallow friendly fire
                let src_pc = self
                    .pos
                    .get_piece(mv.src)
                    .expect("move source should have piece");
                if let Some(dest_pc) = self.pos.get_piece(mv.dest) {
                    return dest_pc.col != src_pc.col;
                }
                true
            })
            .filter(|mv| {
                // disallow moving into check
                let new_pos = mv.make_unlinked(self.pos);
                !is_check(&new_pos, self.pos.turn)
            })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{ToFen, START_POSITION};

    /// Helper to produce test cases.
    fn decondense_moves(
        test_case: (&str, Vec<(&str, Vec<&str>, MoveType)>),
    ) -> (BoardState, Vec<Move>) {
        let (fen, expected) = test_case;
        let board = BoardState::from_fen(fen).unwrap();

        let mut expected_moves = expected
            .iter()
            .map(|(src, dests, move_type)| {
                let src = src.parse::<Square>().unwrap();
                let dests = dests
                    .iter()
                    .map(|x| x.parse::<Square>())
                    .map(|x| x.unwrap());
                dests.map(move |dest| Move {
                    src,
                    dest,
                    move_type: *move_type,
                })
            })
            .flatten()
            .collect::<Vec<Move>>();

        expected_moves.sort_unstable();
        (board, expected_moves)
    }

    /// Generate new test cases by flipping colors on existing ones.
    fn flip_test_case(board: BoardState, moves: &Vec<Move>) -> (BoardState, Vec<Move>) {
        let mut move_vec = moves
            .iter()
            .map(|mv| Move {
                src: mv.src.mirror_vert(),
                dest: mv.dest.mirror_vert(),
                move_type: mv.move_type,
            })
            .collect::<Vec<Move>>();
        move_vec.sort_unstable();
        (board.flip_colors(), move_vec)
    }

    /// Test movegen through contrived positions.
    #[test]
    fn test_movegen() {
        let test_cases = [
            // rook test
            (
                // start position
                "8/8/8/8/8/8/8/R7 w - - 0 1",
                // expected moves
                vec![(
                    // source piece
                    "a1",
                    // destination squares
                    vec![
                        "a2", "a3", "a4", "a5", "a6", "a7", "a8", "b1", "c1", "d1", "e1", "f1",
                        "g1", "h1",
                    ],
                    MoveType::Normal,
                )],
            ),
            // white castle test (blocked)
            (
                "8/8/8/8/8/8/r6r/R3Kn1R w KQ - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    // NOTE: pseudo-legal f1
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                    // NOTE: pseudo-legal d2, e2, f2, f1
                    (
                        "e1",
                        vec!["c1", "d1", "f1", "d2", "e2", "f2"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // white castle test (no rights, blocked)
            (
                "8/8/8/8/8/8/r6r/R3Kn1R w K - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    // NOTE: pseudo-legal f1
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                    // NOTE: pseudo-legal d2, e2, f2, f1
                    ("e1", vec!["d1", "f1", "d2", "e2", "f2"], MoveType::Normal),
                ],
            ),
            // white castle test
            (
                "8/8/8/8/8/8/r6r/R3K2R w KQ - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    // NOTE: pseudo-legal e1
                    ("h1", vec!["g1", "f1", "e1", "h2"], MoveType::Normal),
                    // NOTE: pseudo-legal d2, e2, f2
                    (
                        "e1",
                        vec!["g1", "c1", "d1", "f1", "d2", "e2", "f2"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // black castle test
            (
                "r3k2r/R6R/8/8/8/8/8/8 b kq - 0 1",
                vec![
                    // NOTE: pseudo-legal e8
                    ("a8", vec!["b8", "a7", "c8", "d8", "e8"], MoveType::Normal),
                    // NOTE: pseudo-legal e8
                    ("h8", vec!["g8", "f8", "e8", "h7"], MoveType::Normal),
                    // NOTE: pseudo-legal d7, e7, f7
                    (
                        "e8",
                        vec!["g8", "c8", "d8", "f8", "d7", "e7", "f7"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // horse test
            (
                "8/2r1r3/1n3q2/3N4/8/2p5/8/8 w - - 0 1",
                vec![(
                    "d5",
                    vec!["f6", "e7", "c7", "b6", "b4", "c3", "e3", "f4"],
                    MoveType::Normal,
                )],
            ),
            // horse (blocked by boundary)
            (
                "8/8/8/8/8/6n1/5n2/7N w - - 0 1",
                vec![("h1", vec!["f2", "g3"], MoveType::Normal)],
            ),
            // white pawn promotion
            (
                "q1q5/1P6/8/8/8/8/8/8 w - - 0 1",
                vec![
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b7", vec!["b8"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b7", vec!["a8"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b7", vec!["c8"], MoveType::Promotion(PromotePiece::Knight)),
                ],
            ),
            // black pawn promotion
            (
                "8/8/8/8/8/8/1p6/Q1Q5 b - - 0 1",
                vec![
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b2", vec!["b1"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b2", vec!["a1"], MoveType::Promotion(PromotePiece::Knight)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Rook)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Queen)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Bishop)),
                    ("b2", vec!["c1"], MoveType::Promotion(PromotePiece::Knight)),
                ],
            ),
            // white pawn push/capture
            (
                "8/8/8/8/8/p1p5/1P6/8 w - - 0 1",
                vec![("b2", vec!["a3", "c3", "b3", "b4"], MoveType::Normal)],
            ),
            // white pawn en passant
            (
                "8/8/4p3/3pP3/8/8/8/8 w - d6 0 1",
                vec![("e5", vec!["d6"], MoveType::Normal)],
            ),
            // white pawn blocked
            ("8/8/8/8/8/1p6/1P6/8 w - - 0 1", vec![]),
            // white pawn blocked (partially)
            (
                "8/8/8/8/1p6/8/1P6/8 w - - 0 1",
                vec![("b2", vec!["b3"], MoveType::Normal)],
            ),
            // black pawn push/capture
            (
                "8/1p6/P1P5/8/8/8/8/8 b - - 0 1",
                vec![("b7", vec!["a6", "c6", "b6", "b5"], MoveType::Normal)],
            ),
            // black pawn en passant
            (
                "8/8/8/8/pP6/P7/8/8 b - b3 0 1",
                vec![("a4", vec!["b3"], MoveType::Normal)],
            ),
            // black pawn blocked
            ("8/1p6/1P6/8/8/8/8/8 b - - 0 1", vec![]),
            // king against the boundary
            (
                "3K4/4p3/1q3p2/4p3/1p1r4/8/8/8 w - - 0 1",
                vec![("d8", vec!["c8", "c7", "d7", "e7", "e8"], MoveType::Normal)],
            ),
            // king test
            (
                "8/4p3/1q1K1p2/4p3/1p1r4/8/8/8 w - - 0 1",
                vec![(
                    "d6",
                    vec!["c7", "c6", "c5", "d7", "d5", "e7", "e6", "e5"],
                    MoveType::Normal,
                )],
            ),
            // queen test
            (
                "8/4p3/1q1Q1p2/4p3/1p1r4/8/8/8 w - - 0 1",
                vec![(
                    "d6",
                    vec![
                        "d5", "d4", "d7", "d8", "e7", "c5", "b4", "e6", "f6", "c6", "b6", "e5",
                        "c7", "b8",
                    ],
                    MoveType::Normal,
                )],
            ),
            // rook test (again)
            (
                "8/1p6/8/1R2p3/1p6/8/8/8 w - - 0 1",
                vec![(
                    "b5",
                    vec!["b6", "b7", "b4", "a5", "c5", "d5", "e5"],
                    MoveType::Normal,
                )],
            ),
            // bishop test
            (
                "8/4p3/3B4/4p3/1p6/8/8/8 w - - 0 1",
                vec![(
                    "d6",
                    vec!["e5", "e7", "c5", "b4", "c7", "b8"],
                    MoveType::Normal,
                )],
            ),
            // black test
            (
                "8/3b4/2R1R3/1Q6/1RqRrR2/1QQ5/2R1R3/k7 b - - 0 1",
                vec![
                    ("a1", vec!["a2", "b2", "b1"], MoveType::Normal),
                    (
                        "c4",
                        vec![
                            "b3", "b4", "b5", "c3", "c5", "c6", "d3", "e2", "d4", "d5", "e6",
                        ],
                        MoveType::Normal,
                    ),
                    (
                        "e4",
                        vec!["d4", "f4", "e3", "e2", "e5", "e6"],
                        MoveType::Normal,
                    ),
                    ("d7", vec!["c6", "e6", "c8", "e8"], MoveType::Normal),
                ],
            ),
        ];

        for tc in test_cases {
            let decondensed = decondense_moves(tc);
            let (board, expected_moves) = decondensed;

            let (anti_board, anti_expected_moves) = flip_test_case(board, &expected_moves);

            let mut moves: Vec<Move> = board.gen_pseudo_moves().into_iter().collect();
            let mut anti_moves: Vec<Move> = anti_board.gen_pseudo_moves().into_iter().collect();
            moves.sort_unstable();
            anti_moves.sort_unstable();
            let moves = moves;
            let anti_moves = anti_moves;

            assert_eq!(moves, expected_moves, "failed tc {}", board.to_fen());
            assert_eq!(
                anti_moves,
                anti_expected_moves,
                "failed anti-tc '{}' (originally '{}')",
                anti_board.to_fen(),
                board.to_fen()
            );
        }
    }

    /// Test check checker.
    #[test]
    fn test_is_check() {
        let check_cases = [
            "3r4/8/8/3K4/8/8/8/8 b - - 0 1",
            "8/8/8/3K3r/8/8/8/8 b - - 0 1",
            "8/8/8/3K4/8/8/8/3r4 b - - 0 1",
            "8/8/8/r2K4/8/8/8/8 b - - 0 1",
            "1b6/8/8/3K4/1r6/8/8/k6b b - - 0 1",
            "1b6/8/4p3/3K4/1r6/8/8/k5b1 b - - 0 1",
            "1b6/4n3/3p4/3K4/1r6/8/8/k5b1 b - - 0 1",
            "1b6/2n5/3p4/3K4/1r6/8/8/k5b1 b - - 0 1",
            "1b6/8/3p4/3K4/1r3n2/8/8/k5b1 b - - 0 1",
            "1b6/8/3p4/3K4/1r1k4/5n2/8/6b1 b - - 0 1",
            "8/8/8/4b3/r2b4/rnq5/PP6/KRrr4 w - - 0 1",
        ]
        .map(|tc| (tc, true));

        let not_check_cases = [
            "1b6/8/3p4/3K4/1r6/5n2/8/k5b1 b - - 0 1",
            "1bqnb3/3q1n2/3p4/3K4/1r6/2q1qn2/8/k5b1 b - - 0 1",
            "1bqnb1q1/3q1n2/3p4/1qbKp1q1/1r1b4/2q1qn2/8/k5b1 b - - 0 1",
            "8/8/8/4b3/r2b4/r1q5/PP6/KRrr4 w - - 0 1",
        ]
        .map(|tc| (tc, false));

        let all_cases = check_cases.iter().chain(&not_check_cases);
        for (fen, expected) in all_cases {
            let board = BoardState::from_fen(fen).unwrap();
            assert_eq!(
                is_check(&board, Color::White),
                *expected,
                "failed on {}",
                fen
            );

            let board_anti = board.flip_colors();
            assert_eq!(
                is_check(&board_anti, Color::Black),
                *expected,
                "failed on anti-version of {} ({})",
                fen,
                board_anti.to_fen()
            );
        }
    }

    /// Test legal movegen through contrived positions.
    #[test]
    fn test_legal_movegen() {
        let test_cases = [
            // rook friendly fire test
            (
                // start position
                "8/8/8/8/8/8/rr6/RRr5 w - - 0 1",
                // expected moves
                vec![
                    (
                        // source piece
                        "a1",
                        // destination squares
                        vec!["a2"],
                        MoveType::Normal,
                    ),
                    (
                        // source piece
                        "b1",
                        // destination squares
                        vec!["b2", "c1"],
                        MoveType::Normal,
                    ),
                ],
            ),
            // check test
            (
                "1bqnb1q1/3q1n2/q2p4/2bKp1q1/1r1b4/1q3n2/8/k2q2b1 w - - 0 1",
                vec![("d5", vec!["e4"], MoveType::Normal)],
            ),
            // check test
            (
                "1b1nb1q1/q2q1n2/q2p4/2bKp1q1/1r1p4/1q3n2/8/k2q2b1 w - - 0 1",
                vec![("d5", vec!["e4"], MoveType::Normal)],
            ),
        ];

        for (i, (fen, expected)) in test_cases.iter().enumerate() {
            let node = Node {
                pos: BoardState::from_fen(fen).unwrap(),
                prev: None,
            };

            let mut moves: Vec<Move> = node.gen_moves().into_iter().collect();
            moves.sort_unstable();
            let moves = moves;

            let mut expected_moves = expected
                .iter()
                .map(|(src, dests, move_type)| {
                    let src = src.parse::<Square>().unwrap();
                    let dests = dests
                        .iter()
                        .map(|x| x.parse::<Square>())
                        .map(|x| x.unwrap());
                    dests.map(move |dest| Move {
                        src,
                        dest,
                        move_type: *move_type,
                    })
                })
                .flatten()
                .collect::<Vec<Move>>();

            expected_moves.sort_unstable();
            let expected_moves = expected_moves;

            assert_eq!(moves, expected_moves, "failed test case {i} ({fen})");
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
            let mut node = Rc::new(Node {
                pos: BoardState::from_fen(start_pos).unwrap(),
                prev: None,
            });
            for (move_str, expect_fen) in moves {
                let mv = Move::from_uci_algebraic(move_str).unwrap();
                eprintln!("Moving {move_str}.");
                node = mv.make(node).into();
                assert_eq!(node.pos.to_fen(), expect_fen.to_string())
            }

            // unmake move
            eprintln!("Starting test case {i}, unmake move.");
            for (_, expect_fen) in moves.iter().rev().chain([("", *start_pos)].iter()) {
                eprintln!("{}", expect_fen);
                assert_eq!(*node.pos.to_fen(), expect_fen.to_string());
                if *expect_fen != *start_pos {
                    node = node.unmake();
                }
            }
        }
    }
}
