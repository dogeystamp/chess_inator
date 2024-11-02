/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Move generation.

use crate::fen::ToFen;
use crate::{
    Board, CastleRights, ColPiece, Color, Piece, Square, SquareError, BOARD_HEIGHT, BOARD_WIDTH,
    N_SQUARES,
};

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

impl From<PromotePiece> for char {
    fn from(value: PromotePiece) -> Self {
        Piece::from(value).into()
    }
}

struct NonPromotePiece;

impl TryFrom<Piece> for PromotePiece {
    type Error = NonPromotePiece;

    fn try_from(value: Piece) -> Result<Self, Self::Error> {
        match value {
            Piece::Rook => Ok(PromotePiece::Rook),
            Piece::Bishop => Ok(PromotePiece::Bishop),
            Piece::Knight => Ok(PromotePiece::Knight),
            Piece::Queen => Ok(PromotePiece::Queen),
            Piece::King => Err(NonPromotePiece),
            Piece::Pawn => Err(NonPromotePiece),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AntiMoveType {
    Normal,
    /// En passant.
    EnPassant {
        cap: Square,
    },
    /// Pawn promotion.
    Promotion,
    /// King-rook castle. The king is the one considered to move.
    Castle {
        rook_src: Square,
        rook_dest: Square,
    },
}

/// Information used to reverse (unmake) a move.
#[derive(Debug, Clone, Copy)]
pub struct AntiMove {
    dest: Square,
    src: Square,
    /// Captured piece, always assumed to be of enemy color.
    cap: Option<Piece>,
    move_type: AntiMoveType,
    /// Half-move counter prior to this move
    half_moves: usize,
    /// Castling rights prior to this move.
    castle: CastleRights,
    /// En passant target square prior to this move.
    ep_square: Option<Square>,
}

impl AntiMove {
    /// Undo the move.
    pub fn unmake(self, pos: &mut Board) {
        pos.move_piece(self.dest, self.src);
        pos.half_moves = self.half_moves;
        pos.castle = self.castle;
        pos.ep_square = self.ep_square;

        /// Restore captured piece at a given square.
        macro_rules! cap_sq {
            ($sq: expr) => {
                if let Some(cap_pc) = self.cap {
                    pos.set_piece(
                        $sq,
                        ColPiece {
                            pc: cap_pc,
                            col: pos.turn.flip(),
                        },
                    );
                }
            };
        }

        pos.turn = pos.turn.flip();
        if pos.turn == Color::Black {
            pos.full_moves -= 1;
        }

        match self.move_type {
            AntiMoveType::Normal => {
                cap_sq!(self.dest)
            }
            AntiMoveType::EnPassant { cap } => {
                cap_sq!(cap);
            }
            AntiMoveType::Promotion => {
                cap_sq!(self.dest);
                pos.set_piece(
                    self.src,
                    ColPiece {
                        pc: Piece::Pawn,
                        col: pos.turn,
                    },
                );
            }
            AntiMoveType::Castle {
                rook_src,
                rook_dest,
            } => {
                pos.move_piece(rook_dest, rook_src);
            }
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
    /// Apply move to a position.
    pub fn make(self, pos: &mut Board) -> AntiMove {
        let mut anti_move = AntiMove {
            dest: self.dest,
            src: self.src,
            cap: None,
            move_type: AntiMoveType::Normal,
            half_moves: pos.half_moves,
            castle: pos.castle,
            ep_square: pos.ep_square,
        };

        // reset en passant
        let ep_square = pos.ep_square;
        pos.ep_square = None;

        if pos.turn == Color::Black {
            pos.full_moves += 1;
        }

        /// Get the piece at the source square.
        macro_rules! pc_src {
            ($data: ident) => {
                pos.get_piece($data.src)
                    .expect("Move source should have a piece")
            };
        }
        /// Perform sanity checks.
        macro_rules! pc_asserts {
            ($pc_src: ident) => {
                debug_assert_eq!(
                    $pc_src.col,
                    pos.turn,
                    "Moving piece on wrong turn. Move {} -> {} on board '{}'",
                    self.src,
                    self.dest,
                    pos.to_fen()
                );
                debug_assert_ne!(self.src, self.dest, "Moving piece to itself.");
            };
        }

        match self.move_type {
            MoveType::Promotion(to_piece) => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src);
                debug_assert_eq!(pc_src.pc, Piece::Pawn);

                pos.half_moves = 0;

                anti_move.move_type = AntiMoveType::Promotion;

                pos.del_piece(self.src);
                let cap_pc = pos.set_piece(
                    self.dest,
                    ColPiece {
                        pc: Piece::from(to_piece),
                        col: pc_src.col,
                    },
                );
                anti_move.cap = cap_pc.map(|pc| pc.pc);
            }
            MoveType::Normal => {
                let pc_src = pc_src!(self);
                pc_asserts!(pc_src);

                let pc_dest: Option<ColPiece> = pos.get_piece(self.dest);
                anti_move.cap = pc_dest.map(|pc| pc.pc);

                let (src_row, src_col) = self.src.to_row_col_signed();
                let (dest_row, dest_col) = self.dest.to_row_col_signed();

                if matches!(pc_src.pc, Piece::Pawn) {
                    // pawn moves are irreversible
                    pos.half_moves = 0;

                    // set en-passant target square
                    if src_row.abs_diff(dest_row) == 2 {
                        let ep_col = src_col;
                        debug_assert_eq!(src_col, dest_col);
                        let ep_row = dest_row
                            + match pc_src.col {
                                Color::White => -1,
                                Color::Black => 1,
                            };
                        let ep_targ = Square::from_row_col_signed(ep_row, ep_col)
                            .expect("En-passant target should be valid.");
                        pos.ep_square = Some(ep_targ)
                    } else if pc_dest.is_none() && src_col != dest_col {
                        // we took en passant
                        debug_assert!(src_row.abs_diff(dest_row) == 1);
                        assert_eq!(
                            self.dest,
                            ep_square.expect("ep target should exist if taking ep")
                        );
                        // square to actually capture at
                        let ep_capture = Square::try_from(match pc_src.col {
                            Color::White => usize::from(self.dest.0) - BOARD_WIDTH,
                            Color::Black => usize::from(self.dest.0) + BOARD_WIDTH,
                        })
                        .expect("En-passant capture square should be valid");

                        anti_move.move_type = AntiMoveType::EnPassant { cap: ep_capture };
                        if let Some(pc_cap) = pos.del_piece(ep_capture) {
                            debug_assert_eq!(
                                pc_cap.col,
                                pos.turn.flip(),
                                "attempt to en passant wrong color, pos '{}', move {:?}",
                                pos.to_fen(),
                                self
                            );
                            anti_move.cap = Some(pc_cap.pc);
                        } else {
                            panic!(
                                "En-passant capture square should have piece. Position '{}', move {:?}",
                                pos.to_fen(),
                                self
                            );
                        }
                    }
                } else {
                    pos.half_moves += 1;
                }

                if pc_dest.is_some() {
                    // captures are irreversible
                    pos.half_moves = 0;
                }

                let castle = &mut pos.castle[pc_src.col];
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
                            isize::try_from(BOARD_WIDTH).unwrap() - 1
                        };
                        let rook_dest_col = if src_col > dest_col {
                            dest_col + 1
                        } else {
                            dest_col - 1
                        };
                        let rook_src = Square::from_row_col_signed(rook_row, rook_src_col)
                            .expect("rook castling src square should be valid");
                        let rook_dest = Square::from_row_col_signed(rook_row, rook_dest_col)
                            .expect("rook castling dest square should be valid");
                        debug_assert!(pos.get_piece(rook_src).is_some(), "rook castling src square has no rook (move: {rook_src} -> {rook_dest})");
                        anti_move.move_type = AntiMoveType::Castle {
                            rook_src,
                            rook_dest,
                        };
                        pos.move_piece(rook_src, rook_dest);
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
                            } else if self.src == Square::try_from(BOARD_WIDTH - 1).unwrap() {
                                castle.k = false;
                            };
                        }
                        Color::Black => {
                            if self.src
                                == Square::try_from((BOARD_HEIGHT - 1) * BOARD_WIDTH).unwrap()
                            {
                                castle.q = false;
                            } else if self.src == Square::try_from(N_SQUARES - 1).unwrap() {
                                castle.k = false;
                            };
                        }
                    }
                }

                pos.move_piece(self.src, self.dest);
            }
        };

        pos.turn = pos.turn.flip();

        anti_move
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
    /// Can't promote to a given piece (char at given index).
    InvalidPromotePiece(usize),
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

            let err = Err(MoveAlgebraicError::InvalidCharacter(4));
            let pc = Piece::try_from(promote_char).or(err)?;

            let err = Err(MoveAlgebraicError::InvalidPromotePiece(4));
            move_type = MoveType::Promotion(PromotePiece::try_from(pc).or(err)?);
        }

        Ok(Move {
            src: src_sq,
            dest: dest_sq,
            move_type,
        })
    }
}

impl ToUCIAlgebraic for Move {
    fn to_uci_algebraic(&self) -> String {
        let prom_str = match self.move_type {
            MoveType::Promotion(promote_piece) => char::from(promote_piece).to_string(),
            _ => "".to_string(),
        };

        format!("{}{}{}", self.src, self.dest, prom_str)
    }
}

#[derive(Debug, Clone, Copy)]
enum MoveGenType {
    /// Legal move generation.
    Legal,
    /// Allow capturing friendly pieces, moving into check, but not castling through check.
    Pseudo,
}

/// Internal, slightly more general movegen interface
trait MoveGenInternal {
    fn gen_moves_general(&mut self, gen_type: MoveGenType) -> impl IntoIterator<Item = Move>;
}

pub trait MoveGen {
    /// Legal move generation.
    fn gen_moves(&mut self) -> impl IntoIterator<Item = Move>;
}

impl<T: MoveGenInternal> MoveGen for T {
    fn gen_moves(&mut self) -> impl IntoIterator<Item = Move> {
        self.gen_moves_general(MoveGenType::Legal)
    }
}

pub const DIRS_STRAIGHT: [(isize, isize); 4] = [(0, 1), (1, 0), (-1, 0), (0, -1)];
pub const DIRS_DIAG: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
pub const DIRS_STAR: [(isize, isize); 8] = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1),
];
pub const DIRS_KNIGHT: [(isize, isize); 8] = [
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
    board: &Board,
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
fn is_legal(board: &mut Board, mv: Move) -> bool {
    // mut required for check checking
    // disallow friendly fire
    let src_pc = board
        .get_piece(mv.src)
        .expect("move source should have piece");
    if let Some(dest_pc) = board.get_piece(mv.dest) {
        if dest_pc.col == src_pc.col {
            return false;
        }
    }

    // disallow moving into check
    let anti_move = mv.make(board);
    let is_check = board.is_check(board.turn.flip());
    anti_move.unmake(board);
    if is_check {
        return false;
    }

    true
}

impl MoveGenInternal for Board {
    fn gen_moves_general(&mut self, gen_type: MoveGenType) -> impl IntoIterator<Item = Move> {
        let mut ret = Vec::new();
        let pl = self[self.turn];
        macro_rules! squares {
            ($pc: ident) => {
                pl[Piece::$pc].into_iter()
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
            let (r, c) = src.to_row_col_signed();
            let rights = self.castle[self.turn];
            let castle_sides = [(rights.k, 2, BOARD_WIDTH as isize - 1), (rights.q, -2, 0)];
            for (is_allowed, move_offset, endpoint) in castle_sides {
                if !is_allowed {
                    continue;
                }

                let (rook_r, rook_c) = (r, endpoint);
                let rook_sq = Square::from_row_col_signed(rook_r, rook_c).unwrap();
                let rook_exists = self
                    .get_piece(rook_sq)
                    .map_or(false, |pc| pc.pc == Piece::Rook);
                if !rook_exists {
                    continue;
                }

                let path_range = if c < endpoint {
                    (c + 1)..endpoint
                } else {
                    (endpoint + 1)..c
                };

                debug_assert_ne!(
                    path_range.len(),
                    0,
                    "c {:?}, endpoint {:?}, range {:?}",
                    c,
                    endpoint,
                    path_range
                );

                let mut path_squares =
                    path_range.map(|nc| Square::from_row_col_signed(r, nc).unwrap());
                debug_assert_ne!(path_squares.len(), 0);

                // find first blocking piece
                let is_path_blocked = path_squares.find_map(|sq| self.get_piece(sq)).is_some();
                if is_path_blocked {
                    continue;
                }

                let nc: isize = c + move_offset;
                let dest = Square::from_row_col_signed(r, nc)
                    .expect("Castle destination square should be valid");

                debug_assert!(c.abs_diff(nc) == 2);

                // ensure the path is not being attacked (castle through check)
                let check_range = if c < nc { c..=nc } else { nc..=c };
                debug_assert!(!check_range.is_empty());
                let is_any_checked = check_range
                    .map(|nc| Square::from_row_col_signed(r, nc).unwrap())
                    .map(|dest| {
                        let mut board = *self;
                        board.move_piece(src, dest);
                        board.is_check(self.turn)
                    })
                    .any(|x| x);
                if is_any_checked {
                    continue;
                }

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

            let nr = r + isize::from(self.turn.sign());
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
        ret.into_iter().filter(move |mv| match gen_type {
            MoveGenType::Legal => is_legal(self, *mv),
            MoveGenType::Pseudo => true,
        })
    }
}

/// How many nodes at depth N can be reached from this position.
pub fn perft(depth: usize, pos: &mut Board) -> usize {
    if depth == 0 {
        return 1;
    };

    let mut ans = 0;

    let moves: Vec<Move> = pos.gen_moves().into_iter().collect();
    for mv in moves {
        let anti_move = mv.make(pos);
        ans += perft(depth - 1, pos);
        anti_move.unmake(pos);
    }

    ans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{FromFen, ToFen, START_POSITION};

    #[test]
    /// Ensure that bitboard properly reflects captures.
    fn test_bitboard_capture() {
        let mut pos = Board::from_fen("8/8/8/8/8/8/r7/R7 w - - 0 1").unwrap();
        let mv = Move::from_uci_algebraic("a1a2").unwrap();
        let _anti_move = mv.make(&mut pos);

        use std::collections::hash_set::HashSet;
        use Piece::*;
        for pc in [Rook, Bishop, Knight, Queen, King, Pawn] {
            let white: HashSet<_> = pos[Color::White][pc].into_iter().collect();
            let black: HashSet<_> = pos[Color::Black][pc].into_iter().collect();
            let intersect = white.intersection(&black).collect::<Vec<_>>();
            assert!(
                intersect.is_empty(),
                "Bitboard in illegal state: {pc:?} collides at {}",
                intersect[0]
            );
        }
    }

    /// Helper to produce test cases.
    fn decondense_moves(test_case: (&str, Vec<(&str, Vec<&str>, MoveType)>)) -> (Board, Vec<Move>) {
        let (fen, expected) = test_case;
        let board = Board::from_fen(fen).unwrap();

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
    fn flip_test_case(board: Board, moves: &Vec<Move>) -> (Board, Vec<Move>) {
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
            // white castle test (blocked again)
            (
                "8/8/8/8/8/8/r6r/R3K1nR w KQ - 0 1",
                vec![
                    // NOTE: pseudo-legal e1
                    ("a1", vec!["b1", "a2", "c1", "d1", "e1"], MoveType::Normal),
                    ("h1", vec!["g1", "h2"], MoveType::Normal),
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

        let test_cases = test_cases.map(decondense_moves);

        let augmented_test_cases = test_cases.clone().map(|tc| flip_test_case(tc.0, &tc.1));
        let all_cases = [augmented_test_cases, test_cases].concat();

        for (mut board, expected_moves) in all_cases {
            let mut moves: Vec<Move> = board
                .gen_moves_general(MoveGenType::Pseudo)
                .into_iter()
                .collect();
            moves.sort_unstable();
            let moves = moves;

            assert_eq!(moves, expected_moves, "failed tc {}", board.to_fen());
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
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(board.is_check(Color::White), *expected, "failed on {}", fen);

            let board_anti = board.flip_colors();
            assert_eq!(
                board_anti.is_check(Color::Black),
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
            (
                "rnbqkbnr/p1pppppp/8/1P6/8/8/1PPPPPPP/RNBQKBNR b KQkq - 0 2",
                vec![
                    ("a7", vec!["a6", "a5"], MoveType::Normal),
                    ("c7", vec!["c6", "c5"], MoveType::Normal),
                    ("d7", vec!["d6", "d5"], MoveType::Normal),
                    ("e7", vec!["e6", "e5"], MoveType::Normal),
                    ("f7", vec!["f6", "f5"], MoveType::Normal),
                    ("g7", vec!["g6", "g5"], MoveType::Normal),
                    ("h7", vec!["h6", "h5"], MoveType::Normal),
                    ("g8", vec!["h6", "f6"], MoveType::Normal),
                    ("b8", vec!["c6", "a6"], MoveType::Normal),
                    ("c8", vec!["b7", "a6"], MoveType::Normal),
                ],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/6rr/4K2R w KQ - 0 1",
                vec![
                    ("e1", vec!["d1", "f1"], MoveType::Normal),
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                ],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/5r1r/4K2R w KQ - 0 1",
                vec![
                    ("e1", vec!["d1"], MoveType::Normal),
                    ("h1", vec!["g1", "f1", "h2"], MoveType::Normal),
                ],
            ),
            // castling while checked
            (
                "8/8/8/8/8/8/rrrrr2r/4K2R w KQ - 0 1",
                vec![("e1", vec!["f1"], MoveType::Normal)],
            ),
            // castling while checked
            (
                "8/8/8/8/8/8/r3rrrr/R3K3 w KQ - 0 1",
                vec![("e1", vec!["d1"], MoveType::Normal)],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/r1r5/R3K3 w KQ - 0 1",
                vec![
                    ("e1", vec!["d1", "f1"], MoveType::Normal),
                    ("a1", vec!["a2", "b1", "c1", "d1"], MoveType::Normal),
                ],
            ),
            // castling through check
            (
                "8/8/8/8/8/8/r2r4/R3K3 w KQ - 0 1",
                vec![
                    ("e1", vec!["f1"], MoveType::Normal),
                    ("a1", vec!["a2", "b1", "c1", "d1"], MoveType::Normal),
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

        let test_cases = test_cases.map(|tc| decondense_moves(tc));
        let augmented_test_cases = test_cases.clone().map(|tc| flip_test_case(tc.0, &tc.1));

        let all_cases = [augmented_test_cases, test_cases].concat();

        for (mut board, mut expected_moves) in all_cases {
            eprintln!("on test '{}'", board.to_fen());
            expected_moves.sort_unstable();
            let expected_moves = expected_moves;

            let mut moves: Vec<Move> = board.gen_moves().into_iter().collect();
            moves.sort_unstable();
            let moves = moves;

            assert_eq!(moves, expected_moves);
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
            // promotion unmake regression test
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBPPP3/q4N2/Pp4PP/R2Q1RK1 b kq - 0 1",
                vec![(
                    "b2a1q",
                    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBPPP3/q4N2/P5PP/q2Q1RK1 w kq - 0 2",
                )],
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

            eprintln!("Starting test case {i}, make move.");
            let mut pos = Board::from_fen(start_pos).unwrap();
            for (move_str, expect_fen) in moves {
                let prior_fen = pos.to_fen();
                let mv = Move::from_uci_algebraic(move_str).unwrap();
                eprintln!("Moving {move_str} on {}", prior_fen);
                let anti_move = mv.make(&mut pos);
                eprintln!("Unmaking {move_str} on {}.", pos.to_fen());
                anti_move.unmake(&mut pos);
                assert_eq!(
                    pos.to_fen(),
                    prior_fen.to_string(),
                    "failed unmake with {anti_move:?}"
                );
                eprintln!("Remaking {move_str}.");
                let _anti_move = mv.make(&mut pos);
                assert_eq!(pos.to_fen(), expect_fen.to_string());
            }
        }
    }

    #[test]
    fn test_uci_move_fmt() {
        let test_cases = ["a1e5", "e7e8q", "e7e8r", "e7e8b", "e7e8n"];
        for tc in test_cases {
            let mv = Move::from_uci_algebraic(tc).unwrap();
            assert_eq!(mv.to_uci_algebraic(), tc);
        }
    }

    /// The standard movegen test.
    ///
    /// See https://www.chessprogramming.org/Perft
    #[test]
    fn test_perft() {
        // https://www.chessprogramming.org/Perft_Results
        let test_cases = [
            (
                // fen
                START_POSITION,
                // expected perft values
                vec![1, 20, 400, 8_902, 197_281, 4_865_609, 119_060_324],
                // limit depth when not under `cargo test --release` (unoptimized build too slow)
                4,
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                vec![1, 48, 2_039, 97_862, 4_085_603],
                3,
            ),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                vec![1, 14, 191, 2_812, 43_238, 674_624, 11_030_083],
                4,
            ),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                vec![1, 6, 264, 9467, 422_333, 15_833_292],
                3,
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                vec![1, 44, 1_486, 62_379, 2_103_487, 89_941_194],
                3,
            ),
            (
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
                vec![1, 46, 2_079, 89_890, 3_894_594],
                3,
            ),
        ];
        for (fen, expected_values, _debug_limit_depth) in test_cases {
            let mut pos = Board::from_fen(fen).unwrap();

            for (depth, expected) in expected_values.iter().enumerate() {
                eprintln!("running perft depth {depth} on position '{fen}'");
                #[cfg(debug_assertions)]
                {
                    if depth > _debug_limit_depth {
                        break;
                    }
                }
                assert_eq!(perft(depth, &mut pos), *expected,);
            }
        }
    }
}
