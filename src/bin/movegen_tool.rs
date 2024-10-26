//! Generates moves from the FEN in the argv.

use chess_inator::fen::FromFen;
use chess_inator::movegen::LegalMoveGen;
use chess_inator::Board;
use std::env;

fn main() {
    let fen = env::args().nth(1).unwrap();
    let board = Board::from_fen(&fen).unwrap();
    let mvs = board.gen_moves();
    for mv in mvs.into_iter() {
        println!("{mv:?}")
    }
}
