//! Generates moves from the FEN in the argv.

use chess_inator::Board;
use chess_inator::fen::FromFen;
use chess_inator::movegen::LegalMoveGen;
use std::env;

fn main() {
    for arg in env::args().skip(2) {
        let board = Board::from_fen(&arg).unwrap();
        let mvs = board.gen_moves();
        for mv in mvs.into_iter() {
            println!("{mv:?}")
        }
    }
}
