//! Runs perft at depth for a given FEN.

use chess_inator::Board;
use chess_inator::fen::FromFen;
use chess_inator::movegen::perft;
use std::env;

fn main() {
    let depth = env::args().nth(1).unwrap().parse::<usize>().unwrap();
    let fen = env::args().nth(2).unwrap();
    let mut board = Board::from_fen(&fen).unwrap();
    let res = perft(depth, &mut board);
    println!("{res}")
}
