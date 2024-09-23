use chess_inator::Position;

fn main() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR ";
    let board = Position::from_fen(fen.into()).unwrap();
    println!("{}", board);

    let fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R ";
    let board = Position::from_fen(fen.into()).unwrap();
    print!("{}", board);
}
