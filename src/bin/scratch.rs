use chess_inator::Position;

fn main() {
    let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let board = Position::from_fen(fen.into()).unwrap();
    println!("{}", board);
    println!("{:#?}", board);
}
