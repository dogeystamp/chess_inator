/*

This file is part of chess_inator.
chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Main UCI engine binary.

use chess_inator::eval::eval_metrics;
use chess_inator::fen::FromFen;
use chess_inator::movegen::{FromUCIAlgebraic, Move, ToUCIAlgebraic};
use chess_inator::search::{best_line, InterfaceMsg, SearchEval};
use chess_inator::Board;
use std::io;
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};

/// UCI protocol says to ignore any unknown words.
///
/// This macro exists to avoid copy-pasting this explanation everywhere.
macro_rules! ignore {
    () => {
        continue
    };
}

/// UCI engine metadata query.
fn cmd_uci() -> String {
    let str = "id name chess_inator\n\
                id author dogeystamp\n\
                uciok";
    str.into()
}

/// Parse the `moves` after setting an initial position.
fn cmd_position_moves(mut tokens: std::str::SplitWhitespace<'_>, mut board: Board) -> Board {
    while let Some(token) = tokens.next() {
        match token {
            "moves" => {
                for mv in tokens.by_ref() {
                    let mv = Move::from_uci_algebraic(mv).unwrap();
                    let _ = mv.make(&mut board);
                }
            }
            _ => ignore!(),
        }
    }

    board
}

/// Sets the position.
fn cmd_position(mut tokens: std::str::SplitWhitespace<'_>) -> Board {
    while let Some(token) = tokens.next() {
        match token {
            "fen" => {
                let mut fen = String::with_capacity(64);
                // fen is 6 whitespace-delimited fields
                for i in 0..6 {
                    fen.push_str(tokens.next().expect("FEN missing fields"));
                    if i < 5 {
                        fen.push(' ')
                    }
                }

                let board = Board::from_fen(&fen)
                    .unwrap_or_else(|e| panic!("failed to parse fen '{fen}': {e:?}"));
                let board = cmd_position_moves(tokens, board);

                return board;
            }
            "startpos" => {
                let board = Board::starting_pos();
                let board = cmd_position_moves(tokens, board);

                return board;
            }
            _ => ignore!(),
        }
    }

    panic!("position command was empty")
}

/// Play the game.
fn cmd_go(mut _tokens: std::str::SplitWhitespace<'_>, board: &mut Board) {
    // interface-to-engine
    let (tx1, rx) = channel();
    let tx2 = tx1.clone();

    // timeout
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(1000));
        let _ = tx2.send(InterfaceMsg::Stop);
    });

    let (line, eval) = best_line(board, None, Some(rx));

    let chosen = line.last().copied();
    println!(
        "info pv{}",
        line.iter()
            .rev()
            .map(|mv| mv.to_uci_algebraic())
            .fold(String::new(), |a, b| a + " " + &b)
    );
    match eval {
        SearchEval::Checkmate(n) => println!("info score mate {}", n / 2),
        SearchEval::Centipawns(eval) => {
            println!("info score cp {}", eval,)
        }
    }
    match chosen {
        Some(mv) => println!("bestmove {}", mv.to_uci_algebraic()),
        None => println!("bestmove 0000"),
    }
}

/// Print static evaluation of the position.
fn cmd_eval(mut _tokens: std::str::SplitWhitespace<'_>, board: &mut Board) {
    let res = eval_metrics(board);
    println!("STATIC EVAL (negative black, positive white):\n- pst: {}\n- king distance: {} ({} distance)\n- phase: {}\n- total: {}", res.pst_eval, res.king_distance_eval, res.king_distance, res.phase, res.total_eval);
}

fn main() {
    let stdin = io::stdin();

    let mut board = Board::starting_pos();

    loop {
        let mut line = String::new();
        stdin.read_line(&mut line).unwrap();
        let mut tokens = line.split_whitespace();
        while let Some(token) = tokens.next() {
            match token {
                "uci" => {
                    println!("{}", cmd_uci());
                }
                "isready" => {
                    println!("readyok");
                }
                "ucinewgame" => {
                    board = Board::starting_pos();
                }
                "quit" => {
                    return;
                }
                "position" => {
                    board = cmd_position(tokens);
                }
                "go" => {
                    cmd_go(tokens, &mut board);
                }
                // non-standard command.
                "eval" => {
                    cmd_eval(tokens, &mut board);
                }
                _ => ignore!(),
            }

            break;
        }
    }
}
