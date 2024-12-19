/*

This file is part of chess_inator.
chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Main UCI engine binary.

use chess_inator::prelude::*;
use std::cmp::min;
use std::io;
use std::sync::mpsc::{channel, Sender};
use std::thread;
use std::time::Duration;

/// State machine states.
#[derive(Clone, Copy, Debug)]
enum UCIMode {
    /// It is engine's turn; engine is thinking about a move.
    Think,
    /// It is the opponent's turn; engine is thinking about a move.
    Ponder,
    /// The engine is not doing anything.
    Idle,
}

/// State machine transitions.
#[derive(Clone, Copy, Debug)]
enum UCIModeTransition {
    /// Engine produces a best move result. Thinking to Idle.
    Bestmove,
    /// Engine is stopped via a UCI `stop` command. Thinking/Ponder to Idle.
    Stop,
    /// Engine is asked for a best move through a UCI `go`. Idle -> Thinking.
    Go,
    /// Engine starts pondering on the opponent's time. Idle -> Ponder.
    GoPonder,
    /// While engine ponders, the opponent plays a different move than expected. Ponder -> Thinking
    ///
    /// In UCI, this means that a new `position` command is sent.
    PonderMiss,
    /// While engine ponders, the opponent plays the expected move (`ponderhit`). Ponder -> Thinking
    PonderHit,
}

impl UCIModeTransition {
    /// The state that a transition goes to.
    const fn dest_mode(&self) -> UCIMode {
        use UCIMode::*;
        use UCIModeTransition::*;
        match self {
            Bestmove => Idle,
            Stop => Idle,
            Go => Think,
            GoPonder => Ponder,
            PonderMiss => Think,
            PonderHit => Think,
        }
    }
}

/// State machine for engine's UCI modes.
#[derive(Debug)]
struct UCIModeMachine {
    mode: UCIMode,
}

#[derive(Debug)]
struct InvalidTransitionError {
    /// Original state.
    from: UCIMode,
    /// Desired destination state.
    to: UCIMode,
}

impl Default for UCIModeMachine {
    fn default() -> Self {
        UCIModeMachine {
            mode: UCIMode::Idle,
        }
    }
}

impl UCIModeMachine {
    /// Change state (checked to prevent invalid transitions.)
    fn transition(&mut self, t: UCIModeTransition) -> Result<(), InvalidTransitionError> {
        macro_rules! illegal {
            () => {
                return Err(InvalidTransitionError {
                    from: self.mode,
                    to: t.dest_mode(),
                })
            };
        }
        macro_rules! legal {
            () => {{
                self.mode = t.dest_mode();
                return Ok(());
            }};
        }

        use UCIModeTransition::*;

        match t {
            Bestmove => match self.mode {
                UCIMode::Think => legal!(),
                _ => illegal!(),
            },
            Stop => match self.mode {
                UCIMode::Ponder | UCIMode::Think => legal!(),
                _ => illegal!(),
            },
            Go | GoPonder => match self.mode {
                UCIMode::Idle => legal!(),
                _ => illegal!(),
            },
            PonderMiss => match self.mode {
                UCIMode::Ponder => legal!(),
                _ => illegal!(),
            },
            PonderHit => match self.mode {
                UCIMode::Ponder => legal!(),
                _ => illegal!(),
            },
        }
    }
}

#[cfg(test)]
mod test_state_machine {
    use super::*;

    /// Non-exhaustive test of state machine.
    #[test]
    fn test_transitions() {
        let mut machine = UCIModeMachine {
            mode: UCIMode::Idle,
        };
        assert!(matches!(machine.transition(UCIModeTransition::Go), Ok(())));
        assert!(matches!(machine.mode, UCIMode::Think));
        assert!(matches!(
            machine.transition(UCIModeTransition::Stop),
            Ok(())
        ));
        assert!(matches!(machine.mode, UCIMode::Idle));
        assert!(matches!(machine.transition(UCIModeTransition::Go), Ok(())));
        assert!(matches!(
            machine.transition(UCIModeTransition::Bestmove),
            Ok(())
        ));
        assert!(matches!(machine.mode, UCIMode::Idle));
        assert!(matches!(
            machine.transition(UCIModeTransition::Bestmove),
            Err(_)
        ));
    }
}

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
fn cmd_go(
    mut tokens: std::str::SplitWhitespace<'_>,
    board: &mut Board,
    cache: &mut TranspositionTable,
) {
    // interface-to-engine
    let (tx1, rx) = channel();
    let tx2 = tx1.clone();

    // can expect a 1sec soft timeout to result in more time than that of thinking
    let mut timeout = 1650;

    while let Some(token) = tokens.next() {
        match token {
            "wtime" => {
                if board.get_turn() == Color::White {
                    if let Some(time) = tokens.next() {
                        if let Ok(time) = time.parse::<u64>() {
                            timeout = min(time / 50, timeout);
                        }
                    }
                }
            }
            "btime" => {
                if board.get_turn() == Color::Black {
                    if let Some(time) = tokens.next() {
                        if let Ok(time) = time.parse::<u64>() {
                            timeout = min(time / 50, timeout);
                        }
                    }
                }
            }
            _ => ignore!(),
        }
    }

    // timeout
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(timeout));
        let _ = tx2.send(InterfaceMsg::Stop);
    });

    let mut engine_state = EngineState::new(SearchConfig::default(), rx, cache);
    let (line, eval) = best_line(board, &mut engine_state);

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

/// Read stdin line-by-line in a non-blocking way (in another thread)
///
/// # Arguments
/// - `tx`: channel write end to send lines to
fn task_stdin_reader(tx: Sender<String>) {
    thread::spawn(move || {
        let stdin = io::stdin();

        loop {
            let mut line = String::new();
            stdin.read_line(&mut line).unwrap();
            tx.send(line).unwrap();
        }
    });
}

fn main() {
    let mut board = Board::starting_pos();
    let mut transposition_table = TranspositionTable::new(24);

    let (tx, rx) = channel();
    task_stdin_reader(tx.clone());

    loop {
        let line = rx.recv().unwrap();
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
                    transposition_table = TranspositionTable::new(24);
                }
                "quit" => {
                    return;
                }
                "position" => {
                    board = cmd_position(tokens);
                }
                "go" => {
                    cmd_go(tokens, &mut board, &mut transposition_table);
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
