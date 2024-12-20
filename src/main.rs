/*

This file is part of chess_inator.
chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Main UCI engine binary.
//!
//! # Architecture
//!
//! This runs three threads, Main, Engine, and Stdin. Main coordinates everything, and performs UCI
//! parsing/communication. Stdin is read on a different thread, in order to avoid blocking on it.
//! The Engine is where the actual computation happens. It communicates state (best move, evaluations,
//! board state and configuration) with Main.
//!
//! Main has a single rx (receive) channel. This is so that it can wait for either the Engine to
//! finish a computation, or for Stdin to receive a UCI command. This way, the overall engine
//! program is always listening, even when it is thinking.
//!
//! For every go command, Main sends data, notably the current position and engine configuration,
//! to the Engine. The current position and config are re-sent every time because Main is where the
//! opponent's move, as well as any configuration options, are read and parsed. Meanwhile, internal
//! data, like the transposition table, is owned by the Engine thread.
//!
//! # Notes
//!
//! - The naming scheme for channels here is `tx_main`, `rx_main` for "transmit to Main" and
//!   "receive at Main" respectively. These names would be used for one channel.

use chess_inator::prelude::*;
use std::io;
use std::process::exit;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

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
fn cmd_position(mut tokens: std::str::SplitWhitespace<'_>, state: &mut MainState) {
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

                state.board = board;
                return;
            }
            "startpos" => {
                let board = Board::starting_pos();
                let board = cmd_position_moves(tokens, board);

                state.board = board;
                return;
            }
            _ => ignore!(),
        }
    }

    eprintln!("cmd_position: position command was empty")
}

/// Play the game.
fn cmd_go(mut tokens: std::str::SplitWhitespace<'_>, state: &mut MainState) {
    let mut wtime = 0;
    let mut btime = 0;

    macro_rules! set_time {
        ($color: expr, $var: ident) => {
            if let Some(time) = tokens.next() {
                if let Ok(time) = time.parse::<u64>() {
                    $var = time;
                }
            }
        };
    }

    while let Some(token) = tokens.next() {
        match token {
            "wtime" => {
                set_time!(Color::White, wtime)
            }
            "btime" => {
                set_time!(Color::Black, btime)
            }
            _ => ignore!(),
        }
    }

    let (ourtime_ms, theirtime_ms) = if state.board.get_turn() == Color::White {
        (wtime, btime)
    } else {
        (btime, wtime)
    };

    state
        .tx_engine
        .send(MsgToEngine::Go(Box::new(GoMessage {
            board: state.board,
            config: state.config,
            time_lims: TimeLimits::from_ourtime_theirtime(ourtime_ms, theirtime_ms),
        })))
        .unwrap();
}

/// Print static evaluation of the position.
fn cmd_eval(mut _tokens: std::str::SplitWhitespace<'_>, state: &mut MainState) {
    let res = eval_metrics(&state.board);
    println!("STATIC EVAL (negative black, positive white):\n- pst: {}\n- king distance: {} ({} distance)\n- phase: {}\n- total: {}", res.pst_eval, res.king_distance_eval, res.king_distance, res.phase, res.total_eval);
}

/// Root UCI parser.
fn cmd_root(mut tokens: std::str::SplitWhitespace<'_>, state: &mut MainState) {
    while let Some(token) = tokens.next() {
        match token {
            "uci" => {
                println!("{}", cmd_uci());
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                if matches!(state.uci_mode.mode, UCIMode::Idle) {
                    state.tx_engine.send(MsgToEngine::NewGame).unwrap();
                    state.board = Board::starting_pos();
                }
            }
            "quit" => {
                exit(0);
            }
            "position" => {
                if matches!(state.uci_mode.mode, UCIMode::Idle) {
                    cmd_position(tokens, state);
                }
            }
            "go" => {
                if state.uci_mode.transition(UCIModeTransition::Go).is_ok() {
                    cmd_go(tokens, state);
                }
            }
            "stop" => {
                // actually setting state to stop happens when bestmove is received
                if matches!(state.uci_mode.mode, UCIMode::Think | UCIMode::Ponder) {
                    state.tx_engine.send(MsgToEngine::Stop).unwrap();
                }
            }
            // non-standard command.
            "eval" => {
                cmd_eval(tokens, state);
            }
            _ => ignore!(),
        }

        break;
    }
}

/// Format a bestmove.
fn outp_bestmove(bestmove: MsgBestmove) {
    let chosen = bestmove.pv.last().copied();
    println!(
        "info pv{}",
        bestmove
            .pv
            .iter()
            .rev()
            .map(|mv| mv.to_uci_algebraic())
            .fold(String::new(), |a, b| a + " " + &b)
    );
    match bestmove.eval {
        SearchEval::Checkmate(n) => println!("info score mate {}", n / 2),
        SearchEval::Centipawns(eval) => {
            println!("info score cp {}", eval,)
        }
        SearchEval::Stopped => {
            panic!("info string ERROR: stopped search")
        }
    }
    match chosen {
        Some(mv) => println!("bestmove {}", mv.to_uci_algebraic()),
        None => println!("bestmove 0000"),
    }
}

/// The "Stdin" thread to read stdin while avoiding blocking
///
/// # Arguments
/// - `tx_main`: channel write end to send lines to
fn task_stdin_reader(tx_main: Sender<MsgToMain>) {
    thread::spawn(move || {
        let stdin = io::stdin();

        loop {
            let mut line = String::new();
            stdin.read_line(&mut line).unwrap();
            tx_main.send(MsgToMain::StdinLine(line)).unwrap();
        }
    });
}

/// The "Engine" thread that does all the computation.
fn task_engine(tx_main: Sender<MsgToMain>, rx_engine: Receiver<MsgToEngine>) {
    thread::spawn(move || {
        let conf = SearchConfig::default();
        let mut state = EngineState::new(
            conf,
            rx_engine,
            TranspositionTable::new(conf.transposition_size),
            TimeLimits::default(),
        );

        loop {
            let msg = state.rx_engine.recv().unwrap();
            match msg {
                MsgToEngine::Go(msg_box) => {
                    let mut board = msg_box.board;
                    state.config = msg_box.config;
                    state.time_lims = msg_box.time_lims;
                    let (pv, eval) = best_line(&mut board, &mut state);
                    tx_main
                        .send(MsgToMain::Bestmove(MsgBestmove { pv, eval }))
                        .unwrap();
                }
                MsgToEngine::Stop => {}
                MsgToEngine::NewGame => {
                    state.wipe_state();
                }
            }
        }
    });
}

/// State contained within the main thread.
///
/// This struct helps pass around this thread state.
struct MainState {
    /// Channel to send messages to Engine.
    tx_engine: Sender<MsgToEngine>,
    /// Channel to receive messages from Engine and Stdin.
    rx_main: Receiver<MsgToMain>,
    /// Chessboard.
    board: Board,
    /// Engine configuration settings.
    config: SearchConfig,
    /// UCI mode state machine
    uci_mode: UCIModeMachine,
}

impl MainState {
    fn new(
        tx_engine: Sender<MsgToEngine>,
        rx_main: Receiver<MsgToMain>,
        board: Board,
        config: SearchConfig,
        uci_mode: UCIModeMachine,
    ) -> Self {
        Self {
            tx_engine,
            rx_main,
            board,
            config,
            uci_mode,
        }
    }
}

/// The "Main" thread.
fn main() {
    let (tx_main, rx_main) = channel();
    task_stdin_reader(tx_main.clone());

    let (tx_engine, rx_engine) = channel();
    task_engine(tx_main, rx_engine);

    let mut state = MainState::new(
        tx_engine,
        rx_main,
        Board::starting_pos(),
        SearchConfig::default(),
        UCIModeMachine::default(),
    );

    loop {
        let msg = state.rx_main.recv().unwrap();
        match msg {
            MsgToMain::StdinLine(line) => {
                let tokens = line.split_whitespace();
                cmd_root(tokens, &mut state);
            }
            MsgToMain::Bestmove(msg_bestmove) => {
                state
                    .uci_mode
                    .transition(UCIModeTransition::Bestmove)
                    .unwrap();
                outp_bestmove(msg_bestmove);
            }
        }
    }
}
