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
use std::collections::VecDeque;
use std::io;
use std::ops::Not;
use std::process::exit;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;

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
               option name NNUETrainInfo type check default false\n\
               option name Ponder type check default false\n\
               option name Hash type spin default 16 min 1 max 6200\n\
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
                    board.commit_move();
                    let _ = mv.make(&mut board);
                    // we won't be going back in time, so these states are useless
                    board.discard_nnue();
                }
                board.refresh_nnue();
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
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut override_depth: Option<usize> = None;
    let mut movetime: Option<u64> = None;

    let mut ponder = false;

    macro_rules! set_time {
        ($var: ident) => {
            if let Some(time) = tokens.next() {
                $var = time.parse::<u64>().ok();
            }
        };
    }

    while let Some(token) = tokens.next() {
        match token {
            "wtime" => {
                set_time!(wtime)
            }
            "btime" => {
                set_time!(btime)
            }
            "movetime" => {
                set_time!(movetime)
            }
            "depth" => {
                if let Some(depth) = tokens.next() {
                    override_depth = depth.parse::<usize>().ok();
                }
            }
            "ponder" => {
                ponder = true;
            }
            _ => ignore!(),
        }
    }

    // attempt to transition state, and if it's not allowed then ignore the command
    if ponder {
        if state
            .uci_mode
            .transition(UCIModeTransition::GoPonder)
            .is_ok()
            .not()
        {
            return;
        }
    } else if state
        .uci_mode
        .transition(UCIModeTransition::Go)
        .is_ok()
        .not()
    {
        return;
    }

    let saved_depth = state.config.depth;
    if let Some(depth) = override_depth {
        state.config.depth = depth * chess_inator::search::ONE_PLY;
    }

    state.config.pondering = ponder;

    let (ourtime_ms, theirtime_ms) = if state.board.get_turn() == Color::White {
        (wtime, btime)
    } else {
        (btime, wtime)
    };

    let time_lims = if let Some(movetime) = movetime {
        TimeLimits::from_movetime(movetime)
    } else {
        TimeLimits::from_ourtime_theirtime(
            ourtime_ms.unwrap_or(300_000),
            theirtime_ms.unwrap_or(300_000),
            &state.board,
        )
    };

    state
        .tx_engine
        .send(MsgToEngine::Configure(state.config))
        .unwrap();
    state
        .tx_engine
        .send(MsgToEngine::Go(Box::new(GoMessage {
            board: state.board.clone(),
            time_lims,
        })))
        .unwrap();

    state.config.depth = saved_depth;
}

/// Print static evaluation of the position.
///
/// This calculation is inefficient compared to what happens in the real engine, but it offers more
/// information.
fn cmd_eval(mut _tokens: std::str::SplitWhitespace<'_>, state: &mut MainState) {
    println!("STATIC EVAL");
    println!("board fen: {}", state.board.to_fen());
    println!("{}", state.board.eval());
}

fn match_true_false(s: &str) -> Option<bool> {
    match s {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Set engine options via UCI.
fn cmd_setoption(mut tokens: std::str::SplitWhitespace<'_>, state: &mut MainState) {
    while let Some(token) = tokens.next() {
        fn get_val(mut tokens: std::str::SplitWhitespace<'_>) -> Option<String> {
            if let Some("value") = tokens.next() {
                if let Some(value) = tokens.next() {
                    return Some(value.to_string());
                }
            }
            None
        }

        match token {
            "name" => {
                if let Some(name) = tokens.next() {
                    match name {
                        "NNUETrainInfo" => {
                            if let Some(value) = get_val(tokens) {
                                if let Some(value) = match_true_false(&value) {
                                    state.config.nnue_train_info = value;
                                }
                            }
                        }
                        "Ponder" => {
                            if let Some(value) = get_val(tokens) {
                                if let Some(value) = match_true_false(&value) {
                                    state.config.pondering_enabled = value;
                                }
                            }
                        }
                        "Hash" => {
                            if let Some(value) = get_val(tokens) {
                                if let Ok(value) = value.parse::<usize>() {
                                    state.config.transposition_size = value;
                                }
                            }
                        }
                        _ => {
                            println!("info string Unknown option: {}", name)
                        }
                    }
                }
            }
            _ => ignore!(),
        }

        break;
    }
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
                    state
                        .tx_engine
                        .send(MsgToEngine::Configure(state.config))
                        .unwrap();
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
                } else {
                    eprintln!("err: Can't set position in state {:?}", state.uci_mode.mode)
                }
            }
            "go" => {
                // uci mode transition done inside the command handler, since ponder/go are
                // different
                cmd_go(tokens, state);
            }
            "ponderhit" => {
                if state
                    .uci_mode
                    .transition(UCIModeTransition::PonderHit)
                    .is_ok()
                {
                    state.config.pondering = false;
                    state
                        .tx_engine
                        .send(MsgToEngine::Configure(state.config))
                        .unwrap();
                } else {
                    eprintln!("err: Can't ponderhit in state {:?}", state.uci_mode.mode)
                }
            }
            "stop" => {
                if matches!(state.uci_mode.mode, UCIMode::Think | UCIMode::Ponder) {
                    // disable uci commands until we receive a bestmove
                    state.accept_uci_input = false;
                    state.tx_engine.send(MsgToEngine::Stop).unwrap();
                }
            }
            "setoption" => {
                cmd_setoption(tokens, state);
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

/// Format engine information.
fn outp_info(info: MsgInfo, is_best_move: bool) {
    let pv_str = info
        .pv
        .iter()
        .map(|mv| mv.to_uci_algebraic())
        .fold("pv".to_string(), |a, b| a + " " + &b);

    let score_str = match info.eval {
        Score::Checkmate(n) => format!("score mate {}", n / 2),
        Score::Exact(eval) | Score::Lower(eval) | Score::Upper(eval) => {
            format!("score cp {}", eval,)
        }
        Score::Stopped => {
            panic!("ERROR: attempted to output stopped search")
        }
    };
    println!(
        "info {score_str} time {} depth {} nodes {} nps {} hashfull {} {pv_str}",
        info.time_ms, info.depth, info.nodes, info.nps, info.hashfull,
    );
    for line in info.info {
        println!("info string {line}");
    }

    if is_best_move {
        let mut pv_in_order = info.pv.iter();
        let (chosen, ponder_mv) = (pv_in_order.next(), pv_in_order.next());
        match chosen {
            Some(mv) => print!("bestmove {}", mv.to_uci_algebraic()),
            None => print!("bestmove 0000"),
        }

        if let Some(mv) = ponder_mv {
            print!(" ponder {}", mv.to_uci_algebraic())
        }
        println!();
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
            InterruptMode::default(),
        );

        loop {
            let msg = state.rx_engine.recv().unwrap();
            match msg {
                MsgToEngine::Configure(cfg) => state.config = cfg,
                MsgToEngine::Go(msg_box) => {
                    state.node_count = 0;
                    let mut board = msg_box.board;
                    state.time_lims = msg_box.time_lims;

                    let think_start = Instant::now();
                    let search_res = search(&mut board, &mut state);

                    let mut info: Vec<String> = Vec::new();
                    if state.config.nnue_train_info {
                        let is_quiet = chess_inator::search::is_quiescent_position(
                            &mut board,
                            search_res.eval,
                        );
                        let is_quiet = if is_quiet { "quiet" } else { "non-quiet" };

                        let board_tensor = chess_inator::nnue::InputTensor::from_board(&board);

                        let abs_eval =
                            EvalInt::from(search_res.eval) * EvalInt::from(board.get_turn().sign());
                        info.push(format!("NNUETrainInfo {} {} {}", is_quiet, abs_eval, {
                            board_tensor
                        }))
                    }

                    // elapsed microseconds, plus one to avoid division by zero
                    let elapsed_us = (think_start.elapsed().as_micros() as usize).saturating_add(1);

                    let nps = state.node_count.saturating_mul(1_000_000) / elapsed_us;

                    tx_main
                        .send(MsgToMain::BestMove(MsgInfo {
                            pv: search_res.pv,
                            eval: search_res.eval,
                            info,
                            nodes: state.node_count,
                            nps,
                            hashfull: state.cache.get_hashfull(),
                            time_ms: elapsed_us / 1000,
                            depth: search_res.depth,
                        }))
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
    /// When false, holds commands in a queue until the engine is ready.
    ///
    /// This is useful to allow the Main thread to await an Engine response before listening to
    /// commands from the Stdin thread.
    accept_uci_input: bool,
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
            accept_uci_input: true,
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

    // stdin queue, that we hold messages in
    let mut uci_cmd_queue = VecDeque::<String>::new();

    loop {
        let msg = if !state.accept_uci_input || uci_cmd_queue.is_empty() {
            // listen for a new message
            state.rx_main.recv().unwrap()
        } else {
            // process our backlog of commands
            MsgToMain::StdinLine(uci_cmd_queue.pop_front().unwrap())
        };
        match msg {
            MsgToMain::StdinLine(line) => {
                if state.accept_uci_input {
                    let tokens = line.split_whitespace();
                    cmd_root(tokens, &mut state);
                } else {
                    uci_cmd_queue.push_back(line);
                }
            }
            MsgToMain::BestMove(msg_info) => {
                state
                    .uci_mode
                    .transition(UCIModeTransition::Bestmove)
                    .unwrap();
                outp_info(msg_info, true);
                state.accept_uci_input = true;
            }
            MsgToMain::Info(msg_info) => {
                outp_info(msg_info, false);
            }
        }
    }
}
