/*

This file is part of chess_inator.
chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Threading, state, and flow of information management.
//!
//! This file contains types and helper utilities; see main for actual implementation.

use crate::prelude::*;

/// State machine states.
#[derive(Clone, Copy, Debug)]
pub enum UCIMode {
    /// It is engine's turn; engine is thinking about a move.
    Think,
    /// It is the opponent's turn; engine is thinking about a move.
    Ponder,
    /// The engine is not doing anything.
    Idle,
}

/// State machine transitions.
#[derive(Clone, Copy, Debug)]
pub enum UCIModeTransition {
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
pub struct UCIModeMachine {
    pub mode: UCIMode,
}

#[derive(Debug)]
pub struct InvalidTransitionError {
    /// Original state.
    pub from: UCIMode,
    /// Desired destination state.
    pub to: UCIMode,
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
    pub fn transition(&mut self, t: UCIModeTransition) -> Result<(), InvalidTransitionError> {
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

/// Message (engine->main) to communicate the best move.
pub struct MsgBestmove {
    /// Best line (reversed stack; last element is best current move)
    pub pv: Vec<Move>,
    /// Evaluation of the position
    pub eval: SearchEval,
    /// Extra information (displayed as `info string`).
    pub info: Vec<String>,
}

/// Interface messages that may be received by main's channel.
pub enum MsgToMain {
    StdinLine(String),
    Bestmove(MsgBestmove),
}

pub struct GoMessage {
    pub board: Board,
    pub time_lims: TimeLimits,
}

/// Main -> Engine thread channel message.
pub enum MsgToEngine {
    /// `go` command. Also sends board position and engine configuration to avoid state
    /// synchronization issues (i.e. avoid sending position after a go command, and not before).
    Go(Box<GoMessage>),
    /// Transmit configuration settings.
    Configure(SearchConfig),
    /// Hard stop command. Halt search immediately.
    Stop,
    /// Ask the engine to wipe its state (notably transposition table).
    NewGame,
}
