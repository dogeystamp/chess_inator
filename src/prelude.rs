/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Prelude that you can import entirely to use the library conveniently.

pub use crate::coordination::{
    GoMessage, MsgBestmove, MsgToEngine, MsgToMain, UCIMode, UCIModeMachine, UCIModeTransition,
};
pub use crate::eval::{eval_metrics, Eval, EvalInt, EvalMetrics};
pub use crate::fen::{FromFen, ToFen};
pub use crate::movegen::{FromUCIAlgebraic, GenAttackers, Move, MoveGen, ToUCIAlgebraic};
pub use crate::search::{
    best_line, best_move, EngineState, SearchConfig, SearchEval, TimeLimits, TranspositionTable,
};
pub use crate::{
    Board, ColPiece, Color, Piece, BOARD_HEIGHT, BOARD_WIDTH, N_COLORS, N_PIECES, N_SQUARES, Square,
};
