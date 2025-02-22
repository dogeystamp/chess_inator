/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Neural network tools.
//!
//! # Neural net architecture
//!
//! The NNUE has the following layers:
//!
//! * Input (board features)
//! * Hidden layer / accumulator (N neurons)
//! * Output layer (Single neuron)
//!
//! The input layer is a multi-hot binary tensor that represents the board. It is a product of
//! color (2), piece type (6) and piece position (64), giving a total of 768 elements representing
//! for example "is there a _white_, _pawn_ at _e4_?". This information is not enough to represent
//! the board, but is enough for static evaluation purposes. Our NNUE is only expected to run on
//! quiescent positions, and our traditional minmax algorithm will take care of any exchanges, en
//! passant, and other rules that can be mechanically applied.
//!
//! In the engine, the input layer is imaginary. Because of the nature of NNUE (efficiently
//! updatable neural network), we only store the hidden layer's state, and whenever we want to flip
//! a bit in the input layer, we directly add/subtract the corresponding weight from the hidden
//! layer.
//!
//! For more information about training and the neural net architecture itself, consult the `nnue/`
//! directory's README file.

use crate::prelude::*;
use crate::search::MAX_PLY;
use crate::util::arrayvec::ArrayVec;
use crate::util::serialization::ConstCursor;
use std::fmt::Display;

// alias to easily change precision / data type
pub(crate) type Param = i16;

/// Parameters, in packed binary form.
const WEIGHTS_BIN: &[u8] = include_bytes!("weights.bin");

/// Network architecture string. Reject any weights file that does not fulfill this.
///
/// - Axx: serial number
/// - CReLU: activation function
/// - 768: input size
/// - N: hidden layer size (variable)
/// - 1: output layer size
/// - K: includes sigmoid K parameter
/// - Q: quantized
/// - T: transposed
/// - <i2: quantized to little endian integer, 2 byte
const ARCHITECTURE: &[u8] = "A07_CReLU_768_N_1_KQT<i2\x1b".as_bytes();

const HEADER_DATA: NNUEHeader = NNUEHeader::from_bytes(WEIGHTS_BIN);

/// Size of the input feature tensor.
pub const INP_TENSOR_SIZE: usize = N_COLORS * N_PIECES * N_SQUARES;
/// Size of the output layer.
const OUT_SIZE: usize = 1;
/// Size of the hidden layer (N).
const L1_SIZE: usize = HEADER_DATA.l1_size as usize;

/// Quantization scaling factor (params already scaled; we need to dequantize here)
const L1_SCALE: Param = 255;
/// Quantization scaling factor (params already scaled; we need to dequantize here)
const OUT_SCALE: Param = 64;
/// Output quantization scaling factor (params already scaled; we need to dequantize here)
const DEQUANTIZE_SCALE: i32 = (L1_SCALE * OUT_SCALE) as i32;

/// All weights and biases of the neural network.
#[derive(Debug)]
struct NNUEParameters {
    _sanity_check: [Param; 1],
    k: [Param; 1],
    l1_w: [[Param; L1_SIZE]; INP_TENSOR_SIZE],
    l1_b: [Param; L1_SIZE],
    out_w: [[Param; L1_SIZE]; OUT_SIZE],
    out_b: [Param; OUT_SIZE],
}

/// Version number of the .bin file header.
const HEADER_VERSION: u8 = 0;

/// Header information from the .bin file.
struct NNUEHeader {
    version: u8,
    l1_size: u16,
    // remember to change this padding when the above fields change
    _reserved: [u8; 29],
}

impl NNUEHeader {
    const fn from_bytes(bytes: &[u8]) -> Self {
        let mut cursor = ConstCursor::from_bytes(bytes, 0);
        let ret = NNUEHeader {
            version: cursor.read_single_u8(),
            l1_size: cursor.read_single_u16(),
            _reserved: cursor.read_u8(),
        };

        #[allow(clippy::absurd_extreme_comparisons)]
        if ret.version < HEADER_VERSION || ret.version == b'A' {
            panic!("The weights .bin file has an outdated version.")
        } else if ret.version > HEADER_VERSION {
            panic!("The engine is too outdated to use this weights .bin file.")
        }

        ret
    }
}

impl NNUEParameters {
    const fn from_bytes(bytes: &[u8]) -> Self {
        let mut cursor = ConstCursor::from_bytes(bytes, std::mem::size_of::<NNUEHeader>());
        let arch_string: [u8; ARCHITECTURE.len()] = cursor.read_u8();
        let mut i = 0;
        while i < arch_string.len() {
            if arch_string[i] != ARCHITECTURE[i] {
                panic!("The weights file (.bin) has an incompatible architecture with this engine.")
            }
            i += 1;
        }

        NNUEParameters {
            _sanity_check: cursor.read(),
            k: cursor.read(),
            l1_w: cursor.read2d(),
            l1_b: cursor.read(),
            out_w: cursor.read2d(),
            out_b: cursor.read(),
        }
    }
}

#[allow(long_running_const_eval)]
const WEIGHTS: NNUEParameters = NNUEParameters::from_bytes(WEIGHTS_BIN);
/// Sigmoid scaling factor. Makes the output roughly correspond to centipawns.
///
/// TODO: calculating this in the function and not as a const causes a stack overflow? why?
const SCALE_K: i32 = WEIGHTS.k[0] as i32;

#[derive(Debug, PartialEq, Eq)]
pub struct InputTensor([bool; INP_TENSOR_SIZE]);

/// Input tensor for the NNUE.
///
/// Note that this tensor does not exist at runtime, only during training.
impl InputTensor {
    /// Calculate index within the input tensor of a piece/color/square combination.
    pub fn idx(pc: ColPiece, sq: Square) -> usize {
        let col = pc.col as usize;
        let pc = pc.pc as usize;
        let sq = sq.0 as usize;

        let ret = col * (N_PIECES * N_SQUARES) + pc * (N_SQUARES) + sq;
        debug_assert!((0..INP_TENSOR_SIZE).contains(&ret));
        ret
    }

    /// Create the tensor from a board.
    pub fn from_board(board: &Board) -> Self {
        let mut tensor = [false; INP_TENSOR_SIZE];
        for sq in Board::squares() {
            if let Some(pc) = board.get_piece(sq) {
                let idx = Self::idx(pc, sq);
                tensor[idx] = true;
            }
        }

        InputTensor(tensor)
    }
}

impl Display for InputTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = String::from_iter(self.0.map(|x| if x { '1' } else { '0' }));
        write!(f, "{}", str)
    }
}

/// Neural network.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Nnue {
    /// Accumulator. This is the only persistent data; everything else is conceptual.
    l1: [Param; L1_SIZE],
}

pub(crate) fn crelu(x: Param) -> Param {
    x.clamp(0, L1_SCALE)
}

impl Nnue {
    /// Turn on/off a bit in the input tensor.
    pub fn bit_set(&mut self, i: usize, on: bool) {
        debug_assert!(i < INP_TENSOR_SIZE);
        if on {
            for j in 0..L1_SIZE {
                self.l1[j] += WEIGHTS.l1_w[i][j];
            }
        } else {
            for j in 0..L1_SIZE {
                self.l1[j] -= WEIGHTS.l1_w[i][j];
            }
        }
    }

    pub fn add_piece(&mut self, pc: ColPiece, sq: Square) {
        self.bit_set(InputTensor::idx(pc, sq), true);
    }

    /// Logits from neural net, which should correspond to centipawns.
    pub fn output(&self) -> EvalInt {
        let mut out: [EvalInt; OUT_SIZE] = [0; OUT_SIZE];

        for (k, out_node) in out.iter_mut().enumerate() {
            *out_node = EvalInt::from(WEIGHTS.out_b[k]);
            for j in 0..L1_SIZE {
                *out_node += EvalInt::from(WEIGHTS.out_w[k][j]) * EvalInt::from(crelu(self.l1[j]));
            }
        }

        // scaling factor
        out[0] *= SCALE_K;

        // dequantization step
        out[0] /= DEQUANTIZE_SCALE;

        out[0]
    }

    pub fn new() -> Self {
        Nnue { l1: WEIGHTS.l1_b }
    }
}

impl Default for Nnue {
    fn default() -> Self {
        Nnue::new()
    }
}

/// NNUE indices changed by a move.
///
/// This struct contains the accumulator indices that get turned on and turned off during a move.
/// Moves fall into these categories:
///
/// - quiet move: one off, one on
/// - capture: two off, one on
/// - promotion: one off, two on
/// - castle: two off, two on
#[derive(Clone, Debug)]
pub(crate) struct NnueDelta {
    on: ArrayVec<2, u16>,
    off: ArrayVec<2, u16>,
}

impl NnueDelta {
    fn new() -> Self {
        NnueDelta {
            on: ArrayVec::new(),
            off: ArrayVec::new(),
        }
    }
}

impl Default for NnueDelta {
    fn default() -> Self {
        Self::new()
    }
}

/// History stack for a board to lazily evaluate the NNUE.
///
/// This stores successive accumulator states, as well as the indices changed between these states.
/// When making moves, only the indices are changed. Only when the NNUE output is evaluated will the
/// accumulator be updated to the correct state.
#[derive(Debug, Clone)]
pub(crate) struct NnueHistory {
    /// Accumulator states.
    ///
    /// The first accumulator should correspond to the root position we search from.
    /// This is heap allocated because it's too big to fit on the stack.
    pub(crate) accumulators: Vec<Nnue>,

    /// Indices changed after each move.
    ///
    /// For an index `i` here, that represents a move made on board `i` to get to board `i+1`.
    pub(crate) deltas: ArrayVec<MAX_PLY, NnueDelta>,

    /// When making a move, construct a delta in this field.
    scratch_delta: Option<NnueDelta>,
}

impl NnueHistory {
    pub fn new() -> Self {
        NnueHistory {
            accumulators: Vec::with_capacity(MAX_PLY),
            deltas: ArrayVec::new(),
            scratch_delta: None,
        }
    }

    /// Discard all state.
    pub fn clear(&mut self) {
        self.deltas.clear();
        self.accumulators.clear();
        self.scratch_delta = None;
    }

    /// Regenerate the current accumulator state and evaluate it.
    pub fn output(&mut self) -> EvalInt {
        while self.accumulators.len() <= self.deltas.len() {
            let acc_idx = self.accumulators.len() - 1;

            self.accumulators
                .push(*self.accumulators.last().expect("missing root accumulator"));
            let new_acc = self.accumulators.last_mut().unwrap();

            let delta = &self.deltas[acc_idx];
            for off_idx in delta.off.iter() {
                new_acc.bit_set((*off_idx).into(), false)
            }
            for on_idx in delta.on.iter() {
                new_acc.bit_set((*on_idx).into(), true)
            }
        }

        let cur_acc = self.accumulators.last().unwrap();
        cur_acc.output()
    }

    /// Create a scratch delta.
    ///
    /// Call this at the start of makemove.
    pub fn start_delta(&mut self) {
        self.scratch_delta = Some(NnueDelta::new());
    }

    /// Commit the current scratch delta.
    ///
    /// Call this at the end of makemove.
    ///
    /// ***Panics*** if there is no scratch delta.
    pub fn commit_delta(&mut self) {
        self.deltas.push(
            self.scratch_delta
                .take()
                .expect("Tried to commit a None delta"),
        );
    }

    /// Unmakes a single ply.
    pub fn unmake(&mut self) {
        self.deltas.pop();
        self.accumulators.truncate(self.deltas.len() + 1);
    }

    /// Add a piece to the scratch delta.
    pub fn add_piece(&mut self, pc: ColPiece, sq: Square) {
        if let Some(scratch_delta) = self.scratch_delta.as_mut() {
            scratch_delta.on.push(InputTensor::idx(pc, sq) as u16);
        }
    }

    /// Add a piece to the scratch delta.
    pub fn del_piece(&mut self, pc: ColPiece, sq: Square) {
        if let Some(scratch_delta) = self.scratch_delta.as_mut() {
            scratch_delta.off.push(InputTensor::idx(pc, sq) as u16);
        }
    }
}

impl Default for NnueHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_to_binary_tensor() {
        // more of a sanity check than a test
        let board = Board::from_fen("8/8/8/8/8/8/8/1b6 w - - 0 1").unwrap();
        let tensor = InputTensor::from_board(&board);
        let mut expected = [false; INP_TENSOR_SIZE];
        expected[INP_TENSOR_SIZE / N_COLORS + 1 + N_SQUARES] = true;
        assert_eq!(tensor.0, expected);
    }

    /// Test that weights loaded properly, and our inference works.
    #[test]
    fn test_weight_loading() {
        let mut nnue = Nnue::new();
        for i in 0..INP_TENSOR_SIZE {
            nnue.bit_set(i, true);
        }

        let epsilon = 300;

        let got = nnue.output();
        let expected = EvalInt::from(WEIGHTS._sanity_check[0]);

        assert!(
            (got - expected).abs() < epsilon,
            "NNUE state:\n{:?}\n\ngot {:?}, expected {:?}",
            nnue,
            got,
            expected
        )
    }

    /// Test that lazy updating NNUE state works well.
    #[test]
    fn lazy_eval() {
        let tcs = [
            ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1"),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1c1"),
            (crate::fen::START_POSITION, "a2a3"),
        ];
        for (fen, mvs) in tcs {
            let mut board = Board::from_fen(fen).unwrap();
            let mvs = mvs
                .split_whitespace()
                .map(|x| Move::from_uci_algebraic(x).unwrap());
            let mut anti_mvs = Vec::new();
            for mv in mvs {
                anti_mvs.push(mv.make(&mut board));
            }
            let eval1 = board.eval();
            board.refresh_nnue();
            assert_eq!(eval1, board.eval(), "failed {}", board.to_fen());
            board.refresh_nnue();
            assert_eq!(eval1, board.eval(), "failed {}", board.to_fen());
        }
    }
}
