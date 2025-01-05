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
use crate::serialization::ConstCursor;
use std::fmt::Display;

// alias to easily change precision / data type
pub(crate) type Param = i16;

/// Network architecture string. Reject any weights file that does not fulfill this.
const ARCHITECTURE: &[u8] = "A07_CReLU_768_16_1_K_q<i2\x1b".as_bytes();

/// Size of the input feature tensor.
pub const INP_TENSOR_SIZE: usize = N_COLORS * N_PIECES * N_SQUARES;
/// Size of the hidden layer.
const L1_SIZE: usize = 16;
/// Size of the output layer.
const OUT_SIZE: usize = 1;

/// Quantization scaling factor (params already scaled; we need to dequantize here)
const L1_SCALE: Param = 255;
/// Quantization scaling factor (params already scaled; we need to dequantize here)
const OUT_SCALE: Param = 64;

/// Expected size of the weights binary.
///
/// - Size of all parameters
/// - Size of the ARCHITECTURE string (plus ESC byte)
const BIN_SIZE: usize = std::mem::size_of::<NNUEParameters>() + ARCHITECTURE.len();

/// All weights and biases of the neural network.
#[derive(Debug)]
struct NNUEParameters {
    _sanity_check: [Param; 1],
    k: [Param; 1],
    l1_w: [[Param; INP_TENSOR_SIZE]; L1_SIZE],
    l1_b: [Param; L1_SIZE],
    out_w: [[Param; L1_SIZE]; OUT_SIZE],
    out_b: [Param; OUT_SIZE],
}

/// Parameters, in packed binary form.
///
/// This line may have a mismatched type error if the weights are incompatible, either because of
/// their format, or the version string being different.
const WEIGHTS_BIN: &[u8; BIN_SIZE] = include_bytes!("weights.bin");

impl NNUEParameters {
    const fn from_bytes(bytes: &[u8; BIN_SIZE]) -> Self {
        let mut cursor = ConstCursor::from_bytes(bytes);
        let arch_string: [u8; ARCHITECTURE.len()] = cursor.read_u8();
        let mut i = 0;
        while i < arch_string.len() {
            if arch_string[i] != ARCHITECTURE[i] {
                panic!("Incompatible weights for this version of the engine.")
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

const WEIGHTS: NNUEParameters = NNUEParameters::from_bytes(WEIGHTS_BIN);

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

impl PartialEq for Nnue {
    /// Neural net shouldn't affect board equality, so set always equal
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for Nnue {}

pub(crate) fn crelu(x: Param) -> Param {
    x.clamp(0, L1_SCALE)
}

impl Nnue {
    /// Turn on/off a bit in the input tensor.
    pub fn bit_set(&mut self, i: usize, on: bool) {
        debug_assert!(i < INP_TENSOR_SIZE);
        for j in 0..L1_SIZE {
            if on {
                self.l1[j] += WEIGHTS.l1_w[j][i];
            } else {
                self.l1[j] -= WEIGHTS.l1_w[j][i];
            }
        }
    }

    pub fn add_piece(&mut self, pc: ColPiece, sq: Square) {
        self.bit_set(InputTensor::idx(pc, sq), true);
    }

    pub fn del_piece(&mut self, pc: ColPiece, sq: Square) {
        self.bit_set(InputTensor::idx(pc, sq), false);
    }

    /// Logits from neural net, which should correspond to centipawns.
    pub fn output(&self) -> EvalInt {
        // activations
        let mut z_l1: [Param; L1_SIZE] = [0; L1_SIZE];
        for (j, z) in z_l1.iter_mut().enumerate() {
            *z = crelu(self.l1[j])
        }

        let mut out: [EvalInt; OUT_SIZE] = [0; OUT_SIZE];

        for (k, out_node) in out.iter_mut().enumerate() {
            *out_node = EvalInt::from(WEIGHTS.out_b[k]);
            for (j, z) in z_l1.iter().enumerate() {
                *out_node += EvalInt::from(WEIGHTS.out_w[k][j]) * EvalInt::from(*z);
            }
        }

        // scaling factor
        out[0] *= EvalInt::from(WEIGHTS.k[0]);

        // dequantization step
        out[0] /= EvalInt::from(L1_SCALE * OUT_SCALE);

        out[0]
    }

    pub fn new() -> Self {
        Nnue {
            l1: WEIGHTS.l1_b,
        }
    }
}

impl Default for Nnue {
    fn default() -> Self {
        Nnue::new()
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

        let epsilon = 50;

        let got = nnue.output();
        let expected = EvalInt::from(WEIGHTS._sanity_check[0]);

        assert!((got - expected).abs() < epsilon, "NNUE state:\n{:?}\n\ngot {:?}, expected {:?}", nnue, got, expected)
    }
}
