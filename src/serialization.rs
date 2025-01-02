/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2025 dogeystamp <dogeystamp@disroot.org>
*/

//! Serialization utilities for NNUE parameters.

use std::mem::size_of;

/// Helper to read bytes at compile-time.
pub(crate) struct ConstCursor<'a, const BUF_SIZE: usize> {
    /// Buffer to read from.
    buf: &'a [u8; BUF_SIZE],
    /// Cursor in the buffer where the next byte will be read.
    loc: usize,
}

macro_rules! read_type {
    ($type: ty, $fn_name: ident, $fn2_name: ident) => {
        /// Deserialize data from the buffer.
        ///
        /// Returns an array of `N` elements.
        pub const fn $fn_name<const N: usize>(&mut self) -> [$type; N] {
            const SZ: usize = size_of::<$type>();
            let default = <$type>::from_le_bytes([0; SZ]);
            let mut out_buf: [$type; N] = [default; N];
            let mut i = 0;
            while i < N {
                let elem_buf: [u8; SZ] = self.read_u8::<{SZ}>();
                out_buf[i] = <$type>::from_le_bytes(elem_buf);
                i += 1;
            }
            out_buf
        }

        /// Deserialize 2D array of data from the buffer.
        ///
        /// Returns an array of `M` arrays of `N` elements.
        pub const fn $fn2_name<const N: usize, const M: usize>(&mut self) -> [[$type; N]; M] {
            const SZ: usize = size_of::<$type>();
            let default = <$type>::from_le_bytes([0; SZ]);
            let mut out_buf: [[$type; N]; M] = [[default; N]; M];
            let mut i = 0;
            while i < M {
                out_buf[i] = self.$fn_name();
                i += 1;
            }
            out_buf
        }
    }
}

impl<'a, const BUF_SIZE: usize> ConstCursor<'a, BUF_SIZE> {
    /// Fill an array of bytes by reading from the buffer.
    pub const fn read_u8<const N: usize>(&mut self) -> [u8; N] {
        let mut out_buf: [u8; N] = [0; N];
        let mut i = 0;
        while i < N {
            if i >= BUF_SIZE {
                break;
            }
            out_buf[i] = self.buf[self.loc];
            i += 1;
            self.loc += 1;
        }
        if i != N {
            panic!("Could not fill buffer; ran out of bytes to read.");
        }

        out_buf
    }

    read_type!(crate::nnue::Float, read_float, read2d_float);

    pub const fn from_bytes(buf: &'a[u8; BUF_SIZE]) -> Self {
        Self {
            buf,
            loc: 0,
        }
    }
}

