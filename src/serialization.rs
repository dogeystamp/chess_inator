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
pub(crate) struct ConstCursor<'a> {
    /// Buffer to read from.
    buf: &'a [u8],
    /// Cursor in the buffer where the next byte will be read.
    loc: usize,
}

macro_rules! read_type {
    ($type: ty, $fn0_name: ident, $fn1_name: ident, $fn2_name: ident) => {
        /// Deserialize a single element from the buffer.
        #[allow(dead_code)]
        pub const fn $fn0_name(&mut self) -> $type {
            const SZ: usize = size_of::<$type>();
            let elem_buf: [u8; SZ] = self.read_u8::<{ SZ }>();
            <$type>::from_le_bytes(elem_buf)
        }

        /// Deserialize data from the buffer.
        ///
        /// Returns an array of `N` elements.
        #[allow(dead_code)]
        pub const fn $fn1_name<const N: usize>(&mut self) -> [$type; N] {
            const SZ: usize = size_of::<$type>();
            let default = <$type>::from_le_bytes([0; SZ]);
            let mut out_buf: [$type; N] = [default; N];
            let mut i = 0;
            while i < N {
                out_buf[i] = self.$fn0_name();
                i += 1;
            }
            out_buf
        }

        /// Deserialize 2D array of data from the buffer.
        ///
        /// Returns an array of `M` arrays of `N` elements.
        #[allow(dead_code)]
        pub const fn $fn2_name<const N: usize, const M: usize>(&mut self) -> [[$type; N]; M] {
            const SZ: usize = size_of::<$type>();
            let default = <$type>::from_le_bytes([0; SZ]);
            let mut out_buf: [[$type; N]; M] = [[default; N]; M];
            let mut i = 0;
            while i < M {
                out_buf[i] = self.$fn1_name();
                i += 1;
            }
            out_buf
        }
    };
}

impl<'a> ConstCursor<'a> {
    /// Fill an array of bytes by reading from the buffer.
    pub const fn read_u8<const N: usize>(&mut self) -> [u8; N] {
        let mut out_buf: [u8; N] = [0; N];
        let mut i = 0;
        while i < N {
            out_buf[i] = self.read_single_u8();
            i += 1;
        }
        if i != N {
            panic!("Could not fill buffer; ran out of bytes to read.");
        }

        out_buf
    }

    /// Read a single byte from the buffer.
    pub const fn read_single_u8(&mut self) -> u8 {
        let ret = self.buf[self.loc];
        self.loc += 1;
        ret
    }

    read_type!(crate::nnue::Param, read_single, read, read2d);
    read_type!(u16, read_single_u16, read_u16, read2d_u16);

    /// Initialize a cursor over a buffer, starting the cursor at the index `whence`.
    pub const fn from_bytes(buf: &'a [u8], whence: usize) -> Self {
        Self { buf, loc: whence }
    }
}
