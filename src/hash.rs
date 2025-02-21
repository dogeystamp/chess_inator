/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2024 dogeystamp <dogeystamp@disroot.org>
*/

//! Zobrist hash implementation.

use crate::util::random::Pcg64Random;
use crate::{
    Board, CastleRights, ColPiece, Color, Square, BOARD_WIDTH, N_COLORS, N_PIECES, N_SQUARES,
};
use std::ops::Index;
use std::ops::IndexMut;

const PIECE_KEYS: [[[u64; N_SQUARES]; N_PIECES]; N_COLORS] = [
    Pcg64Random::new(11).random_arr_2d_64(),
    Pcg64Random::new(22).random_arr_2d_64(),
];

// 4 bits in castle perms -> 16 keys
const CASTLE_KEYS: [u64; 16] = Pcg64Random::new(33).random_arr_64();

// ep can be specified by the file
const EP_KEYS: [u64; BOARD_WIDTH] = Pcg64Random::new(44).random_arr_64();

// current turn
const COL_KEY: [u64; N_COLORS] = Pcg64Random::new(55).random_arr_64();

/// Zobrist hash state.
///
/// This is not synced to board state, so ensure that all changes made are reflected in the hash
/// too.
#[derive(PartialEq, Eq, Clone, Copy, Default, Debug)]
pub(crate) struct Zobrist {
    hash: u64,
}

impl Zobrist {
    /// Toggle a piece.
    pub(crate) fn toggle_pc(&mut self, pc: &ColPiece, sq: &Square) {
        let key = PIECE_KEYS[pc.col as usize][pc.pc as usize][usize::from(sq.0)];
        self.hash ^= key;
    }

    /// Toggle an en-passant target square (only square file is used).
    pub(crate) fn toggle_ep(&mut self, sq: Option<Square>) {
        if let Some(sq) = sq {
            let (_r, c) = sq.to_row_col();
            self.hash ^= EP_KEYS[c];
        }
    }

    /// Toggle castle rights key.
    pub(crate) fn toggle_castle(&mut self, castle: &CastleRights) {
        let bits = (castle.0[0].k as u8)
            | (castle.0[0].q as u8) << 1
            | (castle.0[1].k as u8) << 2
            | (castle.0[1].q as u8) << 3;

        self.hash ^= CASTLE_KEYS[bits as usize];
    }

    /// Toggle player to move.
    pub(crate) fn toggle_turn(&mut self, turn: Color) {
        self.hash ^= COL_KEY[turn as usize];
    }

    /// Toggle all of castling rights, en passant and player to move.
    ///
    /// This is done because it's simpler to do this every time at the start and end of a
    /// move/unmove rather than keep track of when castling and ep square and whatever rights
    /// change. Piece moves, unlike this information, have a centralized implementation.
    pub(crate) fn toggle_board_info(pos: &mut Board) {
        pos.zobrist.toggle_ep(pos.ep_square);
        pos.zobrist.toggle_castle(&pos.castle);
        pos.zobrist.toggle_turn(pos.turn);
    }

    /// Convert hash to an index.
    fn truncate_hash(&self, size: usize) -> usize {
        self.hash as usize % size
    }
}

/// Map that takes Zobrist hashes as keys.
///
/// Heap allocated (it's a vector).
#[derive(Debug)]
pub struct ZobristTable<T> {
    data: Vec<(Zobrist, Option<T>)>,
    size: usize,
}

/// Convert a transposition table size in mebibytes to a number of entries.
pub fn mib_to_n<T: Sized>(mib: usize) -> usize {
    let bytes = mib * (1 << 20);
    let entry_size = std::mem::size_of::<(Zobrist, Option<T>)>();

    bytes / entry_size
}

impl<T: Copy> ZobristTable<T> {
    /// Create a Zobrist-keyed table.
    pub fn new(size_mib: usize) -> Self {
        assert!(
            size_mib <= 12000,
            "Attempted to make {size_mib} MiB hash table; aborting to avoid excessive memory usage."
        );
        let size = mib_to_n::<T>(size_mib);
        Self::new_n(size)
    }

    /// Create a table with n entries.
    pub fn new_n(size: usize) -> Self {
        ZobristTable {
            data: vec![(Zobrist { hash: 0 }, None); size],
            size,
        }
    }

    /// Create a table with 2^n entries.
    pub fn new_pow2(size_exp: usize) -> Self {
        ZobristTable {
            data: vec![(Zobrist { hash: 0 }, None); 1 << size_exp],
            size: size_exp,
        }
    }
}

impl<T> IndexMut<Zobrist> for ZobristTable<T> {
    /// Overwrite a table entry (always replace strategy).
    ///
    /// If you `mut`ably index, it will automatically wipe an existing entry,
    /// regardless of it was a cache hit or miss.
    fn index_mut(&mut self, zobrist: Zobrist) -> &mut Self::Output {
        let idx = zobrist.truncate_hash(self.size);
        self.data[idx].0 = zobrist;
        self.data[idx].1 = None;
        &mut self.data[idx].1
    }
}

impl<T> Index<Zobrist> for ZobristTable<T> {
    type Output = Option<T>;

    fn index(&self, zobrist: Zobrist) -> &Self::Output {
        let idx = zobrist.truncate_hash(self.size);
        let data = &self.data[idx];
        if data.0 == zobrist {
            &data.1
        } else {
            // miss
            &None
        }
    }
}

pub trait TableReplacement {
    /// Do we replace `other`?
    fn replaces(&self, other: &Self) -> bool;
}

impl<T: TableReplacement> ZobristTable<T> {
    /// Attempt to save an entry to the Zobrist table.
    ///
    /// If there is an existing entry (due to a hash collision), `Ord` comparison will be used, and
    /// the "biggest" (most important) entry is kept.
    ///
    /// For an "always replace" replacement scheme, try using the `IndexMut` interface to save
    /// entries to the table.
    pub(crate) fn save_entry(&mut self, zobrist: Zobrist, entry: T) {
        let idx = zobrist.truncate_hash(self.size);
        let existing_data = &self.data[idx];

        let mut overwrite = false;

        if let Some(existing_entry) = &existing_data.1 {
            if entry.replaces(existing_entry) {
                overwrite = true;
            }
        } else {
            overwrite = true;
        }

        if overwrite {
            self.data[idx].0 = zobrist;
            self.data[idx].1 = Some(entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::FromFen;
    use crate::movegen::{FromUCIAlgebraic, Move};

    /// Zobrist hashes, and transposition table elements of the same positions should be the same. (basic sanity test)
    #[test]
    fn test_zobrist_equality() {
        let test_cases = [
            (
                "4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1",
                "4k2r/8/8/8/8/8/8/2KR4 b k - 1 1",
                "e1c1",
            ),
            (
                "4k2r/8/8/8/8/8/8/R3K3 b Qk - 0 1",
                "5rk1/8/8/8/8/8/8/R3K3 w Q - 1 2",
                "e8g8",
            ),
            (
                "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1",
                "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                "e2e4",
            ),
            (
                "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                "4k3/8/8/8/8/4p3/8/4K3 w - - 0 2",
                "d4e3",
            ),
        ];
        for (pos1_fen, pos2_fen, mv_uci) in test_cases {
            eprintln!("tc: {}", pos1_fen);
            let mut table = ZobristTable::<usize>::new(4);
            let mut pos1 = Board::from_fen(pos1_fen).unwrap();
            let hash1_orig = pos1.zobrist;
            assert_eq!(table[pos1.zobrist], None);
            table[pos1.zobrist] = Some(100);
            eprintln!("refreshing board 2 '{}'", pos2_fen);
            let pos2 = Board::from_fen(pos2_fen).unwrap();
            table[pos2.zobrist] = Some(200);
            eprintln!("making mv {}", mv_uci);
            let mv = Move::from_uci_algebraic(mv_uci).unwrap();
            let anti_mv = mv.make(&mut pos1);
            assert_eq!(pos1.zobrist, pos2.zobrist);
            assert_eq!(table[pos1.zobrist], Some(200));
            anti_mv.unmake(&mut pos1);
            assert_eq!(pos1.zobrist, hash1_orig);
            assert_eq!(table[pos1.zobrist], Some(100));
        }
    }

    // test that positions are equal when they loop back to the start
    #[test]
    fn test_zobrist_loops() {
        let test_cases = [
            (
                "4k3/4r3/8/8/8/8/3R4/3K4 w - - 0 1",
                "d2f2 e7f7 f2d2 f7e7",
            ),
            (
                "4k3/4r3/8/8/8/8/3R4/3K4 w - - 0 1",
                "d2f2 e7f7 f2d2 f7e7 d2f2 e7f7 f2d2 f7e7 d2f2 e7f7 f2d2 f7e7 d2f2 e7f7 f2d2 f7e7 d2f2 e7f7 f2d2 f7e7 d2f2 e7f7 f2d2 f7e7",
            ),
        ];

        for (fen, mvs_str) in test_cases {
            let pos_orig = Board::from_fen(fen).unwrap();
            let mut pos = pos_orig.clone();
            for mv_str in mvs_str.split_whitespace() {
                let mv = Move::from_uci_algebraic(mv_str).unwrap();
                mv.make(&mut pos);
            }
            pos.irreversible_half = pos_orig.irreversible_half;
            pos.plies = pos_orig.plies;
            assert_eq!(
                pos, pos_orig,
                "test case is incorrect, position should loop back to the original"
            );
            assert_eq!(pos.zobrist, pos_orig.zobrist);
        }
    }

    #[test]
    fn test_table() {
        let mut table = ZobristTable::<usize>::new_pow2(4);

        macro_rules! z {
            ($i: expr) => {
                Zobrist { hash: $i }
            };
        }

        let big_number = 1 << 62;

        table[z!(big_number + 3)] = Some(4);
        table[z!(big_number + 19)] = Some(5);

        // clobbered by newer entry
        assert_eq!(table[z!(big_number + 3)], None);

        assert_eq!(table[z!(big_number + 19)], Some(5));

        eprintln!("{table:?}");
    }

    impl crate::hash::TableReplacement for usize {
        fn replaces(&self, other: &Self) -> bool {
            self >= other
        }
    }

    #[test]
    fn test_replacement() {
        let mut table = ZobristTable::<usize>::new_pow2(4);

        macro_rules! z {
            ($i: expr) => {
                Zobrist { hash: $i }
            };
        }

        let big_number = 1 << 62;

        table.save_entry(z!(big_number + 19), 5);
        table.save_entry(z!(big_number + 3), 4);

        // newer entry is less important, should not clobber
        assert_eq!(table[z!(big_number + 19)], Some(5));

        // should clobber now
        table.save_entry(z!(big_number + 3), 6);
        assert_eq!(table[z!(big_number + 3)], Some(6));
        assert_eq!(table[z!(big_number + 19)], None);

        eprintln!("{table:?}");
    }
}
