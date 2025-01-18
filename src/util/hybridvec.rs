/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2025 dogeystamp <dogeystamp@disroot.org>
*/

use crate::util::arrayvec::{ArrayVec, ArrayVecIntoIter};

/// Vector-like structure that stores on either stack or heap.
///
/// The stack is preferred, and when the space is exhausted, the vector is transferred to heap.
pub struct HybridVec<const N: usize, T: Sized> {
    data: ArrayVec<N, T>,
    heap: Vec<T>,
    is_heap: bool,
}

/// Directly pass through a method to the underlying vector.
macro_rules! pass_through {
    ($(#[$attr:meta])* => $fn_name: ident() -> $ret: ty) => {
        $(#[$attr])*
        pub fn $fn_name(&self) -> $ret {
            if self.is_heap {
                self.heap.$fn_name()
            } else {
                self.data.$fn_name()
            }
        }
    };
    ($(#[$attr:meta])* => mut $fn_name: ident() -> $ret: ty) => {
        $(#[$attr])*
        pub fn $fn_name(&mut self) -> $ret {
            if self.is_heap {
                self.heap.$fn_name()
            } else {
                self.data.$fn_name()
            }
        }
    }
}

impl<const N: usize, T: Sized + Clone> HybridVec<N, T> {
    /// Create a new empty array.
    pub fn new() -> Self {
        Self {
            data: ArrayVec::<N, T>::new(),
            heap: Vec::new(),
            is_heap: false,
        }
    }

    /// If true, this vector is heap-allocated. Otherwise, it is on the stack.
    pub fn is_heap(&self) -> bool {
        self.is_heap
    }

    pub fn push(&mut self, x: T) {
        if !self.is_heap() {
            if self.data.is_full() {
                // move to heap
                self.is_heap = true;
                self.heap.reserve(self.data.capacity() + 1);
                for elem in self.data.iter().cloned() {
                    self.heap.push(elem);
                }
                self.heap.push(x);
            } else {
                self.data.push(x);
            }
        } else {
            self.heap.push(x);
        }
    }

    pass_through!(
        /// Get the number of elements this vector can hold without reallocating.
        => capacity() -> usize
    );

    pass_through!(
        /// Get the number of elements currently in the vector.
        => len() -> usize
    );

    pass_through!(
        /// Get the number of elements currently in the vector.
        => is_empty() -> bool
    );

    pass_through!(
        /// Remove the last element of the vector and return it, if there is one.
        => mut pop() -> Option<T>
    );

    /// Keep only the elements specified by a boolean predicate. Operates in place.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        if self.is_heap() {
            self.heap.retain(f)
        } else {
            self.data.retain(f)
        }
    }

    /// Swap elements at two indices.
    ///
    /// # Panics
    ///
    /// If indices are out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        if self.is_heap() {
            self.heap.swap(a, b)
        } else {
            self.data.swap(a, b)
        }
    }
}

impl<const N: usize, T: Sized + Ord> HybridVec<N, T> {
    /// Sort the vector in place.
    pub fn sort(&mut self) {
        if self.is_heap {
            self.heap.sort()
        } else {
            self.data.selection_sort()
        }
    }
}

/// Iterator type for [`HybridVec`].
pub enum HybridIntoIter<const N: usize, T: Sized> {
    Stack(ArrayVecIntoIter<N, T>),
    Heap(<std::vec::Vec<T> as std::iter::IntoIterator>::IntoIter),
}

impl<const N: usize, T: Sized> Iterator for HybridIntoIter<N, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            HybridIntoIter::Stack(avi) => avi.next(),
            HybridIntoIter::Heap(vi) => vi.next(),
        }
    }
}

impl<const N: usize, T: Sized> IntoIterator for HybridVec<N, T> {
    type Item = T;

    type IntoIter = HybridIntoIter<N, T>;

    fn into_iter(self) -> Self::IntoIter {
        if self.is_heap {
            HybridIntoIter::Heap(self.heap.into_iter())
        } else {
            HybridIntoIter::Stack(self.data.into_iter())
        }
    }
}

impl<const N: usize, T: Sized + Clone> Default for HybridVec<N, T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_vec() {
        let mut hv = HybridVec::<5, usize>::new();

        assert!(!hv.is_heap());

        for i in 0..5 {
            hv.push(i)
        }
        assert!(!hv.is_heap());
        for i in 0..5 {
            hv.push(i)
        }
        assert!(hv.is_heap());

        for i in (0..5).rev() {
            assert_eq!(hv.pop(), Some(i))
        }

        let res = hv.into_iter().collect::<Vec<_>>();
        assert_eq!(res, (0..5).into_iter().collect::<Vec<_>>())
    }
}
