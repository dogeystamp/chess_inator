/*

This file is part of chess_inator.

chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.

Copyright © 2025 dogeystamp <dogeystamp@disroot.org>
*/

use std::mem::MaybeUninit;
use std::ops::Index;
use std::ops::IndexMut;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// Attempted to add more elements to the vector than its capacity.
    CapacityExceeded,
}

/// Vector-like interface, backed by a fixed-size array.
///
/// # Invariants
///
/// 1. All elements in the range `0..len` are initialized.
/// 2. `len` is in the range `0..=N`.
#[derive(Debug)]
pub struct ArrayVec<const N: usize, T: Sized> {
    /// The underlying array.
    data: [MaybeUninit<T>; N],
    /// The length of the vector (i.e. number of elements stored.)
    len: usize,
}

impl<const N: usize, T: Sized> ArrayVec<N, T> {
    /// Create an empty array.
    pub fn new() -> Self {
        ArrayVec {
            data: [const { MaybeUninit::uninit() }; N],
            len: 0,
        }
    }

    /// Get the maximal capacity of this vector (i.e. the size of the backing array).
    pub fn capacity(&self) -> usize {
        N
    }

    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            Some(&self[self.len - 1])
        }
    }

    /// If true, can't push any more elements to this vector.
    pub fn is_full(&self) -> bool {
        self.len() == N
    }

    /// Get the number of elements currently in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is this vector empty, i.e. has no elements?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sets the length (number of elements) in the vector.
    ///
    /// # Safety
    ///
    /// Take care when using this function to uphold Invariant 1.
    unsafe fn set_len(&mut self, x: usize) {
        self.len = x;
    }

    /// Remove the last element of the vector and return it, if there is one.
    ///
    /// See: [`Vec::pop`].
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            // SAFETY: we checked len != 0, and by invariant 1, all elements in the new array will
            // be initialized too
            unsafe { self.set_len(self.len - 1) }
            // SAFETY: this element is outside the array now, so it will no longer be read until we
            // write a new one to this slot. thus, `assume_init_read` will not cause duplication.
            unsafe { Some(self.data[self.len].assume_init_read()) }
        }
    }

    /// Keep only the first `n` elements in the vector, and discard all the others.
    pub fn truncate(&mut self, n: usize) {
        if n >= self.len {
            return;
        }

        let old_len = self.len;

        // SAFETY:
        // 1. all elements from 0..len are initialized, so reducing len doesn't change that
        // 2. reducing len will keep `len` in the range 0..N as long as `n` >= 0
        unsafe { self.set_len(n) }

        for i in self.len..old_len {
            // SAFETY:
            // by invariant 1. all these elements were initialized
            unsafe { self.data[i].assume_init_drop() }
        }
    }

    /// Remove all elements from the vector.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Pushes one element to the end of the vector.
    ///
    /// Returns an error if the capacity is exceeded.
    pub fn try_push(&mut self, x: T) -> Result<(), Error> {
        if self.len == N {
            return Err(Error::CapacityExceeded);
        }

        self.push(x);

        Ok(())
    }

    /// Appends one element to the end of the vector.
    ///
    /// ***Panics*** if the capacity is exceeded, but only through a debug assertion.
    pub fn push(&mut self, x: T) {
        debug_assert!(self.len < N, "ArrayVec capacity (N = {}) exceeded.", N);
        self.data[self.len].write(x);

        // SAFETY: we just wrote into this slot
        unsafe { self.set_len(self.len + 1) }
    }

    /// Keep only the elements specified by a boolean predicate. Operates in place.
    ///
    /// See [`Vec::retain`].
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let mut retained = 0;

        for i in 0..self.len {
            if f(&self[i]) {
                // retain element

                // SAFETY: Given `i` is in `0..len`, values are initialized, by Invariant 1.
                // No duplicated data is in the final array, since:
                // - each element in the ArrayVec is processed exactly once, where it is either
                //   discarded or retained.
                // - retained elements are written to consecutive indices in `0..retained`.
                // - there are only retained elements in the range `0..retained`.
                // - the array is later truncated to `retained`.
                // Therefore, because the final array contains only retained elements, and elements
                // can not be retained multiple times, this function does not duplicate data.
                unsafe {
                    let val = self.data[i].assume_init_read();
                    self.data[retained].write(val);
                }
                retained += 1
            } else {
                // discard element
            }
        }

        // SAFETY: every element from `0..retained` is written, since `retained` starts at 0, and
        // every increment of `retained` is accompanied by a write.
        unsafe { self.set_len(retained) }
    }

    /// Swap elements at two indices.
    ///
    /// See [`Vec::swap`](Vec#method.swap).
    ///
    /// # Panics
    ///
    /// If indices are out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        let pa = &raw mut self[a];
        let pb = &raw mut self[b];
        // SAFETY: `a` and `b` are initialized, since our implementation of Index/IndexMut does
        // bound checking.
        unsafe {
            std::ptr::swap(pa, pb);
        }
    }

    /// Immutable reference iterator over this vector.
    pub fn iter(&self) -> ArrayVecIter<'_, N, T> {
        ArrayVecIter { arr: self, i: 0 }
    }
}

impl<const N: usize, T: Sized + Ord> ArrayVec<N, T> {
    /// Sort in-place.
    ///
    /// "Unstable" does not guarantee the order of equal elements.
    pub fn sort_unstable(&mut self) {
        self.sort_unstable_by(|a, b| a.cmp(b))
    }

    /// Sort in-place using a comparison function.
    ///
    /// "Unstable" does not guarantee the order of equal elements.
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        if self.len == 0 {
            return;
        }

        for i in 0..(self.len - 1) {
            let mut min_idx = i;
            for j in (i + 1)..(self.len) {
                if matches!(compare(&self[j], &self[min_idx]), std::cmp::Ordering::Less) {
                    min_idx = j;
                }
            }
            if min_idx != i {
                self.swap(i, min_idx);
            }
        }
    }

    /// Sort in-place using a function to extract the value of each element.
    ///
    /// "Unstable" does not guarantee the order of equal elements.
    pub fn sort_unstable_by_key<F, K>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_unstable_by(|a, b| f(a).cmp(&f(b)))
    }
}

impl<const N: usize, T: Sized> Default for ArrayVec<N, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, T: Sized + Clone> Clone for ArrayVec<N, T> {
    fn clone(&self) -> Self {
        let mut new = ArrayVec::<N, T>::new();
        for elem in self.iter() {
            new.push(elem.clone());
        }

        new
    }
}

impl<const N: usize, T: Sized> Drop for ArrayVec<N, T> {
    fn drop(&mut self) {
        for i in 0..self.len {
            // SAFETY: Invariant 1 of `ArrayVec`
            unsafe { self.data[i].assume_init_drop() }
        }
    }
}

impl<const N: usize, T: Sized> Index<usize> for ArrayVec<N, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            (0..self.len).contains(&index),
            "Out of range index of ArrayVec"
        );
        // SAFETY: invariant 1 of `ArrayVec`, and above assert
        unsafe { self.data[index].assume_init_ref() }
    }
}

impl<const N: usize, T: Sized> IndexMut<usize> for ArrayVec<N, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            (0..self.len).contains(&index),
            "Out of range index of ArrayVec"
        );
        // SAFETY: invariant 1 of `ArrayVec`, and above assert
        unsafe { self.data[index].assume_init_mut() }
    }
}

impl<const N: usize, T: Sized> IntoIterator for ArrayVec<N, T> {
    type Item = T;

    type IntoIter = ArrayVecIntoIter<N, T>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayVecIntoIter { arr: self, i: 0 }
    }
}

pub struct ArrayVecIntoIter<const N: usize, T: Sized> {
    arr: ArrayVec<N, T>,
    i: usize,
}

impl<const N: usize, T: Sized> Iterator for ArrayVecIntoIter<N, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if (0..self.arr.len).contains(&self.i) {
            // SAFETY:
            // - Invariant 1 ensures the data is initialized.
            // - The iterator will never read the data that has been iterated over, preventing
            //   duplication through `assume_init_read()`.
            unsafe {
                let ret = Some(self.arr.data[self.i].assume_init_read());
                self.i += 1;
                ret
            }
        } else {
            None
        }
    }
}

pub struct ArrayVecIter<'a, const N: usize, T: Sized> {
    arr: &'a ArrayVec<N, T>,
    i: usize,
}

impl<'a, const N: usize, T> Iterator for ArrayVecIter<'a, N, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if (0..self.arr.len).contains(&self.i) {
            let ret = Some(&self.arr[self.i]);
            self.i += 1;
            ret
        } else {
            None
        }
    }
}

impl<const N: usize, A: Sized> FromIterator<A> for ArrayVec<N, A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut ret = ArrayVec::<N, A>::new();
        for item in iter.into_iter() {
            ret.push(item);
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrayvec() {
        let mut v = ArrayVec::<5, u8>::new();
        assert_eq!(v.len(), 0);
        v.push(1);
        assert_eq!(v[0], 1);
        assert_eq!(v.len(), 1);
        let res = v.try_push(2);
        assert_eq!(v.len(), 2);
        assert!(res.is_ok());
        let popped = v.pop();
        assert_eq!(popped, Some(2));
        assert_eq!(v.len(), 1);
        let popped = v.pop();
        assert_eq!(popped, Some(1));
        assert_eq!(v.len(), 0);
        let popped = v.pop();
        assert_eq!(popped, None);
        assert_eq!(v.len(), 0);
        for i in 0..v.capacity() {
            v.push(i as u8);
        }
        assert_eq!(v.len(), v.capacity());
        let res = v.try_push(2);
        assert_eq!(res, Err(Error::CapacityExceeded));
        assert_eq!(v.len(), v.capacity());
        let res = v.try_push(2);
        assert_eq!(res, Err(Error::CapacityExceeded));
        assert_eq!(v.len(), v.capacity());

        for i in (0..v.capacity()).rev() {
            let popped = v.pop();
            assert_eq!(popped, Some(i as u8));
        }
        let popped = v.pop();
        assert_eq!(popped, None);

        for i in 0..v.capacity() {
            v.push(i as u8);
        }
        let expected = (0..(v.capacity() as u8)).into_iter().collect::<Vec<_>>();
        let contents = v.into_iter().collect::<Vec<_>>();
        assert_eq!(contents, expected);
    }

    #[test]
    fn test_swap() {
        let mut av = ArrayVec::<5, u8>::new();
        let mut v: Vec<u8> = vec![4, 2, 3, 0, 1];
        for &n in v.iter() {
            av.push(n);
        }

        for i in [0, 3, 4, 1, 2] {
            for j in [4, 3, 1, 2, 0] {
                av.swap(i, j);
                v.swap(i, j);
            }
        }

        let res = av.into_iter().collect::<Vec<_>>();
        assert_eq!(res, v);
    }

    #[test]
    fn test_sort() {
        let test_cases = [
            vec![4, 12, 5, 56, 2, 34, 123, 4, 32456, 36, 24, 6],
            vec![3, 3, 3, 3, 3, 3, 3, 3, 3],
            vec![],
            vec![3],
            vec![5, 4, 564, 5, 2134, 12, 34, 234, 23, 4, 236],
            vec![0],
            vec![2, 3, 4, 5, 6, 7, 8, 9],
            vec![1, 2, 3, 4],
            vec![0, 1, 2, 3, 4],
            vec![4, 3, 2, 1, 0],
        ];
        for tc in test_cases {
            eprintln!("tc: {:?}", tc);
            let mut av = ArrayVec::<512, i32>::new();
            let mut v: Vec<i32> = tc;
            for &n in v.iter() {
                av.push(n);
            }

            av.sort_unstable();
            v.sort_unstable();
            let res = av.iter().cloned().collect::<Vec<_>>();
            assert_eq!(res, v, "failed sort unstable");

            use std::cmp::Ordering;
            fn gt(a: &i32, b: &i32) -> Ordering {
                match a.cmp(b) {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Equal => Ordering::Equal,
                    Ordering::Greater => Ordering::Less,
                }
            }

            av.sort_unstable_by(gt);
            v.sort_unstable_by(gt);
            let res = av.iter().cloned().collect::<Vec<_>>();
            assert_eq!(res, v, "failed sort unstable by");

            av.sort_unstable_by_key(|x| -x);
            v.sort_unstable_by_key(|x| -x);
            let res = av.iter().cloned().collect::<Vec<_>>();
            assert_eq!(res, v, "failed sort unstable by key");
        }
    }

    #[test]
    fn test_retain() {
        let test_cases = [
            vec![0, 1, 2, 0],
            vec![0, 0, 0, 0],
            vec![],
            vec![1, 2, 3, 0],
            vec![0, 1, 2, 3],
            vec![4, 3, 2, 1],
            vec![19, 0, 2, 1],
            vec![0, 0, 0, 0, 0, 12],
        ];

        for tc in test_cases {
            eprintln!("tc: {:?}", tc);
            let mut av = ArrayVec::<512, usize>::new();
            let mut v: Vec<usize> = tc;
            for &n in v.iter() {
                av.push(n);
            }

            v.retain(|x| *x != 0);
            av.retain(|x| *x != 0);

            assert_eq!(av.len(), v.len());
            let res = av.into_iter().collect::<Vec<_>>();
            assert_eq!(res, v);
        }
    }

    #[test]
    fn test_from_iter() {
        let test_cases = [
            vec![0, 1, 2, 0],
            vec![0, 0, 0, 0],
            vec![],
            vec![1, 2, 3, 0],
            vec![0, 1, 2, 3],
            vec![4, 3, 2, 1],
            vec![19, 0, 2, 1],
            vec![0, 0, 0, 0, 0, 12],
        ];

        for tc in test_cases {
            eprintln!("tc: {:?}", tc);
            let av = tc.clone().into_iter().collect::<ArrayVec<512, _>>();
            assert_eq!(av.len(), tc.len());
            assert_eq!(av.into_iter().collect::<Vec<_>>(), tc);
        }
    }
}
