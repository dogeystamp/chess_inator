//! Rust port by dogeystamp <dogeystamp@disroot.org> of
//! the pcg64 dxsm random number generator (https://dotat.at/@/2023-06-21-pcg64-dxsm.html)

pub struct Pcg64Random {
    state: u128,
    inc: u128,
}

/// Make an RNG state "sane".
const fn pcg64_seed(mut rng: Pcg64Random) -> Pcg64Random {
    // ensure rng.inc is odd
    rng.inc = (rng.inc << 1) | 1;
    rng.state += rng.inc;
    // one iteration of random
    rng.rand();
    rng
}

impl Pcg64Random {
    pub const fn new(seed: u128) -> Self {
        pcg64_seed(Pcg64Random {
            // chosen by fair dice roll
            state: 24437033748623976104561743679864923857,
            inc: seed,
        })
    }

    /// Returns a single random number.
    pub const fn rand(&mut self) -> u64 {
        const MUL: u64 = 15750249268501108917;

        let state: u128 = self.state;
        self.state = state.wrapping_mul(MUL as u128).wrapping_add(self.inc);
        let mut hi: u64 = (state >> 64) as u64;
        let lo: u64 = (state | 1) as u64;
        hi ^= hi >> 32;
        hi &= MUL;
        hi ^= hi >> 48;
        hi = hi.wrapping_mul(lo);

        hi
    }

    /// Generate array of random numbers, based on a seed.
    ///
    /// # Returns
    ///
    /// A tuple with the random number array, and the RNG state afterwards so you can reuse it in later
    /// calls (otherwise you'll get the same result if you're using the same seed.)
    ///
    /// # Example
    ///
    ///```rust
    /// use chess_inator::random::Pcg64Random;
    ///
    /// // generate 3 random numbers
    /// const ARR: [u64; 3] = Pcg64Random::new(123456).random_arr_64();
    /// assert_eq!(ARR, [4526545874411451611, 1124465636717751929, 12699417402402334336])
    ///```
    pub const fn random_arr_64<const N: usize>(&mut self) -> [u64; N] {
        let mut ret = [0; N];
        let mut i = 0;
        while i < N {
            let num = self.rand();
            ret[i] = num;
            i += 1;
        }

        ret
    }

    /// Generate 2D array of random numbers based on a seed.
    pub const fn random_arr_2d_64<const N: usize, const M: usize>(&mut self) -> [[u64; N]; M] {
        let mut ret = [[0; N]; M];
        let mut i = 0;
        while i < M {
            ret[i] = self.random_arr_64();
            i += 1;
        }
        ret
    }
}
