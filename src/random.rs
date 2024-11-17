//! Rust port by dogeystamp <dogeystamp@disroot.org> of
//! the pcg64 dxsm random number generator (https://dotat.at/@/2023-06-21-pcg64-dxsm.html)

struct Pcg64Random {
    state: u128,
    inc: u128,
}

/// Generates an array of random numbers.
///
/// The `rng` parameter only sets the initial state. This function is deterministic and pure.
///
/// # Returns
///
/// The array of random numbers, plus the RNG state at the end.
const fn pcg64_dxsm<const N: usize>(mut rng: Pcg64Random) -> ([u64; N], Pcg64Random) {
    let mut ret = [0; N];

    const MUL: u64 = 15750249268501108917;

    let mut i = 0;
    while i < N {
        let state: u128 = rng.state;
        rng.state = state.wrapping_mul(MUL as u128).wrapping_add(rng.inc);
        let mut hi: u64 = (state >> 64) as u64;
        let lo: u64 = (state | 1) as u64;
        hi ^= hi >> 32;
        hi &= MUL;
        hi ^= hi >> 48;
        hi = hi.wrapping_mul(lo);
        ret[i] = hi;

        i += 1;
    }

    (ret, rng)
}

/// Make an RNG state "sane".
const fn pcg64_seed(mut rng: Pcg64Random) -> Pcg64Random {
    // ensure rng.inc is odd
    rng.inc = (rng.inc << 1) | 1;
    rng.state += rng.inc;
    // one iteration of random
    let (_, rng) = pcg64_dxsm::<1>(rng);
    rng
}

/// Generate array of random numbers, based on a seed.
///
/// This function is pure and deterministic, and also works at compile-time rather than at runtime.
///
/// Example (generate 10 random numbers):
///
///```rust
/// use crate::random::random_arr_64;
/// const ARR: [u64; 10] = random_arr_64(123456);
///```
pub const fn random_arr_64<const N: usize>(seed: u128) -> [u64; N] {
    let rng = pcg64_seed(Pcg64Random {
        // chosen by fair dice roll
        state: 24437033748623976104561743679864923857,
        inc: seed,
    });
    pcg64_dxsm(rng).0
}

/// Generate 2D array of random numbers based on a seed.
pub const fn random_arr_2d_64<const N: usize, const M: usize>(seed: u128) -> [[u64; N]; M] {
    let mut ret = [[0; N]; M];
    let mut i = 0;
    while i < M {
        ret[i] = random_arr_64(seed);
        i += 1;
    }
    ret
}
