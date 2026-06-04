// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared parsing utilities used by multiple format parsers (`NPZ`, `.pth`).

/// Reverses the byte order of each element in `data` in-place.
///
/// Each contiguous `element_size`-byte chunk is reversed, converting
/// big-endian to little-endian (or vice versa). Elements that are not
/// an exact multiple of `element_size` are left untouched (handled by
/// `chunks_exact_mut`).
// VECTORIZED: scalar fallback — chunk.reverse() on a runtime-variable
// element_size prevents auto-vectorization. This is the big-endian path
// (<0.01% of ML files), so scalar performance is acceptable.
// EXPLICIT: in-place mutation on the read buffer avoids allocating a second
// buffer of equal size. CONVENTIONS Rule 6 (separate in/out) is waived here
// because the data is already in a dedicated Vec<u8> that serves as the output.
#[cfg(any(feature = "npz", feature = "pth"))]
pub(crate) fn byteswap_inplace(data: &mut [u8], element_size: usize) {
    for chunk in data.chunks_exact_mut(element_size) {
        chunk.reverse();
    }
}

/// Computes the product of a tensor's shape dimensions with overflow checking.
///
/// Returns `None` if the running product overflows `usize` — an adversarial
/// or malformed header declaring dimensions whose product cannot fit (e.g.
/// `[usize::MAX, 2]`). An empty shape yields `Some(1)` (the scalar
/// element-count convention).
///
/// Callers choose their own policy on `None`: a path that sizes an allocation
/// maps it to `AnamnesisError::Parse`, while a pure eligibility query treats
/// it as "not eligible". This is the
/// checked counterpart to a raw `shape.iter().product()`, which silently wraps
/// in release builds and panics in debug builds on overflow.
#[must_use]
pub(crate) fn checked_num_elements(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_num_elements_basic() {
        assert_eq!(checked_num_elements(&[]), Some(1));
        assert_eq!(checked_num_elements(&[16384, 2304]), Some(16384 * 2304));
        assert_eq!(checked_num_elements(&[0, 5]), Some(0));
    }

    #[test]
    fn checked_num_elements_overflow_is_none() {
        // Two near-`usize::MAX` dims whose product cannot fit.
        assert_eq!(checked_num_elements(&[usize::MAX, 2]), None);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_2byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        byteswap_inplace(&mut data, 2);
        assert_eq!(data, vec![0x02, 0x01, 0x04, 0x03]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_4byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        byteswap_inplace(&mut data, 4);
        assert_eq!(data, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_8byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        byteswap_inplace(&mut data, 8);
        assert_eq!(data, vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_1byte_is_noop() {
        let mut data = vec![0xAA, 0xBB, 0xCC];
        byteswap_inplace(&mut data, 1);
        assert_eq!(data, vec![0xAA, 0xBB, 0xCC]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_empty() {
        let mut data: Vec<u8> = vec![];
        byteswap_inplace(&mut data, 4);
        assert!(data.is_empty());
    }
}
