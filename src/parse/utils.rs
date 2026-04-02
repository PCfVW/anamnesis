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
pub(crate) fn byteswap_inplace(data: &mut [u8], element_size: usize) {
    for chunk in data.chunks_exact_mut(element_size) {
        chunk.reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byteswap_2byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        byteswap_inplace(&mut data, 2);
        assert_eq!(data, vec![0x02, 0x01, 0x04, 0x03]);
    }

    #[test]
    fn byteswap_4byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        byteswap_inplace(&mut data, 4);
        assert_eq!(data, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn byteswap_8byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        byteswap_inplace(&mut data, 8);
        assert_eq!(data, vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn byteswap_1byte_is_noop() {
        let mut data = vec![0xAA, 0xBB, 0xCC];
        byteswap_inplace(&mut data, 1);
        assert_eq!(data, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn byteswap_empty() {
        let mut data: Vec<u8> = vec![];
        byteswap_inplace(&mut data, 4);
        assert!(data.is_empty());
    }
}
