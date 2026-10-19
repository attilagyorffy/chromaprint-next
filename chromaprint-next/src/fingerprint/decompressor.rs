use crate::error::{Error, Result};
use crate::types::Algorithm;

const MAX_NORMAL_VALUE: u8 = 7;

/// Decompress a base64-encoded fingerprint string into raw sub-fingerprints.
pub fn decompress(encoded: &str) -> Result<(Vec<u32>, Algorithm)> {
    let data = crate::codec::base64::decode(encoded)?;
    decompress_raw(&data)
}

/// Decompress raw binary fingerprint data into sub-fingerprints.
pub fn decompress_raw(input: &[u8]) -> Result<(Vec<u32>, Algorithm)> {
    if input.len() < 4 {
        return Err(Error::InvalidFingerprint);
    }

    let algorithm = Algorithm::from_u8(input[0]).ok_or(Error::UnknownAlgorithm(input[0]))?;
    let size = ((input[1] as usize) << 16)
             | ((input[2] as usize) << 8)
             | (input[3] as usize);

    if size == 0 {
        return Ok((Vec::new(), algorithm));
    }

    let offset = 4;
    let mut bits = unpack_int3_array(&input[offset..]);

    // Find the actual end and count exceptional bits
    let mut found_values = 0usize;
    let mut num_exceptional = 0usize;
    let mut actual_len = bits.len();
    for (i, &b) in bits.iter().enumerate() {
        if b == 0 {
            found_values += 1;
            if found_values == size {
                actual_len = i + 1;
                break;
            }
        } else if b == MAX_NORMAL_VALUE {
            num_exceptional += 1;
        }
    }

    if found_values != size {
        return Err(Error::InvalidFingerprint);
    }
    bits.truncate(actual_len);

    // Unpack exceptional bits (mirrors the C++ reference: error when the
    // input does not contain enough exceptional bits).
    let packed3_size = (actual_len * 3).div_ceil(8);
    let exceptional_offset = offset + packed3_size;
    let packed5_size = (num_exceptional * 5).div_ceil(8);
    if input.len() < exceptional_offset + packed5_size {
        return Err(Error::InvalidFingerprint);
    }
    if num_exceptional > 0 {
        let exceptional = unpack_int5_array(&input[exceptional_offset..]);
        let mut j = 0;
        for b in bits.iter_mut() {
            if *b == MAX_NORMAL_VALUE {
                let extra = *exceptional.get(j).ok_or(Error::InvalidFingerprint)?;
                *b += extra;
                j += 1;
            }
        }
    }

    // Reconstruct fingerprint from bit deltas
    let mut output = vec![0u32; size];
    let mut i = 0;
    let mut last_bit: u32 = 0;
    let mut value: u32 = 0;
    for &bit in &bits {
        if bit == 0 {
            output[i] = value;
            last_bit = 0;
            i += 1;
        } else {
            last_bit += bit as u32;
            if last_bit > 32 {
                return Err(Error::InvalidFingerprint);
            }
            value ^= 1u32 << (last_bit - 1);
        }
    }

    Ok((output, algorithm))
}

fn unpack_int3_array(input: &[u8]) -> Vec<u8> {
    let size = input.len();
    let output_size = size * 8 / 3;
    let mut out = Vec::with_capacity(output_size);
    let mut si = 0;

    while si + 3 <= size {
        let s0 = input[si];
        let s1 = input[si + 1];
        let s2 = input[si + 2];
        out.push(s0 & 0x07);
        out.push((s0 & 0x38) >> 3);
        out.push(((s0 & 0xc0) >> 6) | ((s1 & 0x01) << 2));
        out.push((s1 & 0x0e) >> 1);
        out.push((s1 & 0x70) >> 4);
        out.push(((s1 & 0x80) >> 7) | ((s2 & 0x03) << 1));
        out.push((s2 & 0x1c) >> 2);
        out.push((s2 & 0xe0) >> 5);
        si += 3;
    }

    let remaining = size - si;
    if remaining == 2 {
        let s0 = input[si];
        let s1 = input[si + 1];
        out.push(s0 & 0x07);
        out.push((s0 & 0x38) >> 3);
        out.push(((s0 & 0xc0) >> 6) | ((s1 & 0x01) << 2));
        out.push((s1 & 0x0e) >> 1);
        out.push((s1 & 0x70) >> 4);
    } else if remaining == 1 {
        let s0 = input[si];
        out.push(s0 & 0x07);
        out.push((s0 & 0x38) >> 3);
    }

    out
}

fn unpack_int5_array(input: &[u8]) -> Vec<u8> {
    let size = input.len();
    let output_size = size * 8 / 5;
    let mut out = Vec::with_capacity(output_size);
    let mut si = 0;

    while si + 5 <= size {
        let s0 = input[si];
        let s1 = input[si + 1];
        let s2 = input[si + 2];
        let s3 = input[si + 3];
        let s4 = input[si + 4];
        out.push(s0 & 0x1f);
        out.push(((s0 & 0xe0) >> 5) | ((s1 & 0x03) << 3));
        out.push((s1 & 0x7c) >> 2);
        out.push(((s1 & 0x80) >> 7) | ((s2 & 0x0f) << 1));
        out.push(((s2 & 0xf0) >> 4) | ((s3 & 0x01) << 4));
        out.push((s3 & 0x3e) >> 1);
        out.push(((s3 & 0xc0) >> 6) | ((s4 & 0x07) << 2));
        out.push((s4 & 0xf8) >> 3);
        si += 5;
    }

    let remaining = size - si;
    if remaining >= 1 {
        out.push(input[si] & 0x1f);
    }
    if remaining >= 2 {
        out.push(((input[si] & 0xe0) >> 5) | ((input[si + 1] & 0x03) << 3));
        out.push((input[si + 1] & 0x7c) >> 2);
    }
    if remaining >= 3 {
        out.push(((input[si + 1] & 0x80) >> 7) | ((input[si + 2] & 0x0f) << 1));
        out.push(((input[si + 2] & 0xf0) >> 4) | (if remaining >= 4 { (input[si + 3] & 0x01) << 4 } else { 0 }));
    }
    if remaining >= 4 {
        out.push((input[si + 3] & 0x3e) >> 1);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::compressor::compress_raw;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn roundtrip(data: &[u32], algo: u8) {
        let packed = compress_raw(data, algo);
        let (out, a) = decompress_raw(&packed).expect("valid data must decompress");
        assert_eq!(out, data);
        assert_eq!(a as u8, algo);
    }

    #[test]
    fn roundtrip_basic() {
        roundtrip(&[], 1);
        roundtrip(&[0], 1);
        roundtrip(&[0, 0, 0], 0);
        roundtrip(&[1], 1);
        roundtrip(&[0xDEAD_BEEF], 1);
        roundtrip(&[0x8000_0000], 2);
        roundtrip(&[u32::MAX, 0, u32::MAX, 0x8000_0001, 1, 0x8000_0000], 4);
        roundtrip(&[0x0000_0100, 0x0100_0000, 0x8000_0000, 0x0000_0080], 1);
    }

    #[test]
    fn roundtrip_random() {
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for len in [1usize, 2, 3, 7, 8, 9, 50, 300] {
            let data: Vec<u32> = (0..len)
                .map(|_| match rng.next() % 4 {
                    0 => rng.next() as u32,
                    1 => (rng.next() as u32) & (rng.next() as u32) & (rng.next() as u32),
                    2 => 1u32 << (rng.next() % 32),
                    _ => 0,
                })
                .collect();
            roundtrip(&data, 1);
        }
    }

    #[test]
    fn malformed_inputs_error() {
        assert!(decompress_raw(&[]).is_err());
        assert!(decompress_raw(&[1, 0, 0]).is_err());
        // unknown algorithm
        assert!(matches!(
            decompress_raw(&[99, 0, 0, 0]),
            Err(Error::UnknownAlgorithm(99))
        ));
        // claims 5 values but has no data
        assert!(decompress_raw(&[1, 0, 0, 5]).is_err());
        // all 0xFF: every normal value is 7 (exceptional) with no terminator
        assert!(decompress_raw(&[1, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF]).is_err());
        assert!(decompress_raw(&[0xFF; 64]).is_err());
        assert!(decompress_raw(&[1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]).is_err());
        // truncated valid fingerprints
        let packed = compress_raw(&[0xDEAD_BEEF, 0x1234_5678, 0x8000_0000], 1);
        for cut in 0..packed.len() {
            assert!(decompress_raw(&packed[..cut]).is_err(), "cut={cut}");
        }
    }

    #[test]
    fn accumulated_last_bit_overflow_errors() {
        // One value; normal bits 7,7,7,0 with exceptional extras 31 -> last_bit > 32.
        let normal = [7u8, 7, 7, 0];
        let mut out = vec![1u8, 0, 0, 1];
        let mut packed3 = vec![0u8; (normal.len() * 3).div_ceil(8)];
        for (i, &v) in normal.iter().enumerate() {
            for k in 0..3 {
                if ((v >> k) & 1) != 0 {
                    let bit = i * 3 + k;
                    packed3[bit / 8] |= 1 << (bit % 8);
                }
            }
        }
        out.extend_from_slice(&packed3);
        let mut packed5 = vec![0u8; (3 * 5usize).div_ceil(8)];
        for i in 0..3 {
            for k in 0..5 {
                let bit = i * 5 + k;
                packed5[bit / 8] |= 1 << (bit % 8);
            }
        }
        out.extend_from_slice(&packed5);
        assert!(matches!(
            decompress_raw(&out),
            Err(Error::InvalidFingerprint)
        ));
    }

    #[test]
    fn short_exceptional_section_errors() {
        // Two values; normal bits 7,7,7,0,7,7,0 -> 5 exceptional bits, which
        // need (5 * 5).div_ceil(8) = 4 bytes. Supply only 3: unpack_int5_array
        // still yields 5 values from them, so only the length check (as in the
        // C++ reference) rejects this input.
        let normal = [7u8, 7, 7, 0, 7, 7, 0];
        let mut out = vec![1u8, 0, 0, 2];
        let mut packed3 = vec![0u8; (normal.len() * 3).div_ceil(8)];
        for (i, &v) in normal.iter().enumerate() {
            for k in 0..3 {
                if ((v >> k) & 1) != 0 {
                    let bit = i * 3 + k;
                    packed3[bit / 8] |= 1 << (bit % 8);
                }
            }
        }
        out.extend_from_slice(&packed3);
        out.extend_from_slice(&[0u8; 3]);
        assert!(matches!(
            decompress_raw(&out),
            Err(Error::InvalidFingerprint)
        ));
        // With the 4th byte present the same input decodes.
        out.push(0);
        assert!(decompress_raw(&out).is_ok());
    }

    #[test]
    fn fuzz_never_panics() {
        let mut rng = Rng(0xFEED_FACE_1234_5678);
        for iter in 0..20000 {
            let len = (rng.next() % 40) as usize;
            let mut v: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            if iter % 2 == 0 && v.len() >= 4 {
                v[0] = (rng.next() % 5) as u8;
                v[1] = 0;
                v[2] = 0;
                v[3] = (rng.next() % 6) as u8;
            }
            let _ = decompress_raw(&v);
        }
        // mutate valid data
        let packed = compress_raw(&[0xDEAD_BEEF, 0x1234_5678, 0x8000_0000, 0xFFFF_FFFF], 1);
        for _ in 0..5000 {
            let mut v = packed.clone();
            let n = 1 + (rng.next() % 3) as usize;
            for _ in 0..n {
                let i = (rng.next() as usize) % v.len();
                v[i] = rng.next() as u8;
            }
            let _ = decompress_raw(&v);
        }
    }
}
