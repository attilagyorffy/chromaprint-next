use crate::types::Algorithm;

const NORMAL_BITS: u32 = 3;
const MAX_NORMAL_VALUE: u32 = (1 << NORMAL_BITS) - 1; // 7

/// Compress a raw fingerprint into a binary string, then base64 encode it.
pub fn compress(data: &[u32], algorithm: Algorithm) -> String {
    profile_stage!(Compress);
    let compressed = compress_raw(data, algorithm as u8);
    crate::codec::base64::encode(&compressed)
}

/// Compress a raw fingerprint into a binary byte vector.
pub fn compress_raw(data: &[u32], algorithm: u8) -> Vec<u8> {
    let mut normal_bits: Vec<u8> = Vec::with_capacity(data.len() * 8);
    let mut exceptional_bits: Vec<u8> = Vec::with_capacity(data.len() * 2);

    if !data.is_empty() {
        process_subfingerprint(data[0], &mut normal_bits, &mut exceptional_bits);
        for i in 1..data.len() {
            process_subfingerprint(data[i] ^ data[i - 1], &mut normal_bits, &mut exceptional_bits);
        }
    }

    let size = data.len();
    let packed3_size = packed_int3_size(normal_bits.len());
    let packed5_size = packed_int5_size(exceptional_bits.len());

    let mut output = vec![0u8; 4 + packed3_size + packed5_size];
    output[0] = algorithm;
    output[1] = ((size >> 16) & 255) as u8;
    output[2] = ((size >> 8) & 255) as u8;
    output[3] = (size & 255) as u8;

    pack_int3_array(&normal_bits, &mut output[4..]);
    pack_int5_array(&exceptional_bits, &mut output[4 + packed3_size..]);

    output
}

fn process_subfingerprint(mut x: u32, normal: &mut Vec<u8>, exceptional: &mut Vec<u8>) {
    let mut bit: u32 = 1;
    let mut last_bit: u32 = 0;
    while x != 0 {
        if (x & 1) != 0 {
            let value = bit - last_bit;
            if value >= MAX_NORMAL_VALUE {
                normal.push(MAX_NORMAL_VALUE as u8);
                exceptional.push((value - MAX_NORMAL_VALUE) as u8);
            } else {
                normal.push(value as u8);
            }
            last_bit = bit;
        }
        x >>= 1;
        bit += 1;
    }
    normal.push(0);
}

fn packed_int3_size(size: usize) -> usize {
    (size * 3).div_ceil(8)
}

fn packed_int5_size(size: usize) -> usize {
    (size * 5).div_ceil(8)
}

fn pack_int3_array(src: &[u8], dest: &mut [u8]) {
    let mut si = 0;
    let mut di = 0;
    let size = src.len();

    while si + 8 <= size {
        let (s0, s1, s2, s3) = (src[si], src[si + 1], src[si + 2], src[si + 3]);
        let (s4, s5, s6, s7) = (src[si + 4], src[si + 5], src[si + 6], src[si + 7]);
        dest[di] = (s0 & 0x07) | ((s1 & 0x07) << 3) | ((s2 & 0x03) << 6);
        dest[di + 1] = ((s2 & 0x04) >> 2) | ((s3 & 0x07) << 1) | ((s4 & 0x07) << 4) | ((s5 & 0x01) << 7);
        dest[di + 2] = ((s5 & 0x06) >> 1) | ((s6 & 0x07) << 2) | ((s7 & 0x07) << 5);
        si += 8;
        di += 3;
    }

    let remaining = size - si;
    if remaining >= 1 { dest[di] = src[si] & 0x07;
    if remaining >= 2 { dest[di] |= (src[si+1] & 0x07) << 3;
    if remaining >= 3 { dest[di] |= (src[si+2] & 0x03) << 6; di += 1; dest[di] = (src[si+2] & 0x04) >> 2;
    if remaining >= 4 { dest[di] |= (src[si+3] & 0x07) << 1;
    if remaining >= 5 { dest[di] |= (src[si+4] & 0x07) << 4;
    if remaining >= 6 { dest[di] |= (src[si+5] & 0x01) << 7; di += 1; dest[di] = (src[si+5] & 0x06) >> 1;
    if remaining >= 7 { dest[di] |= (src[si+6] & 0x07) << 2;
    }}}}}}}
}

fn pack_int5_array(src: &[u8], dest: &mut [u8]) {
    let mut si = 0;
    let mut di = 0;
    let size = src.len();

    while si + 8 <= size {
        let (s0, s1, s2, s3) = (src[si], src[si + 1], src[si + 2], src[si + 3]);
        let (s4, s5, s6, s7) = (src[si + 4], src[si + 5], src[si + 6], src[si + 7]);
        dest[di] = (s0 & 0x1f) | ((s1 & 0x07) << 5);
        dest[di + 1] = ((s1 & 0x18) >> 3) | ((s2 & 0x1f) << 2) | ((s3 & 0x01) << 7);
        dest[di + 2] = ((s3 & 0x1e) >> 1) | ((s4 & 0x0f) << 4);
        dest[di + 3] = ((s4 & 0x10) >> 4) | ((s5 & 0x1f) << 1) | ((s6 & 0x03) << 6);
        dest[di + 4] = ((s6 & 0x1c) >> 2) | ((s7 & 0x1f) << 3);
        si += 8;
        di += 5;
    }

    let remaining = size - si;
    if remaining >= 1 { dest[di] = src[si] & 0x1f;
    if remaining >= 2 { dest[di] |= (src[si+1] & 0x07) << 5; di += 1; dest[di] = (src[si+1] & 0x18) >> 3;
    if remaining >= 3 { dest[di] |= (src[si+2] & 0x1f) << 2;
    if remaining >= 4 { dest[di] |= (src[si+3] & 0x01) << 7; di += 1; dest[di] = (src[si+3] & 0x1e) >> 1;
    if remaining >= 5 { dest[di] |= (src[si+4] & 0x0f) << 4; di += 1; dest[di] = (src[si+4] & 0x10) >> 4;
    if remaining >= 6 { dest[di] |= (src[si+5] & 0x1f) << 1;
    if remaining >= 7 { dest[di] |= (src[si+6] & 0x03) << 6; di += 1; dest[di] = (src[si+6] & 0x1c) >> 2;
    }}}}}}}
}
