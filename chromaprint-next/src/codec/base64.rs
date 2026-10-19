use crate::error::{Error, Result};

const ENCODE_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

#[rustfmt::skip]
static DECODE_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    table[b'A' as usize] = 0;  table[b'B' as usize] = 1;  table[b'C' as usize] = 2;
    table[b'D' as usize] = 3;  table[b'E' as usize] = 4;  table[b'F' as usize] = 5;
    table[b'G' as usize] = 6;  table[b'H' as usize] = 7;  table[b'I' as usize] = 8;
    table[b'J' as usize] = 9;  table[b'K' as usize] = 10; table[b'L' as usize] = 11;
    table[b'M' as usize] = 12; table[b'N' as usize] = 13; table[b'O' as usize] = 14;
    table[b'P' as usize] = 15; table[b'Q' as usize] = 16; table[b'R' as usize] = 17;
    table[b'S' as usize] = 18; table[b'T' as usize] = 19; table[b'U' as usize] = 20;
    table[b'V' as usize] = 21; table[b'W' as usize] = 22; table[b'X' as usize] = 23;
    table[b'Y' as usize] = 24; table[b'Z' as usize] = 25;
    table[b'a' as usize] = 26; table[b'b' as usize] = 27; table[b'c' as usize] = 28;
    table[b'd' as usize] = 29; table[b'e' as usize] = 30; table[b'f' as usize] = 31;
    table[b'g' as usize] = 32; table[b'h' as usize] = 33; table[b'i' as usize] = 34;
    table[b'j' as usize] = 35; table[b'k' as usize] = 36; table[b'l' as usize] = 37;
    table[b'm' as usize] = 38; table[b'n' as usize] = 39; table[b'o' as usize] = 40;
    table[b'p' as usize] = 41; table[b'q' as usize] = 42; table[b'r' as usize] = 43;
    table[b's' as usize] = 44; table[b't' as usize] = 45; table[b'u' as usize] = 46;
    table[b'v' as usize] = 47; table[b'w' as usize] = 48; table[b'x' as usize] = 49;
    table[b'y' as usize] = 50; table[b'z' as usize] = 51;
    table[b'0' as usize] = 52; table[b'1' as usize] = 53; table[b'2' as usize] = 54;
    table[b'3' as usize] = 55; table[b'4' as usize] = 56; table[b'5' as usize] = 57;
    table[b'6' as usize] = 58; table[b'7' as usize] = 59; table[b'8' as usize] = 60;
    table[b'9' as usize] = 61;
    table[b'-' as usize] = 62;
    table[b'_' as usize] = 63;
    table
};

/// Encode binary data to chromaprint's URL-safe base64 (no padding).
pub fn encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() * 4).div_ceil(3));
    let mut i = 0;
    let size = data.len();

    while i + 3 <= size {
        let s0 = data[i];
        let s1 = data[i + 1];
        let s2 = data[i + 2];
        result.push(ENCODE_TABLE[((s0 >> 2) & 63) as usize] as char);
        result.push(ENCODE_TABLE[(((s0 << 4) | (s1 >> 4)) & 63) as usize] as char);
        result.push(ENCODE_TABLE[(((s1 << 2) | (s2 >> 6)) & 63) as usize] as char);
        result.push(ENCODE_TABLE[(s2 & 63) as usize] as char);
        i += 3;
    }

    let remaining = size - i;
    if remaining == 2 {
        let s0 = data[i];
        let s1 = data[i + 1];
        result.push(ENCODE_TABLE[((s0 >> 2) & 63) as usize] as char);
        result.push(ENCODE_TABLE[(((s0 << 4) | (s1 >> 4)) & 63) as usize] as char);
        result.push(ENCODE_TABLE[((s1 << 2) & 63) as usize] as char);
    } else if remaining == 1 {
        let s0 = data[i];
        result.push(ENCODE_TABLE[((s0 >> 2) & 63) as usize] as char);
        result.push(ENCODE_TABLE[((s0 << 4) & 63) as usize] as char);
    }

    result
}

/// Decode chromaprint's URL-safe base64 (no padding) to binary data.
pub fn decode(input: &str) -> Result<Vec<u8>> {
    let bytes = input.as_bytes();
    let size = bytes.len();
    let mut result = Vec::with_capacity(size * 3 / 4);
    let mut i = 0;

    while i + 4 <= size {
        let b0 = DECODE_TABLE[bytes[i] as usize];
        let b1 = DECODE_TABLE[bytes[i + 1] as usize];
        let b2 = DECODE_TABLE[bytes[i + 2] as usize];
        let b3 = DECODE_TABLE[bytes[i + 3] as usize];
        result.push((b0 << 2) | (b1 >> 4));
        result.push((b1 << 4) | (b2 >> 2));
        result.push((b2 << 6) | b3);
        i += 4;
    }

    let remaining = size - i;
    if remaining == 3 {
        let b0 = DECODE_TABLE[bytes[i] as usize];
        let b1 = DECODE_TABLE[bytes[i + 1] as usize];
        let b2 = DECODE_TABLE[bytes[i + 2] as usize];
        result.push((b0 << 2) | (b1 >> 4));
        result.push((b1 << 4) | (b2 >> 2));
    } else if remaining == 2 {
        let b0 = DECODE_TABLE[bytes[i] as usize];
        let b1 = DECODE_TABLE[bytes[i + 1] as usize];
        result.push((b0 << 2) | (b1 >> 4));
    } else if remaining == 1 {
        return Err(Error::InvalidFingerprint);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_lengths_0_to_10() {
        for len in 0..=10usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            let enc = encode(&data);
            assert_eq!(decode(&enc).unwrap(), data, "len {len}");
        }
    }

    #[test]
    fn known_vectors() {
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"f"), "Zg");
        assert_eq!(encode(b"fo"), "Zm8");
        assert_eq!(encode(b"foo"), "Zm9v");
        assert_eq!(encode(b"foob"), "Zm9vYg");
        assert_eq!(encode(&[0xFB, 0xFF]), "-_8");
        assert_eq!(decode("Zm9vYg").unwrap(), b"foob");
    }

    #[test]
    fn no_padding_emitted() {
        for len in 0..=10usize {
            assert!(!encode(&vec![0xFFu8; len]).contains('='));
        }
    }

    #[test]
    fn decode_rejects_dangling_char() {
        assert!(decode("Z").is_err());
        assert!(decode("Zm9vY").is_err());
    }
}
