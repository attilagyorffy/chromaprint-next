/// Compute SimHash from a raw fingerprint array.
/// Produces a single 32-bit locality-sensitive hash.
pub fn simhash(data: &[u32]) -> u32 {
    if data.is_empty() {
        return 0;
    }

    let mut counts = [0u32; 32];
    for &word in data {
        for (j, c) in counts.iter_mut().enumerate() {
            *c += (word >> j) & 1;
        }
    }

    let threshold = (data.len() as u32) / 2;
    let mut hash: u32 = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c > threshold {
            hash |= 1 << i;
        }
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::simhash;

    #[test]
    fn empty_is_zero() {
        assert_eq!(simhash(&[]), 0);
    }

    #[test]
    fn single_value_is_itself() {
        assert_eq!(simhash(&[0xDEADBEEF]), 0xDEADBEEF);
        assert_eq!(simhash(&[0]), 0);
    }

    #[test]
    fn majority_vote_per_bit() {
        // bit0: 3/3 set, bit1: 2/3 set, bit2: 1/3 set, bit3: 0/3 set
        let data = [0b0111, 0b0011, 0b0001];
        assert_eq!(simhash(&data), 0b0011);
    }

    #[test]
    fn tie_is_not_set() {
        // threshold = len/2 = 1; count must be strictly greater.
        assert_eq!(simhash(&[0b1, 0b0]), 0);
    }
}
