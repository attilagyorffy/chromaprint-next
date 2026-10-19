use std::cell::OnceCell;

/// Number of chroma bands (12 semitones per octave).
pub const NUM_BANDS: usize = 12;

/// A 12-element chroma feature vector.
/// Uses f64 to match the C implementation which uses `double` throughout.
pub type ChromaVector = [f64; NUM_BANDS];

/// Algorithm version selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Algorithm {
    Test1 = 0,
    #[default]
    Test2 = 1,
    Test3 = 2,
    Test4 = 3,
    Test5 = 4,
}

impl Algorithm {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Test1),
            1 => Some(Self::Test2),
            2 => Some(Self::Test3),
            3 => Some(Self::Test4),
            4 => Some(Self::Test5),
            _ => None,
        }
    }
}

/// Result of a fingerprinting operation.
pub struct Fingerprint {
    /// Raw sub-fingerprints (32-bit values).
    pub raw: Vec<u32>,
    /// Algorithm used to generate this fingerprint.
    pub algorithm: Algorithm,
    /// Lazily computed compressed+base64 encoded string.
    encoded: OnceCell<String>,
    /// Lazily computed SimHash.
    hash: OnceCell<u32>,
}

impl Fingerprint {
    pub fn new(raw: Vec<u32>, algorithm: Algorithm) -> Self {
        Self {
            raw,
            algorithm,
            encoded: OnceCell::new(),
            hash: OnceCell::new(),
        }
    }

    /// Get the compressed base64-encoded fingerprint string.
    /// Computed lazily on first access.
    pub fn encoded(&self) -> &str {
        self.encoded.get_or_init(|| {
            crate::fingerprint::compressor::compress(&self.raw, self.algorithm)
        })
    }

    /// Get the SimHash of the fingerprint.
    /// Computed lazily on first access.
    pub fn hash(&self) -> u32 {
        *self.hash.get_or_init(|| {
            crate::fingerprint::simhash::simhash(&self.raw)
        })
    }
}
