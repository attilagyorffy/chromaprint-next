use crate::fingerprint::classifier::Classifier;
use crate::fingerprint::integral_image::RollingIntegralImage;
use crate::types::ChromaVector;

/// Computes 32-bit sub-fingerprints from normalized chroma vectors
/// using an array of 16 classifiers.
pub struct FingerprintCalculator {
    classifiers: &'static [Classifier],
    max_filter_width: usize,
    image: RollingIntegralImage,
    fingerprint: Vec<u32>,
}

impl FingerprintCalculator {
    pub fn new(classifiers: &'static [Classifier]) -> Self {
        let max_filter_width = classifiers
            .iter()
            .map(|c| c.filter.width as usize)
            .max()
            .unwrap_or(0);

        Self {
            classifiers,
            max_filter_width,
            image: RollingIntegralImage::new(256),
            fingerprint: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.image.reset();
        self.fingerprint.clear();
    }

    /// Consume a normalized chroma vector and potentially produce a sub-fingerprint.
    pub fn consume(&mut self, features: &ChromaVector) {
        self.image.add_row(features);

        if self.image.num_rows() < self.max_filter_width {
            return;
        }

        profile_stage!(Classify);
        let offset = self.image.num_rows() - self.max_filter_width;
        let mut bits: u32 = 0;

        for classifier in self.classifiers {
            bits = (bits << 2) | classifier.classify(&self.image, offset);
        }

        self.fingerprint.push(bits);
    }

    /// Get the accumulated fingerprint.
    pub fn fingerprint(&self) -> &[u32] {
        &self.fingerprint
    }
}
