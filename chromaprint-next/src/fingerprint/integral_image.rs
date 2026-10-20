/// Rolling 2D integral image (summed area table) with circular row buffer.
/// Allows O(1) computation of rectangular region sums.
pub struct RollingIntegralImage {
    data: Vec<f64>,
    max_rows: usize,
    num_columns: usize,
    num_rows: usize,
}

impl RollingIntegralImage {
    pub fn new(max_rows: usize) -> Self {
        Self {
            data: Vec::new(),
            max_rows: max_rows + 1,
            num_columns: 0,
            num_rows: 0,
        }
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn reset(&mut self) {
        self.data.clear();
        self.num_rows = 0;
        self.num_columns = 0;
    }

    /// Add a new row of values to the integral image.
    pub fn add_row(&mut self, row: &[f64]) {
        profile_stage!(IntegralImage);
        let size = row.len();
        if self.num_columns == 0 {
            self.num_columns = size;
            self.data.resize(self.max_rows * self.num_columns, 0.0);
        }
        debug_assert_eq!(self.num_columns, size);

        let current_start = self.row_offset(self.num_rows);

        // Compute partial sums of the new row
        let mut running_sum: f64 = 0.0;
        for (&v, slot) in row
            .iter()
            .zip(self.data[current_start..current_start + self.num_columns].iter_mut())
        {
            running_sum += v;
            *slot = running_sum;
        }

        // Add previous row's cumulative sums
        if self.num_rows > 0 {
            let prev_start = self.row_offset(self.num_rows - 1);
            #[allow(clippy::needless_range_loop)] // reads and writes two row offsets of self.data
            for i in 0..self.num_columns {
                self.data[current_start + i] += self.data[prev_start + i];
            }
        }

        self.num_rows += 1;
    }

    /// Compute the sum of a rectangular region.
    /// Region is [r1, r2) x [c1, c2) in the original (non-cumulated) data.
    #[inline]
    pub fn area(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> f64 {
        if r1 == r2 || c1 == c2 {
            return 0.0;
        }

        if r1 == 0 {
            let row2 = self.row_offset(r2 - 1);
            if c1 == 0 {
                self.data[row2 + c2 - 1]
            } else {
                self.data[row2 + c2 - 1] - self.data[row2 + c1 - 1]
            }
        } else {
            let row1 = self.row_offset(r1 - 1);
            let row2 = self.row_offset(r2 - 1);
            if c1 == 0 {
                self.data[row2 + c2 - 1] - self.data[row1 + c2 - 1]
            } else {
                self.data[row2 + c2 - 1] - self.data[row1 + c2 - 1]
                    - self.data[row2 + c1 - 1] + self.data[row1 + c1 - 1]
            }
        }
    }

    #[inline(always)]
    fn row_offset(&self, i: usize) -> usize {
        (i % self.max_rows) * self.num_columns
    }
}
