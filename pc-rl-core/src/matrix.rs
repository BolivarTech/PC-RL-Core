// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Dense matrix operations and vector utilities for neural networks.
//!
//! Provides a custom [`Matrix`] struct and free functions for softmax,
//! argmax, RMS error, categorical sampling, and element-wise vector ops.
//! Pure Rust with no external linear-algebra dependencies.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Maximum absolute value for weight clamping after updates.
pub const WEIGHT_CLIP: f64 = 5.0;

/// Maximum absolute value for gradient clamping.
pub const GRAD_CLIP: f64 = 5.0;

/// A dense row-major matrix of `f64` values.
///
/// Data is stored in a flat `Vec<f64>` of length `rows * cols`.
///
/// # Examples
///
/// ```
/// use pc_rl_core::matrix::Matrix;
///
/// let m = Matrix::zeros(2, 3);
/// assert_eq!(m.rows, 2);
/// assert_eq!(m.cols, 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix {
    /// Flat row-major storage.
    pub data: Vec<f64>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl Matrix {
    /// Creates a matrix filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    ///
    /// # Returns
    ///
    /// A `Matrix` with all elements set to `0.0`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a matrix with Xavier-uniform initialization.
    ///
    /// Elements are drawn uniformly from `[-limit, limit]` where
    /// `limit = sqrt(6.0 / (rows + cols))`.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    /// * `rng` - Mutable reference to a random number generator.
    ///
    /// # Returns
    ///
    /// A `Matrix` with Xavier-initialized values.
    pub fn xavier(rows: usize, cols: usize, rng: &mut impl Rng) -> Self {
        let limit = (6.0 / (rows + cols) as f64).sqrt();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        Self { data, rows, cols }
    }

    /// Returns the element at `(row, col)`.
    ///
    /// Defaults to `0.0` if indices are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index.
    /// * `col` - Column index.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(
            row < self.rows && col < self.cols,
            "Matrix::get out of bounds: ({row}, {col}) for ({}, {})",
            self.rows,
            self.cols
        );
        self.data[row * self.cols + col]
    }

    /// Sets the element at `(row, col)` to `val`.
    ///
    /// Does nothing if indices are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index.
    /// * `col` - Column index.
    /// * `val` - Value to set.
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        assert!(
            row < self.rows && col < self.cols,
            "Matrix::set out of bounds: ({row}, {col}) for ({}, {})",
            self.rows,
            self.cols
        );
        self.data[row * self.cols + col] = val;
    }

    /// Returns the transpose of this matrix.
    ///
    /// # Returns
    ///
    /// A new `Matrix` with rows and columns swapped.
    pub fn transpose(&self) -> Self {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.set(c, r, self.get(r, c));
            }
        }
        result
    }

    /// Multiplies this matrix by a column vector.
    ///
    /// # Arguments
    ///
    /// * `v` - Input vector of length `self.cols`.
    ///
    /// # Returns
    ///
    /// A vector of length `self.rows`.
    ///
    /// # Panics
    ///
    /// Panics with "dimension" if `v.len() != self.cols`.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(
            v.len(),
            self.cols,
            "dimension mismatch: vector length {} != matrix cols {}",
            v.len(),
            self.cols
        );
        (0..self.rows)
            .map(|r| {
                let row_start = r * self.cols;
                self.data[row_start..row_start + self.cols]
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    /// Computes the outer product of two vectors.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector (determines rows).
    /// * `b` - Second vector (determines cols).
    ///
    /// # Returns
    ///
    /// A `Matrix` of shape `(a.len(), b.len())`. Returns a 0x0 matrix
    /// if either vector is empty.
    pub fn outer(a: &[f64], b: &[f64]) -> Self {
        if a.is_empty() || b.is_empty() {
            return Matrix::zeros(0, 0);
        }
        let rows = a.len();
        let cols = b.len();
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[r * cols + c] = a[r] * b[c];
            }
        }
        Self { data, rows, cols }
    }

    /// Adds `scale * other` element-wise and clamps to `[-WEIGHT_CLIP, WEIGHT_CLIP]`.
    ///
    /// # Arguments
    ///
    /// * `other` - Matrix to add (must have same dimensions).
    /// * `scale` - Scalar multiplier for `other`.
    ///
    /// # Panics
    ///
    /// Panics if dimensions do not match.
    pub fn scale_add(&mut self, other: &Matrix, scale: f64) {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "dimension mismatch in scale_add: ({},{}) vs ({},{})",
            self.rows,
            self.cols,
            other.rows,
            other.cols
        );
        for i in 0..self.data.len() {
            self.data[i] += scale * other.data[i];
            self.data[i] = self.data[i].clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
        }
    }
}

/// Numerically stable masked softmax.
///
/// Computes softmax only over indices in `mask`. Non-mask indices are set to zero.
/// Uses max-subtraction trick for numerical stability.
///
/// # Arguments
///
/// * `logits` - Raw scores.
/// * `mask` - Indices to include in the softmax.
///
/// # Returns
///
/// A probability vector of the same length as `logits`. Empty mask returns all zeros.
pub fn softmax_masked(logits: &[f64], mask: &[usize]) -> Vec<f64> {
    let mut result = vec![0.0; logits.len()];
    if mask.is_empty() {
        return result;
    }
    assert!(
        mask.iter().all(|&i| i < logits.len()),
        "softmax_masked: mask index out of bounds (max mask={}, logits len={})",
        mask.iter().max().unwrap_or(&0),
        logits.len()
    );

    let max_val = mask
        .iter()
        .map(|&i| logits[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for &i in mask {
        let exp_val = (logits[i] - max_val).exp();
        result[i] = exp_val;
        sum += exp_val;
    }
    if sum > 0.0 {
        for &i in mask {
            result[i] /= sum;
        }
    }
    result
}

/// Returns the index of the maximum value among masked indices.
///
/// # Arguments
///
/// * `values` - Slice of values.
/// * `mask` - Indices to consider.
///
/// # Panics
///
/// Panics if `mask` is empty.
pub fn argmax_masked(values: &[f64], mask: &[usize]) -> usize {
    assert!(!mask.is_empty(), "argmax_masked: empty mask");
    assert!(
        mask.iter().all(|&i| i < values.len()),
        "argmax_masked: mask index out of bounds (max mask={}, values len={})",
        mask.iter().max().unwrap_or(&0),
        values.len()
    );
    let mut best_idx = mask[0];
    let mut best_val = values[mask[0]];
    for &i in &mask[1..] {
        if values[i] > best_val {
            best_val = values[i];
            best_idx = i;
        }
    }
    best_idx
}

/// Combined RMS error across multiple error vectors.
///
/// # Arguments
///
/// * `error_vecs` - Slice of error vector references.
///
/// # Returns
///
/// The root-mean-square of all elements. Returns `0.0` if empty.
pub fn rms_error(error_vecs: &[&[f64]]) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for v in error_vecs {
        for &e in *v {
            sum_sq += e * e;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt()
}

/// Samples an action index from a probability distribution over masked indices.
///
/// If only one action is valid, returns it directly. If all probabilities among
/// mask indices are zero, falls back to uniform sampling over the mask.
///
/// # Arguments
///
/// * `probs` - Probability vector.
/// * `mask` - Valid action indices.
/// * `rng` - Mutable reference to a random number generator.
///
/// # Panics
///
/// Panics if `mask` is empty.
pub fn sample_from_probs(probs: &[f64], mask: &[usize], rng: &mut impl Rng) -> usize {
    assert!(!mask.is_empty(), "sample_from_probs: empty mask");

    if mask.len() == 1 {
        return mask[0];
    }

    let sum: f64 = mask.iter().map(|&i| probs[i]).sum();
    if sum <= 0.0 {
        // Uniform fallback
        return mask[rng.gen_range(0..mask.len())];
    }

    let threshold: f64 = rng.gen_range(0.0..1.0);
    let mut cumulative = 0.0;
    for &i in mask {
        cumulative += probs[i] / sum;
        if cumulative >= threshold {
            return i;
        }
    }

    // Fallback to last mask element (rounding)
    *mask.last().unwrap()
}

/// Clamps each element of `v` to `[-max_abs, max_abs]` in place.
///
/// # Arguments
///
/// * `v` - Mutable slice to clamp.
/// * `max_abs` - Maximum absolute value.
pub(crate) fn clip_vec(v: &mut [f64], max_abs: f64) {
    for x in v.iter_mut() {
        *x = x.clamp(-max_abs, max_abs);
    }
}

/// Element-wise subtraction: `a - b`.
///
/// # Arguments
///
/// * `a` - First vector.
/// * `b` - Second vector.
///
/// # Returns
///
/// A new vector where each element is `a[i] - b[i]`.
pub(crate) fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "vec_sub: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Element-wise addition: `a + b`.
///
/// # Arguments
///
/// * `a` - First vector.
/// * `b` - Second vector.
///
/// # Returns
///
/// A new vector where each element is `a[i] + b[i]`.
pub(crate) fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "vec_add: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Scales every element of `v` by `s`.
///
/// # Arguments
///
/// * `v` - Input vector.
/// * `s` - Scalar multiplier.
///
/// # Returns
///
/// A new vector where each element is `v[i] * s`.
pub(crate) fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

/// CCA-based neuron alignment between two activation matrices.
///
/// Given activation matrices from parent A and parent B (rows = batch samples,
/// columns = neurons), computes a permutation that aligns B's neurons to A's
/// neurons based on functional similarity via Canonical Correlation Analysis.
///
/// # Arguments
///
/// * `act_a` - Activation matrix for parent A `[batch_size × n_a]`.
/// * `act_b` - Activation matrix for parent B `[batch_size × n_b]`.
///
/// # Returns
///
/// A permutation vector of length `min(n_a, n_b)` where `perm[i]` is the
/// index of the neuron in A that B's neuron `i` maps to.
pub fn cca_neuron_alignment<L: crate::linalg::LinAlg>(
    act_a: &L::Matrix,
    act_b: &L::Matrix,
) -> Vec<usize> {
    let batch_size = L::mat_rows(act_a);
    let n_a = L::mat_cols(act_a);
    let n_b = L::mat_cols(act_b);
    let k = n_a.min(n_b);

    if k == 0 || batch_size < 2 {
        return (0..k).collect();
    }

    // Phase 1: Standardize columns (mean=0, std=1)
    let std_a = standardize_columns::<L>(act_a);
    let std_b = standardize_columns::<L>(act_b);

    let scale = 1.0 / (batch_size as f64 - 1.0);

    // Phase 2: Compute covariance matrices
    let std_a_t = L::mat_transpose(&std_a);
    let std_b_t = L::mat_transpose(&std_b);

    let mut c_a = L::mat_mul(&std_a_t, &std_a); // n_a × n_a
    let mut c_b = L::mat_mul(&std_b_t, &std_b); // n_b × n_b
    let mut c_ab = L::mat_mul(&std_a_t, &std_b); // n_a × n_b

    // Scale by 1/(batch_size - 1)
    scale_matrix::<L>(&mut c_a, n_a, n_a, scale);
    scale_matrix::<L>(&mut c_b, n_b, n_b, scale);
    scale_matrix::<L>(&mut c_ab, n_a, n_b, scale);

    // Phase 3: Compute C_a^(-1/2) and C_b^(-1/2) via SVD
    let c_a_inv_sqrt = mat_inv_sqrt::<L>(&c_a);
    let c_b_inv_sqrt = mat_inv_sqrt::<L>(&c_b);

    // M = C_a^(-1/2) × C_ab × C_b^(-1/2)
    let temp = L::mat_mul(&c_a_inv_sqrt, &c_ab);
    let m = L::mat_mul(&temp, &c_b_inv_sqrt);

    // SVD(M) → U, S, V
    let (u, _s, v) = L::svd(&m);

    // Phase 4: Greedy matching
    greedy_match::<L>(&u, &v, n_a, n_b)
}

/// Scale all elements of a matrix by a scalar.
fn scale_matrix<L: crate::linalg::LinAlg>(m: &mut L::Matrix, rows: usize, cols: usize, s: f64) {
    for r in 0..rows {
        for c in 0..cols {
            let val = L::mat_get(m, r, c);
            L::mat_set(m, r, c, val * s);
        }
    }
}

/// Standardize columns of a matrix to mean=0, std=1.
/// Dead neurons (std < epsilon) get zeroed columns.
fn standardize_columns<L: crate::linalg::LinAlg>(m: &L::Matrix) -> L::Matrix {
    let rows = L::mat_rows(m);
    let cols = L::mat_cols(m);
    let mut result = L::zeros_mat(rows, cols);
    let eps = 1e-12;

    for c in 0..cols {
        // Compute mean
        let mut sum = 0.0;
        for r in 0..rows {
            sum += L::mat_get(m, r, c);
        }
        let mean = sum / rows as f64;

        // Compute std
        let mut var_sum = 0.0;
        for r in 0..rows {
            let diff = L::mat_get(m, r, c) - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / rows as f64).sqrt();

        if std > eps {
            for r in 0..rows {
                L::mat_set(&mut result, r, c, (L::mat_get(m, r, c) - mean) / std);
            }
        }
        // Dead neuron: column stays zero
    }
    result
}

/// Compute M^(-1/2) for a symmetric positive semi-definite matrix via SVD.
/// Eigenvalues below epsilon are treated as zero.
fn mat_inv_sqrt<L: crate::linalg::LinAlg>(m: &L::Matrix) -> L::Matrix {
    let n = L::mat_rows(m);
    let (u, s, _v) = L::svd(m);
    let eps = 1e-10;

    // Build diag(1/sqrt(s_i)) for non-zero singular values
    let k = L::vec_len(&s);
    let mut diag_inv_sqrt = L::zeros_mat(k, k);
    for i in 0..k {
        let si = L::vec_get(&s, i);
        if si > eps {
            L::mat_set(&mut diag_inv_sqrt, i, i, 1.0 / si.sqrt());
        }
    }

    // M^(-1/2) = V × diag(1/sqrt(S)) × U^T
    // For symmetric M: U ≈ V, so M^(-1/2) = U × diag(1/sqrt(S)) × U^T
    let temp = L::mat_mul(&u, &diag_inv_sqrt);
    let ut = L::mat_transpose(&u);
    let mut result = L::mat_mul(&temp, &ut);

    // Ensure result is n×n (SVD may truncate)
    if L::mat_rows(&result) != n || L::mat_cols(&result) != n {
        let mut padded = L::zeros_mat(n, n);
        let r_rows = L::mat_rows(&result);
        let r_cols = L::mat_cols(&result);
        for r in 0..r_rows.min(n) {
            for c in 0..r_cols.min(n) {
                L::mat_set(&mut padded, r, c, L::mat_get(&result, r, c));
            }
        }
        result = padded;
    }

    result
}

/// Greedy matching from CCA canonical directions.
/// Returns permutation[i] = index in A that B's neuron i maps to.
fn greedy_match<L: crate::linalg::LinAlg>(
    u: &L::Matrix,
    v: &L::Matrix,
    n_a: usize,
    n_b: usize,
) -> Vec<usize> {
    let k = n_a.min(n_b);
    let n_canonical = L::mat_cols(u).min(L::mat_cols(v));

    let mut matched_a = vec![false; n_a];
    let mut matched_b = vec![false; n_b];
    let mut perm = vec![0usize; k];
    let mut assigned = vec![false; k];

    // Match by strongest canonical correlation first
    for col in 0..n_canonical {
        // Find neuron in A with largest |u_k| coefficient
        let mut best_a = 0;
        let mut best_a_val = 0.0_f64;
        for (i, &is_matched) in matched_a.iter().enumerate().take(n_a.min(L::mat_rows(u))) {
            let val = L::mat_get(u, i, col).abs();
            if val > best_a_val && !is_matched {
                best_a_val = val;
                best_a = i;
            }
        }

        // Find neuron in B with largest |v_k| coefficient
        let mut best_b = 0;
        let mut best_b_val = 0.0_f64;
        for (i, &is_matched) in matched_b.iter().enumerate().take(n_b.min(L::mat_rows(v))) {
            let val = L::mat_get(v, i, col).abs();
            if val > best_b_val && !is_matched {
                best_b_val = val;
                best_b = i;
            }
        }

        if !matched_a[best_a] && !matched_b[best_b] && best_b < k {
            perm[best_b] = best_a;
            assigned[best_b] = true;
            matched_a[best_a] = true;
            matched_b[best_b] = true;
        }
    }

    // Assign remaining unmatched B neurons to remaining A positions
    let remaining_a: Vec<usize> = (0..n_a).filter(|i| !matched_a[*i]).collect();
    let unassigned_b: Vec<usize> = (0..k).filter(|i| !assigned[*i]).collect();
    for (idx, &b_idx) in unassigned_b.iter().enumerate() {
        if idx < remaining_a.len() {
            perm[b_idx] = remaining_a[idx];
        }
    }

    perm
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // ── Matrix Tests ──────────────────────────────────────────────────

    #[test]
    fn test_zeros_all_zero_correct_dims() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_xavier_variance_approx() {
        let mut rng = StdRng::seed_from_u64(42);
        let m = Matrix::xavier(100, 100, &mut rng);
        let n = m.data.len() as f64;
        let mean = m.data.iter().sum::<f64>() / n;
        let variance = m.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let expected_var = 2.0 / (100.0 + 100.0); // 0.01
        assert!(
            (variance - expected_var).abs() < expected_var * 0.5,
            "variance {} not within 50% of expected {}",
            variance,
            expected_var
        );
    }

    #[test]
    fn test_xavier_all_finite() {
        let mut rng = StdRng::seed_from_u64(42);
        let m = Matrix::xavier(50, 50, &mut rng);
        assert!(m.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_get_set_roundtrip() {
        let mut m = Matrix::zeros(3, 3);
        m.set(1, 2, 42.0);
        assert_eq!(m.get(1, 2), 42.0);
    }

    #[test]
    fn test_get_zero_default() {
        let m = Matrix::zeros(2, 2);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn test_transpose_swaps_dims() {
        let m = Matrix::zeros(3, 5);
        let t = m.transpose();
        assert_eq!(t.rows, 5);
        assert_eq!(t.cols, 3);
    }

    #[test]
    fn test_transpose_repositions_values() {
        let mut m = Matrix::zeros(2, 3);
        m.set(0, 1, 7.0);
        m.set(1, 2, 3.0);
        let t = m.transpose();
        assert_eq!(t.get(1, 0), 7.0);
        assert_eq!(t.get(2, 1), 3.0);
    }

    #[test]
    fn test_transpose_double_is_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let m = Matrix::xavier(3, 5, &mut rng);
        let tt = m.transpose().transpose();
        assert_eq!(m.rows, tt.rows);
        assert_eq!(m.cols, tt.cols);
        for i in 0..m.data.len() {
            assert!((m.data[i] - tt.data[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_mul_vec_known_result() {
        // [[1,2],[3,4]] * [5,6] = [17, 39]
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);
        let result = m.mul_vec(&[5.0, 6.0]);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 17.0).abs() < 1e-10);
        assert!((result[1] - 39.0).abs() < 1e-10);
    }

    #[test]
    fn test_mul_vec_output_length_equals_rows() {
        let m = Matrix::zeros(4, 3);
        let result = m.mul_vec(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 4);
    }

    #[test]
    #[should_panic(expected = "dimension")]
    fn test_mul_vec_panics_wrong_length() {
        let m = Matrix::zeros(2, 3);
        m.mul_vec(&[1.0, 2.0]); // wrong length
    }

    #[test]
    fn test_mul_vec_zero_matrix_returns_zeros() {
        let m = Matrix::zeros(3, 2);
        let result = m.mul_vec(&[5.0, 10.0]);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_outer_dims_and_values() {
        let m = Matrix::outer(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(0, 1) - 4.0).abs() < 1e-10);
        assert!((m.get(0, 2) - 5.0).abs() < 1e-10);
        assert!((m.get(1, 0) - 6.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
        assert!((m.get(1, 2) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_outer_empty_first_returns_zero_matrix() {
        let m = Matrix::outer(&[], &[1.0, 2.0]);
        assert_eq!(m.rows, 0);
        assert_eq!(m.cols, 0);
    }

    #[test]
    fn test_outer_empty_second_returns_zero_matrix() {
        let m = Matrix::outer(&[1.0, 2.0], &[]);
        assert_eq!(m.rows, 0);
        assert_eq!(m.cols, 0);
    }

    #[test]
    fn test_scale_add_basic() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(1, 1, 2.0);
        let mut other = Matrix::zeros(2, 2);
        other.set(0, 0, 0.5);
        other.set(1, 1, 0.5);
        m.scale_add(&other, 2.0);
        assert!((m.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_add_clips_to_weight_clip() {
        let mut m = Matrix::zeros(1, 1);
        m.set(0, 0, 4.0);
        let mut other = Matrix::zeros(1, 1);
        other.set(0, 0, 10.0);
        m.scale_add(&other, 1.0);
        assert!((m.get(0, 0) - WEIGHT_CLIP).abs() < 1e-10);
    }

    #[test]
    fn test_scale_add_negative_clips_to_neg_weight_clip() {
        let mut m = Matrix::zeros(1, 1);
        m.set(0, 0, -4.0);
        let mut other = Matrix::zeros(1, 1);
        other.set(0, 0, -10.0);
        m.scale_add(&other, 1.0);
        assert!((m.get(0, 0) - (-WEIGHT_CLIP)).abs() < 1e-10);
    }

    #[test]
    fn test_scale_add_zero_scale_only_clips() {
        let mut m = Matrix::zeros(1, 1);
        m.set(0, 0, 3.0);
        let other = Matrix::zeros(1, 1);
        m.scale_add(&other, 0.0);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "dimension")]
    fn test_scale_add_panics_on_dimension_mismatch() {
        let mut m = Matrix::zeros(2, 2);
        let other = Matrix::zeros(3, 3);
        m.scale_add(&other, 1.0);
    }

    // ── Softmax Tests ─────────────────────────────────────────────────

    #[test]
    fn test_softmax_masked_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0, 1, 2, 3];
        let probs = softmax_masked(&logits, &mask);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_masked_unmasked_are_zero() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1, 3];
        let probs = softmax_masked(&logits, &mask);
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[2], 0.0);
        assert!(probs[1] > 0.0);
        assert!(probs[3] > 0.0);
    }

    #[test]
    fn test_softmax_masked_single_index_is_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let mask = vec![1];
        let probs = softmax_masked(&logits, &mask);
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_masked_empty_mask_returns_all_zeros() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_masked(&logits, &[]);
        assert!(probs.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_softmax_masked_numerically_stable_large_logits() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let mask = vec![0, 1, 2];
        let probs = softmax_masked(&logits, &mask);
        assert!(probs.iter().all(|p| p.is_finite()));
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_masked_higher_logit_gets_higher_prob() {
        let logits = vec![1.0, 5.0, 2.0];
        let mask = vec![0, 1, 2];
        let probs = softmax_masked(&logits, &mask);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    // ── Argmax Tests ──────────────────────────────────────────────────

    #[test]
    fn test_argmax_masked_returns_highest_in_mask() {
        let values = vec![1.0, 5.0, 3.0, 4.0];
        let mask = vec![0, 2, 3];
        assert_eq!(argmax_masked(&values, &mask), 3);
    }

    #[test]
    fn test_argmax_masked_single_element() {
        let values = vec![1.0, 5.0, 3.0];
        let mask = vec![2];
        assert_eq!(argmax_masked(&values, &mask), 2);
    }

    #[test]
    fn test_argmax_masked_tie_returns_first() {
        let values = vec![3.0, 3.0, 3.0];
        let mask = vec![0, 1, 2];
        assert_eq!(argmax_masked(&values, &mask), 0);
    }

    #[test]
    #[should_panic]
    fn test_argmax_masked_empty_panics() {
        let values = vec![1.0, 2.0];
        argmax_masked(&values, &[]);
    }

    // ── RMS Error Tests ───────────────────────────────────────────────

    #[test]
    fn test_rms_error_empty_returns_zero() {
        assert_eq!(rms_error(&[]), 0.0);
    }

    #[test]
    fn test_rms_error_single_empty_vec_returns_zero() {
        let empty: &[f64] = &[];
        assert_eq!(rms_error(&[empty]), 0.0);
    }

    #[test]
    fn test_rms_error_known_two_vecs() {
        let v1: &[f64] = &[1.0, 0.0];
        let v2: &[f64] = &[0.0, 1.0];
        let rms = rms_error(&[v1, v2]);
        // sum_sq = 1+0+0+1 = 2, count = 4, rms = sqrt(2/4) = sqrt(0.5)
        let expected = (0.5_f64).sqrt();
        assert!((rms - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rms_error_single_vec() {
        let v: &[f64] = &[3.0, 4.0];
        let rms = rms_error(&[v]);
        // sum_sq = 9+16 = 25, count = 2, rms = sqrt(12.5) = 3.5355...
        let expected = (25.0 / 2.0_f64).sqrt();
        assert!((rms - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rms_error_all_zeros_returns_zero() {
        let v: &[f64] = &[0.0, 0.0, 0.0];
        assert_eq!(rms_error(&[v]), 0.0);
    }

    // ── Sample Tests ──────────────────────────────────────────────────

    #[test]
    fn test_sample_from_probs_always_in_mask() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mask = vec![1, 3];
        for _ in 0..20 {
            let idx = sample_from_probs(&probs, &mask, &mut rng);
            assert!(mask.contains(&idx));
        }
    }

    #[test]
    fn test_sample_from_probs_single_action_always_returns_it() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = vec![0.5, 0.5];
        let mask = vec![1];
        for _ in 0..10 {
            assert_eq!(sample_from_probs(&probs, &mask, &mut rng), 1);
        }
    }

    #[test]
    fn test_sample_from_probs_visits_multiple_actions() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = vec![0.5, 0.5];
        let mask = vec![0, 1];
        let mut seen = [false; 2];
        for _ in 0..100 {
            let idx = sample_from_probs(&probs, &mask, &mut rng);
            seen[idx] = true;
        }
        assert!(seen[0] && seen[1], "should visit both actions");
    }

    #[test]
    fn test_sample_from_probs_zero_probs_fallback_is_in_mask() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = vec![0.0, 0.0, 0.0];
        let mask = vec![0, 2];
        for _ in 0..20 {
            let idx = sample_from_probs(&probs, &mask, &mut rng);
            assert!(mask.contains(&idx));
        }
    }

    #[test]
    #[should_panic]
    fn test_sample_from_probs_empty_mask_panics() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = vec![0.5, 0.5];
        sample_from_probs(&probs, &[], &mut rng);
    }

    // ── Vec Utility Tests ─────────────────────────────────────────────

    #[test]
    fn test_vec_sub_known() {
        let result = vec_sub(&[3.0, 1.0], &[1.0, 2.0]);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_vec_add_known() {
        let result = vec_add(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!((result[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_scale_known() {
        let result = vec_scale(&[1.0, -2.0], 3.0);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_clip_vec_clamps_positive() {
        let mut v = vec![10.0, -10.0, 0.5];
        clip_vec(&mut v, 5.0);
        assert!((v[0] - 5.0).abs() < 1e-10);
        assert!((v[1] - (-5.0)).abs() < 1e-10);
        assert!((v[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_vec_sub_panics_on_length_mismatch() {
        vec_sub(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_vec_add_panics_on_length_mismatch() {
        vec_add(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn test_clip_vec_leaves_safe_values() {
        let mut v = vec![1.0, -1.0, 0.0];
        clip_vec(&mut v, 5.0);
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - (-1.0)).abs() < 1e-10);
        assert!((v[2] - 0.0).abs() < 1e-10);
    }

    // ── Defensive: OOB assertions ────────────────────────────────

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_panics_on_oob_row() {
        let m = Matrix::zeros(2, 2);
        m.get(5, 0); // should panic, not return 0.0
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_set_panics_on_oob_row() {
        let mut m = Matrix::zeros(2, 2);
        m.set(5, 0, 1.0); // should panic, not silently do nothing
    }

    #[test]
    #[should_panic(expected = "mask index out of bounds")]
    fn test_softmax_masked_panics_on_oob_mask() {
        let logits = vec![1.0, 2.0, 3.0];
        softmax_masked(&logits, &[0, 5]); // 5 >= logits.len()
    }

    #[test]
    #[should_panic(expected = "mask index out of bounds")]
    fn test_argmax_masked_panics_on_oob_mask() {
        let values = vec![1.0, 2.0, 3.0];
        argmax_masked(&values, &[0, 5]); // 5 >= values.len()
    }

    // ── sample_from_probs distribution ───────────────────────────

    // ── Phase 3 Cycle 3.1: CCA identical activations → identity ────

    #[test]
    fn test_cca_identical_activations_identity_permutation() {
        // Same activations for A and B → identity permutation [0, 1, 2]
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 100;
        let n_neurons = 3;
        let mut rng = StdRng::seed_from_u64(42);

        // Generate random activations (batch_size × n_neurons)
        let mut act_a = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        for r in 0..batch_size {
            for c in 0..n_neurons {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_a, r, c, val);
            }
        }
        let act_b = act_a.clone();

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act_a, &act_b);
        assert_eq!(perm.len(), n_neurons);
        assert_eq!(perm, vec![0, 1, 2]);
    }

    #[test]
    fn test_cca_permutation_length_is_min() {
        // A has 4 neurons, B has 4 neurons → perm length = 4
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 100;
        let mut rng = StdRng::seed_from_u64(42);

        let mut act = CpuLinAlg::zeros_mat(batch_size, 4);
        for r in 0..batch_size {
            for c in 0..4 {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act, r, c, val);
            }
        }

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act, &act);
        assert_eq!(perm.len(), 4);
    }

    // ── Phase 3 Cycle 3.2: CCA recovers permutation ────────────

    #[test]
    fn test_cca_permuted_activations_recovers_permutation() {
        // B = permuted A with columns [2, 0, 1] → perm should map back
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 500;
        let n_neurons = 3;
        let mut rng = StdRng::seed_from_u64(42);

        let mut act_a = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        for r in 0..batch_size {
            for c in 0..n_neurons {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_a, r, c, val);
            }
        }

        // B columns = [A_col2, A_col0, A_col1]
        // So B neuron 0 = A neuron 2, B neuron 1 = A neuron 0, B neuron 2 = A neuron 1
        // permutation[i] = which A neuron maps to B neuron i
        // Expected: [2, 0, 1]
        let mut act_b = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        let col_map = [2, 0, 1]; // B_col_j = A_col_{col_map[j]}
        for r in 0..batch_size {
            for (j, &src_col) in col_map.iter().enumerate() {
                CpuLinAlg::mat_set(&mut act_b, r, j, CpuLinAlg::mat_get(&act_a, r, src_col));
            }
        }

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act_a, &act_b);
        assert_eq!(perm, vec![2, 0, 1]);
    }

    #[test]
    fn test_cca_permuted_with_small_batch() {
        // Same permutation test but with batch_size=50
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 50;
        let n_neurons = 3;
        let mut rng = StdRng::seed_from_u64(99);

        let mut act_a = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        for r in 0..batch_size {
            for c in 0..n_neurons {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_a, r, c, val);
            }
        }

        // Permutation [1, 2, 0]
        let mut act_b = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        let col_map = [1, 2, 0];
        for r in 0..batch_size {
            for (j, &src_col) in col_map.iter().enumerate() {
                CpuLinAlg::mat_set(&mut act_b, r, j, CpuLinAlg::mat_get(&act_a, r, src_col));
            }
        }

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act_a, &act_b);
        assert_eq!(perm, vec![1, 2, 0]);
    }

    #[test]
    fn test_cca_permuted_large_batch() {
        // batch_size=500, verifying robustness
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 500;
        let n_neurons = 4;
        let mut rng = StdRng::seed_from_u64(7);

        let mut act_a = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        for r in 0..batch_size {
            for c in 0..n_neurons {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_a, r, c, val);
            }
        }

        // Permutation [3, 1, 0, 2]
        let mut act_b = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        let col_map = [3, 1, 0, 2];
        for r in 0..batch_size {
            for (j, &src_col) in col_map.iter().enumerate() {
                CpuLinAlg::mat_set(&mut act_b, r, j, CpuLinAlg::mat_get(&act_a, r, src_col));
            }
        }

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act_a, &act_b);
        assert_eq!(perm, vec![3, 1, 0, 2]);
    }

    // ── Phase 3 Cycle 3.3: CCA with different dimensions ────────

    #[test]
    fn test_cca_a_larger_than_b() {
        // A has 4 neurons, B has 3 neurons → permutation length = 3
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 200;
        let mut rng = StdRng::seed_from_u64(42);

        let mut act_a = CpuLinAlg::zeros_mat(batch_size, 4);
        for r in 0..batch_size {
            for c in 0..4 {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_a, r, c, val);
            }
        }

        // B has 3 neurons: B_col_j = A_col_j (first 3 columns)
        let mut act_b = CpuLinAlg::zeros_mat(batch_size, 3);
        for r in 0..batch_size {
            for c in 0..3 {
                CpuLinAlg::mat_set(&mut act_b, r, c, CpuLinAlg::mat_get(&act_a, r, c));
            }
        }

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act_a, &act_b);
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn test_cca_b_larger_than_a() {
        // A has 3 neurons, B has 5 neurons → permutation length = 3 (min)
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 200;
        let mut rng = StdRng::seed_from_u64(42);

        let mut act_a = CpuLinAlg::zeros_mat(batch_size, 3);
        for r in 0..batch_size {
            for c in 0..3 {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_a, r, c, val);
            }
        }

        let mut act_b = CpuLinAlg::zeros_mat(batch_size, 5);
        for r in 0..batch_size {
            for c in 0..5 {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_b, r, c, val);
            }
        }

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act_a, &act_b);
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn test_cca_dead_neuron_excluded() {
        // One neuron in B has zero variance (dead) → still produces valid permutation
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;

        let batch_size = 100;
        let n_neurons = 3;
        let mut rng = StdRng::seed_from_u64(42);

        let mut act_a = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        for r in 0..batch_size {
            for c in 0..n_neurons {
                let val: f64 = rng.gen_range(-1.0..1.0);
                CpuLinAlg::mat_set(&mut act_a, r, c, val);
            }
        }

        // B: neuron 1 is dead (constant 0), neurons 0 and 2 copy from A
        let mut act_b = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        for r in 0..batch_size {
            CpuLinAlg::mat_set(&mut act_b, r, 0, CpuLinAlg::mat_get(&act_a, r, 0));
            CpuLinAlg::mat_set(&mut act_b, r, 1, 0.0); // dead neuron
            CpuLinAlg::mat_set(&mut act_b, r, 2, CpuLinAlg::mat_get(&act_a, r, 2));
        }

        let perm = cca_neuron_alignment::<CpuLinAlg>(&act_a, &act_b);
        // Should produce a valid permutation of length 3, no panic
        assert_eq!(perm.len(), n_neurons);
        // All indices in range [0, n_neurons)
        for &p in &perm {
            assert!(p < n_neurons, "permutation index {p} out of range");
        }
        // All indices unique
        let mut sorted = perm.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n_neurons, "permutation has duplicates");
    }

    #[test]
    fn test_sample_from_probs_distribution_roughly_correct() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = vec![0.7, 0.3];
        let mask = vec![0, 1];
        let mut counts = [0usize; 2];
        let n = 1000;
        for _ in 0..n {
            let idx = sample_from_probs(&probs, &mask, &mut rng);
            counts[idx] += 1;
        }
        let ratio = counts[0] as f64 / n as f64;
        // Should be roughly 0.7, allow 10% tolerance
        assert!(
            (ratio - 0.7).abs() < 0.1,
            "Expected ~0.7 for action 0, got {ratio}"
        );
    }
}
