// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-29

//! CPU backend for the [`LinAlg`] trait.
//!
//! [`CpuLinAlg`] implements all linear algebra operations using the
//! existing [`Matrix`] struct and `Vec<f64>`.
//! All operations are zero-cost wrappers that delegate to the current
//! pure-Rust implementations.

use crate::activation::Activation;
use crate::linalg::LinAlg;
use crate::matrix::Matrix;
use rand::Rng;

/// CPU linear algebra backend.
///
/// Uses `Vec<f64>` for vectors and [`Matrix`] for matrices.
/// All trait methods delegate to existing implementations in
/// [`crate::matrix`] and [`crate::activation`].
#[derive(Debug, Clone)]
pub struct CpuLinAlg;

impl LinAlg for CpuLinAlg {
    type Vector = Vec<f64>;
    type Matrix = Matrix;

    fn zeros_vec(size: usize) -> Self::Vector {
        vec![0.0; size]
    }

    fn zeros_mat(rows: usize, cols: usize) -> Self::Matrix {
        Matrix::zeros(rows, cols)
    }

    fn xavier_mat(rows: usize, cols: usize, rng: &mut impl Rng) -> Self::Matrix {
        Matrix::xavier(rows, cols, rng)
    }

    fn mat_vec_mul(m: &Self::Matrix, v: &Self::Vector) -> Self::Vector {
        m.mul_vec(v)
    }

    fn mat_transpose(m: &Self::Matrix) -> Self::Matrix {
        m.transpose()
    }

    fn outer_product(a: &Self::Vector, b: &Self::Vector) -> Self::Matrix {
        Matrix::outer(a, b)
    }

    fn mat_mul(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix {
        assert_eq!(a.cols, b.rows, "mat_mul: inner dimensions mismatch");
        let mut result = Matrix::zeros(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum = 0.0;
                for k in 0..a.cols {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    fn svd(m: &Self::Matrix) -> crate::linalg::SvdResult<Self> {
        Ok(crate::linalg::golub_kahan::GolubKahanSvd::new().compute(m)?)
    }

    fn mat_scale_add(m: &mut Self::Matrix, other: &Self::Matrix, scale: f64) {
        m.scale_add(other, scale);
    }

    fn mat_rows(m: &Self::Matrix) -> usize {
        m.rows
    }

    fn mat_cols(m: &Self::Matrix) -> usize {
        m.cols
    }

    fn mat_get(m: &Self::Matrix, row: usize, col: usize) -> f64 {
        m.get(row, col)
    }

    fn mat_set(m: &mut Self::Matrix, row: usize, col: usize, val: f64) {
        m.set(row, col, val);
    }

    fn vec_add(a: &Self::Vector, b: &Self::Vector) -> Self::Vector {
        crate::matrix::vec_add(a, b)
    }

    fn vec_sub(a: &Self::Vector, b: &Self::Vector) -> Self::Vector {
        crate::matrix::vec_sub(a, b)
    }

    fn vec_scale(v: &Self::Vector, s: f64) -> Self::Vector {
        crate::matrix::vec_scale(v, s)
    }

    fn vec_hadamard(a: &Self::Vector, b: &Self::Vector) -> Self::Vector {
        assert_eq!(a.len(), b.len(), "vec_hadamard: length mismatch");
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    fn vec_dot(a: &Self::Vector, b: &Self::Vector) -> f64 {
        assert_eq!(a.len(), b.len(), "vec_dot: length mismatch");
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn vec_len(v: &Self::Vector) -> usize {
        v.len()
    }

    fn vec_get(v: &Self::Vector, i: usize) -> f64 {
        v[i]
    }

    fn vec_set(v: &mut Self::Vector, i: usize, val: f64) {
        v[i] = val;
    }

    fn vec_from_slice(s: &[f64]) -> Self::Vector {
        s.to_vec()
    }

    fn vec_to_vec(v: &Self::Vector) -> Vec<f64> {
        v.clone()
    }

    fn vec_as_slice(v: &Self::Vector) -> &[f64] {
        v.as_slice()
    }

    fn clip_vec(v: &mut Self::Vector, max_abs: f64) {
        crate::matrix::clip_vec(v, max_abs);
    }

    fn clip_mat(m: &mut Self::Matrix, max_abs: f64) {
        for x in m.data.iter_mut() {
            *x = x.clamp(-max_abs, max_abs);
        }
    }

    fn apply_activation(v: &Self::Vector, act: Activation) -> Self::Vector {
        v.iter().map(|&x| act.apply(x)).collect()
    }

    fn apply_derivative(v: &Self::Vector, act: Activation) -> Self::Vector {
        v.iter().map(|&fx| act.derivative(fx)).collect()
    }

    fn softmax_masked(logits: &Self::Vector, mask: &[usize]) -> Self::Vector {
        crate::matrix::softmax_masked(logits, mask)
    }

    fn argmax_masked(values: &Self::Vector, mask: &[usize]) -> usize {
        crate::matrix::argmax_masked(values, mask)
    }

    fn sample_from_probs(probs: &Self::Vector, mask: &[usize], rng: &mut impl Rng) -> usize {
        crate::matrix::sample_from_probs(probs, mask, rng)
    }

    fn rms_error(error_vecs: &[&Self::Vector]) -> f64 {
        let slices: Vec<&[f64]> = error_vecs.iter().map(|v| v.as_slice()).collect();
        crate::matrix::rms_error(&slices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    // ── Cycle 1.1: Vector basics ─────────────────────────────────

    #[test]
    fn test_zeros_vec_correct_length() {
        let v = CpuLinAlg::zeros_vec(5);
        assert_eq!(CpuLinAlg::vec_len(&v), 5);
    }

    #[test]
    fn test_zeros_vec_all_zeros() {
        let v = CpuLinAlg::zeros_vec(3);
        for i in 0..3 {
            assert_eq!(CpuLinAlg::vec_get(&v, i), 0.0);
        }
    }

    #[test]
    fn test_zeros_vec_empty() {
        let v = CpuLinAlg::zeros_vec(0);
        assert_eq!(CpuLinAlg::vec_len(&v), 0);
    }

    #[test]
    fn test_vec_get_returns_element() {
        let v = CpuLinAlg::vec_from_slice(&[10.0, 20.0, 30.0]);
        assert_eq!(CpuLinAlg::vec_get(&v, 0), 10.0);
        assert_eq!(CpuLinAlg::vec_get(&v, 1), 20.0);
        assert_eq!(CpuLinAlg::vec_get(&v, 2), 30.0);
    }

    #[test]
    fn test_vec_set_modifies_element() {
        let mut v = CpuLinAlg::zeros_vec(3);
        CpuLinAlg::vec_set(&mut v, 1, 42.0);
        assert_eq!(CpuLinAlg::vec_get(&v, 1), 42.0);
        assert_eq!(CpuLinAlg::vec_get(&v, 0), 0.0);
    }

    #[test]
    fn test_vec_from_slice_creates_vector() {
        let v = CpuLinAlg::vec_from_slice(&[1.0, 2.0]);
        assert_eq!(CpuLinAlg::vec_len(&v), 2);
        assert_eq!(CpuLinAlg::vec_get(&v, 0), 1.0);
        assert_eq!(CpuLinAlg::vec_get(&v, 1), 2.0);
    }

    #[test]
    fn test_vec_from_slice_empty() {
        let v = CpuLinAlg::vec_from_slice(&[]);
        assert_eq!(CpuLinAlg::vec_len(&v), 0);
    }

    #[test]
    fn test_vec_to_vec_returns_owned() {
        let v = CpuLinAlg::vec_from_slice(&[1.0, 2.0, 3.0]);
        let owned = CpuLinAlg::vec_to_vec(&v);
        assert_eq!(owned, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec_as_slice_returns_slice() {
        let v = CpuLinAlg::vec_from_slice(&[4.0, 5.0]);
        let s = CpuLinAlg::vec_as_slice(&v);
        assert_eq!(s, &[4.0, 5.0]);
    }

    #[test]
    fn test_vec_len_matches_creation_size() {
        let v = CpuLinAlg::zeros_vec(7);
        assert_eq!(CpuLinAlg::vec_len(&v), 7);
    }

    // ── Cycle 1.2: Vector arithmetic ─────────────────────────────

    #[test]
    fn test_vec_add_known() {
        let a = CpuLinAlg::vec_from_slice(&[1.0, 2.0]);
        let b = CpuLinAlg::vec_from_slice(&[3.0, 4.0]);
        let r = CpuLinAlg::vec_add(&a, &b);
        assert_eq!(CpuLinAlg::vec_to_vec(&r), vec![4.0, 6.0]);
    }

    #[test]
    fn test_vec_sub_known() {
        let a = CpuLinAlg::vec_from_slice(&[5.0, 3.0]);
        let b = CpuLinAlg::vec_from_slice(&[1.0, 2.0]);
        let r = CpuLinAlg::vec_sub(&a, &b);
        assert_eq!(CpuLinAlg::vec_to_vec(&r), vec![4.0, 1.0]);
    }

    #[test]
    fn test_vec_scale_known() {
        let v = CpuLinAlg::vec_from_slice(&[2.0, 4.0]);
        let r = CpuLinAlg::vec_scale(&v, 0.5);
        assert_eq!(CpuLinAlg::vec_to_vec(&r), vec![1.0, 2.0]);
    }

    #[test]
    fn test_vec_hadamard_known() {
        let a = CpuLinAlg::vec_from_slice(&[2.0, 3.0, 4.0]);
        let b = CpuLinAlg::vec_from_slice(&[0.5, -1.0, 2.0]);
        let r = CpuLinAlg::vec_hadamard(&a, &b);
        assert_eq!(CpuLinAlg::vec_to_vec(&r), vec![1.0, -3.0, 8.0]);
    }

    #[test]
    fn test_clip_vec_clamps() {
        let mut v = CpuLinAlg::vec_from_slice(&[10.0, -10.0, 0.5]);
        CpuLinAlg::clip_vec(&mut v, 5.0);
        assert!((CpuLinAlg::vec_get(&v, 0) - 5.0).abs() < 1e-10);
        assert!((CpuLinAlg::vec_get(&v, 1) - (-5.0)).abs() < 1e-10);
        assert!((CpuLinAlg::vec_get(&v, 2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_clip_vec_leaves_safe_values() {
        let mut v = CpuLinAlg::vec_from_slice(&[1.0, -1.0, 0.0]);
        CpuLinAlg::clip_vec(&mut v, 5.0);
        assert_eq!(CpuLinAlg::vec_to_vec(&v), vec![1.0, -1.0, 0.0]);
    }

    // ── Cycle 1.3: Matrix basics ─────────────────────────────────

    #[test]
    fn test_zeros_mat_correct_dims() {
        let m = CpuLinAlg::zeros_mat(3, 4);
        assert_eq!(CpuLinAlg::mat_rows(&m), 3);
        assert_eq!(CpuLinAlg::mat_cols(&m), 4);
    }

    #[test]
    fn test_zeros_mat_all_zeros() {
        let m = CpuLinAlg::zeros_mat(2, 3);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(CpuLinAlg::mat_get(&m, r, c), 0.0);
            }
        }
    }

    #[test]
    fn test_xavier_mat_correct_dims() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let m = CpuLinAlg::xavier_mat(3, 4, &mut rng);
        assert_eq!(CpuLinAlg::mat_rows(&m), 3);
        assert_eq!(CpuLinAlg::mat_cols(&m), 4);
    }

    #[test]
    fn test_xavier_mat_all_finite() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let m = CpuLinAlg::xavier_mat(10, 10, &mut rng);
        for r in 0..10 {
            for c in 0..10 {
                assert!(CpuLinAlg::mat_get(&m, r, c).is_finite());
            }
        }
    }

    #[test]
    fn test_mat_get_set_roundtrip() {
        let mut m = CpuLinAlg::zeros_mat(3, 3);
        CpuLinAlg::mat_set(&mut m, 1, 2, 42.0);
        assert_eq!(CpuLinAlg::mat_get(&m, 1, 2), 42.0);
        assert_eq!(CpuLinAlg::mat_get(&m, 0, 0), 0.0);
    }

    #[test]
    fn test_mat_transpose_swaps_dims() {
        let m = CpuLinAlg::zeros_mat(3, 5);
        let t = CpuLinAlg::mat_transpose(&m);
        assert_eq!(CpuLinAlg::mat_rows(&t), 5);
        assert_eq!(CpuLinAlg::mat_cols(&t), 3);
    }

    #[test]
    fn test_mat_transpose_repositions_values() {
        let mut m = CpuLinAlg::zeros_mat(2, 3);
        CpuLinAlg::mat_set(&mut m, 0, 1, 7.0);
        CpuLinAlg::mat_set(&mut m, 1, 2, 3.0);
        let t = CpuLinAlg::mat_transpose(&m);
        assert_eq!(CpuLinAlg::mat_get(&t, 1, 0), 7.0);
        assert_eq!(CpuLinAlg::mat_get(&t, 2, 1), 3.0);
    }

    // ── Cycle 1.4: Matrix-vector operations ──────────────────────

    #[test]
    fn test_mat_vec_mul_known() {
        // [[1,2],[3,4]] * [5,6] = [17, 39]
        let mut m = CpuLinAlg::zeros_mat(2, 2);
        CpuLinAlg::mat_set(&mut m, 0, 0, 1.0);
        CpuLinAlg::mat_set(&mut m, 0, 1, 2.0);
        CpuLinAlg::mat_set(&mut m, 1, 0, 3.0);
        CpuLinAlg::mat_set(&mut m, 1, 1, 4.0);
        let v = CpuLinAlg::vec_from_slice(&[5.0, 6.0]);
        let r = CpuLinAlg::mat_vec_mul(&m, &v);
        assert_eq!(CpuLinAlg::vec_len(&r), 2);
        assert!((CpuLinAlg::vec_get(&r, 0) - 17.0).abs() < 1e-10);
        assert!((CpuLinAlg::vec_get(&r, 1) - 39.0).abs() < 1e-10);
    }

    #[test]
    fn test_outer_product_known() {
        let a = CpuLinAlg::vec_from_slice(&[1.0, 2.0]);
        let b = CpuLinAlg::vec_from_slice(&[3.0, 4.0, 5.0]);
        let m = CpuLinAlg::outer_product(&a, &b);
        assert_eq!(CpuLinAlg::mat_rows(&m), 2);
        assert_eq!(CpuLinAlg::mat_cols(&m), 3);
        assert!((CpuLinAlg::mat_get(&m, 0, 0) - 3.0).abs() < 1e-10);
        assert!((CpuLinAlg::mat_get(&m, 1, 2) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_scale_add_basic() {
        let mut m = CpuLinAlg::zeros_mat(2, 2);
        CpuLinAlg::mat_set(&mut m, 0, 0, 1.0);
        CpuLinAlg::mat_set(&mut m, 1, 1, 2.0);
        let mut other = CpuLinAlg::zeros_mat(2, 2);
        CpuLinAlg::mat_set(&mut other, 0, 0, 0.5);
        CpuLinAlg::mat_set(&mut other, 1, 1, 0.5);
        CpuLinAlg::mat_scale_add(&mut m, &other, 2.0);
        assert!((CpuLinAlg::mat_get(&m, 0, 0) - 2.0).abs() < 1e-10);
        assert!((CpuLinAlg::mat_get(&m, 1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_mat_clamps() {
        let mut m = CpuLinAlg::zeros_mat(1, 2);
        CpuLinAlg::mat_set(&mut m, 0, 0, 10.0);
        CpuLinAlg::mat_set(&mut m, 0, 1, -10.0);
        CpuLinAlg::clip_mat(&mut m, 5.0);
        assert!((CpuLinAlg::mat_get(&m, 0, 0) - 5.0).abs() < 1e-10);
        assert!((CpuLinAlg::mat_get(&m, 0, 1) - (-5.0)).abs() < 1e-10);
    }

    // ── Cycle 1.5: Activation + softmax + sampling ───────────────

    #[test]
    fn test_apply_activation_tanh() {
        let v = CpuLinAlg::vec_from_slice(&[0.5, -0.5]);
        let r = CpuLinAlg::apply_activation(&v, Activation::Tanh);
        assert!((CpuLinAlg::vec_get(&r, 0) - 0.5_f64.tanh()).abs() < 1e-12);
        assert!((CpuLinAlg::vec_get(&r, 1) - (-0.5_f64).tanh()).abs() < 1e-12);
    }

    #[test]
    fn test_apply_activation_relu() {
        let v = CpuLinAlg::vec_from_slice(&[1.0, -1.0, 0.0]);
        let r = CpuLinAlg::apply_activation(&v, Activation::Relu);
        assert_eq!(CpuLinAlg::vec_to_vec(&r), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_apply_derivative_tanh() {
        let v = CpuLinAlg::vec_from_slice(&[0.5]);
        let r = CpuLinAlg::apply_derivative(&v, Activation::Tanh);
        // derivative(0.5) = 1 - 0.25 = 0.75
        assert!((CpuLinAlg::vec_get(&r, 0) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_softmax_masked_sums_to_one() {
        let logits = CpuLinAlg::vec_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mask = vec![0, 1, 2, 3];
        let probs = CpuLinAlg::softmax_masked(&logits, &mask);
        let sum: f64 = CpuLinAlg::vec_to_vec(&probs).iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_masked_unmasked_are_zero() {
        let logits = CpuLinAlg::vec_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mask = vec![1, 3];
        let probs = CpuLinAlg::softmax_masked(&logits, &mask);
        assert_eq!(CpuLinAlg::vec_get(&probs, 0), 0.0);
        assert_eq!(CpuLinAlg::vec_get(&probs, 2), 0.0);
        assert!(CpuLinAlg::vec_get(&probs, 1) > 0.0);
        assert!(CpuLinAlg::vec_get(&probs, 3) > 0.0);
    }

    #[test]
    fn test_argmax_masked_returns_highest() {
        let values = CpuLinAlg::vec_from_slice(&[1.0, 5.0, 3.0, 4.0]);
        let mask = vec![0, 2, 3];
        assert_eq!(CpuLinAlg::argmax_masked(&values, &mask), 3);
    }

    #[test]
    fn test_sample_from_probs_in_mask() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let probs = CpuLinAlg::vec_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        let mask = vec![1, 3];
        for _ in 0..20 {
            let idx = CpuLinAlg::sample_from_probs(&probs, &mask, &mut rng);
            assert!(mask.contains(&idx));
        }
    }

    #[test]
    fn test_rms_error_known() {
        let v1 = CpuLinAlg::vec_from_slice(&[1.0, 0.0]);
        let v2 = CpuLinAlg::vec_from_slice(&[0.0, 1.0]);
        let rms = CpuLinAlg::rms_error(&[&v1, &v2]);
        let expected = (0.5_f64).sqrt();
        assert!((rms - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rms_error_empty() {
        let rms = CpuLinAlg::rms_error(&[]);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_vec_dot_known() {
        let a = CpuLinAlg::vec_from_slice(&[1.0, 2.0, 3.0]);
        let b = CpuLinAlg::vec_from_slice(&[4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let dot = CpuLinAlg::vec_dot(&a, &b);
        assert!((dot - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_vec_dot_orthogonal() {
        let a = CpuLinAlg::vec_from_slice(&[1.0, 0.0]);
        let b = CpuLinAlg::vec_from_slice(&[0.0, 1.0]);
        assert!((CpuLinAlg::vec_dot(&a, &b)).abs() < 1e-12);
    }

    // ── Phase 1 Cycle 1.1: mat_mul (matrix × matrix) ────────────

    #[test]
    fn test_mat_mul_2x3_by_3x2() {
        // A = [[1,2,3],[4,5,6]] (2×3)
        // B = [[7,8],[9,10],[11,12]] (3×2)
        // C = A*B = [[58,64],[139,154]] (2×2)
        let mut a = CpuLinAlg::zeros_mat(2, 3);
        CpuLinAlg::mat_set(&mut a, 0, 0, 1.0);
        CpuLinAlg::mat_set(&mut a, 0, 1, 2.0);
        CpuLinAlg::mat_set(&mut a, 0, 2, 3.0);
        CpuLinAlg::mat_set(&mut a, 1, 0, 4.0);
        CpuLinAlg::mat_set(&mut a, 1, 1, 5.0);
        CpuLinAlg::mat_set(&mut a, 1, 2, 6.0);

        let mut b = CpuLinAlg::zeros_mat(3, 2);
        CpuLinAlg::mat_set(&mut b, 0, 0, 7.0);
        CpuLinAlg::mat_set(&mut b, 0, 1, 8.0);
        CpuLinAlg::mat_set(&mut b, 1, 0, 9.0);
        CpuLinAlg::mat_set(&mut b, 1, 1, 10.0);
        CpuLinAlg::mat_set(&mut b, 2, 0, 11.0);
        CpuLinAlg::mat_set(&mut b, 2, 1, 12.0);

        let c = CpuLinAlg::mat_mul(&a, &b);
        assert_eq!(CpuLinAlg::mat_rows(&c), 2);
        assert_eq!(CpuLinAlg::mat_cols(&c), 2);
        assert!((CpuLinAlg::mat_get(&c, 0, 0) - 58.0).abs() < 1e-10);
        assert!((CpuLinAlg::mat_get(&c, 0, 1) - 64.0).abs() < 1e-10);
        assert!((CpuLinAlg::mat_get(&c, 1, 0) - 139.0).abs() < 1e-10);
        assert!((CpuLinAlg::mat_get(&c, 1, 1) - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_mul_identity_left() {
        // I × M = M
        let mut identity = CpuLinAlg::zeros_mat(3, 3);
        CpuLinAlg::mat_set(&mut identity, 0, 0, 1.0);
        CpuLinAlg::mat_set(&mut identity, 1, 1, 1.0);
        CpuLinAlg::mat_set(&mut identity, 2, 2, 1.0);

        let mut m = CpuLinAlg::zeros_mat(3, 2);
        CpuLinAlg::mat_set(&mut m, 0, 0, 1.0);
        CpuLinAlg::mat_set(&mut m, 0, 1, 2.0);
        CpuLinAlg::mat_set(&mut m, 1, 0, 3.0);
        CpuLinAlg::mat_set(&mut m, 1, 1, 4.0);
        CpuLinAlg::mat_set(&mut m, 2, 0, 5.0);
        CpuLinAlg::mat_set(&mut m, 2, 1, 6.0);

        let result = CpuLinAlg::mat_mul(&identity, &m);
        assert_eq!(CpuLinAlg::mat_rows(&result), 3);
        assert_eq!(CpuLinAlg::mat_cols(&result), 2);
        for r in 0..3 {
            for c in 0..2 {
                assert!(
                    (CpuLinAlg::mat_get(&result, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs()
                        < 1e-10
                );
            }
        }
    }

    #[test]
    fn test_mat_mul_identity_right() {
        // M × I = M
        let mut m = CpuLinAlg::zeros_mat(2, 3);
        CpuLinAlg::mat_set(&mut m, 0, 0, 1.0);
        CpuLinAlg::mat_set(&mut m, 0, 1, 2.0);
        CpuLinAlg::mat_set(&mut m, 0, 2, 3.0);
        CpuLinAlg::mat_set(&mut m, 1, 0, 4.0);
        CpuLinAlg::mat_set(&mut m, 1, 1, 5.0);
        CpuLinAlg::mat_set(&mut m, 1, 2, 6.0);

        let mut identity = CpuLinAlg::zeros_mat(3, 3);
        CpuLinAlg::mat_set(&mut identity, 0, 0, 1.0);
        CpuLinAlg::mat_set(&mut identity, 1, 1, 1.0);
        CpuLinAlg::mat_set(&mut identity, 2, 2, 1.0);

        let result = CpuLinAlg::mat_mul(&m, &identity);
        assert_eq!(CpuLinAlg::mat_rows(&result), 2);
        assert_eq!(CpuLinAlg::mat_cols(&result), 3);
        for r in 0..2 {
            for c in 0..3 {
                assert!(
                    (CpuLinAlg::mat_get(&result, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs()
                        < 1e-10
                );
            }
        }
    }

    #[test]
    fn test_mat_mul_result_dimensions() {
        // (4×3) × (3×5) = (4×5)
        let a = CpuLinAlg::zeros_mat(4, 3);
        let b = CpuLinAlg::zeros_mat(3, 5);
        let c = CpuLinAlg::mat_mul(&a, &b);
        assert_eq!(CpuLinAlg::mat_rows(&c), 4);
        assert_eq!(CpuLinAlg::mat_cols(&c), 5);
    }

    // ── Phase 2 Cycle 2.1: SVD of known matrices ────────────────

    /// Helper: build matrix from row-major slice.
    fn mat_from_rows(rows: usize, cols: usize, data: &[f64]) -> Matrix {
        assert_eq!(data.len(), rows * cols);
        let mut m = CpuLinAlg::zeros_mat(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                CpuLinAlg::mat_set(&mut m, r, c, data[r * cols + c]);
            }
        }
        m
    }

    /// Helper: reconstruct M from U, S, V^T → U × diag(S) × V^T.
    fn reconstruct_usv(u: &Matrix, s: &Vec<f64>, v: &Matrix) -> Matrix {
        let rows = CpuLinAlg::mat_rows(u);
        let cols = CpuLinAlg::mat_cols(v);
        let k = CpuLinAlg::vec_len(s);
        // diag(S) × V^T
        let vt = CpuLinAlg::mat_transpose(v);
        let mut sv = CpuLinAlg::zeros_mat(k, cols);
        for i in 0..k {
            for j in 0..cols {
                CpuLinAlg::mat_set(
                    &mut sv,
                    i,
                    j,
                    CpuLinAlg::vec_get(s, i) * CpuLinAlg::mat_get(&vt, i, j),
                );
            }
        }
        // U × (diag(S) × V^T)
        let mut result = CpuLinAlg::zeros_mat(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += CpuLinAlg::mat_get(u, i, l) * CpuLinAlg::mat_get(&sv, l, j);
                }
                CpuLinAlg::mat_set(&mut result, i, j, sum);
            }
        }
        result
    }

    /// Helper: check if matrix is approximately identity.
    fn assert_approx_identity(m: &Matrix, tol: f64) {
        let n = CpuLinAlg::mat_rows(m);
        assert_eq!(n, CpuLinAlg::mat_cols(m), "not square");
        for r in 0..n {
            for c in 0..n {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (CpuLinAlg::mat_get(m, r, c) - expected).abs() < tol,
                    "at ({r},{c}): got {} expected {expected}",
                    CpuLinAlg::mat_get(m, r, c)
                );
            }
        }
    }

    #[test]
    fn test_svd_2x2_diagonal() {
        // diag(5, 3) → U≈I, S=[5,3], V≈I (up to sign)
        let m = mat_from_rows(2, 2, &[5.0, 0.0, 0.0, 3.0]);
        let (u, s, v) = CpuLinAlg::svd(&m).unwrap();

        // S values = [5, 3] sorted descending
        assert!((CpuLinAlg::vec_get(&s, 0) - 5.0).abs() < 1e-10);
        assert!((CpuLinAlg::vec_get(&s, 1) - 3.0).abs() < 1e-10);

        // Reconstruction: U × diag(S) × V^T ≈ M
        let recon = reconstruct_usv(&u, &s, &v);
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (CpuLinAlg::mat_get(&recon, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs() < 1e-10,
                    "reconstruction mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn test_svd_3x3_reconstruction() {
        // Known 3×3 matrix
        let m = mat_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let (u, s, v) = CpuLinAlg::svd(&m).unwrap();

        // Reconstruction within tolerance
        let recon = reconstruct_usv(&u, &s, &v);
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (CpuLinAlg::mat_get(&recon, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs() < 1e-10,
                    "reconstruction mismatch at ({r},{c}): got {} expected {}",
                    CpuLinAlg::mat_get(&recon, r, c),
                    CpuLinAlg::mat_get(&m, r, c)
                );
            }
        }
    }

    #[test]
    fn test_svd_rectangular_3x2_reconstruction() {
        let m = mat_from_rows(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, s, v) = CpuLinAlg::svd(&m).unwrap();

        // U is 3×2, S has 2 elements, V is 2×2
        assert_eq!(CpuLinAlg::mat_rows(&u), 3);
        assert_eq!(CpuLinAlg::mat_cols(&u), 2);
        assert_eq!(CpuLinAlg::vec_len(&s), 2);
        assert_eq!(CpuLinAlg::mat_rows(&v), 2);
        assert_eq!(CpuLinAlg::mat_cols(&v), 2);

        let recon = reconstruct_usv(&u, &s, &v);
        for r in 0..3 {
            for c in 0..2 {
                assert!(
                    (CpuLinAlg::mat_get(&recon, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs() < 1e-10,
                    "reconstruction mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn test_svd_singular_values_non_negative_descending() {
        let m = mat_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let (_u, s, _v) = CpuLinAlg::svd(&m).unwrap();

        for i in 0..CpuLinAlg::vec_len(&s) {
            assert!(
                CpuLinAlg::vec_get(&s, i) >= 0.0,
                "singular value {i} is negative: {}",
                CpuLinAlg::vec_get(&s, i)
            );
        }
        for i in 1..CpuLinAlg::vec_len(&s) {
            assert!(
                CpuLinAlg::vec_get(&s, i - 1) >= CpuLinAlg::vec_get(&s, i) - 1e-12,
                "singular values not descending: s[{}]={} < s[{}]={}",
                i - 1,
                CpuLinAlg::vec_get(&s, i - 1),
                i,
                CpuLinAlg::vec_get(&s, i)
            );
        }
    }

    #[test]
    fn test_svd_orthonormal_columns() {
        let m = mat_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let (u, _s, v) = CpuLinAlg::svd(&m).unwrap();

        // U^T × U ≈ I
        let utu = CpuLinAlg::mat_mul(&CpuLinAlg::mat_transpose(&u), &u);
        assert_approx_identity(&utu, 1e-10);

        // V^T × V ≈ I
        let vtv = CpuLinAlg::mat_mul(&CpuLinAlg::mat_transpose(&v), &v);
        assert_approx_identity(&vtv, 1e-10);
    }

    // ── Phase 2 Cycle 2.2: SVD edge cases ───────────────────────

    #[test]
    fn test_svd_1x1_matrix() {
        let m = mat_from_rows(1, 1, &[7.0]);
        let (_u, s, _v) = CpuLinAlg::svd(&m).unwrap();
        assert_eq!(CpuLinAlg::vec_len(&s), 1);
        assert!((CpuLinAlg::vec_get(&s, 0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_1x1_negative() {
        let m = mat_from_rows(1, 1, &[-3.0]);
        let (u, s, v) = CpuLinAlg::svd(&m).unwrap();
        // S must be non-negative
        assert!(CpuLinAlg::vec_get(&s, 0) >= 0.0);
        assert!((CpuLinAlg::vec_get(&s, 0) - 3.0).abs() < 1e-10);
        // Reconstruction
        let recon = reconstruct_usv(&u, &s, &v);
        assert!((CpuLinAlg::mat_get(&recon, 0, 0) - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_svd_zero_matrix() {
        let m = CpuLinAlg::zeros_mat(3, 3);
        let (_u, s, _v) = CpuLinAlg::svd(&m).unwrap();
        for i in 0..CpuLinAlg::vec_len(&s) {
            assert!(
                CpuLinAlg::vec_get(&s, i).abs() < 1e-12,
                "expected zero singular value, got {}",
                CpuLinAlg::vec_get(&s, i)
            );
        }
    }

    #[test]
    fn test_svd_repeated_singular_values() {
        // diag(4, 4, 2) → S = [4, 4, 2]
        let m = mat_from_rows(3, 3, &[4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0]);
        let (u, s, v) = CpuLinAlg::svd(&m).unwrap();
        assert!((CpuLinAlg::vec_get(&s, 0) - 4.0).abs() < 1e-10);
        assert!((CpuLinAlg::vec_get(&s, 1) - 4.0).abs() < 1e-10);
        assert!((CpuLinAlg::vec_get(&s, 2) - 2.0).abs() < 1e-10);

        let recon = reconstruct_usv(&u, &s, &v);
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (CpuLinAlg::mat_get(&recon, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs() < 1e-10,
                    "reconstruction mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn test_svd_16x16_reconstruction() {
        // Deterministic 16×16 matrix
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let m = CpuLinAlg::xavier_mat(16, 16, &mut rng);
        let (u, s, v) = CpuLinAlg::svd(&m).unwrap();

        let recon = reconstruct_usv(&u, &s, &v);
        for r in 0..16 {
            for c in 0..16 {
                assert!(
                    (CpuLinAlg::mat_get(&recon, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs() < 1e-8,
                    "reconstruction mismatch at ({r},{c}): got {} expected {}",
                    CpuLinAlg::mat_get(&recon, r, c),
                    CpuLinAlg::mat_get(&m, r, c)
                );
            }
        }
    }

    // ── Fix #4: SVD returns Result ──────────────────────────────

    #[test]
    fn test_svd_returns_ok_for_valid_matrix() {
        let m = mat_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let result = CpuLinAlg::svd(&m);
        assert!(result.is_ok(), "SVD of valid matrix should return Ok");
        let (u, s, v) = result.unwrap();
        assert_eq!(CpuLinAlg::vec_len(&s), 3);
        assert_eq!(CpuLinAlg::mat_rows(&u), 3);
        assert_eq!(CpuLinAlg::mat_rows(&v), 3);
    }

    #[test]
    fn test_svd_result_reconstruction() {
        // Verify reconstruction works through Result unwrap
        let m = mat_from_rows(2, 2, &[5.0, 0.0, 0.0, 3.0]);
        let (u, s, v) = CpuLinAlg::svd(&m).unwrap();
        let recon = reconstruct_usv(&u, &s, &v);
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (CpuLinAlg::mat_get(&recon, r, c) - CpuLinAlg::mat_get(&m, r, c)).abs() < 1e-10
                );
            }
        }
    }
}
