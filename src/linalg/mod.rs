// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-29

//! Linear algebra trait abstraction for backend-agnostic compute.
//!
//! Defines the [`LinAlg`] trait that abstracts all vector and matrix
//! operations behind associated types. The default CPU implementation
//! is provided by [`cpu::CpuLinAlg`] in the [`cpu`] submodule.

pub mod cpu;
pub mod golub_kahan;

use crate::activation::Activation;
use crate::error::PcError;
use rand::Rng;
use std::fmt::Debug;

/// Result type for SVD decomposition: `Ok((U, S, V))` or convergence error.
pub type SvdResult<L> = Result<
    (
        <L as LinAlg>::Matrix,
        <L as LinAlg>::Vector,
        <L as LinAlg>::Matrix,
    ),
    PcError,
>;

/// Backend-agnostic linear algebra operations.
///
/// Implementations provide concrete vector and matrix types along with
/// all arithmetic, activation, and sampling operations needed by the
/// PC-Actor-Critic framework. All methods take `&self` to allow GPU
/// backends to carry device state (e.g., CUDA context).
///
/// # Associated Types
///
/// * `Vector` — 1-D array of `f64` values.
/// * `Matrix` — 2-D array of `f64` values (row-major).
///
/// # Backend Requirements
///
/// The `Clone + Send + Sync + 'static` bounds ensure the backend can
/// be used across threads and stored in long-lived structs. For CPU
/// backends like [`CpuLinAlg`](cpu::CpuLinAlg), this is trivially
/// satisfied (zero-sized unit struct). GPU backends holding device
/// handles (e.g., `wgpu::Device`, CUDA contexts) should wrap them in
/// `Arc` to satisfy `Clone` without duplicating the underlying resource.
///
/// # Serde Compatibility
///
/// Backends used with serde-derived structs (e.g., `Layer<L>`) must
/// implement [`Default`] because the `backend` field uses
/// `#[serde(skip, default)]`. `CpuLinAlg` implements `Default`.
/// GPU backends must also implement `Default` (e.g., via a
/// default device selection).
///
/// # Examples
///
/// ```
/// use pc_rl_core::linalg::cpu::CpuLinAlg;
/// use pc_rl_core::linalg::LinAlg;
///
/// let backend = CpuLinAlg::new();
/// let v = backend.zeros_vec(5);
/// assert_eq!(backend.vec_len(&v), 5);
/// ```
pub trait LinAlg: Clone + Send + Sync + 'static {
    /// 1-D vector type.
    type Vector: Clone + Send + Sync + Debug;
    /// 2-D matrix type.
    type Matrix: Clone + Send + Sync + Debug;

    // ── Construction ─────────────────────────────────────────────

    /// Creates a zero-filled vector of the given size.
    fn zeros_vec(&self, size: usize) -> Self::Vector;

    /// Creates a zero-filled matrix with the given dimensions.
    fn zeros_mat(&self, rows: usize, cols: usize) -> Self::Matrix;

    /// Creates a matrix with Xavier-uniform initialization.
    fn xavier_mat(&self, rows: usize, cols: usize, rng: &mut impl Rng) -> Self::Matrix;

    // ── Matrix ops ───────────────────────────────────────────────

    /// Matrix-vector multiplication: `m * v`.
    fn mat_vec_mul(&self, m: &Self::Matrix, v: &Self::Vector) -> Self::Vector;

    /// Returns the transpose of the matrix.
    fn mat_transpose(&self, m: &Self::Matrix) -> Self::Matrix;

    /// Outer product of two vectors: `a * b^T`.
    fn outer_product(&self, a: &Self::Vector, b: &Self::Vector) -> Self::Matrix;

    /// Matrix-matrix multiplication: `a * b`.
    ///
    /// Returns a matrix with dimensions `(rows_a, cols_b)`.
    /// Panics if `cols_a != rows_b`.
    fn mat_mul(&self, a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix;

    /// Singular Value Decomposition: `M = U × diag(S) × V^T`.
    ///
    /// Returns `Ok((U, S, V))` where:
    /// - `U` is `(m, k)` with orthonormal columns,
    /// - `S` is a vector of `k` non-negative singular values in descending order,
    /// - `V` is `(n, k)` with orthonormal columns,
    /// - `k = min(m, n)`.
    ///
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if the decomposition fails to converge.
    fn svd(&self, m: &Self::Matrix) -> SvdResult<Self>;

    /// Adds `scale * other` element-wise to `m` (in place).
    fn mat_scale_add(&self, m: &mut Self::Matrix, other: &Self::Matrix, scale: f64);

    /// Returns the number of rows.
    fn mat_rows(&self, m: &Self::Matrix) -> usize;

    /// Returns the number of columns.
    fn mat_cols(&self, m: &Self::Matrix) -> usize;

    /// Returns the element at `(row, col)`.
    fn mat_get(&self, m: &Self::Matrix, row: usize, col: usize) -> f64;

    /// Sets the element at `(row, col)`.
    fn mat_set(&self, m: &mut Self::Matrix, row: usize, col: usize, val: f64);

    // ── Vector ops ───────────────────────────────────────────────

    /// Element-wise addition: `a + b`.
    fn vec_add(&self, a: &Self::Vector, b: &Self::Vector) -> Self::Vector;

    /// Element-wise subtraction: `a - b`.
    fn vec_sub(&self, a: &Self::Vector, b: &Self::Vector) -> Self::Vector;

    /// Scalar multiplication: `v * s`.
    fn vec_scale(&self, v: &Self::Vector, s: f64) -> Self::Vector;

    /// Element-wise (Hadamard) product: `a[i] * b[i]`.
    fn vec_hadamard(&self, a: &Self::Vector, b: &Self::Vector) -> Self::Vector;

    /// Dot product: `sum(a[i] * b[i])`.
    fn vec_dot(&self, a: &Self::Vector, b: &Self::Vector) -> f64;

    /// Returns the number of elements in the vector.
    fn vec_len(&self, v: &Self::Vector) -> usize;

    /// Returns the element at index `i`.
    fn vec_get(&self, v: &Self::Vector, i: usize) -> f64;

    /// Sets the element at index `i`.
    fn vec_set(&self, v: &mut Self::Vector, i: usize, val: f64);

    /// Creates a vector from a slice of `f64`.
    fn vec_from_slice(&self, s: &[f64]) -> Self::Vector;

    /// Converts the vector to a `Vec<f64>` (copies data).
    fn vec_to_vec(&self, v: &Self::Vector) -> Vec<f64>;

    // ── Clip ─────────────────────────────────────────────────────

    /// Clamps each element to `[-max_abs, max_abs]` in place.
    fn clip_vec(&self, v: &mut Self::Vector, max_abs: f64);

    /// Clamps each matrix element to `[-max_abs, max_abs]` in place.
    fn clip_mat(&self, m: &mut Self::Matrix, max_abs: f64);

    // ── Activation ───────────────────────────────────────────────

    /// Applies an activation function element-wise.
    fn apply_activation(&self, v: &Self::Vector, act: Activation) -> Self::Vector;

    /// Applies an activation derivative element-wise.
    fn apply_derivative(&self, v: &Self::Vector, act: Activation) -> Self::Vector;

    // ── Softmax / sampling ───────────────────────────────────────

    /// Numerically stable masked softmax.
    fn softmax_masked(&self, logits: &Self::Vector, mask: &[usize]) -> Self::Vector;

    /// Returns the index of the maximum value among masked indices.
    fn argmax_masked(&self, values: &Self::Vector, mask: &[usize]) -> usize;

    /// Samples an action index from a probability distribution over masked indices.
    fn sample_from_probs(&self, probs: &Self::Vector, mask: &[usize], rng: &mut impl Rng)
        -> usize;

    /// Combined RMS error across multiple error vectors.
    fn rms_error(&self, error_vecs: &[&Self::Vector]) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuLinAlg;

    /// Escenario 3: Operations use &self
    #[test]
    fn test_linalg_methods_take_self() {
        let backend = CpuLinAlg::new();
        let v = backend.zeros_vec(5);
        assert_eq!(backend.vec_len(&v), 5);
    }

    /// Escenario 9: CpuLinAlg is ZST
    #[test]
    fn test_cpu_linalg_is_zst() {
        assert_eq!(std::mem::size_of::<CpuLinAlg>(), 0);
    }

    /// Escenario 4: mat_vec_mul via instance
    #[test]
    fn test_mat_vec_mul_via_instance() {
        let backend = CpuLinAlg::new();
        // Identity 2x2 * [3,4] = [3,4]
        let mut m = backend.zeros_mat(2, 2);
        backend.mat_set(&mut m, 0, 0, 1.0);
        backend.mat_set(&mut m, 1, 1, 1.0);
        let v = backend.vec_from_slice(&[3.0, 4.0]);
        let result = backend.mat_vec_mul(&m, &v);
        assert_eq!(backend.vec_get(&result, 0), 3.0);
        assert_eq!(backend.vec_get(&result, 1), 4.0);
    }

    /// Escenario 5: vec_as_slice removed from trait.
    /// This is a compile-time negative check. The real verification is that
    /// no generic code (code parameterized over L: LinAlg) can call
    /// backend.vec_as_slice(). This test documents the removal; the compiler
    /// enforces it.
    #[test]
    fn test_trait_has_31_methods() {
        // After removing vec_as_slice, the trait has 31 methods.
        // This is a documentation marker -- the real enforcement is the compiler.
    }
}
