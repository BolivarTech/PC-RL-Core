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

/// Backend-agnostic linear algebra operations.
///
/// Implementations provide concrete vector and matrix types along with
/// all arithmetic, activation, and sampling operations needed by the
/// PC-Actor-Critic framework.
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
/// # Examples
///
/// ```
/// use pc_rl_core::linalg::cpu::CpuLinAlg;
/// use pc_rl_core::linalg::LinAlg;
///
/// let v = CpuLinAlg::zeros_vec(5);
/// assert_eq!(CpuLinAlg::vec_len(&v), 5);
/// ```
/// Result type for SVD decomposition: `Ok((U, S, V))` or convergence error.
pub type SvdResult<L> = Result<
    (
        <L as LinAlg>::Matrix,
        <L as LinAlg>::Vector,
        <L as LinAlg>::Matrix,
    ),
    PcError,
>;

pub trait LinAlg: Clone + Send + Sync + 'static {
    /// 1-D vector type.
    type Vector: Clone + Send + Sync + Debug;
    /// 2-D matrix type.
    type Matrix: Clone + Send + Sync + Debug;

    // ── Construction ─────────────────────────────────────────────

    /// Creates a zero-filled vector of the given size.
    fn zeros_vec(size: usize) -> Self::Vector;

    /// Creates a zero-filled matrix with the given dimensions.
    fn zeros_mat(rows: usize, cols: usize) -> Self::Matrix;

    /// Creates a matrix with Xavier-uniform initialization.
    fn xavier_mat(rows: usize, cols: usize, rng: &mut impl Rng) -> Self::Matrix;

    // ── Matrix ops ───────────────────────────────────────────────

    /// Matrix-vector multiplication: `m * v`.
    fn mat_vec_mul(m: &Self::Matrix, v: &Self::Vector) -> Self::Vector;

    /// Returns the transpose of the matrix.
    fn mat_transpose(m: &Self::Matrix) -> Self::Matrix;

    /// Outer product of two vectors: `a * b^T`.
    fn outer_product(a: &Self::Vector, b: &Self::Vector) -> Self::Matrix;

    /// Matrix-matrix multiplication: `a * b`.
    ///
    /// Returns a matrix with dimensions `(rows_a, cols_b)`.
    /// Panics if `cols_a != rows_b`.
    fn mat_mul(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix;

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
    fn svd(m: &Self::Matrix) -> SvdResult<Self>;

    /// Adds `scale * other` element-wise to `m` (in place).
    fn mat_scale_add(m: &mut Self::Matrix, other: &Self::Matrix, scale: f64);

    /// Returns the number of rows.
    fn mat_rows(m: &Self::Matrix) -> usize;

    /// Returns the number of columns.
    fn mat_cols(m: &Self::Matrix) -> usize;

    /// Returns the element at `(row, col)`.
    fn mat_get(m: &Self::Matrix, row: usize, col: usize) -> f64;

    /// Sets the element at `(row, col)`.
    fn mat_set(m: &mut Self::Matrix, row: usize, col: usize, val: f64);

    // ── Vector ops ───────────────────────────────────────────────

    /// Element-wise addition: `a + b`.
    fn vec_add(a: &Self::Vector, b: &Self::Vector) -> Self::Vector;

    /// Element-wise subtraction: `a - b`.
    fn vec_sub(a: &Self::Vector, b: &Self::Vector) -> Self::Vector;

    /// Scalar multiplication: `v * s`.
    fn vec_scale(v: &Self::Vector, s: f64) -> Self::Vector;

    /// Element-wise (Hadamard) product: `a[i] * b[i]`.
    fn vec_hadamard(a: &Self::Vector, b: &Self::Vector) -> Self::Vector;

    /// Dot product: `sum(a[i] * b[i])`.
    fn vec_dot(a: &Self::Vector, b: &Self::Vector) -> f64;

    /// Returns the number of elements in the vector.
    fn vec_len(v: &Self::Vector) -> usize;

    /// Returns the element at index `i`.
    fn vec_get(v: &Self::Vector, i: usize) -> f64;

    /// Sets the element at index `i`.
    fn vec_set(v: &mut Self::Vector, i: usize, val: f64);

    /// Creates a vector from a slice of `f64`.
    fn vec_from_slice(s: &[f64]) -> Self::Vector;

    /// Converts the vector to a `Vec<f64>` (copies data).
    fn vec_to_vec(v: &Self::Vector) -> Vec<f64>;

    /// Returns a slice view of the vector data (zero-copy for CPU).
    ///
    /// # CPU-only
    ///
    /// This method assumes the vector data lives in host memory. GPU backends
    /// that store data in device memory cannot implement this without a copy.
    /// Prefer [`vec_to_vec`](LinAlg::vec_to_vec) in generic code that must
    /// work across backends. This method exists as an optimization escape
    /// hatch when the caller knows the backend is CPU.
    fn vec_as_slice(v: &Self::Vector) -> &[f64];

    // ── Clip ─────────────────────────────────────────────────────

    /// Clamps each element to `[-max_abs, max_abs]` in place.
    fn clip_vec(v: &mut Self::Vector, max_abs: f64);

    /// Clamps each matrix element to `[-max_abs, max_abs]` in place.
    fn clip_mat(m: &mut Self::Matrix, max_abs: f64);

    // ── Activation ───────────────────────────────────────────────

    /// Applies an activation function element-wise.
    fn apply_activation(v: &Self::Vector, act: Activation) -> Self::Vector;

    /// Applies an activation derivative element-wise.
    fn apply_derivative(v: &Self::Vector, act: Activation) -> Self::Vector;

    // ── Softmax / sampling ───────────────────────────────────────

    /// Numerically stable masked softmax.
    fn softmax_masked(logits: &Self::Vector, mask: &[usize]) -> Self::Vector;

    /// Returns the index of the maximum value among masked indices.
    fn argmax_masked(values: &Self::Vector, mask: &[usize]) -> usize;

    /// Samples an action index from a probability distribution over masked indices.
    fn sample_from_probs(probs: &Self::Vector, mask: &[usize], rng: &mut impl Rng) -> usize;

    /// Combined RMS error across multiple error vectors.
    fn rms_error(error_vecs: &[&Self::Vector]) -> f64;
}
