// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-04

//! Golub-Kahan bidiagonalization SVD algorithm.
//!
//! Provides an O(n^3) SVD implementation as an upgrade path from the
//! existing Jacobi eigendecomposition approach in [`CpuLinAlg`](super::cpu::CpuLinAlg).
//!
//! # Overview
//!
//! The algorithm proceeds in two phases:
//! 1. **Householder bidiagonalization** reduces the input matrix to
//!    upper bidiagonal form.
//! 2. **Implicit QR iteration** with Wilkinson shift extracts singular
//!    values and vectors from the bidiagonal matrix.
//!
//! # Examples
//!
//! ```
//! use pc_rl_core::linalg::golub_kahan::GolubKahanSvd;
//!
//! let svd = GolubKahanSvd::new()
//!     .with_tolerance(1e-12)
//!     .with_max_iter_factor(40);
//! assert!((svd.tol - 1e-12).abs() < f64::EPSILON);
//! assert_eq!(svd.max_iter_factor, 40);
//! ```

use std::fmt;

use crate::error::PcError;
use crate::matrix::Matrix;

/// Error type for SVD decomposition failures.
///
/// Converts to [`PcError::ConfigValidation`] via the [`From`] impl,
/// preserving the error message for upstream callers.
///
/// # Examples
///
/// ```
/// use pc_rl_core::linalg::golub_kahan::SvdError;
///
/// let err = SvdError::Convergence { size: 10, iterations: 300 };
/// assert!(format!("{err}").contains("10"));
/// ```
#[derive(Debug)]
pub enum SvdError {
    /// The iterative solver did not converge within the allowed iterations.
    Convergence {
        /// Matrix dimension (min(rows, cols)) that failed to converge.
        size: usize,
        /// Total iterations attempted before giving up.
        iterations: usize,
    },
    /// The input matrix is invalid (e.g., contains NaN or Inf).
    InvalidInput {
        /// Human-readable description of the problem.
        reason: String,
    },
}

impl fmt::Display for SvdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SvdError::Convergence { size, iterations } => {
                write!(
                    f,
                    "SVD failed to converge for matrix of size {size} \
                     after {iterations} iterations"
                )
            }
            SvdError::InvalidInput { reason } => {
                write!(f, "SVD invalid input: {reason}")
            }
        }
    }
}

impl std::error::Error for SvdError {}

impl From<SvdError> for PcError {
    fn from(e: SvdError) -> Self {
        PcError::ConfigValidation(e.to_string())
    }
}

/// Golub-Kahan bidiagonalization SVD solver.
///
/// Decomposes a matrix `A` into `U * diag(S) * V^T` using Householder
/// bidiagonalization followed by implicit QR iteration with Wilkinson shift.
///
/// # Fields
///
/// * `tol` - Convergence tolerance for off-diagonal elements (default: 1e-14).
/// * `max_iter_factor` - Maximum iterations = `factor * n` where `n = min(rows, cols)` (default: 30).
///
/// # Examples
///
/// ```
/// use pc_rl_core::linalg::golub_kahan::GolubKahanSvd;
///
/// let svd = GolubKahanSvd::new();
/// assert!((svd.tol - 1e-14).abs() < f64::EPSILON);
/// assert_eq!(svd.max_iter_factor, 30);
/// ```
#[derive(Debug, Clone)]
pub struct GolubKahanSvd {
    /// Convergence tolerance for off-diagonal elements.
    pub(crate) tol: f64,
    /// Maximum iterations as a multiple of `min(rows, cols)`.
    pub(crate) max_iter_factor: usize,
}

impl GolubKahanSvd {
    /// Creates a new solver with default parameters.
    ///
    /// Defaults: `tol = 1e-14`, `max_iter_factor = 30`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pc_rl_core::linalg::golub_kahan::GolubKahanSvd;
    ///
    /// let svd = GolubKahanSvd::new();
    /// assert!((svd.tol - 1e-14).abs() < f64::EPSILON);
    /// ```
    pub fn new() -> Self {
        Self {
            tol: 1e-14,
            max_iter_factor: 30,
        }
    }

    /// Sets a custom convergence tolerance.
    ///
    /// # Arguments
    ///
    /// * `tol` - The convergence threshold for off-diagonal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use pc_rl_core::linalg::golub_kahan::GolubKahanSvd;
    ///
    /// let svd = GolubKahanSvd::new().with_tolerance(1e-8);
    /// assert!((svd.tol - 1e-8).abs() < f64::EPSILON);
    /// ```
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets a custom maximum iteration factor.
    ///
    /// Total maximum iterations will be `factor * min(rows, cols)`.
    ///
    /// # Arguments
    ///
    /// * `factor` - Multiplier for the iteration limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use pc_rl_core::linalg::golub_kahan::GolubKahanSvd;
    ///
    /// let svd = GolubKahanSvd::new().with_max_iter_factor(50);
    /// assert_eq!(svd.max_iter_factor, 50);
    /// ```
    pub fn with_max_iter_factor(mut self, factor: usize) -> Self {
        self.max_iter_factor = factor;
        self
    }

    /// Computes the SVD of matrix `a`: `A = U * diag(S) * V^T`.
    ///
    /// Returns `(U, S, V)` where:
    /// - `U` is `(m, k)` with orthonormal columns,
    /// - `S` is a `Vec<f64>` of `k` non-negative singular values in descending order,
    /// - `V` is `(n, k)` with orthonormal columns,
    /// - `k = min(m, n)`.
    ///
    /// # Arguments
    ///
    /// * `a` - The input matrix to decompose.
    ///
    /// # Errors
    ///
    /// Returns [`SvdError::InvalidInput`] if the matrix contains NaN or Inf values.
    /// Returns [`SvdError::Convergence`] if the iterative solver does not converge.
    ///
    /// # Examples
    ///
    /// ```
    /// use pc_rl_core::linalg::golub_kahan::GolubKahanSvd;
    /// use pc_rl_core::matrix::Matrix;
    ///
    /// let svd = GolubKahanSvd::new();
    /// let empty = Matrix::zeros(0, 0);
    /// let (u, s, v) = svd.compute(&empty).unwrap();
    /// assert_eq!(s.len(), 0);
    /// ```
    pub fn compute(&self, a: &Matrix) -> Result<(Matrix, Vec<f64>, Matrix), SvdError> {
        // Validate: reject NaN/Inf
        for &val in &a.data {
            if val.is_nan() || val.is_infinite() {
                return Err(SvdError::InvalidInput {
                    reason: "matrix contains NaN or Inf".to_string(),
                });
            }
        }

        let m = a.rows;
        let n = a.cols;

        // Handle empty matrix
        if m == 0 || n == 0 {
            return Ok((Matrix::zeros(m, 0), Vec::new(), Matrix::zeros(n, 0)));
        }

        // If wide (m < n), transpose and swap U/V at the end
        let transposed = m < n;
        let (work_m, work_n, work_data) = if transposed {
            let mut t = vec![0.0; m * n];
            for r in 0..m {
                for c in 0..n {
                    t[c * m + r] = a.data[r * n + c];
                }
            }
            (n, m, t)
        } else {
            (m, n, a.data.clone())
        };

        // k = min(work_m, work_n) = work_n since work_m >= work_n
        let k = work_n;

        // Handle 1x1 case directly
        if k == 1 {
            // For a single-column tall matrix, SVD is: u = a/||a||, s = ||a||, v = [1]
            let norm: f64 = work_data.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < self.tol {
                let u_mat = make_identity(work_m, 1);
                let v_mat = make_identity(1, 1);
                return Self::finalize(u_mat, vec![0.0], v_mat, work_m, work_n, transposed);
            }
            let sign = if work_data[0] >= 0.0 { 1.0 } else { -1.0 };
            let mut u_data = vec![0.0; work_m];
            for i in 0..work_m {
                u_data[i] = work_data[i] * sign / norm;
            }
            let u_mat = Matrix {
                data: u_data,
                rows: work_m,
                cols: 1,
            };
            let v_mat = Matrix {
                data: vec![sign],
                rows: 1,
                cols: 1,
            };
            return Self::finalize(u_mat, vec![norm], v_mat, work_m, work_n, transposed);
        }

        // Phase 1: Householder bidiagonalization
        // Work on a copy. After this: A is overwritten, diag and superdiag extracted.
        let mut w = work_data;
        let mut u_acc = make_identity(work_m, work_m);
        let mut v_acc = make_identity(work_n, work_n);

        let mut diag = vec![0.0; k];
        let mut superdiag = vec![0.0; k.saturating_sub(1)];

        householder_bidiag(
            &mut w,
            work_m,
            work_n,
            &mut u_acc,
            &mut v_acc,
            &mut diag,
            &mut superdiag,
        );

        // Phase 2: Implicit QR iteration on bidiagonal
        let max_iter = self.max_iter_factor * k * k;
        implicit_qr_svd(
            &mut diag,
            &mut superdiag,
            &mut u_acc,
            &mut v_acc,
            work_m,
            work_n,
            k,
            self.tol,
            max_iter,
        )?;

        // Phase 3: Make singular values non-negative, sort descending
        for (i, d) in diag.iter_mut().enumerate().take(k) {
            if *d < 0.0 {
                *d = -*d;
                // Flip corresponding column of U
                for r in 0..work_m {
                    u_acc.data[r * work_m + i] = -u_acc.data[r * work_m + i];
                }
            }
        }

        // Sort by descending singular value
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a_idx, &b_idx| {
            diag[b_idx]
                .partial_cmp(&diag[a_idx])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_s: Vec<f64> = indices.iter().map(|&i| diag[i]).collect();

        // Extract thin U (work_m x k) with sorted columns
        let mut u_thin = Matrix::zeros(work_m, k);
        for (new_col, &old_col) in indices.iter().enumerate() {
            for r in 0..work_m {
                u_thin.data[r * k + new_col] = u_acc.data[r * work_m + old_col];
            }
        }

        // Extract thin V (work_n x k) with sorted columns
        let mut v_thin = Matrix::zeros(work_n, k);
        for (new_col, &old_col) in indices.iter().enumerate() {
            for r in 0..work_n {
                v_thin.data[r * k + new_col] = v_acc.data[r * work_n + old_col];
            }
        }

        Self::finalize(u_thin, sorted_s, v_thin, work_m, work_n, transposed)
    }

    /// Swaps U and V if the input was transposed, returning the final result.
    ///
    /// # Arguments
    ///
    /// * `u` - Left singular vectors.
    /// * `s` - Singular values.
    /// * `v` - Right singular vectors.
    /// * `work_m` - Working row count (after possible transpose).
    /// * `work_n` - Working column count (after possible transpose).
    /// * `transposed` - Whether the input was transposed.
    fn finalize(
        u: Matrix,
        s: Vec<f64>,
        v: Matrix,
        _work_m: usize,
        _work_n: usize,
        transposed: bool,
    ) -> Result<(Matrix, Vec<f64>, Matrix), SvdError> {
        if transposed {
            Ok((v, s, u))
        } else {
            Ok((u, s, v))
        }
    }
}

/// Creates an identity-like matrix of size `rows x cols`.
///
/// Places 1.0 on the diagonal up to `min(rows, cols)`.
fn make_identity(rows: usize, cols: usize) -> Matrix {
    let mut data = vec![0.0; rows * cols];
    let k = rows.min(cols);
    for i in 0..k {
        data[i * cols + i] = 1.0;
    }
    Matrix { data, rows, cols }
}

/// Householder bidiagonalization: reduces an m x n matrix (m >= n)
/// to upper bidiagonal form, accumulating transformations into U and V.
///
/// After completion, `diag` contains the diagonal and `superdiag` the
/// superdiagonal of the bidiagonal matrix B such that A = U * B * V^T.
///
/// # Arguments
///
/// * `w` - Row-major m x n matrix data, overwritten during computation.
/// * `m` - Number of rows.
/// * `n` - Number of columns.
/// * `u_acc` - m x m identity matrix, accumulates left reflections.
/// * `v_acc` - n x n identity matrix, accumulates right reflections.
/// * `diag` - Output diagonal of bidiagonal B (length n).
/// * `superdiag` - Output superdiagonal of bidiagonal B (length n-1).
fn householder_bidiag(
    w: &mut [f64],
    m: usize,
    n: usize,
    u_acc: &mut Matrix,
    v_acc: &mut Matrix,
    diag: &mut [f64],
    superdiag: &mut [f64],
) {
    for j in 0..n {
        // Left Householder: zero below w[j][j] in column j
        {
            let mut col = vec![0.0; m - j];
            for i in j..m {
                col[i - j] = w[i * n + j];
            }
            let (v_house, beta) = householder_vector(&col);
            if beta != 0.0 {
                // Apply H = I - beta * v * v^T to w[j..m, j..n]
                apply_householder_left(w, m, n, j, j, &v_house, beta);
                // Accumulate into U: U = U * H (apply on right to u_acc)
                apply_householder_right_to_matrix(u_acc, m, m, j, &v_house, beta);
            }
        }
        diag[j] = w[j * n + j];

        // Right Householder: zero beyond w[j][j+1] in row j
        if j + 2 <= n {
            let start = j + 1;
            let mut row = vec![0.0; n - start];
            for c in start..n {
                row[c - start] = w[j * n + c];
            }
            let (v_house, beta) = householder_vector(&row);
            if beta != 0.0 {
                // Apply H = I - beta * v * v^T to w[j..m, (j+1)..n] on the right
                apply_householder_right(w, m, n, j, start, &v_house, beta);
                // Accumulate into V: V = V * H
                apply_householder_right_to_matrix(v_acc, n, n, start, &v_house, beta);
            }
            if j < n - 1 {
                superdiag[j] = w[j * n + j + 1];
            }
        } else if j < n - 1 {
            superdiag[j] = w[j * n + j + 1];
        }
    }
}

/// Computes a Householder vector `v` and scalar `beta` such that
/// `(I - beta * v * v^T) * x` zeroes all elements below the first.
///
/// Returns `(v, beta)` where `v[0] = 1.0`.
///
/// # Arguments
///
/// * `x` - Input vector to reflect.
fn householder_vector(x: &[f64]) -> (Vec<f64>, f64) {
    let len = x.len();
    if len == 0 {
        return (Vec::new(), 0.0);
    }
    if len == 1 {
        return (vec![1.0], 0.0);
    }

    let mut sigma = 0.0;
    for &xi in &x[1..] {
        sigma += xi * xi;
    }

    let mut v = vec![0.0; len];
    v[0] = 1.0;
    v[1..len].copy_from_slice(&x[1..len]);

    if sigma < 1e-300 {
        return (v, 0.0);
    }

    let norm_x = (x[0] * x[0] + sigma).sqrt();
    // Choose sign to avoid cancellation
    if x[0] <= 0.0 {
        v[0] = x[0] - norm_x;
    } else {
        v[0] = -sigma / (x[0] + norm_x);
    }

    let beta = 2.0 * v[0] * v[0] / (sigma + v[0] * v[0]);
    // Normalize so v[0] = 1
    let v0 = v[0];
    for vi in v.iter_mut() {
        *vi /= v0;
    }

    (v, beta)
}

/// Applies a left Householder reflection: w[row_start..m, col_start..n] =
/// (I - beta * v * v^T) * w[row_start..m, col_start..n].
fn apply_householder_left(
    w: &mut [f64],
    _m: usize,
    n: usize,
    row_start: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
) {
    let v_len = v.len();
    let num_cols = n - col_start;
    // Compute p = beta * v^T * W_sub for each column
    let mut p = vec![0.0; num_cols];
    for (vi_idx, &vi) in v.iter().enumerate().take(v_len) {
        let row = row_start + vi_idx;
        for c in 0..num_cols {
            p[c] += vi * w[row * n + col_start + c];
        }
    }
    // W_sub -= v * p^T
    for (vi_idx, &vi) in v.iter().enumerate().take(v_len) {
        let row = row_start + vi_idx;
        for c in 0..num_cols {
            w[row * n + col_start + c] -= beta * vi * p[c];
        }
    }
}

/// Applies a right Householder reflection: w[row_start..m, col_start..n] =
/// w[row_start..m, col_start..n] * (I - beta * v * v^T).
fn apply_householder_right(
    w: &mut [f64],
    m: usize,
    n: usize,
    row_start: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
) {
    let v_len = v.len();
    let num_rows = m - row_start;
    // For each row, compute dot = row . v, then row -= beta * dot * v
    for ri in 0..num_rows {
        let row = row_start + ri;
        let mut dot = 0.0;
        for (vi_idx, &vi) in v.iter().enumerate().take(v_len) {
            dot += w[row * n + col_start + vi_idx] * vi;
        }
        for (vi_idx, &vi) in v.iter().enumerate().take(v_len) {
            w[row * n + col_start + vi_idx] -= beta * dot * vi;
        }
    }
}

/// Applies a Householder reflection on the right to a full accumulator matrix:
/// `acc = acc * H` where H operates on rows `start..start+v.len()`.
///
/// Equivalent to: for each row of acc, update columns start..start+v.len().
fn apply_householder_right_to_matrix(
    acc: &mut Matrix,
    rows: usize,
    cols: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
) {
    let v_len = v.len();
    for r in 0..rows {
        let mut dot = 0.0;
        for (vi_idx, &vi) in v.iter().enumerate().take(v_len) {
            dot += acc.data[r * cols + col_start + vi_idx] * vi;
        }
        for (vi_idx, &vi) in v.iter().enumerate().take(v_len) {
            acc.data[r * cols + col_start + vi_idx] -= beta * dot * vi;
        }
    }
}

/// Computes a Givens rotation `(c, s)` such that
/// `[c s; -s c]^T * [a; b] = [r; 0]`.
///
/// # Arguments
///
/// * `a` - First element.
/// * `b` - Second element (to be zeroed).
///
/// # Returns
///
/// `(c, s)` cosine and sine of the rotation angle.
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-300 {
        return (1.0, 0.0);
    }
    if a.abs() < 1e-300 {
        return (0.0, b.signum());
    }
    if b.abs() > a.abs() {
        let tau = a / b;
        let s = (1.0 + tau * tau).sqrt().recip() * b.signum();
        let c = s * tau;
        (c, s)
    } else {
        let tau = b / a;
        let c = (1.0 + tau * tau).sqrt().recip() * a.signum();
        let s = c * tau;
        (c, s)
    }
}

/// Applies a Givens rotation to columns `i` and `j` of a matrix,
/// operating on all rows: `[col_i, col_j] = [col_i, col_j] * G`.
fn apply_givens_cols(
    mat: &mut Matrix,
    rows: usize,
    stride: usize,
    i: usize,
    j: usize,
    c: f64,
    s: f64,
) {
    for r in 0..rows {
        let a = mat.data[r * stride + i];
        let b = mat.data[r * stride + j];
        mat.data[r * stride + i] = c * a + s * b;
        mat.data[r * stride + j] = -s * a + c * b;
    }
}

/// Implicit QR iteration on a bidiagonal matrix to extract singular values.
///
/// Operates on the diagonal `diag` and superdiagonal `superdiag`, applying
/// Givens rotations accumulated into `u_acc` and `v_acc`.
///
/// # Errors
///
/// Returns [`SvdError::Convergence`] if the iteration does not converge
/// within `max_iter` total sweeps.
#[allow(clippy::too_many_arguments)]
fn implicit_qr_svd(
    diag: &mut [f64],
    superdiag: &mut [f64],
    u_acc: &mut Matrix,
    v_acc: &mut Matrix,
    u_rows: usize,
    v_rows: usize,
    k: usize,
    tol: f64,
    max_iter: usize,
) -> Result<(), SvdError> {
    if k <= 1 {
        return Ok(());
    }

    let mut iter_count = 0usize;

    loop {
        // Find the largest q such that B(k-q-1:k-1, k-q-1:k-1) is diagonal
        // (i.e., superdiag entries from end that are zero)
        let mut q = 0usize;
        while q < k - 1 {
            let idx = k - 2 - q;
            let thresh = tol * (diag[idx].abs() + diag[idx + 1].abs());
            if superdiag[idx].abs() <= thresh.max(tol * 1e-2) {
                superdiag[idx] = 0.0;
                q += 1;
            } else {
                break;
            }
        }

        if q >= k - 1 {
            // All superdiagonal elements are zero — converged
            break;
        }

        // Active block is diag[p..k-q], superdiag[p..k-q-1]
        let block_end = k - q; // exclusive index into diag
                               // Find p: largest index such that superdiag[p-1] is zero (or p=0)
        let mut p = block_end - 1;
        while p > 0 {
            let idx = p - 1;
            let thresh = tol * (diag[idx].abs() + diag[idx + 1].abs());
            if superdiag[idx].abs() <= thresh.max(tol * 1e-2) {
                superdiag[idx] = 0.0;
                break;
            }
            p -= 1;
        }

        let block_size = block_end - p;
        if block_size <= 1 {
            continue;
        }

        // Check for zero diagonal entry in the block — if found, zero the
        // superdiagonal by rotation and restart
        let mut found_zero_diag = false;
        for i in p..block_end {
            if diag[i].abs() < tol * 1e-2 {
                // Zero out superdiag element adjacent to this zero diagonal
                if i < block_end - 1 && superdiag[i].abs() > 0.0 {
                    zero_superdiag_row(diag, superdiag, u_acc, u_rows, i, block_end);
                } else if i > p && superdiag[i - 1].abs() > 0.0 {
                    zero_superdiag_col(diag, superdiag, v_acc, v_rows, i, p);
                }
                found_zero_diag = true;
                break;
            }
        }
        if found_zero_diag {
            iter_count += 1;
            if iter_count > max_iter {
                return Err(SvdError::Convergence {
                    size: k,
                    iterations: max_iter,
                });
            }
            continue;
        }

        // Wilkinson shift from trailing 2x2 of B^T * B
        let n1 = block_end - 1;
        let n2 = block_end - 2;
        let d_n1 = diag[n1];
        let d_n2 = diag[n2];
        let e_n2 = superdiag[n2];
        // T = B^T B trailing 2x2:
        // [d_n2^2 + e_{n2-1}^2,   d_n2 * e_n2          ]
        // [d_n2 * e_n2,            d_n1^2 + e_n2^2      ]
        let e_n3_sq = if n2 > p {
            superdiag[n2 - 1] * superdiag[n2 - 1]
        } else {
            0.0
        };
        let t11 = d_n2 * d_n2 + e_n3_sq;
        let t12 = d_n2 * e_n2;
        let t22 = d_n1 * d_n1 + e_n2 * e_n2;

        let shift = wilkinson_shift(t11, t12, t22);

        // Golub-Kahan SVD step (bulge chase)
        golub_kahan_step(
            diag, superdiag, u_acc, v_acc, u_rows, v_rows, p, block_end, shift,
        );

        iter_count += 1;
        if iter_count > max_iter {
            return Err(SvdError::Convergence {
                size: k,
                iterations: max_iter,
            });
        }
    }

    Ok(())
}

/// Computes the Wilkinson shift for the trailing 2x2 of B^T * B.
///
/// Given the 2x2 matrix `[[a, b], [b, d]]`, returns the eigenvalue
/// closest to `d`.
fn wilkinson_shift(a: f64, b: f64, d: f64) -> f64 {
    let delta = (a - d) * 0.5;
    if delta.abs() < 1e-300 && b.abs() < 1e-300 {
        return d;
    }
    let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
    d - b * b / (delta + sign * (delta * delta + b * b).sqrt())
}

/// Performs one implicit QR step (bulge chase) on the bidiagonal matrix.
///
/// This is the core of the Golub-Kahan SVD iteration (Algorithm 8.6.2
/// from Golub & Van Loan).
#[allow(clippy::too_many_arguments)]
fn golub_kahan_step(
    diag: &mut [f64],
    superdiag: &mut [f64],
    u_acc: &mut Matrix,
    v_acc: &mut Matrix,
    u_rows: usize,
    v_rows: usize,
    p: usize,
    block_end: usize,
    shift: f64,
) {
    let mut y = diag[p] * diag[p] - shift;
    let mut z = diag[p] * superdiag[p];

    for i in p..block_end - 1 {
        // Right Givens rotation to zero z (applied to V)
        let (c, s) = givens_rotation(y, z);
        if i > p {
            superdiag[i - 1] = c * superdiag[i - 1] + s * z;
            // Note: the component that was z is now zero
        }
        let old_d_i = diag[i];
        let old_e_i = superdiag[i];
        diag[i] = c * old_d_i + s * old_e_i;
        superdiag[i] = -s * old_d_i + c * old_e_i;
        let old_d_i1 = diag[i + 1];
        z = s * old_d_i1;
        diag[i + 1] = c * old_d_i1;

        // Accumulate into V
        apply_givens_cols(v_acc, v_rows, v_rows, i, i + 1, c, s);

        // Left Givens rotation to zero z (applied to U)
        let (c, s) = givens_rotation(diag[i], z);
        diag[i] = c * diag[i] + s * z;
        let old_e_i = superdiag[i];
        let old_d_i1 = diag[i + 1];
        superdiag[i] = c * old_e_i + s * old_d_i1;
        diag[i + 1] = -s * old_e_i + c * old_d_i1;
        if i + 1 < block_end - 1 {
            let old_e_i1 = superdiag[i + 1];
            z = s * old_e_i1;
            superdiag[i + 1] = c * old_e_i1;
        }
        y = superdiag[i];

        // Accumulate into U
        apply_givens_cols(u_acc, u_rows, u_rows, i, i + 1, c, s);
    }
}

/// Zeros a superdiagonal element when a diagonal entry is zero,
/// chasing the bulge rightward via left rotations.
///
/// When `diag[zero_idx] ≈ 0`, the left rotation `G^T * B` between rows
/// `zero_idx` and `j+1` zeros the fill-in but creates a new bulge from
/// `superdiag[j+1]`. The bulge is tracked explicitly and passed forward
/// to the next iteration rather than stored back into `superdiag`.
fn zero_superdiag_row(
    diag: &mut [f64],
    superdiag: &mut [f64],
    u_acc: &mut Matrix,
    u_rows: usize,
    zero_idx: usize,
    block_end: usize,
) {
    let mut bulge = superdiag[zero_idx];
    superdiag[zero_idx] = 0.0;

    for j in zero_idx..block_end - 1 {
        let (c, s) = givens_rotation(diag[j + 1], bulge);
        diag[j + 1] = c * diag[j + 1] + s * bulge;
        // bulge position is now zero by construction
        if j + 1 < block_end - 1 {
            let old_e = superdiag[j + 1];
            superdiag[j + 1] = c * old_e;
            bulge = -s * old_e;
            apply_givens_cols(u_acc, u_rows, u_rows, j + 1, zero_idx, c, s);
            if bulge.abs() < 1e-300 {
                break;
            }
        } else {
            apply_givens_cols(u_acc, u_rows, u_rows, j + 1, zero_idx, c, s);
        }
    }
}

/// Zeros a superdiagonal element when a diagonal entry is zero,
/// chasing the bulge leftward via right rotations.
///
/// When `diag[zero_idx] ≈ 0`, the right rotation `B * G` between columns
/// `zero_idx` and `j` zeros the fill-in but creates a new bulge from
/// `superdiag[j-1]`. The bulge is tracked explicitly and passed backward
/// to the next iteration rather than stored back into `superdiag`.
fn zero_superdiag_col(
    diag: &mut [f64],
    superdiag: &mut [f64],
    v_acc: &mut Matrix,
    v_rows: usize,
    zero_idx: usize,
    block_start: usize,
) {
    let mut bulge = superdiag[zero_idx - 1];
    superdiag[zero_idx - 1] = 0.0;

    for j in (block_start..zero_idx).rev() {
        let (c, s) = givens_rotation(diag[j], bulge);
        diag[j] = c * diag[j] + s * bulge;
        // bulge position is now zero by construction
        if j > block_start {
            let old_e = superdiag[j - 1];
            superdiag[j - 1] = c * old_e;
            bulge = -s * old_e;
            apply_givens_cols(v_acc, v_rows, v_rows, j, zero_idx, c, s);
            if bulge.abs() < 1e-300 {
                break;
            }
        } else {
            apply_givens_cols(v_acc, v_rows, v_rows, j, zero_idx, c, s);
        }
    }
}

impl Default for GolubKahanSvd {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_returns_default_parameters() {
        let svd = GolubKahanSvd::new();
        assert!((svd.tol - 1e-14).abs() < f64::EPSILON);
        assert_eq!(svd.max_iter_factor, 30);
    }

    #[test]
    fn test_with_tolerance_sets_custom_tol() {
        let svd = GolubKahanSvd::new().with_tolerance(1e-8);
        assert!((svd.tol - 1e-8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_max_iter_factor_sets_custom_factor() {
        let svd = GolubKahanSvd::new().with_max_iter_factor(50);
        assert_eq!(svd.max_iter_factor, 50);
    }

    #[test]
    fn test_default_trait_matches_new() {
        let a = GolubKahanSvd::new();
        let b = GolubKahanSvd::default();
        assert!((a.tol - b.tol).abs() < f64::EPSILON);
        assert_eq!(a.max_iter_factor, b.max_iter_factor);
    }

    #[test]
    fn test_svd_error_display() {
        let err = SvdError::Convergence {
            size: 10,
            iterations: 300,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains("300"));
    }

    #[test]
    fn test_svd_error_converts_to_pc_error() {
        let err = SvdError::Convergence {
            size: 5,
            iterations: 150,
        };
        let pc_err: crate::error::PcError = err.into();
        assert!(matches!(pc_err, crate::error::PcError::ConfigValidation(_)));
    }

    #[test]
    fn test_empty_matrix() {
        // B6: 0x0 matrix returns empty results
        let a = Matrix::zeros(0, 0);
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_eq!(u.rows, 0);
        assert_eq!(u.cols, 0);
        assert!(s.is_empty());
        assert_eq!(v.rows, 0);
        assert_eq!(v.cols, 0);
    }

    #[test]
    fn test_nan_input_returns_error() {
        // B13: NaN input returns Err
        let a = Matrix {
            data: vec![1.0, f64::NAN, 3.0, 4.0],
            rows: 2,
            cols: 2,
        };
        let result = GolubKahanSvd::new().compute(&a);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SvdError::InvalidInput { .. }));
    }

    #[test]
    fn test_inf_input_returns_error() {
        // B14: Inf input returns Err
        let a = Matrix {
            data: vec![1.0, f64::INFINITY, 3.0, 4.0],
            rows: 2,
            cols: 2,
        };
        let result = GolubKahanSvd::new().compute(&a);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SvdError::InvalidInput { .. }));
    }

    #[test]
    fn test_neg_inf_input_returns_error() {
        let a = Matrix {
            data: vec![f64::NEG_INFINITY, 2.0, 3.0, 4.0],
            rows: 2,
            cols: 2,
        };
        let result = GolubKahanSvd::new().compute(&a);
        assert!(result.is_err());
    }

    // ---- Helper functions for SVD correctness tests ----

    /// Helper: multiply matrices using raw data.
    fn mat_mul_raw(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.cols, b.rows);
        let mut c = Matrix::zeros(a.rows, b.cols);
        for i in 0..a.rows {
            for k in 0..a.cols {
                let aik = a.data[i * a.cols + k];
                for j in 0..b.cols {
                    c.data[i * c.cols + j] += aik * b.data[k * b.cols + j];
                }
            }
        }
        c
    }

    fn transpose_raw(a: &Matrix) -> Matrix {
        let mut t = Matrix::zeros(a.cols, a.rows);
        for r in 0..a.rows {
            for c in 0..a.cols {
                t.data[c * t.cols + r] = a.data[r * a.cols + c];
            }
        }
        t
    }

    fn assert_reconstruction(a: &Matrix, u: &Matrix, s: &[f64], v: &Matrix, tol: f64) {
        let k = s.len();
        let mut diag_s = Matrix::zeros(k, k);
        for (i, &si) in s.iter().enumerate() {
            diag_s.data[i * k + i] = si;
        }
        let us = mat_mul_raw(u, &diag_s);
        let recon = mat_mul_raw(&us, &transpose_raw(v));
        for r in 0..a.rows {
            for c in 0..a.cols {
                let diff = (recon.data[r * recon.cols + c] - a.data[r * a.cols + c]).abs();
                assert!(
                    diff < tol,
                    "reconstruction mismatch at ({r},{c}): got {} expected {}, diff {diff}",
                    recon.data[r * recon.cols + c],
                    a.data[r * a.cols + c]
                );
            }
        }
    }

    fn assert_orthonormal_columns(m: &Matrix, tol: f64) {
        let mtm = mat_mul_raw(&transpose_raw(m), m);
        let k = mtm.rows;
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (mtm.data[i * k + j] - expected).abs();
                assert!(
                    diff < tol,
                    "orthonormality violated at ({i},{j}): got {}, expected {expected}",
                    mtm.data[i * k + j]
                );
            }
        }
    }

    fn assert_singular_values_sorted(s: &[f64]) {
        for (i, &si) in s.iter().enumerate() {
            assert!(si >= -1e-14, "singular value s[{i}] = {si} is negative");
        }
        for i in 1..s.len() {
            assert!(
                s[i - 1] >= s[i] - 1e-12,
                "not descending: s[{}]={} < s[{}]={}",
                i - 1,
                s[i - 1],
                i,
                s[i]
            );
        }
    }

    // ---- SVD correctness tests ----

    #[test]
    fn test_identity_3x3() {
        let mut data = vec![0.0; 9];
        for i in 0..3 {
            data[i * 3 + i] = 1.0;
        }
        let a = Matrix {
            data,
            rows: 3,
            cols: 3,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        for &si in &s {
            assert!((si - 1.0).abs() < 1e-10, "expected 1.0, got {si}");
        }
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_diagonal_matrix() {
        let a = Matrix {
            data: vec![5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            rows: 3,
            cols: 3,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!((s[0] - 5.0).abs() < 1e-10);
        assert!((s[1] - 3.0).abs() < 1e-10);
        assert!((s[2] - 1.0).abs() < 1e-10);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
    }

    #[test]
    fn test_known_2x2() {
        // [[3,2],[2,3]] has singular values 5 and 1
        let a = Matrix {
            data: vec![3.0, 2.0, 2.0, 3.0],
            rows: 2,
            cols: 2,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!((s[0] - 5.0).abs() < 1e-10, "expected s[0]=5, got {}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-10, "expected s[1]=1, got {}", s[1]);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
    }

    #[test]
    fn test_known_3x3() {
        let a = Matrix {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            rows: 3,
            cols: 3,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_known_4x4() {
        let a = Matrix {
            data: vec![
                2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0,
                2.0,
            ],
            rows: 4,
            cols: 4,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_tall_rectangular() {
        let a = Matrix {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
            rows: 5,
            cols: 3,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_eq!(u.rows, 5);
        assert_eq!(u.cols, 3);
        assert_eq!(s.len(), 3);
        assert_eq!(v.rows, 3);
        assert_eq!(v.cols, 3);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
    }

    #[test]
    fn test_wide_rectangular() {
        let a = Matrix {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
            rows: 3,
            cols: 5,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_eq!(u.rows, 3);
        assert_eq!(u.cols, 3);
        assert_eq!(s.len(), 3);
        assert_eq!(v.rows, 5);
        assert_eq!(v.cols, 3);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
    }

    #[test]
    fn test_single_element() {
        let a = Matrix {
            data: vec![7.0],
            rows: 1,
            cols: 1,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!((s[0] - 7.0).abs() < 1e-10);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
    }

    #[test]
    fn test_single_element_negative() {
        let a = Matrix {
            data: vec![-5.0],
            rows: 1,
            cols: 1,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!((s[0] - 5.0).abs() < 1e-10);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
    }

    #[test]
    fn test_single_row() {
        let a = Matrix {
            data: vec![1.0, 2.0, 3.0, 4.0],
            rows: 1,
            cols: 4,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_eq!(s.len(), 1);
        let expected = (1.0f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert!((s[0] - expected).abs() < 1e-10);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
    }

    #[test]
    fn test_single_column() {
        let a = Matrix {
            data: vec![1.0, 2.0, 3.0, 4.0],
            rows: 4,
            cols: 1,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_eq!(s.len(), 1);
        let expected = (1.0f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert!((s[0] - expected).abs() < 1e-10);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
    }

    #[test]
    fn test_zero_matrix() {
        let a = Matrix::zeros(3, 3);
        let (_u, s, _v) = GolubKahanSvd::new().compute(&a).unwrap();
        for &si in &s {
            assert!(si.abs() < 1e-12);
        }
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_rank_deficient() {
        // B8: rank-2 in 3x3 (row 3 = row 1 + row 2)
        let a = Matrix { data: vec![1.0,2.0,3.0, 4.0,5.0,6.0, 5.0,7.0,9.0], rows: 3, cols: 3 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!(s[2] < 1e-10, "third singular value should be ~0, got {}", s[2]);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_rank_one() {
        // B9: outer product [1,2,3] * [4,5]
        let a = Matrix { data: vec![4.0,5.0, 8.0,10.0, 12.0,15.0], rows: 3, cols: 2 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        let norm_u = (1.0f64 + 4.0 + 9.0).sqrt();
        let norm_v = (16.0f64 + 25.0).sqrt();
        let expected_s0 = norm_u * norm_v;
        assert!((s[0] - expected_s0).abs() < 1e-8, "expected s[0]={expected_s0}, got {}", s[0]);
        assert!(s[1] < 1e-10, "expected s[1]~0, got {}", s[1]);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
    }

    #[test]
    fn test_repeated_singular_values() {
        // B10: diag(4, 4, 2)
        let a = Matrix { data: vec![4.0,0.0,0.0, 0.0,4.0,0.0, 0.0,0.0,2.0], rows: 3, cols: 3 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!((s[0] - 4.0).abs() < 1e-10);
        assert!((s[1] - 4.0).abs() < 1e-10);
        assert!((s[2] - 2.0).abs() < 1e-10);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_orthonormal_columns(&u, 1e-10);
        assert_orthonormal_columns(&v, 1e-10);
    }

    #[test]
    fn test_diagonal_with_zeros() {
        // B4, B8: diag(5, 0, 3)
        let a = Matrix { data: vec![5.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,3.0], rows: 3, cols: 3 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!((s[0] - 5.0).abs() < 1e-10);
        assert!((s[1] - 3.0).abs() < 1e-10);
        assert!(s[2] < 1e-10);
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_ill_conditioned() {
        // B11: condition number > 1e10
        let a = Matrix { data: vec![1.0,0.0,0.0, 0.0,1e-12,0.0, 0.0,0.0,1e-6], rows: 3, cols: 3 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!((s[0] - 1.0).abs() < 1e-8);
        assert_reconstruction(&a, &u, &s, &v, 1e-6);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_extreme_small_values() {
        // B12: values near underflow
        let a = Matrix { data: vec![1e-300, 0.0, 0.0, 2e-300], rows: 2, cols: 2 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!(s[0].is_finite());
        assert!(s[1].is_finite());
        assert_singular_values_sorted(&s);
        assert_reconstruction(&a, &u, &s, &v, 1e-290);
    }

    #[test]
    fn test_extreme_large_values() {
        // B12: values near overflow
        let a = Matrix { data: vec![1e+150, 0.0, 0.0, 2e+150], rows: 2, cols: 2 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert!(s[0].is_finite());
        assert!(s[1].is_finite());
        for &val in &u.data { assert!(val.is_finite()); }
        for &val in &v.data { assert!(val.is_finite()); }
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_convergence_64x64() {
        // B15
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f64> = (0..64*64).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let a = Matrix { data, rows: 64, cols: 64 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_reconstruction(&a, &u, &s, &v, 1e-8);
        assert_orthonormal_columns(&u, 1e-8);
        assert_orthonormal_columns(&v, 1e-8);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_convergence_128x128() {
        // B15
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let data: Vec<f64> = (0..128*128).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let a = Matrix { data, rows: 128, cols: 128 };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_reconstruction(&a, &u, &s, &v, 1e-8);
        assert_orthonormal_columns(&u, 1e-8);
        assert_orthonormal_columns(&v, 1e-8);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_almost_bidiagonal() {
        // B16
        let a = Matrix {
            data: vec![5.0,2.0,0.0,0.0, 0.0,4.0,1.0,0.0, 0.0,0.0,3.0,0.5, 0.0,0.0,0.0,1.0],
            rows: 4, cols: 4,
        };
        let (u, s, v) = GolubKahanSvd::new().compute(&a).unwrap();
        assert_reconstruction(&a, &u, &s, &v, 1e-10);
        assert_singular_values_sorted(&s);
    }

    #[test]
    fn test_custom_tolerance() {
        // B17
        let a = Matrix { data: vec![1.0,2.0,3.0, 4.0,5.0,6.0, 7.0,8.0,10.0], rows: 3, cols: 3 };
        let (u, s, v) = GolubKahanSvd::new().with_tolerance(1e-15).compute(&a).unwrap();
        assert_reconstruction(&a, &u, &s, &v, 1e-12);
    }

    #[test]
    fn test_low_max_iter_triggers_error() {
        // B18: factor=0 -> max_iter=0, any non-trivial matrix must fail
        let a = Matrix { data: vec![1.0,2.0,3.0, 4.0,5.0,6.0, 7.0,8.0,10.0], rows: 3, cols: 3 };
        let result = GolubKahanSvd::new().with_max_iter_factor(0).compute(&a);
        assert!(result.is_err(), "expected convergence error with factor=0");
        let err = result.unwrap_err();
        assert!(matches!(err, SvdError::Convergence { .. }));
    }

    #[test]
    fn test_determinism() {
        // B19: same input -> identical output
        let a = Matrix { data: vec![1.0,2.0,3.0, 4.0,5.0,6.0, 7.0,8.0,10.0], rows: 3, cols: 3 };
        let svd = GolubKahanSvd::new();
        let (u1, s1, v1) = svd.compute(&a).unwrap();
        let (u2, s2, v2) = svd.compute(&a).unwrap();
        assert_eq!(s1, s2, "singular values differ");
        assert_eq!(u1.data, u2.data, "U differs");
        assert_eq!(v1.data, v2.data, "V differs");
    }
}
