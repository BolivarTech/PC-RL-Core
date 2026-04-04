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
    /// Returns [`SvdError::Convergence`] if the iterative solver does not converge
    /// (placeholder: current implementation returns zeros for non-empty matrices).
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

        // Placeholder: return zero matrices of correct shape
        let k = m.min(n);
        Ok((Matrix::zeros(m, k), vec![0.0; k], Matrix::zeros(n, k)))
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
}
