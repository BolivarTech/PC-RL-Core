// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Crate-wide error type for `pc_rl_core`.

use std::fmt;

/// Crate-wide error type for `pc_rl_core`.
///
/// # Examples
///
/// ```
/// use pc_rl_core::error::PcError;
///
/// let err = PcError::ConfigValidation("bad topology".to_string());
/// assert!(format!("{err}").contains("bad topology"));
/// ```
#[derive(Debug)]
pub enum PcError {
    /// Matrix or vector dimension mismatch.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
        /// Human-readable context.
        context: &'static str,
    },
    /// Configuration validation failure.
    ConfigValidation(String),
    /// JSON serialization/deserialization error.
    Serialization(String),
    /// File I/O error.
    Io(std::io::Error),
}

impl fmt::Display for PcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcError::DimensionMismatch {
                expected,
                got,
                context,
            } => write!(
                f,
                "dimension mismatch in {context}: expected {expected}, got {got}"
            ),
            PcError::ConfigValidation(msg) => write!(f, "config validation: {msg}"),
            PcError::Serialization(msg) => write!(f, "serialization: {msg}"),
            PcError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for PcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PcError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PcError {
    fn from(e: std::io::Error) -> Self {
        PcError::Io(e)
    }
}

impl From<serde_json::Error> for PcError {
    fn from(e: serde_json::Error) -> Self {
        PcError::Serialization(e.to_string())
    }
}
