// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-09

//! Fisher information state for EWC (Elastic Weight Consolidation).
//!
//! [`FisherState<L>`] holds per-layer Fisher diagonal approximations and
//! reference weights (`theta_snapshot_*`) used to penalize weight drift
//! during continuous learning.

use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;

/// Per-layer Fisher information state for EWC regularization.
///
/// Tracks accumulated Fisher information (`f_total`), current-phase
/// running EMA (`f_ema`), and weight snapshots (`theta_snapshot`).
/// One instance per layer, for both actor and critic.
#[derive(Debug, Clone)]
pub struct FisherState<L: LinAlg = CpuLinAlg> {
    /// Accumulated Fisher information for weights (across reliable phases).
    pub f_total_weights: L::Matrix,
    /// Accumulated Fisher information for biases (across reliable phases).
    pub f_total_bias: L::Vector,
    /// Current-phase running EMA of squared gradients for weights.
    pub f_ema_weights: L::Matrix,
    /// Current-phase running EMA of squared gradients for biases.
    pub f_ema_bias: L::Vector,
    /// Snapshot of weights at last PLASTIC→FROZEN transition.
    pub theta_snapshot_weights: Option<L::Matrix>,
    /// Snapshot of biases at last PLASTIC→FROZEN transition.
    pub theta_snapshot_bias: Option<L::Vector>,
    /// Snapshot of rezero alpha (for residual layers).
    pub theta_snapshot_rezero_alpha: Option<f64>,
    /// Snapshot of skip projection matrix (for heterogeneous residual layers).
    pub theta_snapshot_skip_proj: Option<L::Matrix>,
}

impl<L: LinAlg> FisherState<L> {
    /// Creates a new zeroed Fisher state for a layer with the given dimensions.
    pub fn new(backend: &L, weight_rows: usize, weight_cols: usize, bias_size: usize) -> Self {
        Self {
            f_total_weights: backend.zeros_mat(weight_rows, weight_cols),
            f_total_bias: backend.zeros_vec(bias_size),
            f_ema_weights: backend.zeros_mat(weight_rows, weight_cols),
            f_ema_bias: backend.zeros_vec(bias_size),
            theta_snapshot_weights: None,
            theta_snapshot_bias: None,
            theta_snapshot_rezero_alpha: None,
            theta_snapshot_skip_proj: None,
        }
    }
}
