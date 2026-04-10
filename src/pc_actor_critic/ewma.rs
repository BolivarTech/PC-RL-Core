// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-09

//! EWMA tracker and plasticity state for continuous learning.
//!
//! [`PlasticityState`] is a two-state enum (`Plastic`/`Frozen`) used by
//! the dual-EWMA hysteresis state machine. [`EwmaTracker`] implements
//! warmup-aware exponentially weighted moving average tracking.

use serde::{Deserialize, Serialize};

/// Plasticity state of a network: whether it is actively learning or frozen.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum PlasticityState {
    /// Network weights are frozen (consolidated).
    Frozen,
    /// Network is actively learning.
    #[default]
    Plastic,
}

/// Exponentially weighted moving average tracker with warmup.
///
/// During warmup (`k <= window`), computes the exact arithmetic mean.
/// After warmup, behaves as a standard EMA with `alpha = 1/window`.
///
/// # Examples
///
/// ```
/// use pc_rl_core::EwmaTracker;
///
/// let mut tracker = EwmaTracker::new(5);
/// // Warmup phase: exact arithmetic mean
/// tracker.update(1.0);
/// tracker.update(2.0);
/// tracker.update(3.0);
/// assert!((tracker.value - 2.0).abs() < f64::EPSILON);
/// assert_eq!(tracker.k, 3);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EwmaTracker {
    /// Current EWMA value.
    pub value: f64,
    /// Step counter (monotonic, never resets).
    pub k: u64,
    /// Window size `N`: determines warmup length and steady-state alpha.
    pub window: usize,
}

impl EwmaTracker {
    /// Creates a new tracker with the given window size.
    pub fn new(window: usize) -> Self {
        Self {
            value: 0.0,
            k: 0,
            window,
        }
    }

    /// Updates the EWMA with a new value and returns the updated average.
    ///
    /// During warmup (`k <= window`), computes the exact arithmetic mean.
    /// After warmup, applies `A(k) = A(k-1) + (val - A(k-1)) / window`.
    pub fn update(&mut self, val: f64) -> f64 {
        self.k += 1;
        let divisor = std::cmp::min(self.window as u64, self.k);
        self.value += (val - self.value) / divisor as f64;
        self.value
    }
}
