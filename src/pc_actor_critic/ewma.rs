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
    /// Step counter. Reset to 0 on coupling-forced wake to re-enable warmup guard.
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
    ///
    /// Non-finite inputs (NaN, Inf, -Inf) are silently dropped — the tracker
    /// retains its previous value and `k` is not incremented.
    pub fn update(&mut self, val: f64) -> f64 {
        if !val.is_finite() {
            return self.value;
        }
        self.k += 1;
        let divisor = std::cmp::min(self.window as u64, self.k);
        self.value += (val - self.value) / divisor as f64;
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_warmup_exact_mean() {
        let mut tracker = EwmaTracker::new(10);
        tracker.update(1.0);
        tracker.update(2.0);
        let result = tracker.update(3.0);
        assert!((result - 2.0).abs() < f64::EPSILON);
        assert!((tracker.value - 2.0).abs() < f64::EPSILON);
        assert_eq!(tracker.k, 3);
    }

    #[test]
    fn ewma_warmup_single_value() {
        let mut tracker = EwmaTracker::new(10);
        let result = tracker.update(5.0);
        assert!((result - 5.0).abs() < f64::EPSILON);
        assert_eq!(tracker.k, 1);
    }

    #[test]
    fn ewma_steady_state_ema() {
        let mut tracker = EwmaTracker::new(10);
        // Fill warmup with constant 1.0
        for _ in 0..10 {
            tracker.update(1.0);
        }
        assert_eq!(tracker.k, 10);
        // Now in steady state. Feed 2.0 and verify EMA formula.
        let prev = tracker.value;
        let result = tracker.update(2.0);
        let expected = prev + (2.0 - prev) / 10.0;
        assert!((result - expected).abs() < 1e-12);
        assert_eq!(tracker.k, 11);
    }

    #[test]
    fn ewma_k_monotonic_never_resets() {
        let mut tracker = EwmaTracker::new(50);
        for i in 1..=1000 {
            tracker.update(i as f64);
            assert_eq!(tracker.k, i as u64);
        }
    }

    #[test]
    fn ewma_warmup_to_steady_transition() {
        let n = 5usize;
        let mut tracker = EwmaTracker::new(n);
        // Warmup: feed 1.0, 2.0, 3.0, 4.0, 5.0
        for i in 1..=n {
            tracker.update(i as f64);
        }
        // At k=N=5: value should be mean(1,2,3,4,5) = 3.0
        assert!((tracker.value - 3.0).abs() < f64::EPSILON);
        assert_eq!(tracker.k, n as u64);

        // k=N+1: EMA formula with alpha=1/N=0.2
        let prev = tracker.value;
        let val = 10.0;
        let result = tracker.update(val);
        // divisor = min(5, 6) = 5, so: 3.0 + (10.0 - 3.0) / 5 = 4.4
        let expected = prev + (val - prev) / n as f64;
        assert!((result - expected).abs() < 1e-12);
        assert_eq!(tracker.k, (n + 1) as u64);
    }

    #[test]
    fn plasticity_state_default_is_plastic() {
        let state = PlasticityState::default();
        assert_eq!(state, PlasticityState::Plastic);
    }

    #[test]
    fn plasticity_state_serde_roundtrip() {
        let frozen = PlasticityState::Frozen;
        let json = serde_json::to_string(&frozen).unwrap();
        let loaded: PlasticityState = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded, PlasticityState::Frozen);

        let plastic = PlasticityState::Plastic;
        let json = serde_json::to_string(&plastic).unwrap();
        let loaded: PlasticityState = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded, PlasticityState::Plastic);
    }

    #[test]
    fn ewma_tracker_serde_roundtrip() {
        let mut tracker = EwmaTracker::new(10);
        tracker.update(3.0);
        tracker.update(7.0);
        let json = serde_json::to_string(&tracker).unwrap();
        let loaded: EwmaTracker = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.k, tracker.k);
        assert!((loaded.value - tracker.value).abs() < f64::EPSILON);
        assert_eq!(loaded.window, tracker.window);
    }

    /// MAGI v2 W2: NaN input must not corrupt the EWMA tracker.
    /// NaN/Inf should be silently dropped, preserving the last valid value.
    #[test]
    fn ewma_nan_input_does_not_silently_corrupt() {
        let mut tracker = EwmaTracker::new(10);
        tracker.update(1.0);
        tracker.update(2.0);
        tracker.update(3.0);
        let value_before = tracker.value;
        let k_before = tracker.k;

        // Feed NaN — must NOT corrupt
        let result = tracker.update(f64::NAN);
        assert!(
            result.is_finite(),
            "update(NaN) must return finite value, got {result}"
        );
        assert!(
            (tracker.value - value_before).abs() < f64::EPSILON,
            "NaN must not change EWMA value: before={value_before}, after={}",
            tracker.value
        );
        assert_eq!(tracker.k, k_before, "NaN must not increment k");

        // Feed Inf — must NOT corrupt
        let result = tracker.update(f64::INFINITY);
        assert!(
            result.is_finite(),
            "update(Inf) must return finite value, got {result}"
        );
        assert!(
            (tracker.value - value_before).abs() < f64::EPSILON,
            "Inf must not change EWMA value"
        );

        // Feed -Inf — must NOT corrupt
        let result = tracker.update(f64::NEG_INFINITY);
        assert!(
            result.is_finite(),
            "update(-Inf) must return finite value, got {result}"
        );

        // Normal values must still work after NaN/Inf events
        let result = tracker.update(5.0);
        assert!(result.is_finite(), "Normal update after NaN must work");
        assert!(
            (tracker.value - value_before).abs() > f64::EPSILON,
            "Normal update after NaN must change value"
        );
    }
}
