// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-09

//! Hysteresis state machine for dual-EWMA plasticity transitions.
//!
//! [`HysteresisState`] tracks fast and slow EWMAs and transitions between
//! `Plastic` and `Frozen` states based on configurable wake/sleep fractions.

use serde::{Deserialize, Serialize};

use crate::pc_actor_critic::ewma::{EwmaTracker, PlasticityState};

/// Dual-EWMA state machine for one network (actor or critic).
///
/// Drives automatic FROZEN/PLASTIC transitions based on the ratio
/// between a fast EWMA (responsive) and a slow EWMA (baseline).
///
/// - FROZEN → PLASTIC: `fast > slow × (1 + wake_fraction)`
/// - PLASTIC → FROZEN: `fast < slow × (1 - sleep_fraction)`,
///   guarded by `fast.k >= min_initial_plastic`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HysteresisState {
    /// Fast EWMA (responsive to recent signals).
    pub fast: EwmaTracker,
    /// Slow EWMA (baseline signal level).
    pub slow: EwmaTracker,
    /// Current plasticity state.
    pub state: PlasticityState,
    /// Fraction above slow EWMA that triggers wake (FROZEN → PLASTIC).
    pub wake_fraction: f64,
    /// Fraction below slow EWMA that triggers sleep (PLASTIC → FROZEN).
    pub sleep_fraction: f64,
    /// Minimum fast EWMA steps before sleep is allowed.
    pub min_initial_plastic: u64,
}

impl HysteresisState {
    /// Updates both EWMAs with the new signal and evaluates state transition.
    ///
    /// Returns `Some(new_state)` if a transition occurred, `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `signal` - The input signal (surprise for actor, |TD error| for critic).
    pub fn update(&mut self, signal: f64) -> Option<PlasticityState> {
        self.fast.update(signal);
        self.slow.update(signal);

        match self.state {
            PlasticityState::Frozen => {
                // Wake: fast > slow * (1 + wake_fraction)
                if self.fast.value > self.slow.value * (1.0 + self.wake_fraction) {
                    self.state = PlasticityState::Plastic;
                    return Some(PlasticityState::Plastic);
                }
            }
            PlasticityState::Plastic => {
                // Sleep: fast < slow * (1 - sleep_fraction), guarded by warmup
                if self.fast.k >= self.min_initial_plastic
                    && self.fast.value < self.slow.value * (1.0 - self.sleep_fraction)
                {
                    self.state = PlasticityState::Frozen;
                    return Some(PlasticityState::Frozen);
                }
            }
        }

        None
    }
}
