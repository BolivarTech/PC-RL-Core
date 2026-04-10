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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hysteresis_wake_transition() {
        let mut hyst = HysteresisState {
            fast: EwmaTracker::new(5),
            slow: EwmaTracker::new(20),
            state: PlasticityState::Frozen,
            wake_fraction: 0.5,
            sleep_fraction: 0.3,
            min_initial_plastic: 100,
        };
        // Manually set slow EWMA past warmup at 0.05
        hyst.slow.value = 0.05;
        hyst.slow.k = 50;
        hyst.fast.value = 0.06;
        hyst.fast.k = 10;
        // Feed high signal: fast jumps well above wake threshold
        // fast = 0.06 + (1.0 - 0.06)/5 = 0.248
        // slow = 0.05 + (1.0 - 0.05)/20 = 0.0975
        // Wake: 0.248 > 0.0975 * 1.5 = 0.14625 → yes
        let result = hyst.update(1.0);
        assert_eq!(result, Some(PlasticityState::Plastic));
        assert_eq!(hyst.state, PlasticityState::Plastic);
    }

    #[test]
    fn hysteresis_sleep_transition() {
        let mut hyst = HysteresisState {
            fast: EwmaTracker::new(5),
            slow: EwmaTracker::new(20),
            state: PlasticityState::Plastic,
            wake_fraction: 0.5,
            sleep_fraction: 0.3,
            min_initial_plastic: 100,
        };
        // Past warmup and above min_initial_plastic
        hyst.slow.value = 0.3;
        hyst.slow.k = 200;
        hyst.fast.value = 0.22;
        hyst.fast.k = 200;
        // Feed low signal:
        // fast = 0.22 + (0.0 - 0.22)/5 = 0.176
        // slow = 0.3 + (0.0 - 0.3)/20 = 0.285
        // Sleep: 0.176 < 0.285 * 0.7 = 0.1995 → yes
        // fast.k = 201 >= 100 → guard lifts
        let result = hyst.update(0.0);
        assert_eq!(result, Some(PlasticityState::Frozen));
        assert_eq!(hyst.state, PlasticityState::Frozen);
    }

    #[test]
    fn hysteresis_warmup_guard_suppresses_sleep() {
        let mut hyst = HysteresisState {
            fast: EwmaTracker::new(5),
            slow: EwmaTracker::new(20),
            state: PlasticityState::Plastic,
            wake_fraction: 0.5,
            sleep_fraction: 0.3,
            min_initial_plastic: 100,
        };
        // fast.k < min_initial_plastic — sleep condition met but guard active
        hyst.fast.k = 50;
        hyst.fast.value = 0.1;
        hyst.slow.k = 50;
        hyst.slow.value = 0.5;
        for _ in 0..10 {
            let result = hyst.update(0.05);
            assert_eq!(result, None);
        }
        // fast.k = 60, still < 100
        assert_eq!(hyst.state, PlasticityState::Plastic);
    }

    #[test]
    fn hysteresis_warmup_guard_lifts() {
        let mut hyst = HysteresisState {
            fast: EwmaTracker::new(5),
            slow: EwmaTracker::new(20),
            state: PlasticityState::Plastic,
            wake_fraction: 0.5,
            sleep_fraction: 0.3,
            min_initial_plastic: 100,
        };
        hyst.fast.k = 99;
        hyst.fast.value = 0.1;
        hyst.slow.k = 99;
        hyst.slow.value = 0.5;
        // After update(0.0): fast.k=100 >= 100, guard lifts
        // fast = 0.1 + (0.0 - 0.1)/5 = 0.08
        // slow = 0.5 + (0.0 - 0.5)/20 = 0.475
        // Sleep: 0.08 < 0.475 * 0.7 = 0.3325 → yes
        let result = hyst.update(0.0);
        assert_eq!(result, Some(PlasticityState::Frozen));
        assert_eq!(hyst.state, PlasticityState::Frozen);
    }

    #[test]
    fn hysteresis_no_false_wake_on_noise() {
        let mut hyst = HysteresisState {
            fast: EwmaTracker::new(5),
            slow: EwmaTracker::new(20),
            state: PlasticityState::Frozen,
            wake_fraction: 0.5,
            sleep_fraction: 0.3,
            min_initial_plastic: 100,
        };
        // Feed constant signal — both converge to 0.1, no separation
        for _ in 0..50 {
            let result = hyst.update(0.1);
            assert_eq!(result, None);
        }
        assert_eq!(hyst.state, PlasticityState::Frozen);
    }
}
