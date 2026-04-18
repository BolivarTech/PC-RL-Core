// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-18

//! Manual recovery API for [`PcActorCritic`]: `rollback_soft`,
//! `rollback_hard`, `champion_update`, and `set_rollback_hard_cooldown`.
//!
//! These four methods are the consumer-facing surface of the
//! self-recovery toolkit described in Phase 1 of the self-recovery
//! plan. They mutate the live actor and the distillation anchor slots;
//! the reusable low-level bookkeeping they invoke
//! ([`reset_actor_transient_state`](super::PcActorCritic::reset_actor_transient_state),
//! [`clear_actor_fisher_ema`](super::PcActorCritic::clear_actor_fisher_ema))
//! lives in the parent module.
//!
//! The module exists to decouple recovery semantics from the learning
//! path — a downstream engineer investigating why a `rollback_hard`
//! unlock happens at step N only needs to read this file, not the full
//! 11k-line `mod.rs`.

use crate::error::PcError;
use crate::linalg::LinAlg;
use crate::pc_actor_critic::PcActorCritic;

impl<L: LinAlg> PcActorCritic<L> {
    /// Rolls back the live actor to the Polyak-averaged target weights.
    ///
    /// Copies all weights (layers, biases, ReZero alphas, skip projections)
    /// from the Polyak target into the live actor. Also resets transient
    /// actor state (eligibility trace, plastic step counter, frozen steps,
    /// TD error buffer).
    ///
    /// # When to use — **noise smoothing, NOT cascade recovery**
    ///
    /// The Polyak target tracks the live actor with an exponential moving
    /// average controlled by `polyak_tau`. Its effective lag is on the
    /// order of `1 / polyak_tau` steps (e.g. `polyak_tau = 0.005` →
    /// ~200-step lag). Rolling back to the Polyak target therefore
    /// undoes only the last few hundred steps of drift — it is designed
    /// for smoothing short-horizon policy noise, NOT for recovering from
    /// a cascade of degraded learning that spans thousands of steps.
    ///
    /// For cascade recovery (restoring to a known-good champion
    /// snapshot), use [`rollback_hard`](Self::rollback_hard) instead.
    /// The two methods are complementary, not interchangeable: Polyak
    /// handles variance, Frozen handles drift. Mis-selecting them will
    /// either leave a degraded policy (Polyak when the drift horizon
    /// exceeds `1 / polyak_tau`) or over-correct (Frozen when only a
    /// few recent steps of noise need smoothing).
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if the Polyak target is not
    /// allocated (`distillation_lambda_polyak == 0.0`).
    pub fn rollback_soft(&mut self) -> Result<(), PcError> {
        let polyak = self.polyak_target.as_ref().ok_or_else(|| {
            PcError::ConfigValidation(
                "rollback_soft requires distillation_lambda_polyak > 0".into(),
            )
        })?;
        self.actor.copy_weights_from(polyak)?;
        self.reset_actor_transient_state();
        Ok(())
    }

    /// Rolls back the live actor (and Polyak target, if present) to the
    /// frozen champion weights.
    ///
    /// Restores actor weights from the frozen champion. If a Polyak target
    /// exists, it is also reset to the frozen champion. EWC Fisher running
    /// EMA (`f_ema`) is zeroed while `f_total` and `theta_snapshot` are
    /// preserved. The critic is explicitly NOT touched.
    ///
    /// Subject to a cooldown gate: if fewer than `rollback_hard_cooldown_steps`
    /// learning steps have elapsed since the last successful call, the
    /// method returns an error and performs no mutation.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if the frozen champion is not
    /// allocated (`distillation_lambda_frozen == 0.0`) or if the cooldown
    /// window has not elapsed.
    pub fn rollback_hard(&mut self) -> Result<(), PcError> {
        // Cooldown gate — reject entirely (no mutation) if within window.
        if self.rollback_hard_cooldown_steps > 0
            && self.steps_since_last_rollback_hard < self.rollback_hard_cooldown_steps
        {
            return Err(PcError::ConfigValidation(format!(
                "rollback_hard rejected: cooldown active ({} of {} steps)",
                self.steps_since_last_rollback_hard, self.rollback_hard_cooldown_steps,
            )));
        }
        let frozen = self.frozen_champion.as_ref().ok_or_else(|| {
            PcError::ConfigValidation(
                "rollback_hard requires distillation_lambda_frozen > 0".into(),
            )
        })?;
        let frozen_clone = frozen.clone();

        // 1. Actor weights <- frozen
        self.actor.copy_weights_from(&frozen_clone)?;

        // 2. Polyak <- frozen (if allocated)
        if let Some(ref mut polyak) = self.polyak_target {
            polyak.copy_weights_from(&frozen_clone)?;
        }

        // 3. EWC Fisher: clear f_ema, preserve f_total and theta_snapshot.
        self.clear_actor_fisher_ema();

        // 4. Actor transient state reset.
        self.reset_actor_transient_state();

        // 5. Cooldown reset.
        self.steps_since_last_rollback_hard = 0;

        // Critic is EXPLICITLY NOT TOUCHED.
        Ok(())
    }

    /// Promotes the current live actor weights into the frozen champion slot.
    ///
    /// Copies all weights from the live actor into the frozen champion.
    /// The Polyak target and all learning state are left untouched.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if the frozen champion is not
    /// allocated (`distillation_lambda_frozen == 0.0`).
    pub fn champion_update(&mut self) -> Result<(), PcError> {
        let frozen = self.frozen_champion.as_mut().ok_or_else(|| {
            PcError::ConfigValidation(
                "champion_update requires distillation_lambda_frozen > 0".into(),
            )
        })?;
        frozen.copy_weights_from(&self.actor)?;
        Ok(())
    }

    /// Sets the cooldown window (in learning steps) between consecutive
    /// `rollback_hard()` calls.
    ///
    /// Pass `0` to disable the cooldown entirely.
    ///
    /// # Arguments
    ///
    /// * `steps` - Number of learning steps that must elapse before a
    ///   subsequent `rollback_hard()` is allowed.
    pub fn set_rollback_hard_cooldown(&mut self, steps: u64) {
        self.rollback_hard_cooldown_steps = steps;
    }
}
