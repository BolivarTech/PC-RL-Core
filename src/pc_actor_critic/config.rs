// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-09

//! Configuration types for `PcActorCritic`.
//!
//! Defines [`PcActorCriticConfig`] and all `default_*` helper functions
//! used by serde to populate omitted fields.

use serde::{Deserialize, Serialize};

use crate::mlp_critic::MlpCriticConfig;
use crate::pc_actor::PcActorConfig;

/// Default discount factor.
fn default_gamma() -> f64 {
    0.95
}

/// Default surprise low threshold.
fn default_surprise_low() -> f64 {
    0.02
}

/// Default surprise high threshold.
fn default_surprise_high() -> f64 {
    0.15
}

/// Default for adaptive surprise (enabled).
fn default_adaptive_surprise() -> bool {
    true
}

/// Default surprise buffer size for adaptive surprise.
fn default_surprise_buffer_size() -> usize {
    400
}

/// Default entropy regularization coefficient.
fn default_entropy_coeff() -> f64 {
    0.01
}

/// Default scale floor for surprise-to-learning-rate mapping.
fn default_scale_floor() -> f64 {
    0.0
}

/// Default scale ceiling for surprise-to-learning-rate mapping.
fn default_scale_ceil() -> f64 {
    2.0
}

/// Default for actor hysteresis (disabled).
fn default_actor_hysteresis() -> bool {
    false
}

/// Default fast EWMA window for actor hysteresis.
fn default_actor_fast_window() -> usize {
    20
}

/// Default slow EWMA window for actor hysteresis.
fn default_actor_slow_window() -> usize {
    100
}

/// Default wake fraction for actor hysteresis.
fn default_actor_wake_fraction() -> f64 {
    0.5
}

/// Default sleep fraction for actor hysteresis.
fn default_actor_sleep_fraction() -> f64 {
    0.3
}

/// Default for critic hysteresis (disabled).
fn default_critic_hysteresis() -> bool {
    false
}

/// Default fast EWMA window for critic hysteresis.
fn default_critic_fast_window() -> usize {
    20
}

/// Default slow EWMA window for critic hysteresis.
fn default_critic_slow_window() -> usize {
    100
}

/// Default wake fraction for critic hysteresis.
fn default_critic_wake_fraction() -> f64 {
    0.5
}

/// Default sleep fraction for critic hysteresis.
fn default_critic_sleep_fraction() -> f64 {
    0.3
}

/// Default for actor-wakes-critic coupling (enabled).
fn default_actor_wakes_critic() -> bool {
    true
}

/// Default threshold for actor-wakes-critic coupling.
fn default_actor_wakes_critic_threshold() -> u64 {
    1000
}

/// Default for critic-wakes-actor coupling (enabled).
fn default_critic_wakes_actor() -> bool {
    true
}

/// Default threshold for critic-wakes-actor coupling.
fn default_critic_wakes_actor_threshold() -> u64 {
    1000
}

/// Default actor consolidation decay base (1.0 = no decay).
fn default_consolidation_decay() -> f64 {
    1.0
}

/// Default critic consolidation decay base (1.0 = no decay).
fn default_critic_consolidation_decay() -> f64 {
    1.0
}

/// Default adaptive consolidation flag (disabled).
fn default_adaptive_consolidation() -> bool {
    false
}

/// Default EMA smoothing for per-layer prediction error (M3b).
fn default_consolidation_ema_beta() -> f64 {
    0.99
}

/// Default sigmoid steepness for adaptive consolidation (M3b).
fn default_consolidation_sigmoid_k() -> f64 {
    10.0
}

/// Default sigmoid midpoint for adaptive consolidation (M3b).
fn default_consolidation_error_threshold() -> f64 {
    0.05
}

/// Default EWC regularization strength (0.0 = disabled).
fn default_ewc_lambda() -> f64 {
    0.0
}

/// Default Fisher information decay between consolidation phases.
fn default_fisher_decay() -> f64 {
    0.9
}

/// Default Fisher EMA smoothing factor.
fn default_fisher_ema_beta() -> f64 {
    0.99
}

/// Default logits reversal flag (disabled).
fn default_logits_reversal() -> bool {
    false
}

/// Default TD(n) steps (0 = standard TD(0), zero overhead).
fn default_td_steps() -> usize {
    0
}

/// Default GAE lambda (None — disabled, backward compatible).
/// Recommended: Some(0.95) for short episodes.
fn default_gae_lambda() -> Option<f64> {
    None
}

/// Default Polyak averaging rate for soft target network updates.
/// Standard SAC/DQN value. Must be in (0.0, 1.0].
pub fn default_polyak_tau() -> f64 {
    0.005
}

/// Configuration for the integrated PC Actor-Critic agent.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
/// use pc_rl_core::mlp_critic::MlpCriticConfig;
/// use pc_rl_core::pc_actor::PcActorConfig;
/// use pc_rl_core::pc_actor_critic::PcActorCriticConfig;
///
/// let config = PcActorCriticConfig {
///     actor: PcActorConfig {
///         input_size: 9,
///         hidden_layers: vec![LayerDef { size: 18, activation: Activation::Tanh }],
///         output_size: 9,
///         output_activation: Activation::Tanh,
///         alpha: 0.1, tol: 0.01, min_steps: 1, max_steps: 20,
///         lr_weights: 0.01, synchronous: true, temperature: 1.0,
///         local_lambda: 1.0,
///         residual: false,
///         rezero_init: 0.001,
///     },
///     critic: MlpCriticConfig {
///         input_size: 27,
///         hidden_layers: vec![LayerDef { size: 36, activation: Activation::Tanh }],
///         output_activation: Activation::Linear,
///         lr: 0.005,
///     },
///     gamma: 0.95,
///     surprise_low: 0.02,
///     surprise_high: 0.15,
///     adaptive_surprise: true,
///     surprise_buffer_size: 400,
///     entropy_coeff: 0.01,
///     scale_floor: 0.0,
///     scale_ceil: 2.0,
///     actor_hysteresis: false,
///     actor_fast_window: 20,
///     actor_slow_window: 100,
///     actor_wake_fraction: 0.5,
///     actor_sleep_fraction: 0.3,
///     critic_hysteresis: false,
///     critic_fast_window: 20,
///     critic_slow_window: 100,
///     critic_wake_fraction: 0.5,
///     critic_sleep_fraction: 0.3,
///     actor_wakes_critic: true,
///     actor_wakes_critic_threshold: 1000,
///     critic_wakes_actor: true,
///     critic_wakes_actor_threshold: 1000,
///     consolidation_decay: 1.0,
///     critic_consolidation_decay: 1.0,
///     adaptive_consolidation: false,
///     consolidation_ema_beta: 0.99,
///     consolidation_sigmoid_k: 10.0,
///     consolidation_error_threshold: 0.05,
///     ewc_lambda: 0.0,
///     fisher_decay: 0.9,
///     fisher_ema_beta: 0.99,
///     logits_reversal: false,
///     td_steps: 0,
///     gae_lambda: None,
///     distillation_lambda_polyak: 0.0,
///     polyak_tau: 0.005,
///     distillation_lambda_frozen: 0.0,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcActorCriticConfig {
    /// Actor (PC network) configuration.
    pub actor: PcActorConfig,
    /// Critic (MLP value function) configuration.
    pub critic: MlpCriticConfig,
    /// Discount factor for computing returns. Default: 0.95.
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Surprise threshold below which learning rate is scaled down. Default: 0.02.
    #[serde(default = "default_surprise_low")]
    pub surprise_low: f64,
    /// Surprise threshold above which learning rate is scaled up. Default: 0.15.
    #[serde(default = "default_surprise_high")]
    pub surprise_high: f64,
    /// Whether to adaptively recalibrate surprise thresholds. Default: true.
    #[serde(default = "default_adaptive_surprise")]
    pub adaptive_surprise: bool,
    /// Maximum number of surprise scores in the adaptive buffer.
    /// Only used when `adaptive_surprise` is true. Default: 400.
    #[serde(default = "default_surprise_buffer_size")]
    pub surprise_buffer_size: usize,
    /// Entropy regularization coefficient. Default: 0.01.
    #[serde(default = "default_entropy_coeff")]
    pub entropy_coeff: f64,
    /// Floor of the surprise-to-learning-rate scale mapping.
    /// When surprise is at or below the low threshold, `surprise_scale()`
    /// returns this value. Set to 0.0 for true freeze behavior.
    /// Default: 0.0.
    #[serde(default = "default_scale_floor")]
    pub scale_floor: f64,
    /// Ceiling of the surprise-to-learning-rate scale mapping.
    /// When surprise is at or above the high threshold, `surprise_scale()`
    /// returns this value. Must be strictly greater than `scale_floor`.
    /// Default: 2.0.
    #[serde(default = "default_scale_ceil")]
    pub scale_ceil: f64,
    /// Enable actor dual-EWMA hysteresis state machine. Default: false.
    #[serde(default = "default_actor_hysteresis")]
    pub actor_hysteresis: bool,
    /// Fast EWMA window for actor hysteresis. Default: 20.
    #[serde(default = "default_actor_fast_window")]
    pub actor_fast_window: usize,
    /// Slow EWMA window for actor hysteresis. Default: 100.
    #[serde(default = "default_actor_slow_window")]
    pub actor_slow_window: usize,
    /// Wake threshold fraction for actor (FROZEN → PLASTIC). Default: 0.5.
    #[serde(default = "default_actor_wake_fraction")]
    pub actor_wake_fraction: f64,
    /// Sleep threshold fraction for actor (PLASTIC → FROZEN). Default: 0.3.
    #[serde(default = "default_actor_sleep_fraction")]
    pub actor_sleep_fraction: f64,
    /// Enable critic dual-EWMA hysteresis state machine. Default: false.
    #[serde(default = "default_critic_hysteresis")]
    pub critic_hysteresis: bool,
    /// Fast EWMA window for critic hysteresis. Default: 20.
    #[serde(default = "default_critic_fast_window")]
    pub critic_fast_window: usize,
    /// Slow EWMA window for critic hysteresis. Default: 100.
    #[serde(default = "default_critic_slow_window")]
    pub critic_slow_window: usize,
    /// Wake threshold fraction for critic (FROZEN → PLASTIC). Default: 0.5.
    #[serde(default = "default_critic_wake_fraction")]
    pub critic_wake_fraction: f64,
    /// Sleep threshold fraction for critic (PLASTIC → FROZEN). Default: 0.3.
    #[serde(default = "default_critic_sleep_fraction")]
    pub critic_sleep_fraction: f64,
    /// Enable actor→critic coupling: when actor wakes, force critic to wake
    /// if it has been frozen for at least `actor_wakes_critic_threshold` steps.
    /// EWMA k is reset on coupling-forced wake to prevent immediate re-freeze.
    /// Only effective when both `actor_hysteresis` and `critic_hysteresis` are true.
    /// Default: true.
    #[serde(default = "default_actor_wakes_critic")]
    pub actor_wakes_critic: bool,
    /// Minimum critic frozen steps before actor→critic coupling triggers.
    /// Default: 1000.
    #[serde(default = "default_actor_wakes_critic_threshold")]
    pub actor_wakes_critic_threshold: u64,
    /// Enable critic→actor coupling: when critic wakes (high |TD error|),
    /// force actor to wake if it has been frozen for at least
    /// `critic_wakes_actor_threshold` steps. EWMA k is reset on
    /// coupling-forced wake to prevent immediate re-freeze.
    /// Only effective when both `actor_hysteresis` and `critic_hysteresis` are true.
    /// Default: true.
    #[serde(default = "default_critic_wakes_actor")]
    pub critic_wakes_actor: bool,
    /// Minimum actor frozen steps before critic→actor coupling triggers.
    /// Default: 1000.
    #[serde(default = "default_critic_wakes_actor_threshold")]
    pub critic_wakes_actor_threshold: u64,
    /// Actor fixed decay base for layer-wise consolidation (M3a).
    /// Layer i gets `consolidation_decay^(n_hidden - 1 - i)`.
    /// 1.0 = no decay (default). Must be in [0.0, 1.0].
    #[serde(default = "default_consolidation_decay")]
    pub consolidation_decay: f64,
    /// Critic fixed decay base for layer-wise consolidation (M3a).
    /// Independent from actor decay. Must be in [0.0, 1.0]. Default: 1.0.
    #[serde(default = "default_critic_consolidation_decay")]
    pub critic_consolidation_decay: f64,
    /// Enable adaptive sigmoid decay for actor (M3b). When true,
    /// per-layer prediction error EMA drives decay via sigmoid.
    /// Overrides `consolidation_decay` for actor; critic always uses M3a.
    /// Default: false.
    #[serde(default = "default_adaptive_consolidation")]
    pub adaptive_consolidation: bool,
    /// EMA smoothing factor for per-layer prediction error (M3b).
    /// Must be in (0.0, 1.0) exclusive. Default: 0.99.
    #[serde(default = "default_consolidation_ema_beta")]
    pub consolidation_ema_beta: f64,
    /// Sigmoid steepness for adaptive consolidation (M3b).
    /// Must be > 0.0 when `adaptive_consolidation` is true. Default: 10.0.
    #[serde(default = "default_consolidation_sigmoid_k")]
    pub consolidation_sigmoid_k: f64,
    /// Sigmoid midpoint for adaptive consolidation (M3b).
    /// Must be > 0.0 when `adaptive_consolidation` is true. Default: 0.05.
    #[serde(default = "default_consolidation_error_threshold")]
    pub consolidation_error_threshold: f64,
    /// EWC regularization strength. When 0.0 (default), EWC is fully disabled
    /// and no Fisher state is allocated (zero overhead). When > 0.0, Fisher
    /// information is tracked per layer and EWC correction is applied after
    /// each backward pass.
    #[serde(default = "default_ewc_lambda")]
    pub ewc_lambda: f64,
    /// Fisher information decay factor applied to F_total on each reliable
    /// FROZEN→PLASTIC transition. F_total saturates at F_ema / (1 - fisher_decay).
    /// Must be in [0.0, 1.0]. Default: 0.9.
    #[serde(default = "default_fisher_decay")]
    pub fisher_decay: f64,
    /// EMA smoothing factor for Fisher information accumulation during PLASTIC.
    /// Must be in (0.0, 1.0). Determines min_fisher_phase = ceil(1/(1-beta)).
    /// Default: 0.99.
    #[serde(default = "default_fisher_ema_beta")]
    pub fisher_ema_beta: f64,
    /// Enable logits reversal for Fisher estimation (actor only).
    /// When true, computes Fisher from reversed logits (softmax of -y_conv/T)
    /// rather than the actual policy gradient. Default: false.
    #[serde(default = "default_logits_reversal")]
    pub logits_reversal: bool,
    /// Number of steps for TD(n) return computation.
    /// 0 = standard TD(0) (default, zero overhead). No buffer allocated.
    /// n > 0 = accumulate n real reward steps before bootstrapping with V(s_{t+n}).
    /// Recommended: 0 (TD(0)), 4-5 (TicTacToe), 8-10 (longer games).
    /// Avoid td_steps=1 — use 0 for standard TD(0) or >= 2 for multi-step.
    /// Memory: O(td_steps × network_size) per agent.
    #[serde(default = "default_td_steps")]
    pub td_steps: usize,
    /// GAE lambda for eligibility traces. Mutually exclusive with `td_steps > 0`.
    /// - `Some(λ)`: GAE(λ) with output-level eligibility traces. λ=0.0 ≈ TD(0), λ=1.0 ≈ MC.
    /// - `None`: Disabled — standard TD(0) or TD(n) via `td_steps`.
    ///
    /// Default: `None` (backward compatible). Recommended: `Some(0.95)` for short episodes.
    #[serde(default = "default_gae_lambda")]
    pub gae_lambda: Option<f64>,
    /// Strength of KL divergence distillation toward the Polyak-averaged
    /// target actor. 0.0 (default) disables Polyak distillation entirely
    /// — no target is allocated. When > 0.0, a slow-moving copy of the actor
    /// is maintained and its distribution regularizes the live policy gradient.
    #[serde(default)]
    pub distillation_lambda_polyak: f64,
    /// Polyak averaging rate for soft target network updates.
    /// `target = (1 - tau) * target + tau * live` after each actor weight update.
    /// Must be in (0.0, 1.0] when `distillation_lambda_polyak > 0`.
    /// Default: 0.005.
    #[serde(default = "default_polyak_tau")]
    pub polyak_tau: f64,
    /// Strength of KL divergence distillation toward a frozen snapshot actor.
    /// 0.0 (default) disables frozen distillation entirely.
    /// Allocated/deallocated by Task 3 (Phase 1 commit 5-6).
    #[serde(default)]
    pub distillation_lambda_frozen: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::error::PcError;
    use crate::layer::LayerDef;
    use crate::linalg::cpu::CpuLinAlg;
    use crate::mlp_critic::MlpCriticConfig;
    use crate::pc_actor::PcActorConfig;
    use crate::pc_actor_critic::PcActorCritic;

    /// Minimal valid config for distillation validation tests.
    fn base_config() -> PcActorCriticConfig {
        PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: 4,
                hidden_layers: vec![LayerDef {
                    size: 8,
                    activation: Activation::Tanh,
                }],
                output_size: 4,
                output_activation: Activation::Linear,
                alpha: 0.1,
                tol: 0.01,
                min_steps: 1,
                max_steps: 5,
                lr_weights: 0.01,
                synchronous: true,
                temperature: 1.0,
                local_lambda: 1.0,
                residual: false,
                rezero_init: 0.001,
            },
            critic: MlpCriticConfig {
                input_size: 12,
                hidden_layers: vec![LayerDef {
                    size: 16,
                    activation: Activation::Tanh,
                }],
                output_activation: Activation::Linear,
                lr: 0.005,
            },
            gamma: 0.95,
            surprise_low: 0.02,
            surprise_high: 0.15,
            adaptive_surprise: false,
            surprise_buffer_size: 100,
            entropy_coeff: 0.0,
            scale_floor: 0.1,
            scale_ceil: 2.0,
            actor_hysteresis: false,
            actor_fast_window: 20,
            actor_slow_window: 100,
            actor_wake_fraction: 0.5,
            actor_sleep_fraction: 0.3,
            critic_hysteresis: false,
            critic_fast_window: 20,
            critic_slow_window: 100,
            critic_wake_fraction: 0.5,
            critic_sleep_fraction: 0.3,
            actor_wakes_critic: true,
            actor_wakes_critic_threshold: 1000,
            critic_wakes_actor: true,
            critic_wakes_actor_threshold: 1000,
            consolidation_decay: 1.0,
            critic_consolidation_decay: 1.0,
            adaptive_consolidation: false,
            consolidation_ema_beta: 0.99,
            consolidation_sigmoid_k: 10.0,
            consolidation_error_threshold: 0.05,
            ewc_lambda: 0.0,
            fisher_decay: 0.9,
            fisher_ema_beta: 0.99,
            logits_reversal: false,
            td_steps: 0,
            gae_lambda: None,
            distillation_lambda_polyak: 0.0,
            polyak_tau: 0.005,
            distillation_lambda_frozen: 0.0,
        }
    }

    #[test]
    fn test_default_polyak_tau_is_005() {
        assert!(
            (default_polyak_tau() - 0.005).abs() < 1e-12,
            "default_polyak_tau must be 0.005"
        );
    }

    #[test]
    fn test_validate_distillation_lambda_polyak_negative_rejected() {
        let mut cfg = base_config();
        cfg.distillation_lambda_polyak = -0.1;
        let result = PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), cfg, 42);
        assert!(
            result.is_err(),
            "negative distillation_lambda_polyak must be rejected"
        );
        match result.unwrap_err() {
            PcError::ConfigValidation(msg) => {
                assert!(
                    msg.contains("distillation_lambda_polyak"),
                    "error must mention field name: {msg}"
                );
            }
            other => panic!("expected ConfigValidation, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_polyak_tau_out_of_range_rejected() {
        let mut cfg = base_config();
        cfg.polyak_tau = 1.5;
        let result = PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), cfg, 42);
        assert!(result.is_err(), "polyak_tau=1.5 must be rejected");
        match result.unwrap_err() {
            PcError::ConfigValidation(msg) => {
                assert!(
                    msg.contains("polyak_tau"),
                    "error must mention field name: {msg}"
                );
            }
            other => panic!("expected ConfigValidation, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_polyak_tau_zero_with_lambda_positive_rejected() {
        let mut cfg = base_config();
        cfg.distillation_lambda_polyak = 0.1;
        cfg.polyak_tau = 0.0;
        let result = PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), cfg, 42);
        assert!(
            result.is_err(),
            "polyak_tau=0 with lambda>0 must be rejected"
        );
        match result.unwrap_err() {
            PcError::ConfigValidation(msg) => {
                assert!(
                    msg.contains("polyak_tau"),
                    "error must mention field name: {msg}"
                );
            }
            other => panic!("expected ConfigValidation, got {other:?}"),
        }
    }
}
