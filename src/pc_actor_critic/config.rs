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

/// Tolerance for detecting the `-1.0` sentinel value in the replay
/// floor fields (`scale_floor_replay`, `critic_floor_replay`) after
/// JSON round-trip. Exactly `-1.0` is representable in f64 and should
/// round-trip bit-perfectly, but defensive tolerance prevents
/// misdetection if a future serde backend introduces any ULP-level
/// noise.
pub(crate) const SENTINEL_EPSILON: f64 = 1e-9;

/// Action space discriminator (v4.0.0). Determines the policy gradient
/// form, the public step API, and the replay buffer transition schema.
///
/// Brainstorm Q1: this enum is **not** directly serialized in
/// `ReplayTransition::action` — that is a separate `Action` enum with
/// `#[serde(untagged)]` representation. `ActionSpace` lives in config
/// only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ActionSpace {
    /// Discrete: actor logits → softmax → sample/argmax over a finite set.
    /// REINFORCE policy gradient. v3.x default; preserved bit-for-bit.
    #[default]
    Discrete,
    /// Continuous: actor output is the mean μ(s) of a fixed-σ Gaussian
    /// policy. Sampled action `a = μ + σ·ε`, gradient `(a − μ) / σ²`.
    Continuous,
}

/// Default `policy_sigma` for continuous Gaussian policy.
fn default_policy_sigma() -> f64 {
    0.1
}

/// Returns true when `value` matches the `-1.0` "opt-in not provided"
/// sentinel within `SENTINEL_EPSILON` tolerance.
///
/// Shared between `scale_floor_replay` (v2.2.1) and
/// `critic_floor_replay` (v3.0.0) — both use the same tri-state
/// sentinel semantics. Inlined to keep the hot-path gate branch-free
/// after monomorphization.
#[inline]
pub(crate) fn is_replay_floor_sentinel(value: f64) -> bool {
    (value - (-1.0)).abs() < SENTINEL_EPSILON
}

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

/// Default positive-reward-only filter flag for the replay buffer.
fn default_replay_positive_only() -> bool {
    true
}

/// Default batch size for each `replay_learn()` call.
fn default_replay_batch_size() -> usize {
    64
}

/// Default `scale_floor_replay` sentinel.
///
/// Returns `-1.0` to signal "opt-in not provided" — replay-under-FROZEN
/// inherits the conservative no-op semantics. Set to a value in
/// `[0.0, 10*scale_ceil]` to opt in: that value becomes the actor's
/// effective scale during `replay_learn()` regardless of hysteresis state.
fn default_scale_floor_replay() -> f64 {
    -1.0
}

/// Default `critic_floor_replay` sentinel (v3.0.0).
///
/// Returns `-1.0` to signal "opt-in not provided". Mirror of
/// `default_scale_floor_replay()` for the critic-side knob added in
/// v3.0.0 alongside the new `critic_hysteresis` weight-update gate.
fn default_critic_floor_replay() -> f64 {
    -1.0
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
/// use pc_rl_core::ActionSpace;
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
///     replay_training_capacity: 0,
///     replay_recent_capacity: 0,
///     replay_positive_only: true,
///     replay_batch_size: 64,
///     scale_floor_replay: -1.0,
///     critic_floor_replay: -1.0,
///     action_space: ActionSpace::Discrete,
///     policy_sigma: 0.1,
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
    /// Capacity of the training-phase replay compartment.
    /// 0 disables the replay buffer entirely. Default: 0.
    #[serde(default)]
    pub replay_training_capacity: usize,
    /// Capacity of the recent-stress replay compartment.
    /// 0 disables stress-phase recording. Default: 0.
    #[serde(default)]
    pub replay_recent_capacity: usize,
    /// If true, only transitions with reward > 0 are stored. Default: true.
    #[serde(default = "default_replay_positive_only")]
    pub replay_positive_only: bool,
    /// Batch size for each `replay_learn()` call. Default: 64.
    #[serde(default = "default_replay_batch_size")]
    pub replay_batch_size: usize,
    /// Replay-only scale floor: opt-in override of the actor's effective
    /// scale during `replay_learn()`.
    ///
    /// `-1.0` (default) acts as a sentinel meaning "opt-in not provided" —
    /// replay-under-FROZEN inherits the conservative no-op semantics
    /// (actor weights frozen for replay, only critic updates). Any value
    /// in `[0.0, 10 * scale_ceil]` opts in: that value becomes the
    /// effective actor scale during replay regardless of hysteresis
    /// state, allowing controlled actor learning from positive-reward
    /// memories even while FROZEN. Values in `(-1.0, 0.0)`, non-finite
    /// values, and values above `10 * scale_ceil` are rejected by
    /// `validate_config`.
    ///
    /// # Upper-bound rationale (`10 * scale_ceil`)
    ///
    /// The validator caps the opt-in at `10 × scale_ceil` rather than at
    /// `scale_ceil` itself for two reasons:
    ///
    /// 1. **Replay is batched**: a single `replay_learn()` call applies
    ///    gradients from `replay_batch_size` transitions. A replay scale
    ///    equal to the online `scale_ceil` would under-correct for the
    ///    higher information content in a curated positive-reward batch,
    ///    so mild super-unity (2×–5×) is a legitimate tuning regime.
    ///
    /// 2. **Typo safety net**: users occasionally set `scale_floor_replay`
    ///    to raw learning-rate numbers (e.g. `0.5`, `1.0`) rather than the
    ///    scale-of-scale semantics. `10 × scale_ceil` gives wide latitude
    ///    for legitimate tuning while still rejecting an obvious order-of-
    ///    magnitude typo (e.g. `100.0` where `1.0` was meant).
    ///
    /// Users who need a cap tighter than `10 × scale_ceil` should simply
    /// choose their `scale_floor_replay` value accordingly — the validator
    /// enforces the outer bound, not the recommended operating point.
    ///
    /// Detection of the `-1.0` sentinel uses `SENTINEL_EPSILON`
    /// (crate-private constant) tolerance so future serde backends with
    /// ULP-level round-trip noise still match the sentinel.
    #[serde(default = "default_scale_floor_replay")]
    pub scale_floor_replay: f64,
    /// Replay-only scale floor for the **critic** under FROZEN
    /// hysteresis (v3.0.0).
    ///
    /// Mirror of [`scale_floor_replay`](Self::scale_floor_replay) for
    /// the critic side. As of v3.0.0, `critic_hysteresis = true` now
    /// enforces the FROZEN state on critic weight updates (this was
    /// tracked but unenforced in v2.x — see CHANGELOG breaking notes).
    /// This field provides the symmetric replay opt-in.
    ///
    /// `-1.0` (default) acts as the "opt-in not provided" sentinel —
    /// replay-under-FROZEN-critic clamps to `scale_floor` (no critic
    /// update under defaults). Any value in `[0.0, 10 * scale_ceil]`
    /// opts in: that value becomes the effective critic scale during
    /// `replay_learn()` regardless of `critic_hysteresis.state`.
    /// Strict-positive (`> 0.0`) constitutes a real opt-in.
    ///
    /// # Recommended pairing with `scale_floor_replay`
    ///
    /// For coherent actor-critic dynamics under FROZEN-replay, set
    /// this field IN LOCKSTEP with
    /// [`scale_floor_replay`](Self::scale_floor_replay). Partial
    /// opt-in (one knob active, the other at sentinel) produces
    /// actor-critic desynchronization — the moving side learns from
    /// storage while the gated side stays still, drifting the two
    /// networks apart. Recommended symmetric pairs:
    ///
    /// - `(-1.0, -1.0)` — full stress protection (default).
    /// - `(0.3, 0.3)` — mild symmetric recovery.
    /// - `(1.0, 1.0)` — aggressive symmetric recovery.
    ///
    /// Asymmetric opt-in is syntactically allowed but not recommended
    /// (see CHANGELOG v3.0.0 §"Note on pairing").
    ///
    /// # Sentinel detection and validation
    ///
    /// Detection of the `-1.0` sentinel uses `SENTINEL_EPSILON`
    /// (crate-private constant) tolerance, identical to
    /// `scale_floor_replay`. Validation rejects values in
    /// `(-1.0, 0.0)`, non-finite values, and values above
    /// `10 * scale_ceil` at both `PcActorCritic::new` and
    /// `apply_config` paths.
    #[serde(default = "default_critic_floor_replay")]
    pub critic_floor_replay: f64,
    /// v4.0.0 — action space discriminator. Default `Discrete` preserves
    /// v3.x behavior for save files written before this field existed.
    /// Brainstorm Q1: see `Action` enum in replay.rs for the per-transition
    /// representation.
    #[serde(default)]
    pub action_space: ActionSpace,
    /// v4.0.0 — Gaussian policy std-dev for continuous mode. Ignored when
    /// `action_space == Discrete`. Must be `> 0.0 && finite` when continuous.
    /// Default 0.1.
    #[serde(default = "default_policy_sigma")]
    pub policy_sigma: f64,
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
            replay_training_capacity: 0,
            replay_recent_capacity: 0,
            replay_positive_only: true,
            replay_batch_size: 64,
            scale_floor_replay: -1.0,
            critic_floor_replay: -1.0,
            action_space: ActionSpace::Discrete,
            policy_sigma: 0.1,
        }
    }

    #[test]
    fn test_scale_floor_replay_default_is_minus_one_sentinel() {
        // The default sentinel `-1.0` signals "opt-in not provided".
        // Replay-under-FROZEN preserves the conservative no-op semantics
        // unless the user explicitly opts in by setting this field to a
        // value in `[0.0, 10 * scale_ceil]`. Validation and gate logic
        // live in `validate_config` and `effective_actor_scale_for_mode`
        // respectively.
        let cfg = base_config();
        // `base_config` mirrors the user-facing default for this field.
        assert!(
            is_replay_floor_sentinel(cfg.scale_floor_replay),
            "default scale_floor_replay must be -1.0 sentinel, got {}",
            cfg.scale_floor_replay
        );
        // The serde-default helper must agree with the struct literal.
        assert!(
            is_replay_floor_sentinel(default_scale_floor_replay()),
            "default_scale_floor_replay() must return -1.0 sentinel"
        );
    }

    #[test]
    fn test_critic_floor_replay_default_is_minus_one_sentinel() {
        // v3.0.0 — critic-side mirror of the scale_floor_replay sentinel.
        // The default `-1.0` signals "opt-in not provided" — the new
        // critic-hysteresis-FROZEN gate (also v3.0.0) clamps the critic
        // to `scale_floor` during replay unless this field is explicitly
        // set to a value in `[0.0, 10 * scale_ceil]`.
        let cfg = base_config();
        assert!(
            is_replay_floor_sentinel(cfg.critic_floor_replay),
            "default critic_floor_replay must be -1.0 sentinel, got {}",
            cfg.critic_floor_replay
        );
        assert!(
            is_replay_floor_sentinel(default_critic_floor_replay()),
            "default_critic_floor_replay() must return -1.0 sentinel"
        );
    }

    #[test]
    fn test_action_space_default_is_discrete() {
        // Brainstorm Q1: default Discrete preserves v3.x behavior bit-for-bit
        // when consumers don't set the field explicitly.
        let cfg = base_config();
        assert_eq!(cfg.action_space, ActionSpace::Discrete);
        assert_eq!(ActionSpace::default(), ActionSpace::Discrete);
    }

    #[test]
    fn test_policy_sigma_default_is_0_1() {
        let cfg = base_config();
        assert!(
            (cfg.policy_sigma - 0.1).abs() < 1e-12,
            "default policy_sigma must be 0.1, got {}",
            cfg.policy_sigma
        );
        assert!(
            (default_policy_sigma() - 0.1).abs() < 1e-12,
            "default_policy_sigma() helper must return 0.1"
        );
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
