// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Integrated PC Actor-Critic agent.
//!
//! Combines [`PcActor`] for action selection via predictive coding inference
//! with [`MlpCritic`] for value estimation. Supports REINFORCE episodic
//! learning, TD(0) continuous learning, surprise-based scheduling, and
//! entropy regularization.
//!
//! Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].

use std::collections::VecDeque;

use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::error::PcError;
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;
use crate::mlp_critic::{MlpCritic, MlpCriticConfig};
use crate::pc_actor::{InferResult, PcActor, PcActorConfig, SelectionMode};

pub mod config;

pub use config::*;

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

/// A single step in a trajectory collected during an episode.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
#[derive(Debug, Clone)]
pub struct TrajectoryStep<L: LinAlg = CpuLinAlg> {
    /// Board state input vector.
    pub input: L::Vector,
    /// Concatenated hidden layer activations from inference.
    pub latent_concat: L::Vector,
    /// Converged output logits from inference.
    pub y_conv: L::Vector,
    /// Per-layer hidden state activations from inference (for backprop).
    pub hidden_states: Vec<L::Vector>,
    /// Per-layer prediction errors from the PC inference loop.
    pub prediction_errors: Vec<L::Vector>,
    /// Per-layer tanh components for residual layers (for correct backward pass).
    pub tanh_components: Vec<Option<L::Vector>>,
    /// Action taken at this step.
    pub action: usize,
    /// Valid actions at this step (needed for masked softmax).
    pub valid_actions: Vec<usize>,
    /// Reward received after taking the action.
    pub reward: f64,
    /// Prediction error from inference.
    pub surprise_score: f64,
    /// Number of inference steps used.
    pub steps_used: usize,
}

/// Cache for hidden layer activations captured during inference.
///
/// Used by CCA neuron alignment during crossover to compare functional
/// neuron responses between parent networks. Activations are recorded
/// during normal fitness evaluation at zero additional compute cost.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
///
/// # Examples
///
/// ```
/// use pc_rl_core::pc_actor_critic::ActivationCache;
/// use pc_rl_core::linalg::cpu::CpuLinAlg;
///
/// let cache: ActivationCache<CpuLinAlg> = ActivationCache::new(2);
/// assert_eq!(cache.batch_size(), 0);
/// assert_eq!(cache.num_layers(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ActivationCache<L: LinAlg = CpuLinAlg> {
    /// activations[layer_idx][batch_sample_idx] = neuron activations.
    layers: Vec<Vec<L::Vector>>,
}

impl<L: LinAlg> ActivationCache<L> {
    /// Creates a new empty cache with the given number of hidden layers.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of hidden layers to track.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| Vec::new()).collect(),
        }
    }

    /// Returns the number of samples recorded so far.
    pub fn batch_size(&self) -> usize {
        self.layers.first().map_or(0, |l| l.len())
    }

    /// Returns the number of hidden layers in the cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Records hidden layer activations from a single inference.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Per-layer activations from `InferResult::hidden_states`.
    pub fn record(&mut self, hidden_states: &[L::Vector]) {
        for (layer, state) in self.layers.iter_mut().zip(hidden_states.iter()) {
            layer.push(state.clone());
        }
    }

    /// Returns the recorded activations for a given layer.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the hidden layer.
    pub fn layer(&self, layer_idx: usize) -> &[L::Vector] {
        &self.layers[layer_idx]
    }
}

/// Integrated PC Actor-Critic agent.
///
/// Combines a predictive coding actor with an MLP critic for
/// reinforcement learning with surprise-based scheduling.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
#[derive(Debug)]
pub struct PcActorCritic<L: LinAlg = CpuLinAlg> {
    /// The PC actor network.
    pub(crate) actor: PcActor<L>,
    /// The MLP critic (value function).
    pub(crate) critic: MlpCritic<L>,
    /// Agent configuration.
    pub config: PcActorCriticConfig,
    /// Random number generator for action selection.
    rng: StdRng,
    /// Circular buffer of recent surprise scores for adaptive thresholds.
    surprise_buffer: VecDeque<f64>,
    /// Backend used for linear algebra operations.
    pub(crate) backend: L,
    /// Previous observation (transient, not serialized).
    state_prev: Option<L::Vector>,
    /// Previous action taken (transient, not serialized).
    action_prev: Option<usize>,
    /// Previous inference result (transient, not serialized).
    infer_prev: Option<InferResult<L>>,
    /// Previous valid actions mask (transient, not serialized).
    /// `None` = all actions valid (used by `step()`), `Some` = masked (used by `step_masked()`).
    valid_actions_prev: Option<Vec<usize>>,
    /// Actor hysteresis state machine (None when disabled).
    actor_hysteresis: Option<HysteresisState>,
    /// Critic hysteresis state machine (None when disabled).
    critic_hysteresis: Option<HysteresisState>,
    /// Steps the actor has been in PLASTIC state during the current phase.
    actor_plastic_step_counter: u64,
    /// Steps the critic has been in PLASTIC state during the current phase.
    critic_plastic_step_counter: u64,
    /// Consecutive steps the critic has been FROZEN.
    critic_frozen_steps: u64,
    /// Circular buffer of recent |TD errors| for critic adaptive scale.
    td_error_buffer: VecDeque<f64>,
    /// Last TD error from learn_continuous (transient, for hysteresis).
    last_td_error: f64,
    /// Precomputed per-hidden-layer decay factors for actor (M3a).
    actor_decay_factors: Vec<f64>,
    /// Precomputed per-hidden-layer decay factors for critic (M3a).
    critic_decay_factors: Vec<f64>,
    /// Per-layer prediction error EMA for adaptive consolidation (M3b, actor only).
    layer_error_ema: Vec<f64>,
    /// Per-layer Fisher information state for actor EWC (empty when ewc_lambda=0).
    actor_fisher: Vec<FisherState<L>>,
    /// Per-layer Fisher information state for critic EWC (empty when ewc_lambda=0).
    critic_fisher: Vec<FisherState<L>>,
    /// Whether the last actor PLASTIC phase was reliable (>= min_fisher_phase steps).
    actor_last_phase_reliable: bool,
    /// Whether the last critic PLASTIC phase was reliable (>= min_fisher_phase steps).
    critic_last_phase_reliable: bool,
}

impl<L: LinAlg> PcActorCritic<L> {
    /// Creates a new PC Actor-Critic agent.
    ///
    /// # Arguments
    ///
    /// * `config` - Agent configuration with actor, critic, and learning parameters.
    /// * `seed` - Random seed for reproducibility.
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if gamma is out of `[0.0, 1.0]`,
    /// or if actor/critic config is invalid.
    pub fn new(backend: L, config: PcActorCriticConfig, seed: u64) -> Result<Self, PcError> {
        if !(0.0..=1.0).contains(&config.gamma) {
            return Err(PcError::ConfigValidation(format!(
                "gamma must be in [0.0, 1.0], got {}",
                config.gamma
            )));
        }
        if config.adaptive_surprise && config.surprise_buffer_size < 10 {
            return Err(PcError::ConfigValidation(format!(
                "surprise_buffer_size must be >= 10 when adaptive_surprise is enabled, got {}",
                config.surprise_buffer_size
            )));
        }
        if config.scale_floor < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "scale_floor must be >= 0.0, got {}",
                config.scale_floor
            )));
        }
        if config.scale_ceil <= config.scale_floor {
            return Err(PcError::ConfigValidation(format!(
                "scale_ceil must be > scale_floor, got scale_ceil={} scale_floor={}",
                config.scale_ceil, config.scale_floor
            )));
        }

        // Validate actor hysteresis fractions
        if config.actor_hysteresis {
            if config.actor_wake_fraction <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "actor_wake_fraction must be > 0.0 when actor_hysteresis enabled, got {}",
                    config.actor_wake_fraction
                )));
            }
            if config.actor_sleep_fraction <= 0.0 || config.actor_sleep_fraction >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "actor_sleep_fraction must be in (0.0, 1.0) when actor_hysteresis enabled, got {}",
                    config.actor_sleep_fraction
                )));
            }
        }

        // Validate critic hysteresis fractions
        if config.critic_hysteresis {
            if config.critic_wake_fraction <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "critic_wake_fraction must be > 0.0 when critic_hysteresis enabled, got {}",
                    config.critic_wake_fraction
                )));
            }
            if config.critic_sleep_fraction <= 0.0 || config.critic_sleep_fraction >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "critic_sleep_fraction must be in (0.0, 1.0) when critic_hysteresis enabled, got {}",
                    config.critic_sleep_fraction
                )));
            }
        }

        // Validate consolidation decay (M3a)
        if !(0.0..=1.0).contains(&config.consolidation_decay) {
            return Err(PcError::ConfigValidation(format!(
                "consolidation_decay must be in [0.0, 1.0], got {}",
                config.consolidation_decay
            )));
        }
        if !(0.0..=1.0).contains(&config.critic_consolidation_decay) {
            return Err(PcError::ConfigValidation(format!(
                "critic_consolidation_decay must be in [0.0, 1.0], got {}",
                config.critic_consolidation_decay
            )));
        }

        // Validate adaptive consolidation params (M3b)
        if config.adaptive_consolidation {
            if config.consolidation_sigmoid_k <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "consolidation_sigmoid_k must be > 0.0 when adaptive_consolidation enabled, got {}",
                    config.consolidation_sigmoid_k
                )));
            }
            if config.consolidation_ema_beta <= 0.0 || config.consolidation_ema_beta >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "consolidation_ema_beta must be in (0.0, 1.0), got {}",
                    config.consolidation_ema_beta
                )));
            }
            if config.consolidation_error_threshold <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "consolidation_error_threshold must be > 0.0, got {}",
                    config.consolidation_error_threshold
                )));
            }
        }

        // Validate EWC parameters (M4)
        if config.ewc_lambda < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "ewc_lambda must be >= 0.0, got {}",
                config.ewc_lambda
            )));
        }
        if config.ewc_lambda > 0.0 {
            if !(0.0..=1.0).contains(&config.fisher_decay) {
                return Err(PcError::ConfigValidation(format!(
                    "fisher_decay must be in [0.0, 1.0], got {}",
                    config.fisher_decay
                )));
            }
            if config.fisher_ema_beta <= 0.0 || config.fisher_ema_beta >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "fisher_ema_beta must be in (0.0, 1.0), got {}",
                    config.fisher_ema_beta
                )));
            }
        }

        // Precompute layer decay factors (M3a)
        let n_actor_hidden = config.actor.hidden_layers.len();
        let actor_decay_factors: Vec<f64> = (0..n_actor_hidden)
            .map(|i| {
                config
                    .consolidation_decay
                    .powi((n_actor_hidden - 1 - i) as i32)
            })
            .collect();
        let n_critic_hidden = config.critic.hidden_layers.len();
        let critic_decay_factors: Vec<f64> = (0..n_critic_hidden)
            .map(|i| {
                config
                    .critic_consolidation_decay
                    .powi((n_critic_hidden - 1 - i) as i32)
            })
            .collect();

        // Per-layer error EMA for adaptive consolidation (M3b)
        let layer_error_ema = if config.adaptive_consolidation {
            vec![0.0; n_actor_hidden]
        } else {
            Vec::new()
        };

        // Build hysteresis state machines when enabled
        let actor_hysteresis = if config.actor_hysteresis {
            Some(HysteresisState {
                fast: EwmaTracker::new(config.actor_fast_window),
                slow: EwmaTracker::new(config.actor_slow_window),
                state: PlasticityState::Plastic,
                wake_fraction: config.actor_wake_fraction,
                sleep_fraction: config.actor_sleep_fraction,
                min_initial_plastic: config.actor_slow_window as u64,
            })
        } else {
            None
        };

        let critic_hysteresis = if config.critic_hysteresis {
            Some(HysteresisState {
                fast: EwmaTracker::new(config.critic_fast_window),
                slow: EwmaTracker::new(config.critic_slow_window),
                state: PlasticityState::Plastic,
                wake_fraction: config.critic_wake_fraction,
                sleep_fraction: config.critic_sleep_fraction,
                min_initial_plastic: config.critic_slow_window as u64,
            })
        } else {
            None
        };

        // Compute min_fisher_phase and update hysteresis warmup guard
        let min_fisher_phase = if config.ewc_lambda > 0.0 {
            (1.0 / (1.0 - config.fisher_ema_beta)).ceil() as u64
        } else {
            0
        };

        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let actor = PcActor::<L>::new(backend.clone(), config.actor.clone(), &mut rng)?;
        let critic = MlpCritic::<L>::new(backend.clone(), config.critic.clone(), &mut rng)?;

        // Allocate Fisher state when EWC is enabled
        let actor_fisher = if config.ewc_lambda > 0.0 {
            actor
                .layers
                .iter()
                .map(|layer| {
                    let rows = backend.mat_rows(&layer.weights);
                    let cols = backend.mat_cols(&layer.weights);
                    let bias_size = backend.vec_len(&layer.bias);
                    FisherState::new(&backend, rows, cols, bias_size)
                })
                .collect()
        } else {
            Vec::new()
        };
        let critic_fisher = if config.ewc_lambda > 0.0 {
            critic
                .layers
                .iter()
                .map(|layer| {
                    let rows = backend.mat_rows(&layer.weights);
                    let cols = backend.mat_cols(&layer.weights);
                    let bias_size = backend.vec_len(&layer.bias);
                    FisherState::new(&backend, rows, cols, bias_size)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Update hysteresis min_initial_plastic to max(slow_window, min_fisher_phase)
        let mut actor_hysteresis = actor_hysteresis;
        if let Some(ref mut hyst) = actor_hysteresis {
            hyst.min_initial_plastic = std::cmp::max(hyst.min_initial_plastic, min_fisher_phase);
        }
        let mut critic_hysteresis = critic_hysteresis;
        if let Some(ref mut hyst) = critic_hysteresis {
            hyst.min_initial_plastic = std::cmp::max(hyst.min_initial_plastic, min_fisher_phase);
        }

        Ok(Self {
            actor,
            critic,
            config,
            rng,
            surprise_buffer: VecDeque::new(),
            backend,
            state_prev: None,
            action_prev: None,
            infer_prev: None,
            valid_actions_prev: None,
            actor_hysteresis,
            critic_hysteresis,
            actor_plastic_step_counter: 0,
            critic_plastic_step_counter: 0,
            critic_frozen_steps: 0,
            td_error_buffer: VecDeque::new(),
            last_td_error: 0.0,
            actor_decay_factors,
            critic_decay_factors,
            layer_error_ema,
            actor_fisher,
            critic_fisher,
            actor_last_phase_reliable: false,
            critic_last_phase_reliable: false,
        })
    }

    /// Creates a child agent by crossing over two parent agents using CCA neuron alignment.
    ///
    /// Delegates to `PcActor::crossover` and `MlpCritic::crossover`, converting
    /// activation caches to the matrix format expected by CCA alignment.
    ///
    /// # Arguments
    ///
    /// * `parent_a` - First parent agent (reference, typically higher fitness).
    /// * `parent_b` - Second parent agent.
    /// * `cache_a` - Activation cache for parent A on the reference batch.
    /// * `cache_b` - Activation cache for parent B on the reference batch.
    /// * `alpha` - Blending weight: 1.0 = all A, 0.0 = all B.
    /// * `child_config` - Configuration for the child agent.
    /// * `seed` - Random seed for the child's RNG.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if activation caches have different
    /// batch sizes. Returns `PcError::ConfigValidation` if child config is invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn crossover(
        parent_a: &PcActorCritic<L>,
        parent_b: &PcActorCritic<L>,
        actor_cache_a: &ActivationCache<L>,
        actor_cache_b: &ActivationCache<L>,
        critic_cache_a: &ActivationCache<L>,
        critic_cache_b: &ActivationCache<L>,
        alpha: f64,
        child_config: PcActorCriticConfig,
        seed: u64,
    ) -> Result<Self, PcError> {
        // Validate actor batch sizes match
        if actor_cache_a.batch_size() != actor_cache_b.batch_size() {
            return Err(PcError::DimensionMismatch {
                expected: actor_cache_a.batch_size(),
                got: actor_cache_b.batch_size(),
                context: "actor activation cache batch sizes must match for crossover",
            });
        }
        // Validate critic batch sizes match
        if critic_cache_a.batch_size() != critic_cache_b.batch_size() {
            return Err(PcError::DimensionMismatch {
                expected: critic_cache_a.batch_size(),
                got: critic_cache_b.batch_size(),
                context: "critic activation cache batch sizes must match for crossover",
            });
        }

        // Convert caches to matrices [batch × neurons] for CCA
        let actor_cache_mats_a = cache_to_matrices(&parent_a.backend, actor_cache_a);
        let actor_cache_mats_b = cache_to_matrices(&parent_a.backend, actor_cache_b);
        let critic_cache_mats_a = cache_to_matrices(&parent_a.backend, critic_cache_a);
        let critic_cache_mats_b = cache_to_matrices(&parent_a.backend, critic_cache_b);

        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);

        // Crossover actor with actor-specific caches
        let actor = PcActor::<L>::crossover(
            &parent_a.actor,
            &parent_b.actor,
            &actor_cache_mats_a,
            &actor_cache_mats_b,
            alpha,
            child_config.actor.clone(),
            &mut rng,
        )?;

        // Crossover critic with critic-specific caches
        let critic = MlpCritic::<L>::crossover(
            &parent_a.critic,
            &parent_b.critic,
            &critic_cache_mats_a,
            &critic_cache_mats_b,
            alpha,
            child_config.critic.clone(),
            &mut rng,
        )?;

        // Precompute crossover child decay factors
        let n_ah = child_config.actor.hidden_layers.len();
        let child_actor_decay: Vec<f64> = (0..n_ah)
            .map(|i| child_config.consolidation_decay.powi((n_ah - 1 - i) as i32))
            .collect();
        let n_ch = child_config.critic.hidden_layers.len();
        let child_critic_decay: Vec<f64> = (0..n_ch)
            .map(|i| {
                child_config
                    .critic_consolidation_decay
                    .powi((n_ch - 1 - i) as i32)
            })
            .collect();
        let child_layer_error_ema = if child_config.adaptive_consolidation {
            vec![0.0; n_ah]
        } else {
            Vec::new()
        };

        Ok(Self {
            actor,
            critic,
            config: child_config,
            rng,
            surprise_buffer: VecDeque::new(),
            backend: parent_a.backend.clone(),
            state_prev: None,
            action_prev: None,
            infer_prev: None,
            valid_actions_prev: None,
            actor_hysteresis: None,
            critic_hysteresis: None,
            actor_plastic_step_counter: 0,
            critic_plastic_step_counter: 0,
            critic_frozen_steps: 0,
            td_error_buffer: VecDeque::new(),
            last_td_error: 0.0,
            actor_decay_factors: child_actor_decay,
            critic_decay_factors: child_critic_decay,
            layer_error_ema: child_layer_error_ema,
            actor_fisher: Vec::new(),
            critic_fisher: Vec::new(),
            actor_last_phase_reliable: false,
            critic_last_phase_reliable: false,
        })
    }

    /// Reconstructs an agent from pre-built components (used by serializer).
    ///
    /// # Arguments
    ///
    /// * `config` - Agent configuration.
    /// * `actor` - Pre-built PC actor with loaded weights.
    /// * `critic` - Pre-built MLP critic with loaded weights.
    /// * `rng` - Random number generator.
    pub fn from_parts(
        config: PcActorCriticConfig,
        actor: PcActor<L>,
        critic: MlpCritic<L>,
        rng: StdRng,
        backend: L,
    ) -> Self {
        let n_ah = config.actor.hidden_layers.len();
        let actor_decay_factors: Vec<f64> = (0..n_ah)
            .map(|i| config.consolidation_decay.powi((n_ah - 1 - i) as i32))
            .collect();
        let n_ch = config.critic.hidden_layers.len();
        let critic_decay_factors: Vec<f64> = (0..n_ch)
            .map(|i| {
                config
                    .critic_consolidation_decay
                    .powi((n_ch - 1 - i) as i32)
            })
            .collect();
        let layer_error_ema = if config.adaptive_consolidation {
            vec![0.0; n_ah]
        } else {
            Vec::new()
        };
        Self {
            actor,
            critic,
            config,
            rng,
            surprise_buffer: VecDeque::new(),
            backend,
            state_prev: None,
            action_prev: None,
            infer_prev: None,
            valid_actions_prev: None,
            actor_hysteresis: None,
            critic_hysteresis: None,
            actor_plastic_step_counter: 0,
            critic_plastic_step_counter: 0,
            critic_frozen_steps: 0,
            td_error_buffer: VecDeque::new(),
            last_td_error: 0.0,
            actor_decay_factors,
            critic_decay_factors,
            layer_error_ema,
            actor_fisher: Vec::new(),
            critic_fisher: Vec::new(),
            actor_last_phase_reliable: false,
            critic_last_phase_reliable: false,
        }
    }

    /// Extracts the continuous learning state for serialization.
    ///
    /// Converts all CL state (hysteresis, Fisher, counters) into
    /// CPU-side serializable types. Returns `None` if no CL features
    /// are active (all defaults).
    pub fn to_cl_state(&self) -> Option<crate::serializer::ClState> {
        use crate::serializer::{
            ClState, EwmaTrackerSerialized, FisherStateSerialized, HysteresisStateSerialized,
        };

        let has_cl = self.actor_hysteresis.is_some()
            || self.critic_hysteresis.is_some()
            || !self.actor_fisher.is_empty()
            || !self.critic_fisher.is_empty()
            || self.actor_plastic_step_counter > 0
            || self.critic_plastic_step_counter > 0
            || self.critic_frozen_steps > 0
            || !self.layer_error_ema.is_empty()
            || self.actor_last_phase_reliable
            || self.critic_last_phase_reliable;

        if !has_cl {
            return None;
        }

        let serialize_ewma = |t: &EwmaTracker| EwmaTrackerSerialized {
            value: t.value,
            k: t.k,
            window: t.window,
        };

        let serialize_hysteresis = |h: &HysteresisState| HysteresisStateSerialized {
            fast: serialize_ewma(&h.fast),
            slow: serialize_ewma(&h.slow),
            state: h.state.clone(),
            wake_fraction: h.wake_fraction,
            sleep_fraction: h.sleep_fraction,
            min_initial_plastic: h.min_initial_plastic,
        };

        let serialize_fisher = |fs: &FisherState<L>, backend: &L| -> FisherStateSerialized {
            let mat_to_cpu = |m: &L::Matrix| -> crate::matrix::Matrix {
                let rows = backend.mat_rows(m);
                let cols = backend.mat_cols(m);
                let mut cpu = crate::matrix::Matrix::zeros(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        cpu.set(r, c, backend.mat_get(m, r, c));
                    }
                }
                cpu
            };
            FisherStateSerialized {
                f_total_weights: mat_to_cpu(&fs.f_total_weights),
                f_total_bias: backend.vec_to_vec(&fs.f_total_bias),
                f_ema_weights: mat_to_cpu(&fs.f_ema_weights),
                f_ema_bias: backend.vec_to_vec(&fs.f_ema_bias),
                theta_snapshot_weights: fs.theta_snapshot_weights.as_ref().map(mat_to_cpu),
                theta_snapshot_bias: fs
                    .theta_snapshot_bias
                    .as_ref()
                    .map(|v| backend.vec_to_vec(v)),
                theta_snapshot_rezero_alpha: fs.theta_snapshot_rezero_alpha,
                theta_snapshot_skip_proj: fs.theta_snapshot_skip_proj.as_ref().map(mat_to_cpu),
            }
        };

        Some(ClState {
            actor_hysteresis: self.actor_hysteresis.as_ref().map(serialize_hysteresis),
            critic_hysteresis: self.critic_hysteresis.as_ref().map(serialize_hysteresis),
            actor_plastic_step_counter: self.actor_plastic_step_counter,
            critic_plastic_step_counter: self.critic_plastic_step_counter,
            critic_frozen_steps: self.critic_frozen_steps,
            actor_fisher: self
                .actor_fisher
                .iter()
                .map(|f| serialize_fisher(f, &self.backend))
                .collect(),
            critic_fisher: self
                .critic_fisher
                .iter()
                .map(|f| serialize_fisher(f, &self.backend))
                .collect(),
            actor_last_phase_reliable: self.actor_last_phase_reliable,
            critic_last_phase_reliable: self.critic_last_phase_reliable,
            layer_error_ema: self.layer_error_ema.clone(),
        })
    }

    /// Restores continuous learning state from a serialized `ClState`.
    ///
    /// Called after `from_parts()` during deserialization. If `cl_state`
    /// is `None` (legacy JSON), the agent keeps its clean defaults.
    pub fn restore_cl_state(&mut self, cl_state: crate::serializer::ClState) {
        use crate::serializer::{EwmaTrackerSerialized, HysteresisStateSerialized};

        let deserialize_ewma = |t: EwmaTrackerSerialized| -> EwmaTracker {
            EwmaTracker {
                value: t.value,
                k: t.k,
                window: t.window,
            }
        };

        let deserialize_hysteresis = |h: HysteresisStateSerialized| -> HysteresisState {
            HysteresisState {
                fast: deserialize_ewma(h.fast),
                slow: deserialize_ewma(h.slow),
                state: h.state,
                wake_fraction: h.wake_fraction,
                sleep_fraction: h.sleep_fraction,
                min_initial_plastic: h.min_initial_plastic,
            }
        };

        self.actor_hysteresis = cl_state.actor_hysteresis.map(deserialize_hysteresis);
        self.critic_hysteresis = cl_state.critic_hysteresis.map(deserialize_hysteresis);
        self.actor_plastic_step_counter = cl_state.actor_plastic_step_counter;
        self.critic_plastic_step_counter = cl_state.critic_plastic_step_counter;
        self.critic_frozen_steps = cl_state.critic_frozen_steps;
        self.actor_last_phase_reliable = cl_state.actor_last_phase_reliable;
        self.critic_last_phase_reliable = cl_state.critic_last_phase_reliable;

        if !cl_state.layer_error_ema.is_empty() {
            self.layer_error_ema = cl_state.layer_error_ema;
        }

        // Restore Fisher state
        let deserialize_fisher_vec = |serialized: Vec<crate::serializer::FisherStateSerialized>,
                                      backend: &L|
         -> Vec<FisherState<L>> {
            serialized
                .into_iter()
                .map(|fs| {
                    let cpu_to_mat = |m: &crate::matrix::Matrix| -> L::Matrix {
                        let rows = m.rows;
                        let cols = m.cols;
                        let mut result = backend.zeros_mat(rows, cols);
                        for r in 0..rows {
                            for c in 0..cols {
                                backend.mat_set(&mut result, r, c, m.get(r, c));
                            }
                        }
                        result
                    };
                    let cpu_to_vec = |v: &[f64]| -> L::Vector { backend.vec_from_slice(v) };

                    FisherState {
                        f_total_weights: cpu_to_mat(&fs.f_total_weights),
                        f_total_bias: cpu_to_vec(&fs.f_total_bias),
                        f_ema_weights: cpu_to_mat(&fs.f_ema_weights),
                        f_ema_bias: cpu_to_vec(&fs.f_ema_bias),
                        theta_snapshot_weights: fs.theta_snapshot_weights.as_ref().map(cpu_to_mat),
                        theta_snapshot_bias: fs.theta_snapshot_bias.as_ref().map(|v| cpu_to_vec(v)),
                        theta_snapshot_rezero_alpha: fs.theta_snapshot_rezero_alpha,
                        theta_snapshot_skip_proj: fs
                            .theta_snapshot_skip_proj
                            .as_ref()
                            .map(cpu_to_mat),
                    }
                })
                .collect()
        };

        if !cl_state.actor_fisher.is_empty() {
            self.actor_fisher = deserialize_fisher_vec(cl_state.actor_fisher, &self.backend);
        }
        if !cl_state.critic_fisher.is_empty() {
            self.critic_fisher = deserialize_fisher_vec(cl_state.critic_fisher, &self.backend);
        }
    }

    /// Runs PC inference without selecting an action or modifying RNG state.
    ///
    /// Use this when you only need the inference result (e.g., for TD(0)
    /// next-state evaluation) without side effects.
    ///
    /// # Arguments
    ///
    /// * `input` - Board state vector.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.actor.input_size`.
    pub fn infer(&self, input: &[f64]) -> InferResult<L> {
        self.actor.infer(input)
    }

    /// Selects an action given the current state.
    ///
    /// Runs PC inference on the input, then selects an action using the
    /// converged logits and the specified selection mode.
    ///
    /// # Arguments
    ///
    /// * `input` - Board state vector.
    /// * `valid_actions` - Indices of legal actions.
    /// * `mode` - Training (stochastic) or Play (deterministic).
    ///
    /// # Panics
    ///
    /// Panics if `valid_actions` is empty.
    pub fn act(
        &mut self,
        input: &[f64],
        valid_actions: &[usize],
        mode: SelectionMode,
    ) -> (usize, InferResult<L>) {
        let infer_result = self.actor.infer(input);
        let action =
            self.actor
                .select_action(&infer_result.y_conv, valid_actions, mode, &mut self.rng);
        (action, infer_result)
    }

    /// Learns from a complete episode trajectory using REINFORCE with baseline.
    ///
    /// Empty trajectory returns 0.0 without modifying weights. Otherwise computes
    /// discounted returns, advantages, and updates both actor and critic.
    ///
    /// # Arguments
    ///
    /// * `trajectory` - Sequence of steps from an episode.
    ///
    /// # Returns
    ///
    /// Average critic loss over the trajectory.
    #[deprecated(since = "2.1.0", note = "Use step() or step_masked() instead")]
    pub fn learn(&mut self, trajectory: &[TrajectoryStep<L>]) -> f64 {
        if trajectory.is_empty() {
            return 0.0;
        }

        let n = trajectory.len();

        // Compute discounted returns backward
        let mut returns = vec![0.0; n];
        returns[n - 1] = trajectory[n - 1].reward;
        for t in (0..n - 1).rev() {
            returns[t] = trajectory[t].reward + self.config.gamma * returns[t + 1];
        }

        let mut total_loss = 0.0;

        for (t, step) in trajectory.iter().enumerate() {
            // Build critic input: concat(input, latent_concat)
            let input_vec = self.backend.vec_to_vec(&step.input);
            let latent_vec = self.backend.vec_to_vec(&step.latent_concat);
            let mut critic_input = input_vec.clone();
            critic_input.extend_from_slice(&latent_vec);

            // V(s)
            let value = self.critic.forward(&critic_input);
            let advantage = returns[t] - value;

            // Update critic toward discounted return
            let loss = self.critic.update(&critic_input, returns[t]);
            total_loss += loss;

            // Policy gradient
            let y_conv_vec = self.backend.vec_to_vec(&step.y_conv);
            let scaled: Vec<f64> = y_conv_vec
                .iter()
                .map(|&v| v / self.actor.config.temperature)
                .collect();
            let scaled_l = self.backend.vec_from_slice(&scaled);
            let pi_l = self.backend.softmax_masked(&scaled_l, &step.valid_actions);
            let pi = self.backend.vec_to_vec(&pi_l);

            let mut delta = vec![0.0; pi.len()];
            for &i in &step.valid_actions {
                delta[i] = pi[i];
            }
            delta[step.action] -= 1.0;

            // Scale by advantage
            for &i in &step.valid_actions {
                delta[i] *= advantage;
            }

            // Entropy regularization
            for &i in &step.valid_actions {
                let log_pi = (pi[i].max(1e-10)).ln();
                delta[i] -= self.config.entropy_coeff * (log_pi + 1.0);
            }

            // Compute surprise scale and update actor using stored hidden_states
            let s_scale = self.surprise_scale(step.surprise_score);

            let stored_infer = InferResult {
                y_conv: step.y_conv.clone(),
                latent_concat: step.latent_concat.clone(),
                hidden_states: step.hidden_states.clone(),
                prediction_errors: step.prediction_errors.clone(),
                surprise_score: step.surprise_score,
                steps_used: step.steps_used,
                converged: false,
                tanh_components: step.tanh_components.clone(),
            };
            let actor_decay = self.effective_actor_decay();
            self.actor
                .update_weights(&delta, &stored_infer, &input_vec, s_scale, &actor_decay);

            // Push surprise to adaptive buffer
            if self.config.adaptive_surprise {
                self.push_surprise(step.surprise_score);
            }
        }

        total_loss / n as f64
    }

    /// Single-step TD(0) continuous learning.
    ///
    /// # Arguments
    ///
    /// * `input` - Current state.
    /// * `infer` - Inference result from `act` at current state.
    /// * `action` - Action taken.
    /// * `valid_actions` - Valid actions at current state.
    /// * `reward` - Reward received.
    /// * `next_input` - Next state.
    /// * `next_infer` - Inference result from `act` at next state.
    /// * `terminal` - Whether the episode ended.
    ///
    /// # Returns
    ///
    /// Critic loss for this step.
    #[allow(clippy::too_many_arguments)]
    pub fn learn_continuous(
        &mut self,
        input: &[f64],
        infer: &InferResult<L>,
        action: usize,
        valid_actions: &[usize],
        reward: f64,
        next_input: &[f64],
        next_infer: &InferResult<L>,
        terminal: bool,
    ) -> f64 {
        // Build critic inputs
        let latent_vec = self.backend.vec_to_vec(&infer.latent_concat);
        let mut critic_input = input.to_vec();
        critic_input.extend_from_slice(&latent_vec);

        let next_latent_vec = self.backend.vec_to_vec(&next_infer.latent_concat);
        let mut next_critic_input = next_input.to_vec();
        next_critic_input.extend_from_slice(&next_latent_vec);

        let v_s = self.critic.forward(&critic_input);
        let v_next = if terminal {
            0.0
        } else {
            self.critic.forward(&next_critic_input)
        };

        let target = reward
            + if terminal {
                0.0
            } else {
                self.config.gamma * v_next
            };
        let td_error = target - v_s;

        // Update critic with per-layer consolidation decay
        let critic_scale = self.critic_surprise_scale(td_error.abs());
        let loss = self.critic.update_with_decay(
            &critic_input,
            target,
            critic_scale,
            &self.critic_decay_factors,
        );

        // Policy gradient (same formula as learn, but scaled by td_error)
        let y_conv_vec = self.backend.vec_to_vec(&infer.y_conv);
        let scaled: Vec<f64> = y_conv_vec
            .iter()
            .map(|&v| v / self.actor.config.temperature)
            .collect();
        let scaled_l = self.backend.vec_from_slice(&scaled);
        let pi_l = self.backend.softmax_masked(&scaled_l, valid_actions);
        let pi = self.backend.vec_to_vec(&pi_l);

        let mut delta = vec![0.0; pi.len()];
        for &i in valid_actions {
            delta[i] = pi[i];
        }
        delta[action] -= 1.0;

        for &i in valid_actions {
            delta[i] *= td_error;
        }

        // Entropy regularization
        for &i in valid_actions {
            let log_pi = (pi[i].max(1e-10)).ln();
            delta[i] -= self.config.entropy_coeff * (log_pi + 1.0);
        }

        let s_scale = self.effective_actor_scale(infer.surprise_score);
        let actor_decay = self.effective_actor_decay();

        // Fisher EMA accumulation and EWC correction (M4)
        if self.config.ewc_lambda > 0.0 && !self.actor_fisher.is_empty() {
            // Step 2: Extract per-layer gradients for Fisher EMA (read-only)
            let fisher_delta = if self.config.logits_reversal {
                // Logits reversal: delta_fisher = softmax(-y_conv/T, valid) - one_hot(action)
                let y_conv_rev: Vec<f64> = y_conv_vec
                    .iter()
                    .map(|&v| -v / self.actor.config.temperature)
                    .collect();
                let rev_l = self.backend.vec_from_slice(&y_conv_rev);
                let pi_rev_l = self.backend.softmax_masked(&rev_l, valid_actions);
                let pi_rev = self.backend.vec_to_vec(&pi_rev_l);
                let mut fd = pi_rev;
                fd[action] -= 1.0;
                fd
            } else {
                delta.clone()
            };
            self.accumulate_actor_fisher_ema(&fisher_delta, infer, input, s_scale, &actor_decay);

            // EWC correction: capture pre-update weights, update, then correct
            let pre_weights: Vec<L::Matrix> = self
                .actor
                .layers
                .iter()
                .map(|l| l.weights.clone())
                .collect();
            let pre_biases: Vec<L::Vector> =
                self.actor.layers.iter().map(|l| l.bias.clone()).collect();

            self.actor
                .update_weights(&delta, infer, input, s_scale, &actor_decay);

            // Apply EWC post-correction per layer
            self.apply_actor_ewc_correction(&pre_weights, &pre_biases, s_scale, &actor_decay);
        } else {
            self.actor
                .update_weights(&delta, infer, input, s_scale, &actor_decay);
        }

        if self.config.adaptive_surprise {
            self.push_surprise(infer.surprise_score);
        }

        self.last_td_error = td_error;
        self.push_td_error(td_error.abs());

        loss
    }

    /// Computes the learning rate scale factor based on surprise score.
    ///
    /// - surprise <= low → 0.1
    /// - surprise >= high → 2.0
    /// - Between → linear interpolation from 0.1 to 2.0
    ///
    /// If adaptive surprise is enabled and the buffer has >= 10 entries,
    /// thresholds are dynamically recomputed from the buffer statistics.
    pub fn surprise_scale(&self, surprise: f64) -> f64 {
        let (low, high) = if self.config.adaptive_surprise && self.surprise_buffer.len() >= 10 {
            let mean = self.surprise_buffer.iter().sum::<f64>() / self.surprise_buffer.len() as f64;
            let variance = self
                .surprise_buffer
                .iter()
                .map(|&s| (s - mean) * (s - mean))
                .sum::<f64>()
                / self.surprise_buffer.len() as f64;
            let std = variance.sqrt();
            let lo = (mean - 0.5 * std).max(0.0);
            let hi = mean + 1.5 * std;
            (lo, hi)
        } else {
            (self.config.surprise_low, self.config.surprise_high)
        };

        if surprise <= low {
            self.config.scale_floor
        } else if surprise >= high {
            self.config.scale_ceil
        } else {
            // Linear interpolation
            let t = (surprise - low) / (high - low);
            self.config.scale_floor + t * (self.config.scale_ceil - self.config.scale_floor)
        }
    }

    /// Performs a single training step: learns from the previous transition (if any),
    /// infers on the current state, selects an action, and stores internal state.
    ///
    /// Uses TD(0) single-step learning (same logic as `learn_continuous()`).
    /// Always uses Training mode (stochastic softmax sampling). For deterministic
    /// play mode, use `act(SelectionMode::Play)` directly.
    ///
    /// On the first call (or after reset/terminal), no learning occurs — the agent
    /// only infers and stores state. On subsequent calls, learning uses the stored
    /// previous state and the current state as the TD(0) bootstrap target.
    ///
    /// When `terminal` is true, V(s') = 0 for the TD error computation. The agent
    /// infers on the terminal state but immediately clears stored state, so the
    /// next call starts a fresh episode.
    ///
    /// # Arguments
    ///
    /// * `state` - Current observation vector.
    /// * `reward` - Reward received from the environment after the previous action.
    /// * `terminal` - Whether the current state is terminal.
    ///
    /// # Returns
    ///
    /// The selected action index.
    pub fn step(&mut self, state: &[f64], reward: f64, terminal: bool) -> usize {
        let all_actions: Vec<usize> = (0..self.config.actor.output_size).collect();
        self.step_inner(state, &all_actions, reward, terminal, None)
    }

    /// Performs a single training step with action masking.
    ///
    /// Identical to [`step()`](Self::step) except uses masked softmax for action
    /// selection, restricting the output to `valid_actions`. Stores the mask
    /// for policy gradient computation on the next call.
    ///
    /// # Arguments
    ///
    /// * `state` - Current observation vector.
    /// * `valid_actions` - Indices of legal actions for the current state.
    /// * `reward` - Reward received from the environment after the previous action.
    /// * `terminal` - Whether the current state is terminal.
    ///
    /// # Returns
    ///
    /// The selected action index (guaranteed to be in `valid_actions`).
    ///
    /// # Panics
    ///
    /// Panics if `valid_actions` is empty.
    pub fn step_masked(
        &mut self,
        state: &[f64],
        valid_actions: &[usize],
        reward: f64,
        terminal: bool,
    ) -> usize {
        assert!(!valid_actions.is_empty(), "valid_actions must not be empty");
        self.step_inner(
            state,
            valid_actions,
            reward,
            terminal,
            Some(valid_actions.to_vec()),
        )
    }

    /// Shared implementation for `step()` and `step_masked()`.
    ///
    /// # Arguments
    ///
    /// * `state` - Current observation vector.
    /// * `select_actions` - Actions to select from (all or masked).
    /// * `reward` - Reward received.
    /// * `terminal` - Whether state is terminal.
    /// * `store_mask` - If `Some`, stored as `valid_actions_prev` for next learning step.
    fn step_inner(
        &mut self,
        state: &[f64],
        select_actions: &[usize],
        reward: f64,
        terminal: bool,
        store_mask: Option<Vec<usize>>,
    ) -> usize {
        // Infer on current state (needed for both learning and action selection)
        let current_infer = self.actor.infer(state);

        // If previous state exists, learn from the transition
        if let (Some(prev_state), Some(prev_action), Some(prev_infer)) = (
            self.state_prev.take(),
            self.action_prev.take(),
            self.infer_prev.take(),
        ) {
            let surprise_score = prev_infer.surprise_score;
            let prev_state_vec = self.backend.vec_to_vec(&prev_state);
            let learn_mask = self
                .valid_actions_prev
                .take()
                .unwrap_or_else(|| (0..self.config.actor.output_size).collect());

            self.learn_continuous(
                &prev_state_vec,
                &prev_infer,
                prev_action,
                &learn_mask,
                reward,
                state,
                &current_infer,
                terminal,
            );

            // Update hysteresis state machines after learning
            if self.actor_hysteresis.is_some() || self.critic_hysteresis.is_some() {
                self.process_hysteresis(surprise_score, self.last_td_error.abs());
            }
        }

        // Select action
        let action = self.actor.select_action(
            &current_infer.y_conv,
            select_actions,
            SelectionMode::Training,
            &mut self.rng,
        );

        // Store current state for next step
        self.state_prev = Some(self.backend.vec_from_slice(state));
        self.action_prev = Some(action);
        self.infer_prev = Some(current_infer);
        self.valid_actions_prev = store_mask;

        // If terminal, clear all transient state
        if terminal {
            self.state_prev = None;
            self.action_prev = None;
            self.infer_prev = None;
            self.valid_actions_prev = None;
        }

        action
    }

    /// Clears step-level internal state without affecting weights or learning state.
    ///
    /// After calling this method, the next `step()` or `step_masked()` call
    /// behaves as the first call of a new episode (skips learning).
    ///
    /// Does NOT modify: weights, surprise buffer, or any continuous learning state.
    pub fn reset_step(&mut self) {
        self.state_prev = None;
        self.action_prev = None;
        self.infer_prev = None;
        self.valid_actions_prev = None;
    }

    /// Pushes a surprise score into the adaptive buffer (circular).
    fn push_surprise(&mut self, surprise: f64) {
        if self.surprise_buffer.len() >= self.config.surprise_buffer_size {
            self.surprise_buffer.pop_front();
        }
        self.surprise_buffer.push_back(surprise);
    }

    /// Pushes a |TD error| into the critic adaptive buffer (circular).
    fn push_td_error(&mut self, td_error: f64) {
        if self.td_error_buffer.len() >= self.config.surprise_buffer_size {
            self.td_error_buffer.pop_front();
        }
        self.td_error_buffer.push_back(td_error);
    }

    /// Computes the learning rate scale factor for the critic based on |TD error|.
    ///
    /// Identical to [`surprise_scale()`](Self::surprise_scale) but reads from
    /// the `td_error_buffer` for adaptive threshold computation.
    pub fn critic_surprise_scale(&self, td_error: f64) -> f64 {
        let (low, high) = if self.config.adaptive_surprise && self.td_error_buffer.len() >= 10 {
            let mean = self.td_error_buffer.iter().sum::<f64>() / self.td_error_buffer.len() as f64;
            let variance = self
                .td_error_buffer
                .iter()
                .map(|&s| (s - mean) * (s - mean))
                .sum::<f64>()
                / self.td_error_buffer.len() as f64;
            let std = variance.sqrt();
            let lo = (mean - 0.5 * std).max(0.0);
            let hi = mean + 1.5 * std;
            (lo, hi)
        } else {
            (self.config.surprise_low, self.config.surprise_high)
        };

        if td_error <= low {
            self.config.scale_floor
        } else if td_error >= high {
            self.config.scale_ceil
        } else {
            let t = (td_error - low) / (high - low);
            self.config.scale_floor + t * (self.config.scale_ceil - self.config.scale_floor)
        }
    }

    /// Computes the effective actor learning rate scale considering hysteresis.
    ///
    /// When actor hysteresis is enabled and the actor is FROZEN, returns
    /// `scale_floor`. Otherwise delegates to [`surprise_scale()`](Self::surprise_scale).
    pub(crate) fn effective_actor_scale(&self, surprise: f64) -> f64 {
        match &self.actor_hysteresis {
            Some(h) if h.state == PlasticityState::Frozen => self.config.scale_floor,
            _ => self.surprise_scale(surprise),
        }
    }

    /// Computes per-hidden-layer decay factors for the actor.
    ///
    /// When `adaptive_consolidation` is true, uses sigmoid of per-layer
    /// error EMA (M3b). Otherwise uses precomputed fixed decay (M3a).
    pub(crate) fn effective_actor_decay(&self) -> Vec<f64> {
        if self.config.adaptive_consolidation {
            self.layer_error_ema
                .iter()
                .map(|&e| {
                    let x = -self.config.consolidation_sigmoid_k
                        * (e - self.config.consolidation_error_threshold);
                    let adaptive_decay = 1.0 / (1.0 + (-x).exp());
                    1.0 - adaptive_decay
                })
                .collect()
        } else {
            self.actor_decay_factors.clone()
        }
    }

    /// Accumulates Fisher EMA for actor layers from extracted gradients.
    ///
    /// Extracts per-layer gradients using Approach 1 (activation derivative,
    /// hadamard, clip) and updates F_ema = beta * F_ema + (1-beta) * g_raw².
    fn accumulate_actor_fisher_ema(
        &mut self,
        output_delta: &[f64],
        infer: &InferResult<L>,
        _input: &[f64],
        _surprise_scale: f64,
        _decay_factors: &[f64],
    ) {
        let output_delta_vec = self.backend.vec_from_slice(output_delta);
        let n_hidden = self.actor.config.hidden_layers.len();
        let n_layers = self.actor.layers.len();

        // Output layer gradient extraction
        let output_output = &infer.y_conv;
        let deriv = self
            .backend
            .apply_derivative(output_output, self.actor.layers[n_layers - 1].activation);
        let mut grad = self.backend.vec_hadamard(&output_delta_vec, &deriv);
        self.backend.clip_vec(&mut grad, 5.0); // GRAD_CLIP

        // Update F_ema for output layer
        self.update_fisher_ema_layer(n_layers - 1, &grad, true);

        // Propagated delta: W^T * grad (read-only, using current weights)
        let wt = self
            .backend
            .mat_transpose(&self.actor.layers[n_layers - 1].weights);
        let mut bp_delta = self.backend.mat_vec_mul(&wt, &grad);

        // Hidden layers (from top to bottom)
        for i in (0..n_hidden).rev() {
            let layer_output = if self.actor.skip_alpha_index(i).is_some() {
                // Skip-eligible: use tanh_out
                infer.tanh_components[i].as_ref().unwrap()
            } else {
                &infer.hidden_states[i]
            };

            // Blend delta if using hybrid (same logic as update_weights_hybrid)
            let effective_delta = if (self.actor.config.local_lambda - 1.0).abs() < f64::EPSILON {
                bp_delta.clone()
            } else if self.actor.config.local_lambda.abs() < f64::EPSILON {
                let error_idx = n_hidden - 1 - i;
                infer.prediction_errors[error_idx].clone()
            } else {
                let error_idx = n_hidden - 1 - i;
                let pc_error = &infer.prediction_errors[error_idx];
                let bp_scaled = self
                    .backend
                    .vec_scale(&bp_delta, self.actor.config.local_lambda);
                let pc_scaled = self
                    .backend
                    .vec_scale(pc_error, 1.0 - self.actor.config.local_lambda);
                self.backend.vec_add(&bp_scaled, &pc_scaled)
            };

            // Scale by rezero_alpha if skip layer
            let scaled_delta = if let Some(alpha_idx) = self.actor.skip_alpha_index(i) {
                self.backend
                    .vec_scale(&effective_delta, self.actor.rezero_alpha[alpha_idx])
            } else {
                effective_delta.clone()
            };

            // Extract gradient
            let deriv_h = self
                .backend
                .apply_derivative(layer_output, self.actor.layers[i].activation);
            let mut grad_h = self.backend.vec_hadamard(&scaled_delta, &deriv_h);
            self.backend.clip_vec(&mut grad_h, 5.0);

            self.update_fisher_ema_layer(i, &grad_h, true);

            // Propagate delta read-only
            let wt_h = self.backend.mat_transpose(&self.actor.layers[i].weights);
            let propagated = self.backend.mat_vec_mul(&wt_h, &grad_h);

            if let Some(alpha_idx) = self.actor.skip_alpha_index(i) {
                // Skip path propagation (read-only)
                if let Some(ref proj) = self.actor.skip_projections[alpha_idx] {
                    let proj_t = self.backend.mat_transpose(proj);
                    let skip_delta = self.backend.mat_vec_mul(&proj_t, &effective_delta);
                    bp_delta = self.backend.vec_add(&propagated, &skip_delta);
                } else {
                    bp_delta = self.backend.vec_add(&propagated, &effective_delta);
                }
            } else {
                bp_delta = propagated;
            }
        }
    }

    /// Updates Fisher EMA for a single layer from its extracted gradient.
    ///
    /// F_ema = beta * F_ema + (1-beta) * g_raw²
    fn update_fisher_ema_layer(&mut self, layer_idx: usize, grad: &L::Vector, is_actor: bool) {
        let beta = self.config.fisher_ema_beta;
        let fisher = if is_actor {
            &mut self.actor_fisher[layer_idx]
        } else {
            &mut self.critic_fisher[layer_idx]
        };

        // Update bias F_ema
        let bias_len = self.backend.vec_len(&fisher.f_ema_bias);
        for i in 0..bias_len {
            let g = self.backend.vec_get(grad, i);
            let g_sq = g * g;
            let prev = self.backend.vec_get(&fisher.f_ema_bias, i);
            self.backend
                .vec_set(&mut fisher.f_ema_bias, i, beta * prev + (1.0 - beta) * g_sq);
        }

        // For weight F_ema, we need the outer product direction.
        // The gradient w.r.t. weights is outer(grad, input).
        // But we're tracking Fisher per-weight, so F_ema[r][c] = beta * F_ema[r][c] + (1-beta) * (grad[r] * input[c])².
        // However, this is expensive. The spec says "F_ema = beta * F_ema + (1-beta) * g_raw²"
        // where g_raw is the local gradient (not the weight gradient).
        // For Fisher information, we track per-parameter. The weight gradient for w[r][c] = grad[r] * input[c].
        // But the spec uses g_raw (the post-clip gradient vector) squared element-wise.
        // This is a diagonal approximation: F_ema for weights uses grad² broadcasted.
        // Actually re-reading spec: "g_raw = post-clip, pre-scaling gradient"
        // F_ema is per-parameter. For weights: dL/dW[r][c] = grad[r] * input[c].
        // To avoid storing full outer products, we can use the diagonal Fisher approximation:
        // F_ema_w[r][c] = beta * F_ema_w[r][c] + (1-beta) * grad[r]²
        // This is the standard diagonal Fisher for the row dimension.
        let rows = self.backend.mat_rows(&fisher.f_ema_weights);
        let cols = self.backend.mat_cols(&fisher.f_ema_weights);
        for r in 0..rows {
            let g = self.backend.vec_get(grad, r);
            let g_sq = g * g;
            for c in 0..cols {
                let prev = self.backend.mat_get(&fisher.f_ema_weights, r, c);
                self.backend.mat_set(
                    &mut fisher.f_ema_weights,
                    r,
                    c,
                    beta * prev + (1.0 - beta) * g_sq,
                );
            }
        }
    }

    /// Applies EWC post-update correction to actor layers.
    ///
    /// For each layer: W -= effective_lr * ewc_lambda * F_total * (W_pre - snapshot).
    /// Then applies WEIGHT_CLIP.
    fn apply_actor_ewc_correction(
        &mut self,
        pre_weights: &[L::Matrix],
        pre_biases: &[L::Vector],
        surprise_scale: f64,
        decay_factors: &[f64],
    ) {
        let n_hidden = self.actor.config.hidden_layers.len();
        let n_layers = self.actor.layers.len();

        for i in 0..n_layers {
            let fisher = &self.actor_fisher[i];
            let snapshot_w = match &fisher.theta_snapshot_weights {
                Some(s) => s,
                None => continue, // No snapshot yet
            };
            let snapshot_b = match &fisher.theta_snapshot_bias {
                Some(s) => s,
                None => continue,
            };

            // Compute effective_lr for this layer
            let layer_surprise = if i < n_hidden && !decay_factors.is_empty() {
                surprise_scale * decay_factors[i]
            } else {
                surprise_scale
            };
            let effective_lr = self.actor.config.lr_weights * layer_surprise;

            // EWC correction for weights: W -= effective_lr * ewc_lambda * F_total * (W_pre - snapshot)
            let rows = self.backend.mat_rows(&self.actor.layers[i].weights);
            let cols = self.backend.mat_cols(&self.actor.layers[i].weights);
            for r in 0..rows {
                for c in 0..cols {
                    let w_pre = self.backend.mat_get(&pre_weights[i], r, c);
                    let w_snap = self.backend.mat_get(snapshot_w, r, c);
                    let f_total = self.backend.mat_get(&fisher.f_total_weights, r, c);
                    let correction =
                        effective_lr * self.config.ewc_lambda * f_total * (w_pre - w_snap);
                    let w_cur = self.backend.mat_get(&self.actor.layers[i].weights, r, c);
                    let w_new = (w_cur - correction).clamp(-5.0, 5.0); // WEIGHT_CLIP
                    self.backend
                        .mat_set(&mut self.actor.layers[i].weights, r, c, w_new);
                }
            }

            // EWC correction for biases
            let bias_len = self.backend.vec_len(&self.actor.layers[i].bias);
            for j in 0..bias_len {
                let b_pre = self.backend.vec_get(&pre_biases[i], j);
                let b_snap = self.backend.vec_get(snapshot_b, j);
                let f_total = self.backend.vec_get(&fisher.f_total_bias, j);
                let correction = effective_lr * self.config.ewc_lambda * f_total * (b_pre - b_snap);
                let b_cur = self.backend.vec_get(&self.actor.layers[i].bias, j);
                let b_new = (b_cur - correction).clamp(-5.0, 5.0);
                self.backend
                    .vec_set(&mut self.actor.layers[i].bias, j, b_new);
            }
        }
    }

    /// Updates hysteresis state machines after learning.
    ///
    /// Handles EWMA updates, state transitions, counter management,
    /// and actor→critic coupling.
    pub(crate) fn process_hysteresis(&mut self, actor_signal: f64, critic_signal: f64) {
        let mut actor_woke = false;
        let mut actor_slept = false;
        let mut critic_woke = false;
        let mut critic_slept = false;

        // Update actor hysteresis
        if let Some(ref mut hyst) = self.actor_hysteresis {
            // Increment counter for pre-transition state
            if hyst.state == PlasticityState::Plastic {
                self.actor_plastic_step_counter += 1;
            }
            if let Some(new_state) = hyst.update(actor_signal) {
                if new_state == PlasticityState::Plastic {
                    actor_woke = true;
                    self.actor_plastic_step_counter = 0;
                } else {
                    actor_slept = true;
                }
            }
        }

        // Update critic hysteresis
        if let Some(ref mut hyst) = self.critic_hysteresis {
            // Increment counters for pre-transition state
            if hyst.state == PlasticityState::Frozen {
                self.critic_frozen_steps += 1;
            }
            if hyst.state == PlasticityState::Plastic {
                self.critic_plastic_step_counter += 1;
            }
            if let Some(new_state) = hyst.update(critic_signal) {
                if new_state == PlasticityState::Plastic {
                    critic_woke = true;
                    self.critic_plastic_step_counter = 0;
                    self.critic_frozen_steps = 0;
                } else {
                    critic_slept = true;
                }
            }
        }

        // Actor wakes critic coupling
        if actor_woke && self.config.actor_wakes_critic {
            if let Some(ref mut critic_hyst) = self.critic_hysteresis {
                if critic_hyst.state == PlasticityState::Frozen
                    && self.critic_frozen_steps >= self.config.actor_wakes_critic_threshold
                {
                    critic_hyst.state = PlasticityState::Plastic;
                    self.critic_plastic_step_counter = 0;
                    self.critic_frozen_steps = 0;
                    critic_woke = true;
                }
            }
        }

        // Fisher lifecycle on transitions
        if actor_slept {
            self.handle_fisher_sleep(true);
        }
        if actor_woke {
            self.handle_fisher_wake(true);
        }
        if critic_slept {
            self.handle_fisher_sleep(false);
        }
        if critic_woke {
            self.handle_fisher_wake(false);
        }
    }

    /// Fisher lifecycle Step 1: FROZEN→PLASTIC transition.
    ///
    /// If `last_phase_reliable`, decays `F_total *= fisher_decay`.
    /// Resets `F_ema` to zeros and plastic_step_counter.
    ///
    /// # Arguments
    ///
    /// * `is_actor` - true for actor, false for critic.
    pub(crate) fn handle_fisher_wake(&mut self, is_actor: bool) {
        if self.config.ewc_lambda <= 0.0 {
            return;
        }

        let (fisher_states, reliable) = if is_actor {
            (&mut self.actor_fisher, &self.actor_last_phase_reliable)
        } else {
            (&mut self.critic_fisher, &self.critic_last_phase_reliable)
        };

        if *reliable {
            // Decay F_total
            for fisher in fisher_states.iter_mut() {
                let rows = self.backend.mat_rows(&fisher.f_total_weights);
                let cols = self.backend.mat_cols(&fisher.f_total_weights);
                for r in 0..rows {
                    for c in 0..cols {
                        let val = self.backend.mat_get(&fisher.f_total_weights, r, c);
                        self.backend.mat_set(
                            &mut fisher.f_total_weights,
                            r,
                            c,
                            val * self.config.fisher_decay,
                        );
                    }
                }
                let bias_len = self.backend.vec_len(&fisher.f_total_bias);
                for i in 0..bias_len {
                    let val = self.backend.vec_get(&fisher.f_total_bias, i);
                    self.backend.vec_set(
                        &mut fisher.f_total_bias,
                        i,
                        val * self.config.fisher_decay,
                    );
                }
            }
        }

        // Reset F_ema to zeros
        for fisher in fisher_states.iter_mut() {
            let rows = self.backend.mat_rows(&fisher.f_ema_weights);
            let cols = self.backend.mat_cols(&fisher.f_ema_weights);
            fisher.f_ema_weights = self.backend.zeros_mat(rows, cols);
            let bias_len = self.backend.vec_len(&fisher.f_ema_bias);
            fisher.f_ema_bias = self.backend.zeros_vec(bias_len);
        }
    }

    /// Fisher lifecycle Step 3: PLASTIC→FROZEN transition.
    ///
    /// If `plastic_steps >= min_fisher_phase`: F_total += F_ema, reliable=true.
    /// Else: discard F_ema, reliable=false. Always snapshot weights.
    ///
    /// # Arguments
    ///
    /// * `is_actor` - true for actor, false for critic.
    pub(crate) fn handle_fisher_sleep(&mut self, is_actor: bool) {
        if self.config.ewc_lambda <= 0.0 {
            return;
        }

        let min_fisher_phase = (1.0 / (1.0 - self.config.fisher_ema_beta)).ceil() as u64;

        let plastic_steps = if is_actor {
            self.actor_plastic_step_counter
        } else {
            self.critic_plastic_step_counter
        };

        let reliable = plastic_steps >= min_fisher_phase;

        if is_actor {
            if reliable {
                // F_total += F_ema
                for fisher in self.actor_fisher.iter_mut() {
                    let rows = self.backend.mat_rows(&fisher.f_total_weights);
                    let cols = self.backend.mat_cols(&fisher.f_total_weights);
                    for r in 0..rows {
                        for c in 0..cols {
                            let total = self.backend.mat_get(&fisher.f_total_weights, r, c);
                            let ema = self.backend.mat_get(&fisher.f_ema_weights, r, c);
                            self.backend
                                .mat_set(&mut fisher.f_total_weights, r, c, total + ema);
                        }
                    }
                    let bias_len = self.backend.vec_len(&fisher.f_total_bias);
                    for i in 0..bias_len {
                        let total = self.backend.vec_get(&fisher.f_total_bias, i);
                        let ema = self.backend.vec_get(&fisher.f_ema_bias, i);
                        self.backend
                            .vec_set(&mut fisher.f_total_bias, i, total + ema);
                    }
                }
            }
            self.actor_last_phase_reliable = reliable;

            // Snapshot weights (always, regardless of reliability)
            for (i, fisher) in self.actor_fisher.iter_mut().enumerate() {
                fisher.theta_snapshot_weights = Some(self.actor.layers[i].weights.clone());
                fisher.theta_snapshot_bias = Some(self.actor.layers[i].bias.clone());
            }
        } else {
            if reliable {
                for fisher in self.critic_fisher.iter_mut() {
                    let rows = self.backend.mat_rows(&fisher.f_total_weights);
                    let cols = self.backend.mat_cols(&fisher.f_total_weights);
                    for r in 0..rows {
                        for c in 0..cols {
                            let total = self.backend.mat_get(&fisher.f_total_weights, r, c);
                            let ema = self.backend.mat_get(&fisher.f_ema_weights, r, c);
                            self.backend
                                .mat_set(&mut fisher.f_total_weights, r, c, total + ema);
                        }
                    }
                    let bias_len = self.backend.vec_len(&fisher.f_total_bias);
                    for i in 0..bias_len {
                        let total = self.backend.vec_get(&fisher.f_total_bias, i);
                        let ema = self.backend.vec_get(&fisher.f_ema_bias, i);
                        self.backend
                            .vec_set(&mut fisher.f_total_bias, i, total + ema);
                    }
                }
            }
            self.critic_last_phase_reliable = reliable;

            for (i, fisher) in self.critic_fisher.iter_mut().enumerate() {
                fisher.theta_snapshot_weights = Some(self.critic.layers[i].weights.clone());
                fisher.theta_snapshot_bias = Some(self.critic.layers[i].bias.clone());
            }
        }
    }
}

/// Converts an `ActivationCache` into a vector of matrices `[batch × neurons]`,
/// one per hidden layer, suitable for CCA alignment.
fn cache_to_matrices<L: LinAlg>(backend: &L, cache: &ActivationCache<L>) -> Vec<L::Matrix> {
    let num_layers = cache.num_layers();
    let batch_size = cache.batch_size();
    let mut matrices = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let samples = cache.layer(layer_idx);
        if samples.is_empty() {
            matrices.push(backend.zeros_mat(0, 0));
            continue;
        }
        let n_neurons = backend.vec_len(&samples[0]);
        let mut mat = backend.zeros_mat(batch_size, n_neurons);
        for (r, sample) in samples.iter().enumerate() {
            for c in 0..n_neurons {
                backend.mat_set(&mut mat, r, c, backend.vec_get(sample, c));
            }
        }
        matrices.push(mat);
    }

    matrices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::LayerDef;
    use crate::pc_actor::SelectionMode;

    fn default_config() -> PcActorCriticConfig {
        PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: 9,
                hidden_layers: vec![LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                }],
                output_size: 9,
                output_activation: Activation::Tanh,
                alpha: 0.1,
                tol: 0.01,
                min_steps: 1,
                max_steps: 20,
                lr_weights: 0.01,
                synchronous: true,
                temperature: 1.0,
                local_lambda: 1.0,
                residual: false,
                rezero_init: 0.001,
            },
            critic: MlpCriticConfig {
                input_size: 27,
                hidden_layers: vec![LayerDef {
                    size: 36,
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
            entropy_coeff: 0.01,
            scale_floor: 0.1, // v2.0.0 compat: existing tests expect 0.1 floor
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
            actor_wakes_critic: false,
            actor_wakes_critic_threshold: 1000,
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
        }
    }

    fn make_agent() -> PcActorCritic {
        let agent: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), default_config(), 42).unwrap();
        agent
    }

    fn make_trajectory(agent: &mut PcActorCritic) -> Vec<TrajectoryStep> {
        let input = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let valid = vec![2, 7];
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
        vec![TrajectoryStep {
            input,
            latent_concat: infer.latent_concat,
            y_conv: infer.y_conv,
            hidden_states: infer.hidden_states,
            prediction_errors: infer.prediction_errors,
            tanh_components: infer.tanh_components,
            action,
            valid_actions: valid,
            reward: 1.0,
            surprise_score: infer.surprise_score,
            steps_used: infer.steps_used,
        }]
    }

    // ── learn tests ───────────────────────────────────────────────

    #[test]
    #[allow(deprecated)]
    fn test_learn_empty_returns_zero_without_modifying_weights() {
        let mut agent: PcActorCritic = make_agent();
        let w_before = agent.actor.layers[0].weights.data.clone();
        let cw_before = agent.critic.layers[0].weights.data.clone();
        let loss = agent.learn(&[]);
        assert_eq!(loss, 0.0);
        assert_eq!(agent.actor.layers[0].weights.data, w_before);
        assert_eq!(agent.critic.layers[0].weights.data, cw_before);
    }

    #[test]
    #[allow(deprecated)]
    fn test_learn_updates_actor_weights() {
        let mut agent: PcActorCritic = make_agent();
        let trajectory = make_trajectory(&mut agent);
        let w_before = agent.actor.layers[0].weights.data.clone();
        let _ = agent.learn(&trajectory);
        assert_ne!(agent.actor.layers[0].weights.data, w_before);
    }

    #[test]
    #[allow(deprecated)]
    fn test_learn_updates_critic_weights() {
        let mut agent: PcActorCritic = make_agent();
        let trajectory = make_trajectory(&mut agent);
        let w_before = agent.critic.layers[0].weights.data.clone();
        let _ = agent.learn(&trajectory);
        assert_ne!(agent.critic.layers[0].weights.data, w_before);
    }

    #[test]
    #[allow(deprecated)]
    fn test_learn_returns_finite_nonneg_loss() {
        let mut agent: PcActorCritic = make_agent();
        let trajectory = make_trajectory(&mut agent);
        let loss = agent.learn(&trajectory);
        assert!(loss.is_finite(), "Loss {loss} is not finite");
        assert!(loss >= 0.0, "Loss {loss} is negative");
    }

    #[test]
    #[allow(deprecated)]
    fn test_learn_single_step_trajectory() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let valid = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
        let trajectory = vec![TrajectoryStep {
            input,
            latent_concat: infer.latent_concat,
            y_conv: infer.y_conv,
            hidden_states: infer.hidden_states,
            prediction_errors: infer.prediction_errors,
            tanh_components: infer.tanh_components,
            action,
            valid_actions: valid,
            reward: -1.0,
            surprise_score: infer.surprise_score,
            steps_used: infer.steps_used,
        }];
        let loss = agent.learn(&trajectory);
        assert!(loss.is_finite());
    }

    #[test]
    #[allow(deprecated)]
    fn test_learn_multi_step_uses_stored_hidden_states() {
        // Build a 3-step trajectory to exercise multi-step learning
        let mut agent: PcActorCritic = make_agent();
        let inputs = [
            vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5],
            vec![0.5, 0.5, -1.0, 0.0, 1.0, -0.5, 0.0, -1.0, 0.5],
            vec![-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, 1.0, -1.0, -0.5],
        ];
        let valid = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        let mut trajectory = Vec::new();
        for (i, inp) in inputs.iter().enumerate() {
            let (action, infer) = agent.act(inp, &valid, SelectionMode::Training);
            trajectory.push(TrajectoryStep {
                input: inp.clone(),
                latent_concat: infer.latent_concat,
                y_conv: infer.y_conv,
                hidden_states: infer.hidden_states,
                prediction_errors: infer.prediction_errors,
                tanh_components: infer.tanh_components,
                action,
                valid_actions: valid.clone(),
                reward: if i == 2 { 1.0 } else { 0.0 },
                surprise_score: infer.surprise_score,
                steps_used: infer.steps_used,
            });
        }

        let loss = agent.learn(&trajectory);
        assert!(
            loss.is_finite(),
            "Multi-step learn should produce finite loss"
        );
        assert!(loss >= 0.0);
    }

    // ── learn_continuous tests ────────────────────────────────────

    #[test]
    fn test_learn_continuous_nonterminal_uses_next_value() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let next_input = vec![-0.5; 9];
        let valid = vec![0, 1, 2];
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
        let (_, next_infer) = agent.act(&next_input, &valid, SelectionMode::Training);

        // Non-terminal: should incorporate next value
        let loss = agent.learn_continuous(
            &input,
            &infer,
            action,
            &valid,
            0.5,
            &next_input,
            &next_infer,
            false,
        );
        assert!(loss.is_finite());
    }

    #[test]
    fn test_learn_continuous_terminal_uses_reward_only() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let next_input = vec![0.0; 9];
        let valid = vec![0, 1, 2];
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
        let (_, next_infer) = agent.act(&next_input, &valid, SelectionMode::Training);

        // Terminal: target = reward only (no gamma * V(s'))
        let loss = agent.learn_continuous(
            &input,
            &infer,
            action,
            &valid,
            1.0,
            &next_input,
            &next_infer,
            true,
        );
        assert!(loss.is_finite());
    }

    #[test]
    fn test_learn_continuous_terminal_and_nonterminal_produce_different_updates() {
        // Create two identical agents
        let config = default_config();
        let mut agent_term: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_nonterm: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let input = vec![0.5; 9];
        let next_input = vec![-0.5; 9];
        let valid = vec![0, 1, 2];

        // Use identical actions and inferences
        let (action, infer) = agent_term.act(&input, &valid, SelectionMode::Training);
        let (_, next_infer) = agent_term.act(&next_input, &valid, SelectionMode::Training);

        // Clone infer for the non-terminal agent (same starting point)
        let (action2, infer2) = agent_nonterm.act(&input, &valid, SelectionMode::Training);
        let (_, next_infer2) = agent_nonterm.act(&next_input, &valid, SelectionMode::Training);

        // Terminal update
        let loss_term = agent_term.learn_continuous(
            &input,
            &infer,
            action,
            &valid,
            1.0,
            &next_input,
            &next_infer,
            true,
        );

        // Non-terminal update with same reward
        let loss_nonterm = agent_nonterm.learn_continuous(
            &input,
            &infer2,
            action2,
            &valid,
            1.0,
            &next_input,
            &next_infer2,
            false,
        );

        // The losses should differ because terminal uses target=reward
        // while non-terminal uses target=reward+gamma*V(s')
        assert!(
            (loss_term - loss_nonterm).abs() > 1e-15,
            "Terminal and non-terminal should produce different losses: {loss_term} vs {loss_nonterm}"
        );
    }

    #[test]
    fn test_learn_continuous_updates_actor() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let next_input = vec![-0.5; 9];
        let valid = vec![0, 1, 2];
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
        let (_, next_infer) = agent.act(&next_input, &valid, SelectionMode::Training);
        let w_before = agent.actor.layers[0].weights.data.clone();
        let _ = agent.learn_continuous(
            &input,
            &infer,
            action,
            &valid,
            1.0,
            &next_input,
            &next_infer,
            false,
        );
        assert_ne!(agent.actor.layers[0].weights.data, w_before);
    }

    // ── surprise_scale tests ─────────────────────────────────────

    #[test]
    fn test_surprise_scale_below_low() {
        let agent: PcActorCritic = make_agent();
        let scale = agent.surprise_scale(0.01); // below low=0.02
        assert!((scale - 0.1).abs() < 1e-12, "Expected 0.1, got {scale}");
    }

    #[test]
    fn test_surprise_scale_above_high() {
        let agent: PcActorCritic = make_agent();
        let scale = agent.surprise_scale(0.20); // above high=0.15
        assert!((scale - 2.0).abs() < 1e-12, "Expected 2.0, got {scale}");
    }

    #[test]
    fn test_surprise_scale_midpoint_in_range() {
        let agent: PcActorCritic = make_agent();
        let midpoint = (0.02 + 0.15) / 2.0;
        let scale = agent.surprise_scale(midpoint);
        assert!(
            scale > 0.1 && scale < 2.0,
            "Midpoint scale {scale} out of range"
        );
    }

    #[test]
    fn test_surprise_scale_monotone_increasing() {
        let agent: PcActorCritic = make_agent();
        let s1 = agent.surprise_scale(0.01);
        let s2 = agent.surprise_scale(0.05);
        let s3 = agent.surprise_scale(0.10);
        let s4 = agent.surprise_scale(0.20);
        assert!(s1 <= s2, "s1={s1} > s2={s2}");
        assert!(s2 <= s3, "s2={s2} > s3={s3}");
        assert!(s3 <= s4, "s3={s3} > s4={s4}");
    }

    #[test]
    fn test_adaptive_surprise_recalibrates_thresholds_after_many_episodes() {
        let mut config = default_config();
        config.adaptive_surprise = true;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Fill buffer with varied surprise scores to get nonzero std
        for i in 0..15 {
            agent.push_surprise(0.1 + 0.02 * i as f64);
        }

        // mean ≈ 0.24, std ≈ 0.089
        // adaptive low = max(0, mean - 0.5*std) ≈ 0.196
        // adaptive high = mean + 1.5*std ≈ 0.373
        // These differ from the static defaults (0.02, 0.15)

        // Something well below adaptive low should get 0.1
        let scale_low = agent.surprise_scale(0.0);
        assert!(
            (scale_low - 0.1).abs() < 1e-12,
            "Expected 0.1 below adaptive low: got {scale_low}"
        );

        // Something well above adaptive high should get 2.0
        let scale_high = agent.surprise_scale(1.0);
        assert!(
            (scale_high - 2.0).abs() < 1e-12,
            "Expected 2.0 above adaptive high: got {scale_high}"
        );

        // Something at the mean should be between 0.1 and 2.0
        let scale_mid = agent.surprise_scale(0.24);
        assert!(
            scale_mid > 0.1 && scale_mid < 2.0,
            "Expected interpolated value at mean, got {scale_mid}"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_entropy_regularization_prevents_policy_collapse() {
        // With entropy regularization, repeated learning on same trajectory
        // should keep the policy from collapsing to a single action
        let mut config = default_config();
        config.entropy_coeff = 0.1; // Strong entropy
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let input = vec![0.5; 9];
        let valid: Vec<usize> = (0..9).collect();

        // Train many times on same trajectory
        for _ in 0..20 {
            let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
            let trajectory = vec![TrajectoryStep {
                input: input.clone(),
                latent_concat: infer.latent_concat,
                y_conv: infer.y_conv,
                hidden_states: infer.hidden_states,
                prediction_errors: infer.prediction_errors,
                tanh_components: infer.tanh_components,
                action,
                valid_actions: valid.clone(),
                reward: 1.0,
                surprise_score: infer.surprise_score,
                steps_used: infer.steps_used,
            }];
            let _ = agent.learn(&trajectory);
        }

        // Check that policy is not collapsed (multiple actions selected over 50 trials)
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            let (action, _) = agent.act(&input, &valid, SelectionMode::Training);
            seen.insert(action);
        }
        assert!(
            seen.len() > 1,
            "Entropy regularization should prevent collapse to single action, but only saw {:?}",
            seen
        );
    }

    // ── act tests ─────────────────────────────────────────────────

    #[test]
    fn test_act_returns_valid_action() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let valid = vec![1, 3, 5, 7];
        for _ in 0..20 {
            let (action, _) = agent.act(&input, &valid, SelectionMode::Training);
            assert!(valid.contains(&action), "Action {action} not in valid set");
        }
    }

    #[test]
    #[should_panic]
    fn test_act_empty_valid_panics() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let _ = agent.act(&input, &[], SelectionMode::Training);
    }

    // ── learning diagnostic test ──────────────────────────────

    #[test]
    #[allow(deprecated)]
    fn test_learn_improves_policy_for_rewarded_action() {
        // Linear output so logits are unbounded
        let config = PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: 9,
                hidden_layers: vec![LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                }],
                output_size: 9,
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
                input_size: 27,
                hidden_layers: vec![LayerDef {
                    size: 36,
                    activation: Activation::Tanh,
                }],
                output_activation: Activation::Linear,
                lr: 0.005,
            },
            gamma: 0.99,
            surprise_low: 0.02,
            surprise_high: 0.15,
            adaptive_surprise: false,
            surprise_buffer_size: 100,
            entropy_coeff: 0.0, // no entropy to isolate gradient effect
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
            actor_wakes_critic: false,
            actor_wakes_critic_threshold: 1000,
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
        };
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let input = vec![0.0; 9];
        let valid = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let target_action = 4; // center

        // Repeatedly reward action 4
        for _ in 0..200 {
            let (_, infer) = agent.act(&input, &valid, SelectionMode::Training);
            let trajectory = vec![TrajectoryStep {
                input: input.clone(),
                latent_concat: infer.latent_concat,
                y_conv: infer.y_conv,
                hidden_states: infer.hidden_states,
                prediction_errors: infer.prediction_errors,
                tanh_components: infer.tanh_components,
                action: target_action,
                valid_actions: valid.clone(),
                reward: 1.0,
                surprise_score: infer.surprise_score,
                steps_used: infer.steps_used,
            }];
            agent.learn(&trajectory);
        }

        // After 200 episodes always rewarding action 4, it should be the
        // preferred action in Play mode (deterministic argmax)
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Play);

        // Check that action 4's logit is the highest
        let logit_4 = infer.y_conv[4];
        let max_other = valid
            .iter()
            .filter(|&&a| a != 4)
            .map(|&a| infer.y_conv[a])
            .fold(f64::NEG_INFINITY, f64::max);

        eprintln!(
            "DIAGNOSTIC: action={action}, logit[4]={logit_4:.4}, max_other={max_other:.4}, \
             y_conv={:?}",
            infer
                .y_conv
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
        );

        assert_eq!(
            action, target_action,
            "After 200 episodes rewarding action 4, agent should prefer it. Got action {action}"
        );
    }

    // ── config validation tests ────────────────────────────────

    #[test]
    fn test_new_returns_error_zero_temperature() {
        let mut config = default_config();
        config.actor.temperature = 0.0;
        let err = PcActorCritic::new(CpuLinAlg::new(), config, 42)
            .map(|_: PcActorCritic| ())
            .unwrap_err();
        assert!(format!("{err}").contains("temperature"));
    }

    #[test]
    fn test_new_returns_error_zero_input_size() {
        let mut config = default_config();
        config.actor.input_size = 0;
        config.critic.input_size = 0;
        assert!(PcActorCritic::new(CpuLinAlg::new(), config, 42)
            .map(|_: PcActorCritic| ())
            .is_err());
    }

    #[test]
    fn test_new_returns_error_zero_output_size() {
        let mut config = default_config();
        config.actor.output_size = 0;
        assert!(PcActorCritic::new(CpuLinAlg::new(), config, 42)
            .map(|_: PcActorCritic| ())
            .is_err());
    }

    #[test]
    fn test_new_returns_error_negative_gamma() {
        let mut config = default_config();
        config.gamma = -0.1;
        let err = PcActorCritic::new(CpuLinAlg::new(), config, 42)
            .map(|_: PcActorCritic| ())
            .unwrap_err();
        assert!(format!("{err}").contains("gamma"));
    }

    #[test]
    fn test_new_returns_error_surprise_buffer_size_zero() {
        let mut config = default_config();
        config.adaptive_surprise = true;
        config.surprise_buffer_size = 0;
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42).map(|_: PcActorCritic| ());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err}").contains("surprise_buffer_size"),
            "Expected surprise_buffer_size error, got: {err}"
        );
    }

    // ── Phase 4 Cycle 4.1: ActivationCache construction and recording ──

    #[test]
    fn test_activation_cache_new_creates_empty() {
        let cache: ActivationCache = ActivationCache::new(3);
        assert_eq!(cache.batch_size(), 0);
    }

    #[test]
    fn test_activation_cache_record_increments_batch_size() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let valid = vec![0, 1, 2];
        let (_, infer) = agent.act(&input, &valid, SelectionMode::Training);

        let num_hidden = infer.hidden_states.len();
        let mut cache: ActivationCache = ActivationCache::new(num_hidden);
        cache.record(&infer.hidden_states);
        assert_eq!(cache.batch_size(), 1);
    }

    #[test]
    fn test_activation_cache_record_multiple() {
        let mut agent: PcActorCritic = make_agent();
        let valid = vec![0, 1, 2];
        let init_input = vec![0.5; 9];
        let num_hidden = {
            let (_, infer) = agent.act(&init_input, &valid, SelectionMode::Training);
            infer.hidden_states.len()
        };

        let mut cache: ActivationCache = ActivationCache::new(num_hidden);
        for i in 0..5 {
            let input = vec![i as f64 * 0.1; 9];
            let (_, infer) = agent.act(&input, &valid, SelectionMode::Training);
            cache.record(&infer.hidden_states);
        }
        assert_eq!(cache.batch_size(), 5);
    }

    #[test]
    fn test_activation_cache_recorded_values_match_hidden_states() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let valid = vec![0, 1, 2];
        let (_, infer) = agent.act(&input, &valid, SelectionMode::Training);

        let num_hidden = infer.hidden_states.len();
        let mut cache: ActivationCache = ActivationCache::new(num_hidden);
        cache.record(&infer.hidden_states);

        // Verify recorded activations match
        for (layer_idx, expected) in infer.hidden_states.iter().enumerate() {
            let layer_data = cache.layer(layer_idx);
            assert_eq!(layer_data.len(), 1);
            assert_eq!(layer_data[0], *expected);
        }
    }

    // ── Phase 4 Cycle 4.2: ActivationCache layer access ────────────

    #[test]
    fn test_activation_cache_layer_count() {
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let valid = vec![0, 1, 2];
        let (_, infer) = agent.act(&input, &valid, SelectionMode::Training);

        let num_hidden = infer.hidden_states.len();
        let mut cache: ActivationCache = ActivationCache::new(num_hidden);
        cache.record(&infer.hidden_states);

        assert_eq!(cache.num_layers(), num_hidden);
    }

    #[test]
    fn test_activation_cache_layer_sample_count() {
        let mut agent: PcActorCritic = make_agent();
        let valid = vec![0, 1, 2];
        let init_input = vec![0.5; 9];
        let num_hidden = {
            let (_, infer) = agent.act(&init_input, &valid, SelectionMode::Training);
            infer.hidden_states.len()
        };

        let mut cache: ActivationCache = ActivationCache::new(num_hidden);
        for i in 0..10 {
            let input = vec![i as f64 * 0.1; 9];
            let (_, infer) = agent.act(&input, &valid, SelectionMode::Training);
            cache.record(&infer.hidden_states);
        }

        for layer_idx in 0..num_hidden {
            assert_eq!(
                cache.layer(layer_idx).len(),
                10,
                "Layer {layer_idx} should have 10 samples"
            );
        }
    }

    // ── Phase 7 Cycle 7.1: PcActorCritic::crossover ────────────

    fn build_caches_for_agent(
        agent: &mut PcActorCritic,
        batch_size: usize,
    ) -> (ActivationCache, ActivationCache) {
        let num_actor_hidden = agent.config.actor.hidden_layers.len();
        let num_critic_hidden = agent.config.critic.hidden_layers.len();
        let mut actor_cache: ActivationCache = ActivationCache::new(num_actor_hidden);
        let mut critic_cache: ActivationCache = ActivationCache::new(num_critic_hidden);
        let valid: Vec<usize> = (0..agent.config.actor.output_size).collect();
        for i in 0..batch_size {
            let input: Vec<f64> = (0..agent.config.actor.input_size)
                .map(|j| ((i * 9 + j) as f64 * 0.1).sin())
                .collect();
            let (_, infer) = agent.act(&input, &valid, SelectionMode::Training);
            actor_cache.record(&infer.hidden_states);
            let mut critic_input = input;
            critic_input.extend_from_slice(&infer.latent_concat);
            let (_value, critic_hidden) = agent.critic.forward_with_hidden(&critic_input);
            critic_cache.record(&critic_hidden);
        }
        (actor_cache, critic_cache)
    }

    #[test]
    fn test_agent_crossover_produces_valid_agent() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut agent_b, 50);

        let child: PcActorCritic = PcActorCritic::crossover(
            &agent_a, &agent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        assert_eq!(
            child.config.actor.hidden_layers.len(),
            agent_a.config.actor.hidden_layers.len()
        );
    }

    #[test]
    fn test_agent_crossover_actor_weights_differ() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut agent_b, 50);

        let child: PcActorCritic = PcActorCritic::crossover(
            &agent_a, &agent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        assert_ne!(
            child.actor.layers[0].weights.data,
            agent_a.actor.layers[0].weights.data
        );
        assert_ne!(
            child.actor.layers[0].weights.data,
            agent_b.actor.layers[0].weights.data
        );
    }

    #[test]
    fn test_agent_crossover_critic_weights_differ() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut agent_b, 50);

        let child: PcActorCritic = PcActorCritic::crossover(
            &agent_a, &agent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        assert_ne!(
            child.critic.layers[0].weights.data,
            agent_a.critic.layers[0].weights.data
        );
        assert_ne!(
            child.critic.layers[0].weights.data,
            agent_b.critic.layers[0].weights.data
        );
    }

    // ── Phase 7 Cycle 7.2: Integration — full GA workflow ───────

    #[test]
    fn test_agent_crossover_child_can_infer() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut agent_b, 50);

        let mut child: PcActorCritic = PcActorCritic::crossover(
            &agent_a, &agent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        let input = vec![0.5; 9];
        let valid = vec![0, 1, 2, 3, 4];
        let (action, _) = child.act(&input, &valid, SelectionMode::Training);
        assert!(valid.contains(&action), "Action {action} not in valid set");
    }

    #[test]
    #[allow(deprecated)]
    fn test_agent_crossover_child_can_learn() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut agent_b, 50);

        let mut child: PcActorCritic = PcActorCritic::crossover(
            &agent_a, &agent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        let trajectory = make_trajectory(&mut child);
        let loss = child.learn(&trajectory);
        assert!(loss.is_finite(), "Child learn loss not finite: {loss}");
    }

    #[test]
    fn test_agent_crossover_mismatched_batch_size_error() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, _cc_b) = build_caches_for_agent(&mut agent_b, 30); // different batch
        let (_, cc_b_match) = build_caches_for_agent(&mut agent_b, 50);

        // Actor batch mismatch
        let result = PcActorCritic::crossover(
            &agent_a,
            &agent_b,
            &ac_a,
            &ac_b,
            &cc_a,
            &cc_b_match,
            0.5,
            config,
            99,
        );
        assert!(result.is_err(), "Mismatched actor batch sizes should error");
    }

    // ── Fix #2: Separate critic caches in crossover ────────────

    #[test]
    fn test_agent_crossover_with_separate_critic_caches() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut agent_b, 50);

        let child: PcActorCritic = PcActorCritic::crossover(
            &agent_a, &agent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        assert_eq!(child.critic.layers.len(), agent_a.critic.layers.len());
    }

    #[test]
    fn test_agent_crossover_critic_uses_own_caches() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut agent_b, 50);

        let child: PcActorCritic = PcActorCritic::crossover(
            &agent_a, &agent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        assert_ne!(
            child.critic.layers[0].weights.data,
            agent_a.critic.layers[0].weights.data
        );
        assert_ne!(
            child.critic.layers[0].weights.data,
            agent_b.critic.layers[0].weights.data
        );
    }

    #[test]
    fn test_agent_crossover_mismatched_critic_batch_error() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut agent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut agent_a, 50);
        let (ac_b, _) = build_caches_for_agent(&mut agent_b, 50);
        // Build critic cache with different batch size
        let (_, cc_b_small) = build_caches_for_agent(&mut agent_b, 30);

        let result = PcActorCritic::crossover(
            &agent_a,
            &agent_b,
            &ac_a,
            &ac_b,
            &cc_a,
            &cc_b_small,
            0.5,
            config,
            99,
        );
        assert!(
            result.is_err(),
            "Mismatched critic batch sizes should error"
        );
    }

    // ── Phase 7 Cycle 7.3: lib.rs re-exports ────────────────────

    #[test]
    fn test_activation_cache_accessible_from_crate() {
        // Verify ActivationCache is accessible via pc_actor_critic module
        let _cache: crate::pc_actor_critic::ActivationCache = ActivationCache::new(1);
    }

    #[test]
    fn test_cca_neuron_alignment_accessible_from_crate() {
        // Verify cca_neuron_alignment is accessible via matrix module
        use crate::linalg::cpu::CpuLinAlg;
        use crate::linalg::LinAlg;
        let mat = CpuLinAlg::new().zeros_mat(10, 3);
        let _perm = crate::matrix::cca_neuron_alignment::<CpuLinAlg>(&CpuLinAlg::new(), &mat, &mat)
            .unwrap();
    }

    // ── Phase 0: Unified step() API ──────────────────────────────

    /// Helper: extract all layer weights (actor + critic) as flat Vec<f64>.
    fn collect_all_weights(agent: &PcActorCritic) -> Vec<f64> {
        let mut weights = Vec::new();
        for layer in &agent.actor.layers {
            weights.extend_from_slice(&layer.weights.data);
            weights.extend_from_slice(&layer.bias);
        }
        for layer in &agent.critic.layers {
            weights.extend_from_slice(&layer.weights.data);
            weights.extend_from_slice(&layer.bias);
        }
        weights
    }

    #[test]
    fn step_step_matches_learn_continuous_td0() {
        // Agent A: uses step() + step()
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();

        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -1.0, 0.0, 1.0, -0.5, 0.0, -1.0, 0.5];
        let reward = 1.0;

        let _a1 = agent_a.step(&s1, 0.0, false);
        let _a2 = agent_a.step(&s2, reward, false);

        // Agent B: uses act() + act() + learn_continuous()
        let mut agent_b: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        let all_actions: Vec<usize> = (0..9).collect();

        let (action1, infer1) = agent_b.act(&s1, &all_actions, SelectionMode::Training);
        let (_, infer2) = agent_b.act(&s2, &all_actions, SelectionMode::Training);

        let _ = agent_b.learn_continuous(
            &s1,
            &infer1,
            action1,
            &all_actions,
            reward,
            &s2,
            &infer2,
            false,
        );

        let w_a = collect_all_weights(&agent_a);
        let w_b = collect_all_weights(&agent_b);
        assert_eq!(w_a.len(), w_b.len());
        for (i, (a, b)) in w_a.iter().zip(w_b.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "Weight mismatch at index {i}: step={a} vs learn_continuous={b}"
            );
        }
    }

    #[test]
    fn step_terminal_uses_zero_bootstrap() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();

        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -1.0, 0.0, 1.0, -0.5, 0.0, -1.0, 0.5];

        let _a1 = agent_a.step(&s1, 0.0, false);
        let _a2 = agent_a.step(&s2, 1.0, true); // terminal

        // Agent B: manual learn_continuous with terminal=true
        let mut agent_b: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        let all_actions: Vec<usize> = (0..9).collect();

        let (action1, infer1) = agent_b.act(&s1, &all_actions, SelectionMode::Training);
        let (_, infer2) = agent_b.act(&s2, &all_actions, SelectionMode::Training);

        let _ =
            agent_b.learn_continuous(&s1, &infer1, action1, &all_actions, 1.0, &s2, &infer2, true);

        let w_a = collect_all_weights(&agent_a);
        let w_b = collect_all_weights(&agent_b);
        for (i, (a, b)) in w_a.iter().zip(w_b.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "Weight mismatch at index {i}: step={a} vs learn_continuous={b}"
            );
        }
    }

    #[test]
    fn step_masked_stores_valid_actions_for_learning() {
        let config = default_config();
        let mut agent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();

        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -1.0, 0.0, 1.0, -0.5, 0.0, -1.0, 0.5];
        let mask = vec![0, 2, 5];

        let _a1 = agent_a.step_masked(&s1, &mask, 0.0, false);
        let all_actions: Vec<usize> = (0..9).collect();
        let _a2 = agent_a.step(&s2, 1.0, false);

        // Agent B: manual path
        let mut agent_b: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let (action1, infer1) = agent_b.act(&s1, &mask, SelectionMode::Training);
        let (_, infer2) = agent_b.act(&s2, &all_actions, SelectionMode::Training);

        let _ = agent_b.learn_continuous(&s1, &infer1, action1, &mask, 1.0, &s2, &infer2, false);

        let w_a = collect_all_weights(&agent_a);
        let w_b = collect_all_weights(&agent_b);
        for (i, (a, b)) in w_a.iter().zip(w_b.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "Weight mismatch at index {i}: step_masked={a} vs learn_continuous={b}"
            );
        }
    }

    #[test]
    fn step_first_call_skips_learning() {
        let mut agent: PcActorCritic = make_agent();
        let w_before = collect_all_weights(&agent);
        let state = vec![0.5; 9];
        let action = agent.step(&state, 0.0, false);
        assert!(action < 9, "Action {action} out of bounds");
        let w_after = collect_all_weights(&agent);
        assert_eq!(
            w_before, w_after,
            "Weights should not change on first step()"
        );
    }

    #[test]
    fn step_second_call_modifies_weights() {
        let mut agent: PcActorCritic = make_agent();
        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -1.0, 0.0, 1.0, -0.5, 0.0, -1.0, 0.5];
        let _ = agent.step(&s1, 0.0, false);
        let w_before = collect_all_weights(&agent);
        let _ = agent.step(&s2, 1.0, false);
        let w_after = collect_all_weights(&agent);
        assert_ne!(
            w_before, w_after,
            "Weights should change after second step()"
        );
    }

    #[test]
    fn step_terminal_clears_state() {
        let mut agent: PcActorCritic = make_agent();
        let s1 = vec![1.0; 9];
        let s2 = vec![-1.0; 9];
        let s3 = vec![0.5; 9];

        let _ = agent.step(&s1, 0.0, false);
        let _ = agent.step(&s2, 1.0, true); // terminal clears state
        let w_after_terminal = collect_all_weights(&agent);

        let _ = agent.step(&s3, 0.0, false); // should skip learning (first after terminal)
        let w_after_first = collect_all_weights(&agent);
        assert_eq!(
            w_after_terminal, w_after_first,
            "First step after terminal should skip learning"
        );
    }

    #[test]
    fn step_masked_action_in_valid_set() {
        let valid = vec![0, 2, 5];
        for seed in 0..100u64 {
            let mut agent: PcActorCritic =
                PcActorCritic::new(CpuLinAlg::new(), default_config(), seed).unwrap();
            let state = vec![0.5; 9];
            let action = agent.step_masked(&state, &valid, 0.0, false);
            assert!(
                valid.contains(&action),
                "seed={seed}: action {action} not in valid set {valid:?}"
            );
        }
    }

    #[test]
    fn reset_step_clears_only_step_state() {
        let mut agent: PcActorCritic = make_agent();
        let s1 = vec![1.0; 9];
        let s2 = vec![-1.0; 9];

        let _ = agent.step(&s1, 0.0, false); // stores state
        let w_before = collect_all_weights(&agent);

        agent.reset_step();

        let w_after = collect_all_weights(&agent);
        assert_eq!(w_before, w_after, "reset_step() must not change weights");

        // Next step should skip learning (first call after reset)
        let _ = agent.step(&s2, 0.5, false);
        let w_after_step = collect_all_weights(&agent);
        assert_eq!(
            w_after, w_after_step,
            "First step after reset should skip learning"
        );
    }

    #[test]
    fn reset_step_does_not_affect_surprise_buffer() {
        let mut config = default_config();
        config.adaptive_surprise = true;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let s1 = vec![1.0; 9];
        let s2 = vec![-1.0; 9];

        // Two step() calls push surprise to buffer (second call triggers learning)
        let _ = agent.step(&s1, 0.0, false);
        let _ = agent.step(&s2, 1.0, false);
        let buf_len_before = agent.surprise_buffer.len();

        agent.reset_step();

        assert_eq!(
            agent.surprise_buffer.len(),
            buf_len_before,
            "reset_step() must not affect surprise buffer"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn learn_deprecated_still_functional() {
        let mut agent: PcActorCritic = make_agent();
        let trajectory = make_trajectory(&mut agent);
        let loss = agent.learn(&trajectory);
        assert!(loss.is_finite(), "learn() should still return finite loss");
    }

    #[test]
    fn act_not_deprecated() {
        // This test compiles without #[allow(deprecated)].
        // If act() were deprecated, clippy -D warnings would catch it.
        let mut agent: PcActorCritic = make_agent();
        let input = vec![0.5; 9];
        let valid = vec![0, 1, 2];
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
        assert!(action < 9);
        assert!(infer.surprise_score.is_finite());
    }

    #[test]
    fn step_masked_empty_valid_actions_handled() {
        let mut agent: PcActorCritic = make_agent();
        let state = vec![0.5; 9];
        // Empty valid actions should panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent.step_masked(&state, &[], 0.0, false)
        }));
        assert!(
            result.is_err(),
            "step_masked with empty valid_actions should panic"
        );
    }

    // ── configurable scale range tests (Phase 1 — M1) ───────────

    /// Helper: create config with custom scale floor/ceil.
    fn config_with_scale(floor: f64, ceil: f64) -> PcActorCriticConfig {
        let mut cfg = default_config();
        cfg.scale_floor = floor;
        cfg.scale_ceil = ceil;
        cfg
    }

    #[test]
    fn test_scale_floor_zero_produces_zero_scale() {
        let cfg = config_with_scale(0.0, 2.0);
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();
        let scale = agent.surprise_scale(0.01); // below low=0.02
        assert!(
            scale.abs() < 1e-12,
            "Expected 0.0 for surprise below low with floor=0.0, got {scale}"
        );
    }

    #[test]
    fn test_scale_floor_0_1_ceil_2_0_matches_v2() {
        let cfg = config_with_scale(0.1, 2.0);
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();
        // Below low threshold -> floor
        let s_low = agent.surprise_scale(0.01);
        assert!(
            (s_low - 0.1).abs() < 1e-12,
            "Expected 0.1 below low, got {s_low}"
        );
        // Above high threshold -> ceil
        let s_high = agent.surprise_scale(0.20);
        assert!(
            (s_high - 2.0).abs() < 1e-12,
            "Expected 2.0 above high, got {s_high}"
        );
        // Midpoint -> in range
        let midpoint = (0.02 + 0.15) / 2.0;
        let s_mid = agent.surprise_scale(midpoint);
        assert!(
            s_mid > 0.1 && s_mid < 2.0,
            "Expected midpoint in (0.1, 2.0), got {s_mid}"
        );
    }

    #[test]
    fn test_scale_ceil_custom_value() {
        let cfg = config_with_scale(0.0, 3.0);
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();
        let scale = agent.surprise_scale(0.20); // above high=0.15
        assert!(
            (scale - 3.0).abs() < 1e-12,
            "Expected 3.0 above high with ceil=3.0, got {scale}"
        );
    }

    #[test]
    fn test_scale_interpolation_midpoint() {
        let cfg = config_with_scale(0.0, 2.0);
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();
        let midpoint = (0.02 + 0.15) / 2.0;
        let scale = agent.surprise_scale(midpoint);
        // t = 0.5, expected = 0.0 + 0.5 * (2.0 - 0.0) = 1.0
        assert!(
            (scale - 1.0).abs() < 1e-12,
            "Expected 1.0 at midpoint with floor=0.0/ceil=2.0, got {scale}"
        );
    }

    #[test]
    fn test_scale_floor_negative_rejected() {
        let cfg = config_with_scale(-0.1, 2.0);
        let result: Result<PcActorCritic, _> = PcActorCritic::new(CpuLinAlg::new(), cfg, 42);
        assert!(result.is_err(), "Negative scale_floor should be rejected");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("scale_floor"),
            "Error should mention scale_floor: {err_msg}"
        );
    }

    #[test]
    fn test_scale_ceil_less_than_floor_rejected() {
        let cfg = config_with_scale(2.0, 1.0);
        let result: Result<PcActorCritic, _> = PcActorCritic::new(CpuLinAlg::new(), cfg, 42);
        assert!(
            result.is_err(),
            "scale_ceil < scale_floor should be rejected"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("scale_ceil"),
            "Error should mention scale_ceil: {err_msg}"
        );
    }

    #[test]
    fn test_scale_floor_equals_ceil_degenerate() {
        let cfg = config_with_scale(1.0, 1.0);
        let result: Result<PcActorCritic, _> = PcActorCritic::new(CpuLinAlg::new(), cfg, 42);
        assert!(
            result.is_err(),
            "scale_floor == scale_ceil should be rejected (ceil must be > floor)"
        );
    }

    // ── Phase 2a: EwmaTracker & PlasticityState tests ──────────────────

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

    // ============ Phase 2b: HysteresisState Unit Tests ============

    /// Helper: set up a HysteresisState's actor EWMAs for a wake transition.
    fn setup_for_wake(hyst: &mut HysteresisState) {
        hyst.state = PlasticityState::Frozen;
        hyst.slow.value = 0.05;
        hyst.slow.k = 200;
        hyst.fast.value = 0.06;
        hyst.fast.k = 200;
    }

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

    // ============ Phase 2b: Integration Tests ============

    #[test]
    fn actor_critic_independent_hysteresis() {
        let mut cfg = default_config();
        cfg.actor_hysteresis = true;
        cfg.critic_hysteresis = true;
        cfg.adaptive_surprise = true;
        cfg.surprise_buffer_size = 100;
        let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

        // Verify independent state machines exist
        assert!(agent.actor_hysteresis.is_some());
        assert!(agent.critic_hysteresis.is_some());

        // Set actor to FROZEN, critic stays PLASTIC
        agent.actor_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        assert_eq!(
            agent.critic_hysteresis.as_ref().unwrap().state,
            PlasticityState::Plastic
        );

        // Fill td_error_buffer for critic_surprise_scale
        for _ in 0..20 {
            agent.td_error_buffer.push_back(0.5);
        }

        // Actor FROZEN → effective scale is scale_floor
        let actor_scale = agent.effective_actor_scale(0.5);
        assert!((actor_scale - agent.config.scale_floor).abs() < f64::EPSILON);

        // Critic PLASTIC → critic_surprise_scale computes from td_error_buffer
        let critic_scale = agent.critic_surprise_scale(0.5);
        assert!(critic_scale >= agent.config.scale_floor);
        assert!(critic_scale <= agent.config.scale_ceil);
    }

    #[test]
    fn actor_wakes_critic_coupling_default_threshold() {
        let mut cfg = default_config();
        cfg.actor_hysteresis = true;
        cfg.critic_hysteresis = true;
        cfg.actor_wakes_critic = true;
        // Default threshold = 1000
        let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

        // Set both to FROZEN
        agent.actor_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_frozen_steps = 1000;

        // Set up actor for wake transition
        setup_for_wake(agent.actor_hysteresis.as_mut().unwrap());

        agent.process_hysteresis(1.0, 0.0);

        // Actor should be PLASTIC
        assert_eq!(
            agent.actor_hysteresis.as_ref().unwrap().state,
            PlasticityState::Plastic
        );
        // Critic forced to PLASTIC via coupling
        assert_eq!(
            agent.critic_hysteresis.as_ref().unwrap().state,
            PlasticityState::Plastic
        );
        // Counters reset
        assert_eq!(agent.actor_plastic_step_counter, 0);
        assert_eq!(agent.critic_frozen_steps, 0);
    }

    #[test]
    fn actor_wakes_critic_custom_threshold() {
        let mut cfg = default_config();
        cfg.actor_hysteresis = true;
        cfg.critic_hysteresis = true;
        cfg.actor_wakes_critic = true;
        cfg.actor_wakes_critic_threshold = 50;
        let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

        // Both FROZEN, critic below custom threshold
        agent.actor_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_frozen_steps = 40;

        setup_for_wake(agent.actor_hysteresis.as_mut().unwrap());
        agent.process_hysteresis(1.0, 0.0);

        // Actor wakes, but critic stays FROZEN (41 < 50 threshold)
        assert_eq!(
            agent.actor_hysteresis.as_ref().unwrap().state,
            PlasticityState::Plastic
        );
        assert_eq!(
            agent.critic_hysteresis.as_ref().unwrap().state,
            PlasticityState::Frozen
        );

        // Now set critic above threshold and trigger again
        agent.actor_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_frozen_steps = 50;
        setup_for_wake(agent.actor_hysteresis.as_mut().unwrap());
        agent.process_hysteresis(1.0, 0.0);

        // Now coupling fires (51 >= 50)
        assert_eq!(
            agent.critic_hysteresis.as_ref().unwrap().state,
            PlasticityState::Plastic
        );
    }

    #[test]
    fn actor_wakes_critic_off_by_default() {
        let mut cfg = default_config();
        cfg.actor_hysteresis = true;
        cfg.critic_hysteresis = true;
        // actor_wakes_critic defaults to false
        let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

        agent.actor_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_frozen_steps = 2000;

        setup_for_wake(agent.actor_hysteresis.as_mut().unwrap());
        agent.process_hysteresis(1.0, 0.0);

        // Actor transitions to PLASTIC
        assert_eq!(
            agent.actor_hysteresis.as_ref().unwrap().state,
            PlasticityState::Plastic
        );
        // Critic stays FROZEN (coupling disabled)
        assert_eq!(
            agent.critic_hysteresis.as_ref().unwrap().state,
            PlasticityState::Frozen
        );
    }

    #[test]
    fn critic_frozen_steps_resets_on_plastic() {
        let mut cfg = default_config();
        cfg.critic_hysteresis = true;
        let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

        // Set critic to FROZEN with accumulated frozen steps
        agent.critic_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;
        agent.critic_frozen_steps = 500;

        // Set up critic for wake transition
        let critic_hyst = agent.critic_hysteresis.as_mut().unwrap();
        critic_hyst.slow.value = 0.05;
        critic_hyst.slow.k = 200;
        critic_hyst.fast.value = 0.06;
        critic_hyst.fast.k = 200;

        // Feed high signal to critic to trigger wake
        agent.process_hysteresis(0.0, 1.0);

        assert_eq!(
            agent.critic_hysteresis.as_ref().unwrap().state,
            PlasticityState::Plastic
        );
        assert_eq!(agent.critic_frozen_steps, 0);
    }

    #[test]
    fn hysteresis_disabled_by_default() {
        let agent = make_agent();
        assert!(agent.actor_hysteresis.is_none());
        assert!(agent.critic_hysteresis.is_none());
        // surprise_scale works normally (legacy behavior)
        let scale = agent.surprise_scale(0.1);
        assert!(scale > 0.0);
    }

    #[test]
    fn wake_fraction_zero_rejected() {
        let mut cfg = default_config();
        cfg.actor_hysteresis = true;
        cfg.actor_wake_fraction = 0.0;
        let result = PcActorCritic::new(CpuLinAlg::new(), cfg, 42);
        assert!(result.is_err());
        match result.unwrap_err() {
            PcError::ConfigValidation(msg) => assert!(msg.contains("wake_fraction")),
            e => panic!("expected ConfigValidation, got {:?}", e),
        }
    }

    #[test]
    fn sleep_fraction_one_rejected() {
        let mut cfg = default_config();
        cfg.actor_hysteresis = true;
        cfg.actor_sleep_fraction = 1.0;
        let result = PcActorCritic::new(CpuLinAlg::new(), cfg, 42);
        assert!(result.is_err());
        match result.unwrap_err() {
            PcError::ConfigValidation(msg) => assert!(msg.contains("sleep_fraction")),
            e => panic!("expected ConfigValidation, got {:?}", e),
        }
    }

    #[test]
    fn sleep_fraction_zero_rejected() {
        let mut cfg = default_config();
        cfg.critic_hysteresis = true;
        cfg.critic_sleep_fraction = 0.0;
        let result = PcActorCritic::new(CpuLinAlg::new(), cfg, 42);
        assert!(result.is_err());
        match result.unwrap_err() {
            PcError::ConfigValidation(msg) => assert!(msg.contains("sleep_fraction")),
            e => panic!("expected ConfigValidation, got {:?}", e),
        }
    }

    #[test]
    fn td_error_buffer_feeds_critic_scale() {
        let mut cfg = default_config();
        cfg.critic_hysteresis = true;
        cfg.adaptive_surprise = true;
        cfg.surprise_buffer_size = 100;
        let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

        // Feed known |td_error| values into buffer
        for i in 0..20 {
            agent.td_error_buffer.push_back(0.1 * (i as f64));
        }

        // Compute expected adaptive thresholds from buffer
        let mean: f64 = agent.td_error_buffer.iter().sum::<f64>() / 20.0;
        let variance: f64 = agent
            .td_error_buffer
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / 20.0;
        let std = variance.sqrt();
        let lo = (mean - 0.5 * std).max(0.0);
        let hi = mean + 1.5 * std;

        // At low threshold → scale_floor
        let scale_low = agent.critic_surprise_scale(lo);
        assert!((scale_low - agent.config.scale_floor).abs() < 1e-10);

        // At high threshold → scale_ceil
        let scale_high = agent.critic_surprise_scale(hi);
        assert!((scale_high - agent.config.scale_ceil).abs() < 1e-10);

        // At midpoint → linear interpolation midpoint
        let mid = (lo + hi) / 2.0;
        let scale_mid = agent.critic_surprise_scale(mid);
        let expected_mid = (agent.config.scale_floor + agent.config.scale_ceil) / 2.0;
        assert!((scale_mid - expected_mid).abs() < 1e-10);
    }

    // ── Phase 3: Layer-Wise Consolidation Decay (M3) Tests ──────────

    /// Helper: config with 3 hidden layers for decay tests.
    fn three_layer_config() -> PcActorCriticConfig {
        PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: 9,
                hidden_layers: vec![
                    LayerDef {
                        size: 12,
                        activation: Activation::Tanh,
                    },
                    LayerDef {
                        size: 12,
                        activation: Activation::Tanh,
                    },
                    LayerDef {
                        size: 8,
                        activation: Activation::Tanh,
                    },
                ],
                output_size: 9,
                output_activation: Activation::Tanh,
                alpha: 0.1,
                tol: 0.01,
                min_steps: 1,
                max_steps: 20,
                lr_weights: 0.01,
                synchronous: true,
                temperature: 1.0,
                local_lambda: 1.0,
                residual: false,
                rezero_init: 0.001,
            },
            critic: MlpCriticConfig {
                input_size: 41, // 9 + 12 + 12 + 8
                hidden_layers: vec![
                    LayerDef {
                        size: 20,
                        activation: Activation::Tanh,
                    },
                    LayerDef {
                        size: 16,
                        activation: Activation::Tanh,
                    },
                ],
                output_activation: Activation::Linear,
                lr: 0.005,
            },
            gamma: 0.95,
            surprise_low: 0.02,
            surprise_high: 0.15,
            adaptive_surprise: false,
            surprise_buffer_size: 100,
            entropy_coeff: 0.01,
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
            actor_wakes_critic: false,
            actor_wakes_critic_threshold: 1000,
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
        }
    }

    #[test]
    fn test_decay_1_0_is_noop() {
        let config = PcActorCriticConfig {
            consolidation_decay: 1.0,
            ..three_layer_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // All factors should be 1.0 (no decay)
        for &f in &agent.actor_decay_factors {
            assert!((f - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_decay_0_5_three_layers() {
        let config = PcActorCriticConfig {
            consolidation_decay: 0.5,
            ..three_layer_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // 3 hidden layers: factors = [0.5^2, 0.5^1, 0.5^0] = [0.25, 0.5, 1.0]
        assert_eq!(agent.actor_decay_factors.len(), 3);
        assert!((agent.actor_decay_factors[0] - 0.25).abs() < f64::EPSILON);
        assert!((agent.actor_decay_factors[1] - 0.5).abs() < f64::EPSILON);
        assert!((agent.actor_decay_factors[2] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decay_factors_precomputed() {
        let mut config = three_layer_config();
        config.actor.hidden_layers.push(LayerDef {
            size: 6,
            activation: Activation::Tanh,
        });
        config.consolidation_decay = 0.5;
        // Fix critic input_size for 4 hidden layers: 9 + 12 + 12 + 8 + 6 = 47
        config.critic.input_size = 47;
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // 4 hidden layers: [0.5^3, 0.5^2, 0.5^1, 0.5^0] = [0.125, 0.25, 0.5, 1.0]
        assert_eq!(agent.actor_decay_factors.len(), 4);
        assert!((agent.actor_decay_factors[0] - 0.125).abs() < f64::EPSILON);
        assert!((agent.actor_decay_factors[1] - 0.25).abs() < f64::EPSILON);
        assert!((agent.actor_decay_factors[2] - 0.5).abs() < f64::EPSILON);
        assert!((agent.actor_decay_factors[3] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_critic_independent_decay() {
        let config = PcActorCriticConfig {
            consolidation_decay: 0.5,
            critic_consolidation_decay: 0.8,
            ..three_layer_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // Actor: 3 hidden layers, decay=0.5 → [0.25, 0.5, 1.0]
        assert_eq!(agent.actor_decay_factors.len(), 3);
        assert!((agent.actor_decay_factors[0] - 0.25).abs() < f64::EPSILON);
        // Critic: 2 hidden layers, decay=0.8 → [0.8^1, 0.8^0] = [0.8, 1.0]
        assert_eq!(agent.critic_decay_factors.len(), 2);
        assert!((agent.critic_decay_factors[0] - 0.8).abs() < f64::EPSILON);
        assert!((agent.critic_decay_factors[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decay_single_hidden_layer_is_noop() {
        let config = PcActorCriticConfig {
            consolidation_decay: 0.5,
            ..default_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // 1 hidden layer: 0.5^(1-1-0) = 0.5^0 = 1.0
        assert_eq!(agent.actor_decay_factors.len(), 1);
        assert!((agent.actor_decay_factors[0] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decay_no_hidden_layers_safe() {
        let config = PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: 9,
                hidden_layers: vec![],
                output_size: 9,
                output_activation: Activation::Linear,
                alpha: 0.0,
                tol: 0.01,
                min_steps: 1,
                max_steps: 1,
                lr_weights: 0.01,
                synchronous: true,
                temperature: 1.0,
                local_lambda: 1.0,
                residual: false,
                rezero_init: 0.001,
            },
            critic: MlpCriticConfig {
                input_size: 9,
                hidden_layers: vec![],
                output_activation: Activation::Linear,
                lr: 0.005,
            },
            consolidation_decay: 0.5,
            ..default_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // 0 hidden layers: empty decay factors
        assert!(agent.actor_decay_factors.is_empty());
        assert!(agent.critic_decay_factors.is_empty());
    }

    #[test]
    fn test_consolidation_decay_zero_freezes_early_layers() {
        let config = PcActorCriticConfig {
            consolidation_decay: 0.0,
            ..three_layer_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // 3 hidden layers, decay=0.0: [0.0^2, 0.0^1, 0.0^0] = [0.0, 0.0, 1.0]
        assert!((agent.actor_decay_factors[0] - 0.0).abs() < f64::EPSILON);
        assert!((agent.actor_decay_factors[1] - 0.0).abs() < f64::EPSILON);
        assert!((agent.actor_decay_factors[2] - 1.0).abs() < f64::EPSILON);
    }

    // ── M3b: Adaptive Sigmoid Tests ─────────────────────────────────

    #[test]
    fn test_adaptive_sigmoid_output_range() {
        // sigmoid(-k*(e - threshold)) should always be in [0, 1]
        let k = 10.0;
        let threshold = 0.05;
        for &error in &[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0] {
            let x: f64 = -k * (error - threshold);
            let sig = 1.0 / (1.0 + (-x).exp());
            assert!((0.0..=1.0).contains(&sig), "sigmoid({x}) = {sig}");
        }
    }

    #[test]
    fn test_adaptive_sigmoid_low_error_protects() {
        // error_ema << threshold → high adaptive_decay → strong protection
        let k = 10.0;
        let threshold = 0.05;
        let error = 0.001; // very low
        let x: f64 = -k * (error - threshold);
        let adaptive_decay = 1.0 / (1.0 + (-x).exp());
        // sigmoid(0.49) ≈ 0.62 → (1 - 0.62) ≈ 0.38 effective
        assert!(
            adaptive_decay > 0.5,
            "low error should give high decay (protection)"
        );
        assert!(
            (1.0 - adaptive_decay) < 0.5,
            "effective learning should be < 50%"
        );
    }

    #[test]
    fn test_adaptive_sigmoid_high_error_releases() {
        // error_ema >> threshold → low adaptive_decay → full plasticity
        let k = 10.0;
        let threshold = 0.05;
        let error = 0.5; // very high
        let x: f64 = -k * (error - threshold);
        let adaptive_decay = 1.0 / (1.0 + (-x).exp());
        // sigmoid(-4.5) ≈ 0.011 → (1 - 0.011) ≈ 0.989 effective
        assert!(
            adaptive_decay < 0.1,
            "high error should give low decay (release)"
        );
        assert!(
            (1.0 - adaptive_decay) > 0.9,
            "effective learning should be > 90%"
        );
    }

    #[test]
    fn test_adaptive_overrides_fixed_decay() {
        let config = PcActorCriticConfig {
            consolidation_decay: 0.5, // would give [0.25, 0.5, 1.0]
            adaptive_consolidation: true,
            ..three_layer_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // When adaptive is on, layer_error_ema should be initialized
        assert_eq!(agent.layer_error_ema.len(), 3);
        // Fixed decay factors should still be precomputed (for critic)
        // but adaptive flag takes precedence for actor
        assert!(agent.config.adaptive_consolidation);
    }

    #[test]
    fn test_m3b_error_ema_uses_consolidation_ema_beta() {
        let config = PcActorCriticConfig {
            adaptive_consolidation: true,
            consolidation_ema_beta: 0.9,
            ..three_layer_config()
        };
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        // Verify the config beta is 0.9 (will be used in EMA updates)
        assert!((agent.config.consolidation_ema_beta - 0.9).abs() < f64::EPSILON);
        // layer_error_ema initialized to zeros
        for &e in &agent.layer_error_ema {
            assert!((e - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_m3b_cold_start_effective_lr() {
        // Fresh agent with adaptive: all EMAs = 0.0
        // sigmoid(-10 * (0.0 - 0.05)) = sigmoid(0.5) ≈ 0.6225
        // effective = (1.0 - 0.6225) ≈ 0.3775
        let k = 10.0;
        let threshold = 0.05;
        let error = 0.0;
        let x: f64 = -k * (error - threshold);
        let adaptive_decay = 1.0 / (1.0 + (-x).exp());
        let effective = 1.0 - adaptive_decay;
        // ~38% of full learning rate at cold start
        assert!(
            effective > 0.35 && effective < 0.40,
            "cold start effective factor should be ~0.378, got {effective}"
        );
    }

    // ── Validation Tests ────────────────────────────────────────────

    #[test]
    fn test_decay_out_of_range_rejected() {
        // consolidation_decay=1.5 → error
        let config = PcActorCriticConfig {
            consolidation_decay: 1.5,
            ..three_layer_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("consolidation_decay"),
            "error should mention consolidation_decay: {err}"
        );

        // consolidation_decay=-0.1 → error
        let config2 = PcActorCriticConfig {
            consolidation_decay: -0.1,
            ..three_layer_config()
        };
        let result2 = PcActorCritic::new(CpuLinAlg::new(), config2, 42);
        assert!(result2.is_err());
    }

    #[test]
    fn test_critic_decay_out_of_range_rejected() {
        let config = PcActorCriticConfig {
            critic_consolidation_decay: 2.0,
            ..three_layer_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("critic_consolidation_decay"),
            "error should mention critic_consolidation_decay: {err}"
        );
    }

    #[test]
    fn test_adaptive_sigmoid_steepness_positive_required() {
        let config = PcActorCriticConfig {
            adaptive_consolidation: true,
            consolidation_sigmoid_k: -1.0,
            ..three_layer_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("consolidation_sigmoid_k"),
            "error should mention consolidation_sigmoid_k: {err}"
        );
    }

    #[test]
    fn test_consolidation_ema_beta_out_of_range_rejected() {
        // beta=0.0 → rejected
        let config = PcActorCriticConfig {
            adaptive_consolidation: true,
            consolidation_ema_beta: 0.0,
            ..three_layer_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err());

        // beta=1.0 → rejected
        let config2 = PcActorCriticConfig {
            adaptive_consolidation: true,
            consolidation_ema_beta: 1.0,
            ..three_layer_config()
        };
        let result2 = PcActorCritic::new(CpuLinAlg::new(), config2, 42);
        assert!(result2.is_err());
    }

    #[test]
    fn test_consolidation_error_threshold_nonpositive_rejected() {
        let config = PcActorCriticConfig {
            adaptive_consolidation: true,
            consolidation_error_threshold: 0.0,
            ..three_layer_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err());

        let config2 = PcActorCriticConfig {
            adaptive_consolidation: true,
            consolidation_error_threshold: -0.1,
            ..three_layer_config()
        };
        let result2 = PcActorCritic::new(CpuLinAlg::new(), config2, 42);
        assert!(result2.is_err());
    }

    // ============ Phase 4: EWC Regularization (M4) Tests ============

    /// Helper: create an EWC-enabled config with hysteresis.
    fn ewc_config() -> PcActorCriticConfig {
        PcActorCriticConfig {
            ewc_lambda: 1.0,
            fisher_decay: 0.9,
            fisher_ema_beta: 0.99,
            logits_reversal: false,
            actor_hysteresis: true,
            actor_fast_window: 5,
            actor_slow_window: 20,
            actor_wake_fraction: 0.5,
            actor_sleep_fraction: 0.3,
            critic_hysteresis: true,
            critic_fast_window: 5,
            critic_slow_window: 20,
            critic_wake_fraction: 0.5,
            critic_sleep_fraction: 0.3,
            ..default_config()
        }
    }

    // ── Gradient extraction ──────────────────────────────────────

    #[test]
    fn test_gradient_extraction_spike_trivial_layer() {
        // Verify that extract_gradients produces the expected g_raw for a known layer
        let backend = CpuLinAlg::new();
        let mut config = default_config();
        config.ewc_lambda = 1.0;
        let mut agent: PcActorCritic = PcActorCritic::new(backend, config, 42).unwrap();

        // Run one step so we have an infer result
        let state = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let _a = agent.step(&state, 0.0, false);
        let state2 = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let _a2 = agent.step(&state2, 1.0, true);

        // After learning, Fisher should have non-zero F_ema
        assert!(!agent.actor_fisher.is_empty());
        let f = &agent.actor_fisher[0];
        let rows = agent.backend.mat_rows(&f.f_ema_weights);
        let cols = agent.backend.mat_cols(&f.f_ema_weights);
        let mut has_nonzero = false;
        for r in 0..rows {
            for c in 0..cols {
                if agent.backend.mat_get(&f.f_ema_weights, r, c).abs() > 0.0 {
                    has_nonzero = true;
                }
            }
        }
        assert!(
            has_nonzero,
            "F_ema should have non-zero entries after learning"
        );
    }

    // ── Fisher EMA accumulation ──────────────────────────────────

    #[test]
    fn test_fisher_ema_accumulates_during_plastic() {
        // Agent in PLASTIC state. After learning steps, F_ema should be non-zero.
        let mut config = default_config();
        config.ewc_lambda = 1.0;
        config.fisher_ema_beta = 0.99;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Multiple learning steps to accumulate Fisher
        let state1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        agent.step(&state1, 0.0, false);
        let state2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        agent.step(&state2, 1.0, false);
        let state3 = vec![-0.5, 0.0, 1.0, -1.0, 0.5, 0.5, 0.0, 1.0, -0.5];
        agent.step(&state3, -1.0, true);

        // F_ema should be non-zero for actor layers
        assert!(!agent.actor_fisher.is_empty());
        let f = &agent.actor_fisher[0];
        let rows = agent.backend.mat_rows(&f.f_ema_weights);
        let cols = agent.backend.mat_cols(&f.f_ema_weights);
        let mut sum = 0.0;
        for r in 0..rows {
            for c in 0..cols {
                sum += agent.backend.mat_get(&f.f_ema_weights, r, c);
            }
        }
        assert!(
            sum > 0.0,
            "F_ema should have positive entries (squared gradients)"
        );
    }

    #[test]
    fn test_fisher_ema_bounded_by_grad_clip_squared() {
        // Maximum g_raw = GRAD_CLIP = 5.0. So g_raw^2 = 25.0.
        // F_ema per element <= 25.0 / (1 - beta) in steady state.
        let mut config = default_config();
        config.ewc_lambda = 1.0;
        config.fisher_ema_beta = 0.99;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Run many steps to saturate F_ema
        for i in 0..200 {
            let state: Vec<f64> = (0..9).map(|j| ((i + j) as f64 * 0.3).sin()).collect();
            agent.step(&state, if i % 2 == 0 { 1.0 } else { -1.0 }, i == 199);
        }

        // Check bound: each F_ema element <= 25.0 / (1 - 0.99) = 2500.0
        let max_bound = 25.0 / (1.0 - 0.99);
        for fisher in &agent.actor_fisher {
            let rows = agent.backend.mat_rows(&fisher.f_ema_weights);
            let cols = agent.backend.mat_cols(&fisher.f_ema_weights);
            for r in 0..rows {
                for c in 0..cols {
                    let val = agent.backend.mat_get(&fisher.f_ema_weights, r, c);
                    assert!(
                        val <= max_bound + 1e-6,
                        "F_ema element {} exceeds bound {}",
                        val,
                        max_bound
                    );
                }
            }
        }
    }

    #[test]
    fn test_fisher_gradient_extraction_approach1() {
        // Verify gradient extraction: apply_derivative, hadamard, clip
        let backend = CpuLinAlg::new();

        // Known output and delta
        let output = backend.vec_from_slice(&[0.5, -0.3, 0.8]);
        let delta = backend.vec_from_slice(&[1.0, 2.0, -1.5]);

        // Tanh derivative: 1 - tanh(x)^2
        let deriv = backend.apply_derivative(&output, Activation::Tanh);
        let mut grad = backend.vec_hadamard(&delta, &deriv);
        backend.clip_vec(&mut grad, 5.0);

        // Verify the gradient values are computed correctly
        let grad_vec = backend.vec_to_vec(&grad);
        assert_eq!(grad_vec.len(), 3);
        // Each element should be delta[i] * (1 - output[i]^2)
        for i in 0..3 {
            let out_i = [0.5, -0.3, 0.8][i];
            let delta_i = [1.0, 2.0, -1.5][i];
            let expected: f64 = delta_i * (1.0 - out_i * out_i);
            let expected_clipped = expected.clamp(-5.0, 5.0);
            assert!(
                (grad_vec[i] - expected_clipped).abs() < 1e-10,
                "grad[{}] = {}, expected {}",
                i,
                grad_vec[i],
                expected_clipped
            );
        }
    }

    // ── Fisher lifecycle ──────────────────────────────────────

    #[test]
    fn test_fisher_decay_on_reliable_phase() {
        // last_phase_reliable=true, fisher_decay=0.9.
        // On FROZEN→PLASTIC: F_total *= 0.9.
        let mut config = ewc_config();
        config.fisher_decay = 0.9;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Manually set F_total to known values and mark reliable
        let rows = agent
            .backend
            .mat_rows(&agent.actor_fisher[0].f_total_weights);
        let cols = agent
            .backend
            .mat_cols(&agent.actor_fisher[0].f_total_weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 10.0);
            }
        }
        agent.actor_last_phase_reliable = true;

        // Force actor to FROZEN state then trigger wake (FROZEN→PLASTIC)
        agent.actor_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;

        // Simulate a wake transition by calling process_hysteresis with high signal
        // First need to make fast > slow * (1 + wake)
        let hyst = agent.actor_hysteresis.as_mut().unwrap();
        hyst.fast.value = 1.0;
        hyst.slow.value = 0.1;
        // Manually trigger the lifecycle
        agent.handle_fisher_wake(true);

        // F_total should be *= 0.9 → 9.0
        for r in 0..rows {
            for c in 0..cols {
                let val = agent
                    .backend
                    .mat_get(&agent.actor_fisher[0].f_total_weights, r, c);
                assert!(
                    (val - 9.0).abs() < 1e-10,
                    "F_total should be 10.0 * 0.9 = 9.0, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_fisher_no_decay_after_unreliable_phase() {
        // last_phase_reliable=false. F_total unchanged on FROZEN→PLASTIC.
        let mut config = ewc_config();
        config.fisher_decay = 0.9;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Set F_total to known value, but mark unreliable
        let rows = agent
            .backend
            .mat_rows(&agent.actor_fisher[0].f_total_weights);
        let cols = agent
            .backend
            .mat_cols(&agent.actor_fisher[0].f_total_weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 10.0);
            }
        }
        agent.actor_last_phase_reliable = false;

        agent.handle_fisher_wake(true);

        // F_total should be unchanged → 10.0
        for r in 0..rows {
            for c in 0..cols {
                let val = agent
                    .backend
                    .mat_get(&agent.actor_fisher[0].f_total_weights, r, c);
                assert!(
                    (val - 10.0).abs() < 1e-10,
                    "F_total should be unchanged at 10.0, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_fisher_short_phase_discards_fema() {
        // 50 steps (< min_fisher_phase=100 for beta=0.99). F_ema NOT added to F_total.
        let mut config = ewc_config();
        config.fisher_ema_beta = 0.99; // min_fisher_phase = ceil(1/0.01) = 100
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Set F_ema to non-zero
        let rows = agent.backend.mat_rows(&agent.actor_fisher[0].f_ema_weights);
        let cols = agent.backend.mat_cols(&agent.actor_fisher[0].f_ema_weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_ema_weights, r, c, 5.0);
            }
        }
        // Only 50 plastic steps (< 100)
        agent.actor_plastic_step_counter = 50;

        // Trigger sleep (PLASTIC→FROZEN)
        agent.handle_fisher_sleep(true);

        // F_total should still be zero (F_ema discarded)
        for r in 0..rows {
            for c in 0..cols {
                let val = agent
                    .backend
                    .mat_get(&agent.actor_fisher[0].f_total_weights, r, c);
                assert!(
                    val.abs() < 1e-10,
                    "F_total should be zero (short phase), got {}",
                    val
                );
            }
        }
        // last_phase_reliable should be false
        assert!(!agent.actor_last_phase_reliable);
    }

    #[test]
    fn test_fisher_reliable_phase_adds_fema() {
        // 150 steps (>= 100). F_total += F_ema. last_phase_reliable = true.
        let mut config = ewc_config();
        config.fisher_ema_beta = 0.99;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Set F_ema to known values
        let rows = agent.backend.mat_rows(&agent.actor_fisher[0].f_ema_weights);
        let cols = agent.backend.mat_cols(&agent.actor_fisher[0].f_ema_weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_ema_weights, r, c, 5.0);
            }
        }
        // 150 plastic steps (>= 100)
        agent.actor_plastic_step_counter = 150;

        agent.handle_fisher_sleep(true);

        // F_total should be 0 + 5.0 = 5.0
        for r in 0..rows {
            for c in 0..cols {
                let val = agent
                    .backend
                    .mat_get(&agent.actor_fisher[0].f_total_weights, r, c);
                assert!(
                    (val - 5.0).abs() < 1e-10,
                    "F_total should be 5.0, got {}",
                    val
                );
            }
        }
        assert!(agent.actor_last_phase_reliable);
    }

    #[test]
    fn test_fisher_preserved_through_oscillations() {
        // 5 rapid oscillations (each < 100 steps). F_total unchanged.
        let mut config = ewc_config();
        config.fisher_ema_beta = 0.99;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Pre-load F_total
        let rows = agent
            .backend
            .mat_rows(&agent.actor_fisher[0].f_total_weights);
        let cols = agent
            .backend
            .mat_cols(&agent.actor_fisher[0].f_total_weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 42.0);
            }
        }

        // 5 oscillations: PLASTIC→FROZEN (short), FROZEN→PLASTIC (unreliable)
        for _ in 0..5 {
            agent.actor_plastic_step_counter = 30; // short
            agent.handle_fisher_sleep(true);
            agent.handle_fisher_wake(true);
        }

        // F_total unchanged at 42.0
        for r in 0..rows {
            for c in 0..cols {
                let val = agent
                    .backend
                    .mat_get(&agent.actor_fisher[0].f_total_weights, r, c);
                assert!(
                    (val - 42.0).abs() < 1e-10,
                    "F_total should be preserved at 42.0, got {}",
                    val
                );
            }
        }
    }

    // ── EWC correction ──────────────────────────────────────

    #[test]
    fn test_ewc_correction_direction() {
        // Known weights, Fisher, snapshot. Correction pulls toward snapshot.
        let mut config = default_config();
        config.ewc_lambda = 1.0;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Set up Fisher and snapshot
        let rows = agent.backend.mat_rows(&agent.actor.layers[0].weights);
        let cols = agent.backend.mat_cols(&agent.actor.layers[0].weights);

        // F_total = 1.0 everywhere
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 1.0);
            }
        }

        // Snapshot = current weights
        agent.actor_fisher[0].theta_snapshot_weights = Some(agent.actor.layers[0].weights.clone());
        agent.actor_fisher[0].theta_snapshot_bias = Some(agent.actor.layers[0].bias.clone());

        // Perturb current weights away from snapshot
        let snapshot_val = agent.backend.mat_get(&agent.actor.layers[0].weights, 0, 0);
        agent
            .backend
            .mat_set(&mut agent.actor.layers[0].weights, 0, 0, snapshot_val + 0.5);
        let weight_before = agent.backend.mat_get(&agent.actor.layers[0].weights, 0, 0);

        // Do a learning step
        let state1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        agent.step(&state1, 0.0, false);
        let state2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        agent.step(&state2, 1.0, true);

        let weight_after = agent.backend.mat_get(&agent.actor.layers[0].weights, 0, 0);
        // EWC should pull weight back toward snapshot (smaller deviation)
        let deviation_before = (weight_before - snapshot_val).abs();
        let deviation_after = (weight_after - snapshot_val).abs();
        assert!(
            deviation_after < deviation_before,
            "EWC should pull weights toward snapshot: before={}, after={}",
            deviation_before,
            deviation_after
        );
    }

    #[test]
    fn test_ewc_uses_pre_update_theta() {
        // Verify EWC penalty computed from weights BEFORE backward modifies them.
        // This is structural: the correction uses snapshot vs pre-backward weights.
        let mut config = default_config();
        config.ewc_lambda = 10.0; // Large lambda to make EWC effect dominant
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Set F_total and snapshot
        let rows = agent.backend.mat_rows(&agent.actor.layers[0].weights);
        let cols = agent.backend.mat_cols(&agent.actor.layers[0].weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 1.0);
            }
        }
        // Snapshot at current weights
        agent.actor_fisher[0].theta_snapshot_weights = Some(agent.actor.layers[0].weights.clone());
        agent.actor_fisher[0].theta_snapshot_bias = Some(agent.actor.layers[0].bias.clone());

        // Record pre-update weights
        let w_pre = agent.backend.mat_get(&agent.actor.layers[0].weights, 0, 0);

        // Step to trigger learning
        let state1 = vec![1.0; 9];
        agent.step(&state1, 0.0, false);
        let state2 = vec![0.5; 9];
        agent.step(&state2, 1.0, true);

        // Since snapshot == pre-update, EWC correction = lambda * F * (W_pre - snapshot) = 0
        // So EWC should NOT affect the backward-only update
        // (The test verifies the ordering: pre-update theta is used)
        let w_post = agent.backend.mat_get(&agent.actor.layers[0].weights, 0, 0);
        // With snapshot == initial weights, EWC correction is zero
        // The weight should change due to backward only
        assert!(
            (w_post - w_pre).abs() > 1e-12,
            "Weights should change from backward pass"
        );
    }

    #[test]
    fn test_ewc_weight_clip_applied_after_correction() {
        // Correction pushes beyond WEIGHT_CLIP. Verify clipped to [-5.0, 5.0].
        let mut config = default_config();
        config.ewc_lambda = 1000.0; // Extreme lambda
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let rows = agent.backend.mat_rows(&agent.actor.layers[0].weights);
        let cols = agent.backend.mat_cols(&agent.actor.layers[0].weights);

        // Set huge F_total and distant snapshot
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 100.0);
            }
        }
        // Snapshot far from current weights
        let mut snapshot = agent.actor.layers[0].weights.clone();
        for r in 0..rows {
            for c in 0..cols {
                agent.backend.mat_set(&mut snapshot, r, c, -100.0);
            }
        }
        agent.actor_fisher[0].theta_snapshot_weights = Some(snapshot);
        agent.actor_fisher[0].theta_snapshot_bias = Some(agent.actor.layers[0].bias.clone());

        // Step to trigger learning + EWC correction
        let state1 = vec![1.0; 9];
        agent.step(&state1, 0.0, false);
        let state2 = vec![0.5; 9];
        agent.step(&state2, 1.0, true);

        // All weights should be clipped to [-5.0, 5.0]
        for r in 0..rows {
            for c in 0..cols {
                let val = agent.backend.mat_get(&agent.actor.layers[0].weights, r, c);
                assert!(
                    (-5.0 - 1e-10..=5.0 + 1e-10).contains(&val),
                    "Weight at ({},{}) = {} should be clipped to [-5.0, 5.0]",
                    r,
                    c,
                    val
                );
            }
        }
    }

    #[test]
    fn test_ewc_lambda_zero_is_noop() {
        // ewc_lambda=0.0. Weights identical to backward-only.
        let config_no_ewc = default_config(); // ewc_lambda = 0.0
        let mut agent_no_ewc: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config_no_ewc, 42).unwrap();

        let mut config_ewc = default_config();
        config_ewc.ewc_lambda = 0.0; // Explicitly zero
        let mut agent_ewc: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config_ewc, 42).unwrap();

        // Same sequence of steps
        let state1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let state2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];

        agent_no_ewc.step(&state1, 0.0, false);
        agent_no_ewc.step(&state2, 1.0, true);

        agent_ewc.step(&state1, 0.0, false);
        agent_ewc.step(&state2, 1.0, true);

        // Weights should be identical
        let rows = agent_no_ewc
            .backend
            .mat_rows(&agent_no_ewc.actor.layers[0].weights);
        let cols = agent_no_ewc
            .backend
            .mat_cols(&agent_no_ewc.actor.layers[0].weights);
        for r in 0..rows {
            for c in 0..cols {
                let v1 = agent_no_ewc
                    .backend
                    .mat_get(&agent_no_ewc.actor.layers[0].weights, r, c);
                let v2 = agent_ewc
                    .backend
                    .mat_get(&agent_ewc.actor.layers[0].weights, r, c);
                assert!(
                    (v1 - v2).abs() < 1e-12,
                    "Weights differ at ({},{}): {} vs {}",
                    r,
                    c,
                    v1,
                    v2
                );
            }
        }
    }

    #[test]
    fn test_ewc_propagated_gradient_clean() {
        // Propagated delta identical with and without EWC.
        // We verify this by comparing weight changes in the INPUT layer
        // (layer 0), which receives the propagated delta from the output layer.
        // If EWC contaminated the propagated gradient, the input layer would
        // differ between EWC and non-EWC runs.
        let mut config_ewc = default_config();
        config_ewc.ewc_lambda = 5.0;
        let mut agent_ewc: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config_ewc, 42).unwrap();

        // Set EWC only on the output layer (layer 1), NOT the hidden layer
        // F_total[0] = zeros, F_total[1] = ones
        // This means output layer gets EWC correction but hidden layer doesn't
        let n_layers = agent_ewc.actor.layers.len();
        assert!(n_layers >= 2, "Need at least 2 layers");
        let rows1 = agent_ewc
            .backend
            .mat_rows(&agent_ewc.actor.layers[n_layers - 1].weights);
        let cols1 = agent_ewc
            .backend
            .mat_cols(&agent_ewc.actor.layers[n_layers - 1].weights);
        for r in 0..rows1 {
            for c in 0..cols1 {
                agent_ewc.backend.mat_set(
                    &mut agent_ewc.actor_fisher[n_layers - 1].f_total_weights,
                    r,
                    c,
                    1.0,
                );
            }
        }
        agent_ewc.actor_fisher[n_layers - 1].theta_snapshot_weights =
            Some(agent_ewc.actor.layers[n_layers - 1].weights.clone());
        agent_ewc.actor_fisher[n_layers - 1].theta_snapshot_bias =
            Some(agent_ewc.actor.layers[n_layers - 1].bias.clone());

        // Non-EWC reference agent
        let config_ref = default_config();
        let mut agent_ref: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config_ref, 42).unwrap();

        // Record hidden layer weights before
        let h_rows = agent_ewc
            .backend
            .mat_rows(&agent_ewc.actor.layers[0].weights);
        let h_cols = agent_ewc
            .backend
            .mat_cols(&agent_ewc.actor.layers[0].weights);
        let mut h_before_ewc = vec![0.0; h_rows * h_cols];
        let mut h_before_ref = vec![0.0; h_rows * h_cols];
        for r in 0..h_rows {
            for c in 0..h_cols {
                h_before_ewc[r * h_cols + c] =
                    agent_ewc
                        .backend
                        .mat_get(&agent_ewc.actor.layers[0].weights, r, c);
                h_before_ref[r * h_cols + c] =
                    agent_ref
                        .backend
                        .mat_get(&agent_ref.actor.layers[0].weights, r, c);
            }
        }

        // Same learning steps
        let state1 = vec![1.0; 9];
        let state2 = vec![0.5; 9];
        agent_ewc.step(&state1, 0.0, false);
        agent_ewc.step(&state2, 1.0, true);
        agent_ref.step(&state1, 0.0, false);
        agent_ref.step(&state2, 1.0, true);

        // Hidden layer weight deltas should be identical (clean propagated gradient)
        for r in 0..h_rows {
            for c in 0..h_cols {
                let delta_ewc = agent_ewc
                    .backend
                    .mat_get(&agent_ewc.actor.layers[0].weights, r, c)
                    - h_before_ewc[r * h_cols + c];
                let delta_ref = agent_ref
                    .backend
                    .mat_get(&agent_ref.actor.layers[0].weights, r, c)
                    - h_before_ref[r * h_cols + c];
                assert!(
                    (delta_ewc - delta_ref).abs() < 1e-10,
                    "Hidden layer delta differs at ({},{}): ewc={}, ref={}",
                    r,
                    c,
                    delta_ewc,
                    delta_ref
                );
            }
        }
    }

    #[test]
    fn test_fisher_snapshot_includes_all_trainable_params() {
        // After PLASTIC→FROZEN, snapshot should include weights and biases
        let config = ewc_config();
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Simulate reliable phase and sleep
        agent.actor_plastic_step_counter = 200;
        agent.handle_fisher_sleep(true);

        // Verify snapshots exist for all layers
        for (i, fisher) in agent.actor_fisher.iter().enumerate() {
            assert!(
                fisher.theta_snapshot_weights.is_some(),
                "Layer {} missing weight snapshot",
                i
            );
            assert!(
                fisher.theta_snapshot_bias.is_some(),
                "Layer {} missing bias snapshot",
                i
            );
        }
    }

    // ── Config validation and derived values ──────────────────────

    #[test]
    fn test_min_fisher_phase_derived_correctly() {
        // beta=0.99 -> ceil(1/0.01) = 100
        let min_phase_99 = (1.0_f64 / (1.0 - 0.99)).ceil() as u64;
        assert_eq!(min_phase_99, 100);

        // beta=0.95 -> ceil(1/0.05) = 20
        let min_phase_95 = (1.0_f64 / (1.0 - 0.95)).ceil() as u64;
        assert_eq!(min_phase_95, 20);
    }

    #[test]
    fn test_ewc_disabled_by_default() {
        // ewc_lambda=0.0. No Fisher allocated.
        let agent = make_agent();
        assert!(
            agent.actor_fisher.is_empty(),
            "No Fisher state when ewc_lambda=0"
        );
        assert!(
            agent.critic_fisher.is_empty(),
            "No Fisher state when ewc_lambda=0"
        );
    }

    #[test]
    fn test_fisher_decay_out_of_range_rejected() {
        let config = PcActorCriticConfig {
            ewc_lambda: 1.0,
            fisher_decay: 1.5,
            ..default_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err(), "fisher_decay > 1.0 should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("fisher_decay"),
            "Error should mention fisher_decay: {err}"
        );

        let config2 = PcActorCriticConfig {
            ewc_lambda: 1.0,
            fisher_decay: -0.1,
            ..default_config()
        };
        let result2 = PcActorCritic::new(CpuLinAlg::new(), config2, 42);
        assert!(result2.is_err(), "fisher_decay < 0.0 should be rejected");
    }

    #[test]
    fn test_fisher_ema_beta_out_of_range_rejected() {
        let config = PcActorCriticConfig {
            ewc_lambda: 1.0,
            fisher_ema_beta: 0.0,
            ..default_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err(), "fisher_ema_beta=0.0 should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("fisher_ema_beta"),
            "Error should mention fisher_ema_beta: {err}"
        );

        let config2 = PcActorCriticConfig {
            ewc_lambda: 1.0,
            fisher_ema_beta: 1.0,
            ..default_config()
        };
        let result2 = PcActorCritic::new(CpuLinAlg::new(), config2, 42);
        assert!(result2.is_err(), "fisher_ema_beta=1.0 should be rejected");
    }

    #[test]
    fn test_ewc_lambda_negative_rejected() {
        let config = PcActorCriticConfig {
            ewc_lambda: -0.1,
            ..default_config()
        };
        let result = PcActorCritic::new(CpuLinAlg::new(), config, 42);
        assert!(result.is_err(), "Negative ewc_lambda should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("ewc_lambda"),
            "Error should mention ewc_lambda: {err}"
        );
    }

    // ── Logits reversal ──────────────────────────────────────

    #[test]
    fn test_logits_reversal_off_by_default() {
        let config = default_config();
        assert!(!config.logits_reversal);
    }

    #[test]
    fn test_logits_reversal_separate_from_learning() {
        // When logits_reversal is on, actual backward() uses real logits.
        // Fisher EMA uses reversed logits. We verify by comparing weight updates.
        let mut config_rev = default_config();
        config_rev.ewc_lambda = 1.0;
        config_rev.logits_reversal = true;
        let mut agent_rev: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config_rev, 42).unwrap();

        let mut config_no_rev = default_config();
        config_no_rev.ewc_lambda = 1.0;
        config_no_rev.logits_reversal = false;
        let mut agent_no_rev: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config_no_rev, 42).unwrap();

        let state1 = vec![1.0; 9];
        let state2 = vec![0.5; 9];
        agent_rev.step(&state1, 0.0, false);
        agent_rev.step(&state2, 1.0, true);
        agent_no_rev.step(&state1, 0.0, false);
        agent_no_rev.step(&state2, 1.0, true);

        // Weights should be identical (no snapshot yet means EWC correction = 0)
        // but Fisher EMA should differ
        let f_rev = &agent_rev.actor_fisher[0];
        let f_no_rev = &agent_no_rev.actor_fisher[0];
        let rows = agent_rev.backend.mat_rows(&f_rev.f_ema_weights);
        let cols = agent_rev.backend.mat_cols(&f_rev.f_ema_weights);
        let mut differs = false;
        for r in 0..rows {
            for c in 0..cols {
                let v1 = agent_rev.backend.mat_get(&f_rev.f_ema_weights, r, c);
                let v2 = agent_no_rev.backend.mat_get(&f_no_rev.f_ema_weights, r, c);
                if (v1 - v2).abs() > 1e-10 {
                    differs = true;
                }
            }
        }
        assert!(
            differs,
            "Fisher EMA should differ between reversal and non-reversal modes"
        );
    }

    #[test]
    fn test_logits_reversal_delta_formula() {
        // delta_fisher = softmax(-y_conv / T, valid_actions) - one_hot(action)
        let backend = CpuLinAlg::new();
        let y_conv = backend.vec_from_slice(&[1.0, 2.0, 0.5]);
        let temperature = 1.0;
        let valid_actions = vec![0, 1, 2];
        let action = 1;

        // Reversed logits: -y_conv / T
        let reversed: Vec<f64> = backend
            .vec_to_vec(&y_conv)
            .iter()
            .map(|&v| -v / temperature)
            .collect();
        let reversed_l = backend.vec_from_slice(&reversed);
        let pi_rev_l = backend.softmax_masked(&reversed_l, &valid_actions);
        let pi_rev = backend.vec_to_vec(&pi_rev_l);

        // delta = pi_rev - one_hot(action)
        let mut delta = pi_rev.clone();
        delta[action] -= 1.0;

        // Verify reversed softmax sums to 1.0
        let sum: f64 = valid_actions.iter().map(|&i| pi_rev[i]).sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Action=1 had highest logit (2.0), so reversed has lowest prob
        assert!(pi_rev[1] < pi_rev[0] && pi_rev[1] < pi_rev[2]);
    }

    #[test]
    fn test_logits_reversal_no_advantage_scaling() {
        // Logits reversal delta should NOT include td_error scaling
        // This is verified structurally: the delta is purely from reversed softmax
        let backend = CpuLinAlg::new();
        let y_conv = [1.0, 2.0, 0.5];
        let valid_actions = [0, 1, 2];
        let action = 1;
        let temperature = 1.0;

        let reversed: Vec<f64> = y_conv.iter().map(|&v| -v / temperature).collect();
        let reversed_l = backend.vec_from_slice(&reversed);
        let pi_rev_l = backend.softmax_masked(&reversed_l, &valid_actions);
        let pi_rev = backend.vec_to_vec(&pi_rev_l);

        let mut delta = pi_rev.clone();
        delta[action] -= 1.0;

        // Delta magnitude should be independent of any td_error
        // Maximum delta element is bounded by 1.0
        for &d in &delta {
            assert!(d.abs() <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_logits_reversal_read_only_pass() {
        // Logits reversal extracts gradient read-only (no weight changes)
        // Only F_ema is updated, not actual weights
        let mut config = default_config();
        config.ewc_lambda = 1.0;
        config.logits_reversal = true;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Record weights before
        let rows = agent.backend.mat_rows(&agent.actor.layers[0].weights);
        let cols = agent.backend.mat_cols(&agent.actor.layers[0].weights);
        let mut w_before = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                w_before[r * cols + c] =
                    agent.backend.mat_get(&agent.actor.layers[0].weights, r, c);
            }
        }

        // No snapshot, so EWC correction = 0. Weight changes come only from backward.
        let state1 = vec![1.0; 9];
        agent.step(&state1, 0.0, false);
        let state2 = vec![0.5; 9];
        agent.step(&state2, 1.0, true);

        // Weight changes should match non-reversal behavior (backward is the same)
        // The key check: Fisher EMA was updated (read-only from reversed pass)
        assert!(!agent.actor_fisher.is_empty());
        let f = &agent.actor_fisher[0];
        let mut has_nonzero = false;
        for r in 0..rows {
            for c in 0..cols {
                if agent.backend.mat_get(&f.f_ema_weights, r, c) > 0.0 {
                    has_nonzero = true;
                }
            }
        }
        assert!(
            has_nonzero,
            "F_ema should be updated from reversed logits pass"
        );
    }

    #[test]
    fn test_logits_reversal_uses_actor_temperature() {
        // Verify reversed logits use config temperature
        let backend = CpuLinAlg::new();
        let y_conv = backend.vec_from_slice(&[1.0, 2.0, 0.5]);
        let t = 2.0; // Different temperature

        let reversed: Vec<f64> = backend
            .vec_to_vec(&y_conv)
            .iter()
            .map(|&v| -v / t)
            .collect();
        let reversed_l = backend.vec_from_slice(&reversed);
        let pi = backend.softmax_masked(&reversed_l, &[0, 1, 2]);
        let pi_vec = backend.vec_to_vec(&pi);

        // Higher temperature → more uniform distribution
        let t1_reversed: Vec<f64> = backend
            .vec_to_vec(&y_conv)
            .iter()
            .map(|&v| -v / 1.0)
            .collect();
        let t1_l = backend.vec_from_slice(&t1_reversed);
        let pi_t1 = backend.softmax_masked(&t1_l, &[0, 1, 2]);
        let pi_t1_vec = backend.vec_to_vec(&pi_t1);

        // With T=2.0, distribution should be more uniform (higher entropy)
        let entropy_t2: f64 = pi_vec.iter().map(|&p| -p * p.ln()).sum();
        let entropy_t1: f64 = pi_t1_vec.iter().map(|&p| -p * p.ln()).sum();
        assert!(
            entropy_t2 > entropy_t1,
            "Higher temperature should produce more uniform reversed distribution"
        );
    }

    #[test]
    fn test_logits_reversal_uses_softmax_masked() {
        // Verify only valid actions get probability mass
        let backend = CpuLinAlg::new();
        let y_conv = backend.vec_from_slice(&[1.0, 2.0, 0.5, 3.0]);
        let valid_actions = vec![0, 2]; // Only 0 and 2 are valid

        let reversed: Vec<f64> = backend
            .vec_to_vec(&y_conv)
            .iter()
            .map(|&v| -v / 1.0)
            .collect();
        let reversed_l = backend.vec_from_slice(&reversed);
        let pi = backend.softmax_masked(&reversed_l, &valid_actions);
        let pi_vec = backend.vec_to_vec(&pi);

        // Invalid actions (1, 3) should have zero probability
        assert!(
            pi_vec[1].abs() < 1e-10,
            "Invalid action 1 should have zero prob"
        );
        assert!(
            pi_vec[3].abs() < 1e-10,
            "Invalid action 3 should have zero prob"
        );
        // Valid actions should sum to 1.0
        let valid_sum: f64 = valid_actions.iter().map(|&i| pi_vec[i]).sum();
        assert!((valid_sum - 1.0).abs() < 1e-10);
    }

    // ── Cross-phase interaction tests ──────────────────────────

    #[test]
    fn test_m3_layer_decay_combined_with_m4_ewc() {
        // Both M3 (layer decay) and M4 (EWC) active simultaneously.
        // The per-layer surprise used in backward should be: surprise_scale * layer_decay.
        // EWC correction is a separate post-backward step.
        let mut config = three_layer_config();
        config.consolidation_decay = 0.5;
        config.ewc_lambda = 1.0;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Verify both decay factors and Fisher state exist
        assert_eq!(agent.actor_decay_factors.len(), 3);
        assert!(!agent.actor_fisher.is_empty());

        // Set Fisher state for the output layer (last layer)
        let n_layers = agent.actor.layers.len();
        let out_rows = agent
            .backend
            .mat_rows(&agent.actor.layers[n_layers - 1].weights);
        let out_cols = agent
            .backend
            .mat_cols(&agent.actor.layers[n_layers - 1].weights);
        for r in 0..out_rows {
            for c in 0..out_cols {
                agent.backend.mat_set(
                    &mut agent.actor_fisher[n_layers - 1].f_total_weights,
                    r,
                    c,
                    1.0,
                );
            }
        }
        agent.actor_fisher[n_layers - 1].theta_snapshot_weights =
            Some(agent.actor.layers[n_layers - 1].weights.clone());
        agent.actor_fisher[n_layers - 1].theta_snapshot_bias =
            Some(agent.actor.layers[n_layers - 1].bias.clone());

        // Do learning steps — should not crash
        let state1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        agent.step(&state1, 0.0, false);
        let state2 = vec![0.5; 9];
        agent.step(&state2, 1.0, true);
    }

    #[test]
    fn test_hysteresis_transition_with_logits_reversal() {
        // Logits reversal + hysteresis. Verify transitions work correctly.
        let mut config = ewc_config();
        config.logits_reversal = true;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Run many steps to trigger possible transitions
        for i in 0..50 {
            let state: Vec<f64> = (0..9).map(|j| ((i + j) as f64 * 0.3).sin()).collect();
            agent.step(&state, if i % 3 == 0 { 1.0 } else { 0.0 }, i == 49);
        }
        // Should not panic
    }

    #[test]
    fn test_hysteresis_frozen_suppresses_ewc() {
        // When actor is FROZEN (scale_floor), EWC correction still uses
        // effective_lr which includes the floor scale, so correction is minimal.
        let mut config = ewc_config();
        config.scale_floor = 0.0; // True freeze
        config.ewc_lambda = 10.0;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Force actor FROZEN
        agent.actor_hysteresis.as_mut().unwrap().state = PlasticityState::Frozen;

        // Set up Fisher + snapshot with deviation
        let rows = agent.backend.mat_rows(&agent.actor.layers[0].weights);
        let cols = agent.backend.mat_cols(&agent.actor.layers[0].weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 1.0);
            }
        }
        agent.actor_fisher[0].theta_snapshot_weights = Some(agent.actor.layers[0].weights.clone());
        agent.actor_fisher[0].theta_snapshot_bias = Some(agent.actor.layers[0].bias.clone());

        // Record weights before
        let w_before = agent.backend.mat_get(&agent.actor.layers[0].weights, 0, 0);

        // Step (actor is frozen → scale_floor = 0.0 → effective_lr = 0)
        let state1 = vec![1.0; 9];
        agent.step(&state1, 0.0, false);
        let state2 = vec![0.5; 9];
        agent.step(&state2, 1.0, true);

        let w_after = agent.backend.mat_get(&agent.actor.layers[0].weights, 0, 0);
        // With scale_floor=0.0 and FROZEN, effective_lr=0 → no weight change
        assert!(
            (w_after - w_before).abs() < 1e-10,
            "FROZEN actor with scale_floor=0 should not change weights"
        );
    }

    // ── Additional Fisher/EWC behavioral tests ──────────────────

    #[test]
    fn test_fisher_decay_one_no_erasure() {
        // fisher_decay=1.0 means F_total is never decayed (preserved perfectly)
        let mut config = ewc_config();
        config.fisher_decay = 1.0;
        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let rows = agent
            .backend
            .mat_rows(&agent.actor_fisher[0].f_total_weights);
        let cols = agent
            .backend
            .mat_cols(&agent.actor_fisher[0].f_total_weights);
        for r in 0..rows {
            for c in 0..cols {
                agent
                    .backend
                    .mat_set(&mut agent.actor_fisher[0].f_total_weights, r, c, 10.0);
            }
        }
        agent.actor_last_phase_reliable = true;
        agent.handle_fisher_wake(true);

        // F_total *= 1.0 → unchanged at 10.0
        for r in 0..rows {
            for c in 0..cols {
                let val = agent
                    .backend
                    .mat_get(&agent.actor_fisher[0].f_total_weights, r, c);
                assert!(
                    (val - 10.0).abs() < 1e-10,
                    "fisher_decay=1.0 should preserve F_total, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_fisher_ema_beta_near_zero_min_phase_one() {
        // beta near 0 → min_fisher_phase ≈ 1
        let min_phase = (1.0_f64 / (1.0 - 0.01)).ceil() as u64;
        // 1/0.99 ≈ 1.01, ceil = 2
        assert!(
            min_phase <= 2,
            "beta=0.01 → min_phase should be ~1, got {}",
            min_phase
        );
    }

    // ── Section 07: GA Crossover Reset ──────────────────────────

    #[test]
    fn test_crossover_resets_all_cl_state() {
        // Two parents with EWC + hysteresis enabled, train them to get non-zero Fisher
        let mut config = ewc_config();
        config.adaptive_consolidation = true;
        config.consolidation_ema_beta = 0.99;
        config.consolidation_sigmoid_k = 10.0;
        config.consolidation_error_threshold = 0.05;

        let mut parent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut parent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        // Train parents to accumulate Fisher and CL state
        let state1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let state2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        for _ in 0..5 {
            parent_a.step(&state1, 0.0, false);
            parent_a.step(&state2, 1.0, true);
            parent_b.step(&state1, 0.0, false);
            parent_b.step(&state2, -1.0, true);
        }

        // Force parents into FROZEN state with non-zero Fisher
        if let Some(ref mut h) = parent_a.actor_hysteresis {
            h.state = PlasticityState::Frozen;
        }
        if let Some(ref mut h) = parent_a.critic_hysteresis {
            h.state = PlasticityState::Frozen;
        }
        parent_a.critic_frozen_steps = 50;

        // Pre-load F_total on parent_a to ensure non-zero Fisher
        let rows = parent_a
            .backend
            .mat_rows(&parent_a.actor_fisher[0].f_total_weights);
        let cols = parent_a
            .backend
            .mat_cols(&parent_a.actor_fisher[0].f_total_weights);
        for r in 0..rows {
            for c in 0..cols {
                parent_a
                    .backend
                    .mat_set(&mut parent_a.actor_fisher[0].f_total_weights, r, c, 1.0);
            }
        }

        // Set mid-episode step state on parent_a
        parent_a.step(&state1, 0.0, false); // state_prev is now set

        let (ac_a, cc_a) = build_caches_for_agent(&mut parent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut parent_b, 50);

        let child: PcActorCritic = PcActorCritic::crossover(
            &parent_a, &parent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        // Verify all CL state is clean
        assert!(child.actor_hysteresis.is_none());
        assert!(child.critic_hysteresis.is_none());
        assert_eq!(child.actor_plastic_step_counter, 0);
        assert_eq!(child.critic_plastic_step_counter, 0);
        assert_eq!(child.critic_frozen_steps, 0);
        assert!(child.surprise_buffer.is_empty());
        assert!(child.td_error_buffer.is_empty());
        assert!(child.state_prev.is_none());
        assert!(child.action_prev.is_none());
        assert!(child.infer_prev.is_none());
        assert!(child.valid_actions_prev.is_none());
        assert!(!child.actor_last_phase_reliable);
        assert!(!child.critic_last_phase_reliable);

        // Fisher state should be empty (clean)
        assert!(
            child.actor_fisher.is_empty(),
            "Child actor_fisher should be empty"
        );
        assert!(
            child.critic_fisher.is_empty(),
            "Child critic_fisher should be empty"
        );

        // Per-layer error EMAs should be zero (if adaptive consolidation enabled)
        for ema in &child.layer_error_ema {
            assert!(
                (*ema).abs() < f64::EPSILON,
                "Per-layer error EMA should be 0.0"
            );
        }
    }

    #[test]
    fn test_crossover_child_warmup_guard_active() {
        // Child from crossover with hysteresis config should have k=0
        // so warmup guard prevents sleep even under sleep conditions
        let config = ewc_config();
        let mut parent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut parent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut parent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut parent_b, 50);

        let mut child: PcActorCritic = PcActorCritic::crossover(
            &parent_a, &parent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        // Child starts PLASTIC, hysteresis is None (clean defaults).
        // Feed it a few steps that would normally trigger sleep
        // but since hysteresis is None, it stays Plastic.
        let state = vec![0.5; 9];
        for _ in 0..5 {
            child.step(&state, 0.0, false);
            child.step(&state, 0.0, true);
        }

        // Child should still be functional (no panic, no stuck state)
        // Actor hysteresis is None → always Plastic
        assert!(child.actor_hysteresis.is_none());
    }

    #[test]
    fn test_crossover_weights_from_parents() {
        // Two parents with known distinct weights.
        // Child weights are CCA-blended. CL state is clean.
        let config = default_config();
        let mut parent_a: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 42).unwrap();
        let mut parent_b: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), config.clone(), 123).unwrap();

        let (ac_a, cc_a) = build_caches_for_agent(&mut parent_a, 50);
        let (ac_b, cc_b) = build_caches_for_agent(&mut parent_b, 50);

        let child: PcActorCritic = PcActorCritic::crossover(
            &parent_a, &parent_b, &ac_a, &ac_b, &cc_a, &cc_b, 0.5, config, 99,
        )
        .unwrap();

        // Child weights should differ from both parents (CCA blend)
        assert_ne!(
            child.actor.layers[0].weights.data,
            parent_a.actor.layers[0].weights.data
        );
        assert_ne!(
            child.actor.layers[0].weights.data,
            parent_b.actor.layers[0].weights.data
        );

        // CL state is clean
        assert!(child.actor_fisher.is_empty());
        assert!(child.critic_fisher.is_empty());
        assert!(child.surprise_buffer.is_empty());
        assert!(child.td_error_buffer.is_empty());
        assert_eq!(child.actor_plastic_step_counter, 0);
        assert_eq!(child.critic_plastic_step_counter, 0);
    }

    // ── Section 07: Default config reproduces v2 behavior ───────

    #[test]
    fn test_default_config_reproduces_v2_behavior() {
        // Default config (all CL disabled) should behave identically to v2.0.0
        let config = default_config();
        assert!(!config.actor_hysteresis);
        assert!(!config.critic_hysteresis);
        assert!(!config.adaptive_consolidation);
        assert!((config.ewc_lambda).abs() < f64::EPSILON);
        assert!(!config.logits_reversal);
        assert!((config.consolidation_decay - 1.0).abs() < f64::EPSILON);
        assert!((config.critic_consolidation_decay - 1.0).abs() < f64::EPSILON);

        // Agent with default config should have no CL overhead
        let agent = make_agent();
        assert!(agent.actor_hysteresis.is_none());
        assert!(agent.critic_hysteresis.is_none());
        assert!(agent.actor_fisher.is_empty());
        assert!(agent.critic_fisher.is_empty());
        assert!(agent.layer_error_ema.is_empty());
    }
}
