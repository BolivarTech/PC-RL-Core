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

/// Default for actor-wakes-critic coupling (disabled).
fn default_actor_wakes_critic() -> bool {
    false
}

/// Default threshold for actor-wakes-critic coupling.
fn default_actor_wakes_critic_threshold() -> u64 {
    1000
}

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
///     actor_wakes_critic: false,
///     actor_wakes_critic_threshold: 1000,
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
    /// Default: false.
    #[serde(default = "default_actor_wakes_critic")]
    pub actor_wakes_critic: bool,
    /// Minimum critic frozen steps before actor→critic coupling triggers.
    /// Default: 1000.
    #[serde(default = "default_actor_wakes_critic_threshold")]
    pub actor_wakes_critic_threshold: u64,
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

        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let actor = PcActor::<L>::new(backend.clone(), config.actor.clone(), &mut rng)?;
        let critic = MlpCritic::<L>::new(backend.clone(), config.critic.clone(), &mut rng)?;
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
            self.actor
                .update_weights(&delta, &stored_infer, &input_vec, s_scale);

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

        // Update critic
        let loss = self.critic.update(&critic_input, target);

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
        self.actor.update_weights(&delta, infer, input, s_scale);

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

    /// Updates hysteresis state machines after learning.
    ///
    /// Handles EWMA updates, state transitions, counter management,
    /// and actor→critic coupling.
    pub(crate) fn process_hysteresis(&mut self, actor_signal: f64, critic_signal: f64) {
        let mut actor_woke = false;

        // Update actor hysteresis
        if let Some(ref mut hyst) = self.actor_hysteresis {
            // Increment counter for pre-transition state
            if hyst.state == PlasticityState::Plastic {
                self.actor_plastic_step_counter += 1;
            }
            if let Some(new_state) = hyst.update(actor_signal) {
                if new_state == PlasticityState::Plastic {
                    self.actor_plastic_step_counter = 0;
                    actor_woke = true;
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
                    self.critic_plastic_step_counter = 0;
                    self.critic_frozen_steps = 0;
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
                }
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
}
