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

        let s_scale = self.surprise_scale(infer.surprise_score);
        self.actor.update_weights(&delta, infer, input, s_scale);

        if self.config.adaptive_surprise {
            self.push_surprise(infer.surprise_score);
        }

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
            0.1
        } else if surprise >= high {
            2.0
        } else {
            // Linear interpolation
            let t = (surprise - low) / (high - low);
            0.1 + t * (2.0 - 0.1)
        }
    }

    /// Pushes a surprise score into the adaptive buffer (circular).
    fn push_surprise(&mut self, surprise: f64) {
        if self.surprise_buffer.len() >= self.config.surprise_buffer_size {
            self.surprise_buffer.pop_front();
        }
        self.surprise_buffer.push_back(surprise);
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
    fn test_learn_updates_actor_weights() {
        let mut agent: PcActorCritic = make_agent();
        let trajectory = make_trajectory(&mut agent);
        let w_before = agent.actor.layers[0].weights.data.clone();
        let _ = agent.learn(&trajectory);
        assert_ne!(agent.actor.layers[0].weights.data, w_before);
    }

    #[test]
    fn test_learn_updates_critic_weights() {
        let mut agent: PcActorCritic = make_agent();
        let trajectory = make_trajectory(&mut agent);
        let w_before = agent.critic.layers[0].weights.data.clone();
        let _ = agent.learn(&trajectory);
        assert_ne!(agent.critic.layers[0].weights.data, w_before);
    }

    #[test]
    fn test_learn_returns_finite_nonneg_loss() {
        let mut agent: PcActorCritic = make_agent();
        let trajectory = make_trajectory(&mut agent);
        let loss = agent.learn(&trajectory);
        assert!(loss.is_finite(), "Loss {loss} is not finite");
        assert!(loss >= 0.0, "Loss {loss} is negative");
    }

    #[test]
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
}
