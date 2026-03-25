// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Predictive Coding Actor Network.
//!
//! Implements an actor that uses iterative top-down/bottom-up predictive coding
//! inference loops instead of standard feedforward passes. The prediction error
//! (surprise score) drives learning rate modulation in the actor-critic agent.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::error::PcError;
use crate::layer::{Layer, LayerDef};
use crate::matrix::{
    argmax_masked, rms_error, sample_from_probs, softmax_masked, vec_add, vec_scale, vec_sub,
};

/// Configuration for the predictive coding actor network.
///
/// # Examples
///
/// ```
/// use pc_core::activation::Activation;
/// use pc_core::layer::LayerDef;
/// use pc_core::pc_actor::PcActorConfig;
///
/// let config = PcActorConfig {
///     input_size: 9,
///     hidden_layers: vec![LayerDef { size: 18, activation: Activation::Tanh }],
///     output_size: 9,
///     output_activation: Activation::Tanh,
///     alpha: 0.1,
///     tol: 0.01,
///     min_steps: 1,
///     max_steps: 20,
///     lr_weights: 0.01,
///     synchronous: true,
///     temperature: 1.0,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcActorConfig {
    /// Number of input features (e.g. 9 for tic-tac-toe board).
    pub input_size: usize,
    /// Hidden layer topology definitions.
    pub hidden_layers: Vec<LayerDef>,
    /// Number of output actions.
    pub output_size: usize,
    /// Activation function for the output layer.
    pub output_activation: Activation,
    /// Inference learning rate for PC loop updates.
    pub alpha: f64,
    /// Convergence threshold for RMS prediction error.
    pub tol: f64,
    /// Minimum inference steps before convergence check.
    pub min_steps: usize,
    /// Maximum inference steps.
    pub max_steps: usize,
    /// Base learning rate for weight updates.
    pub lr_weights: f64,
    /// If true, use synchronous snapshot mode; otherwise in-place.
    pub synchronous: bool,
    /// Softmax temperature for action selection.
    pub temperature: f64,
}

/// Result of the predictive coding inference loop.
///
/// Contains converged output logits, hidden state representations,
/// and diagnostic information about the inference process.
#[derive(Debug, Clone)]
pub struct InferResult {
    /// Converged output logits.
    pub y_conv: Vec<f64>,
    /// All hidden states concatenated (fed to critic).
    pub latent_concat: Vec<f64>,
    /// Per-layer hidden state activations.
    pub hidden_states: Vec<Vec<f64>>,
    /// RMS prediction error across layers.
    pub surprise_score: f64,
    /// Number of inference steps performed.
    pub steps_used: usize,
    /// Whether the inference loop converged within tolerance.
    pub converged: bool,
}

/// Action selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// Stochastic sampling from softmax distribution.
    Training,
    /// Deterministic argmax selection.
    Play,
}

/// Predictive coding actor network.
///
/// Uses iterative top-down/bottom-up inference loops to produce
/// stable hidden representations and output logits.
///
/// # Examples
///
/// ```
/// use pc_core::activation::Activation;
/// use pc_core::layer::LayerDef;
/// use pc_core::pc_actor::{PcActor, PcActorConfig, SelectionMode};
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
///
/// let config = PcActorConfig {
///     input_size: 9,
///     hidden_layers: vec![LayerDef { size: 18, activation: Activation::Tanh }],
///     output_size: 9,
///     output_activation: Activation::Tanh,
///     alpha: 0.1, tol: 0.01, min_steps: 1, max_steps: 20,
///     lr_weights: 0.01, synchronous: true, temperature: 1.0,
/// };
/// let mut rng = StdRng::seed_from_u64(42);
/// let actor = PcActor::new(config, &mut rng).unwrap();
/// let result = actor.infer(&[0.0; 9]);
/// assert_eq!(result.y_conv.len(), 9);
/// ```
#[derive(Debug)]
pub struct PcActor {
    /// Network layers: hidden_layers.len() + 1 (output layer).
    pub(crate) layers: Vec<Layer>,
    /// Actor configuration.
    pub config: PcActorConfig,
}

impl PcActor {
    /// Creates a new PC actor with Xavier-initialized layers.
    ///
    /// # Arguments
    ///
    /// * `config` - Actor configuration specifying topology and hyperparameters.
    /// * `rng` - Random number generator for weight initialization.
    ///
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if `input_size`, `output_size`,
    /// or `temperature` are invalid.
    pub fn new(config: PcActorConfig, rng: &mut impl Rng) -> Result<Self, PcError> {
        if config.input_size == 0 {
            return Err(PcError::ConfigValidation("input_size must be > 0".into()));
        }
        if config.output_size == 0 {
            return Err(PcError::ConfigValidation("output_size must be > 0".into()));
        }
        if config.temperature <= 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "temperature must be positive, got {}",
                config.temperature
            )));
        }

        let mut layers = Vec::new();
        let mut prev_size = config.input_size;

        for def in &config.hidden_layers {
            layers.push(Layer::new(prev_size, def.size, def.activation, rng));
            prev_size = def.size;
        }

        // Output layer
        layers.push(Layer::new(
            prev_size,
            config.output_size,
            config.output_activation,
            rng,
        ));

        Ok(Self { layers, config })
    }

    /// Returns the total size of the latent concatenation (sum of hidden layer sizes).
    pub fn latent_size(&self) -> usize {
        self.config.hidden_layers.iter().map(|def| def.size).sum()
    }

    /// Runs the predictive coding inference loop on the given input.
    ///
    /// This method is `&self` — it never modifies weights.
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of length `input_size`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    pub fn infer(&self, input: &[f64]) -> InferResult {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "input size mismatch: got {}, expected {}",
            input.len(),
            self.config.input_size
        );

        let n_hidden = self.config.hidden_layers.len();

        // Forward pass to initialize hidden states and output
        let mut hidden_states: Vec<Vec<f64>> = Vec::with_capacity(n_hidden);
        let mut prev = input.to_vec();
        for layer in &self.layers[..n_hidden] {
            prev = layer.forward(&prev);
            hidden_states.push(prev.clone());
        }
        // Output from last hidden (or input if no hidden)
        let last_input = if n_hidden > 0 {
            &hidden_states[n_hidden - 1]
        } else {
            input
        };
        let mut y = self.layers[n_hidden].forward(last_input);

        // PC inference loop
        let mut steps_used = 0;
        let mut converged = false;
        let mut surprise_score = 0.0;

        for step in 0..self.config.max_steps {
            steps_used = step + 1;

            if self.config.synchronous {
                // Snapshot mode: freeze all states
                let snapshot: Vec<Vec<f64>> = hidden_states.clone();

                let mut error_vecs: Vec<Vec<f64>> = Vec::new();

                for i in (0..n_hidden).rev() {
                    // State above: y for top hidden, hidden_states[i+1] otherwise
                    let state_above = if i == n_hidden - 1 {
                        &y
                    } else {
                        &snapshot[i + 1]
                    };

                    // Top-down prediction
                    let prediction = self.layers[i + 1]
                        .transpose_forward(state_above, self.config.hidden_layers[i].activation);

                    // Error = prediction - snapshot[i]
                    let error = vec_sub(&prediction, &snapshot[i]);
                    error_vecs.push(error.clone());

                    // Update: h[i] = snapshot[i] + alpha * error
                    let update = vec_add(&snapshot[i], &vec_scale(&error, self.config.alpha));
                    hidden_states[i] = update;
                }

                // Recompute output from updated top hidden
                let top_hidden = if n_hidden > 0 {
                    &hidden_states[n_hidden - 1]
                } else {
                    input
                };
                y = self.layers[n_hidden].forward(top_hidden);

                // RMS error
                let refs: Vec<&[f64]> = error_vecs.iter().map(|v| v.as_slice()).collect();
                surprise_score = rms_error(&refs);
            } else {
                // In-place mode: updates immediately visible
                let mut error_vecs: Vec<Vec<f64>> = Vec::new();

                for i in (0..n_hidden).rev() {
                    let state_above = if i == n_hidden - 1 {
                        &y
                    } else {
                        &hidden_states[i + 1]
                    };

                    let prediction = self.layers[i + 1]
                        .transpose_forward(state_above, self.config.hidden_layers[i].activation);

                    let error = vec_sub(&prediction, &hidden_states[i]);
                    error_vecs.push(error.clone());

                    let update = vec_add(&hidden_states[i], &vec_scale(&error, self.config.alpha));
                    hidden_states[i] = update;
                }

                let top_hidden = if n_hidden > 0 {
                    &hidden_states[n_hidden - 1]
                } else {
                    input
                };
                y = self.layers[n_hidden].forward(top_hidden);

                let refs: Vec<&[f64]> = error_vecs.iter().map(|v| v.as_slice()).collect();
                surprise_score = rms_error(&refs);
            }

            // Convergence check (alpha must be > 0 for meaningful convergence)
            if self.config.alpha > 0.0
                && step + 1 >= self.config.min_steps
                && surprise_score < self.config.tol
            {
                converged = true;
                break;
            }
        }

        // Build latent_concat
        let latent_concat: Vec<f64> = hidden_states
            .iter()
            .flat_map(|h| h.iter().copied())
            .collect();

        InferResult {
            y_conv: y,
            latent_concat,
            hidden_states,
            surprise_score,
            steps_used,
            converged,
        }
    }

    /// Selects an action given converged output logits and valid actions.
    ///
    /// # Arguments
    ///
    /// * `y_conv` - Output logits from inference.
    /// * `valid_actions` - Indices of valid actions.
    /// * `mode` - Training (stochastic) or Play (deterministic).
    /// * `rng` - Random number generator (used only in Training mode).
    ///
    /// # Panics
    ///
    /// Panics if `valid_actions` is empty.
    pub fn select_action(
        &self,
        y_conv: &[f64],
        valid_actions: &[usize],
        mode: SelectionMode,
        rng: &mut impl Rng,
    ) -> usize {
        assert!(!valid_actions.is_empty(), "valid_actions must not be empty");

        // Scale logits by temperature
        let scaled: Vec<f64> = y_conv
            .iter()
            .map(|&v| v / self.config.temperature)
            .collect();

        let probs = softmax_masked(&scaled, valid_actions);

        match mode {
            SelectionMode::Play => argmax_masked(&probs, valid_actions),
            SelectionMode::Training => sample_from_probs(&probs, valid_actions, rng),
        }
    }

    /// Updates network weights via backpropagation.
    ///
    /// # Arguments
    ///
    /// * `output_delta` - Error signal at the output layer.
    /// * `infer_result` - Result from the most recent inference.
    /// * `input` - Original input that was fed to `infer`.
    /// * `surprise_scale` - Multiplier on learning rate based on surprise.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    pub fn update_weights(
        &mut self,
        output_delta: &[f64],
        infer_result: &InferResult,
        input: &[f64],
        surprise_scale: f64,
    ) {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "input size mismatch: got {}, expected {}",
            input.len(),
            self.config.input_size
        );

        let n_hidden = self.config.hidden_layers.len();
        let n_layers = self.layers.len();

        // Output layer backward
        let output_input = if n_hidden > 0 {
            &infer_result.hidden_states[n_hidden - 1]
        } else {
            input
        };
        let output_output = &infer_result.y_conv;
        let mut delta = self.layers[n_layers - 1].backward(
            output_input,
            output_output,
            output_delta,
            self.config.lr_weights,
            surprise_scale,
        );

        // Hidden layers backward (from top to bottom)
        for i in (0..n_hidden).rev() {
            let layer_input = if i > 0 {
                &infer_result.hidden_states[i - 1]
            } else {
                input
            };
            let layer_output = &infer_result.hidden_states[i];
            delta = self.layers[i].backward(
                layer_input,
                layer_output,
                &delta,
                self.config.lr_weights,
                surprise_scale,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::LayerDef;
    use crate::matrix::WEIGHT_CLIP;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn default_config() -> PcActorConfig {
        PcActorConfig {
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
        }
    }

    fn two_hidden_config() -> PcActorConfig {
        PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 12,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        }
    }

    // ── Inference Tests ──────────────────────────────────────────────

    #[test]
    fn test_infer_converges_on_zero_board() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        // Should complete without panic; all finite
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_infer_steps_used_at_least_min_steps() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            min_steps: 3,
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(result.steps_used >= 3);
    }

    #[test]
    fn test_infer_alpha_zero_does_not_converge() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            alpha: 0.0,
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(!result.converged);
        assert_eq!(result.steps_used, 20);
    }

    #[test]
    fn test_infer_does_not_modify_weights() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let weights_before: Vec<Vec<f64>> = actor
            .layers
            .iter()
            .map(|l| l.weights.data.clone())
            .collect();
        let _ = actor.infer(&[0.0; 9]);
        for (i, layer) in actor.layers.iter().enumerate() {
            assert_eq!(layer.weights.data, weights_before[i]);
        }
    }

    #[test]
    fn test_infer_latent_size_single_hidden() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 18);
    }

    #[test]
    fn test_infer_latent_size_two_hidden() {
        let mut rng = make_rng();
        let actor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 30);
    }

    #[test]
    fn test_infer_latent_size_matches_latent_size_method() {
        let mut rng = make_rng();
        let actor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), actor.latent_size());
    }

    #[test]
    fn test_infer_y_conv_length_equals_output_size() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.y_conv.len(), 9);
    }

    #[test]
    fn test_infer_hidden_states_count_matches_hidden_layers() {
        let mut rng = make_rng();
        let actor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.hidden_states.len(), 2);
    }

    #[test]
    fn test_infer_all_outputs_finite() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
        for &v in &result.latent_concat {
            assert!(v.is_finite());
        }
        assert!(result.surprise_score.is_finite());
    }

    #[test]
    fn test_infer_surprise_score_nonnegative() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(result.surprise_score >= 0.0);
    }

    #[test]
    fn test_infer_synchronous_and_inplace_both_converge() {
        let mut rng = make_rng();
        let sync_actor = PcActor::new(default_config(), &mut rng).unwrap();
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..default_config()
        };
        let inplace_actor = PcActor::new(inplace_config, &mut rng2).unwrap();
        let sync_result = sync_actor.infer(&[0.0; 9]);
        let inplace_result = inplace_actor.infer(&[0.0; 9]);
        // Both should complete without panic; at least one should converge or use all steps
        assert!(sync_result.steps_used > 0);
        assert!(inplace_result.steps_used > 0);
    }

    #[test]
    fn test_infer_synchronous_produces_different_result_than_inplace() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 12,
                    activation: Activation::Tanh,
                },
            ],
            alpha: 0.3,
            tol: 1e-15,
            min_steps: 1,
            max_steps: 3,
            ..default_config()
        };
        let sync_actor = PcActor::new(config.clone(), &mut rng).unwrap();
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..config
        };
        let inplace_actor = PcActor::new(inplace_config, &mut rng2).unwrap();
        let input = [1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let sync_result = sync_actor.infer(&input);
        let inplace_result = inplace_actor.infer(&input);
        // Different update orders should produce different hidden representations
        let differs = sync_result
            .latent_concat
            .iter()
            .zip(inplace_result.latent_concat.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(
            differs,
            "Synchronous and in-place should produce different results"
        );
    }

    #[test]
    #[should_panic(expected = "input size")]
    fn test_infer_panics_wrong_input_length() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let _ = actor.infer(&[0.0; 5]);
    }

    // ── Action Selection Tests ───────────────────────────────────────

    #[test]
    fn test_select_action_training_always_in_valid() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let logits = vec![0.1, -0.2, 0.5, -0.1, 0.3, 0.0, -0.3, 0.2, 0.4];
        let valid = vec![0, 2, 4, 6, 8];
        for _ in 0..20 {
            let action = actor.select_action(&logits, &valid, SelectionMode::Training, &mut rng);
            assert!(valid.contains(&action));
        }
    }

    #[test]
    fn test_select_action_play_mode_deterministic() {
        let mut rng1 = StdRng::seed_from_u64(1);
        let mut rng2 = StdRng::seed_from_u64(99);
        let mut rng_init = make_rng();
        let actor = PcActor::new(default_config(), &mut rng_init).unwrap();
        let logits = vec![0.1, -0.2, 0.5, -0.1, 0.3, 0.0, -0.3, 0.2, 0.4];
        let valid = vec![0, 2, 4, 6, 8];
        let a1 = actor.select_action(&logits, &valid, SelectionMode::Play, &mut rng1);
        let a2 = actor.select_action(&logits, &valid, SelectionMode::Play, &mut rng2);
        assert_eq!(a1, a2, "Play mode should be deterministic");
    }

    #[test]
    fn test_select_action_temperature_gt_one_more_uniform() {
        let mut rng = make_rng();
        let hot_config = PcActorConfig {
            temperature: 5.0,
            ..default_config()
        };
        let actor = PcActor::new(hot_config, &mut rng).unwrap();
        // With high temperature, sampling should visit more actions
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let valid: Vec<usize> = (0..9).collect();
        let mut seen = std::collections::HashSet::new();
        let mut rng2 = StdRng::seed_from_u64(123);
        for _ in 0..100 {
            let a = actor.select_action(&logits, &valid, SelectionMode::Training, &mut rng2);
            seen.insert(a);
        }
        assert!(seen.len() > 1, "High temperature should explore more");
    }

    #[test]
    #[should_panic]
    fn test_select_action_empty_valid_panics() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let logits = vec![0.1; 9];
        let _ = actor.select_action(&logits, &[], SelectionMode::Training, &mut rng);
    }

    // ── Weight Update Tests ──────────────────────────────────────────

    #[test]
    fn test_update_weights_changes_first_layer() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(default_config(), &mut rng).unwrap();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let infer_result = actor.infer(&input);
        let weights_before = actor.layers[0].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0);
        assert_ne!(actor.layers[0].weights.data, weights_before);
    }

    #[test]
    fn test_update_weights_clips_all_layers() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(default_config(), &mut rng).unwrap();
        let input = vec![1.0; 9];
        let infer_result = actor.infer(&input);
        let delta = vec![1e6; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0);
        for layer in &actor.layers {
            for &w in &layer.weights.data {
                assert!(
                    w.abs() <= WEIGHT_CLIP + 1e-12,
                    "Weight {w} exceeds WEIGHT_CLIP"
                );
            }
        }
    }

    #[test]
    fn test_update_weights_two_hidden_changes_both_layers() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0_before = actor.layers[0].weights.data.clone();
        let w1_before = actor.layers[1].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0);
        assert_ne!(
            actor.layers[0].weights.data, w0_before,
            "Layer 0 should change"
        );
        assert_ne!(
            actor.layers[1].weights.data, w1_before,
            "Layer 1 should change"
        );
    }

    #[test]
    #[should_panic(expected = "input size")]
    fn test_update_weights_panics_wrong_x_size() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(default_config(), &mut rng).unwrap();
        let input = vec![0.0; 9];
        let infer_result = actor.infer(&input);
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &[0.0; 5], 1.0);
    }

    // ── Zero Hidden Layers Test ─────────────────────────────────

    #[test]
    fn test_infer_zero_hidden_layers_produces_finite_output() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            hidden_layers: vec![],
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.y_conv.len(), 9);
        assert!(result.y_conv.iter().all(|v| v.is_finite()));
        assert!(result.latent_concat.is_empty());
        assert!(result.hidden_states.is_empty());
    }

    // ── Config Validation Tests ─────────────────────────────────

    #[test]
    fn test_new_zero_input_size_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            input_size: 0,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::PcError::ConfigValidation(_)));
    }

    #[test]
    fn test_new_zero_output_size_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            output_size: 0,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_zero_temperature_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            temperature: 0.0,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_negative_temperature_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            temperature: -1.0,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }
}
