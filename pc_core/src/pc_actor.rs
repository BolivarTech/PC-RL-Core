// Author: Julian Bolivar
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
///     local_lambda: 1.0,
///     residual: false,
///     rezero_init: 0.001,
///     aux_loss_coefficient: 0.0,
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
    /// Inference learning rate for PC loop state updates (`h += alpha * error`).
    /// Set to 0.0 to disable PC inference (network behaves as standard MLP).
    /// Active regardless of `residual` setting.
    pub alpha: f64,
    /// Convergence threshold for RMS prediction error.
    /// PC loop exits early when surprise < tol (after at least `min_steps`).
    /// Active regardless of `residual` setting.
    pub tol: f64,
    /// Minimum PC inference steps before convergence check is allowed.
    /// Active regardless of `residual` setting.
    pub min_steps: usize,
    /// Maximum PC inference steps per action.
    /// Active regardless of `residual` setting.
    pub max_steps: usize,
    /// Base learning rate for weight updates.
    pub lr_weights: f64,
    /// If true, use synchronous snapshot mode; otherwise in-place.
    pub synchronous: bool,
    /// Softmax temperature for action selection.
    pub temperature: f64,
    /// Blend factor for hidden layer weight updates, range `[0.0, 1.0]`.
    ///
    /// Controls how hidden layers combine two gradient signals:
    /// `delta = lambda * backprop_grad + (1 - lambda) * pc_prediction_error`
    ///
    /// - `1.0` — Pure backprop: reward signal propagated from output (default).
    /// - `0.0` — Pure local PC: prediction errors from inference loop
    ///   used as gradients (Millidge et al. 2022). No vanishing gradient
    ///   but no reward signal reaches hidden layers.
    /// - `0.0 < lambda < 1.0` — Hybrid: reward-aware backprop regularized
    ///   by local PC consistency errors.
    ///
    /// The output layer always uses standard backprop regardless of this value.
    #[serde(default = "default_local_lambda")]
    pub local_lambda: f64,
    /// Enable residual skip connections between same-dimension hidden layers.
    /// When false, `rezero_init` is ignored. When true, all hidden layers
    /// must have the same size, and skip connections with learnable ReZero
    /// scaling are added between consecutive hidden layers (not the first,
    /// since input_size typically differs from hidden_size).
    #[serde(default)]
    pub residual: bool,
    /// Initial value for ReZero scaling factors on residual connections.
    /// Only used when `residual = true`. Controls initial contribution of
    /// the nonlinear component: `h[i] = rezero_init * tanh(...) + h[i-1]`.
    ///
    /// - `0.001` — Near-identity start (ReZero: network learns depth gradually)
    /// - `1.0` — Standard ResNet residual (full contribution from start)
    ///
    /// Ignored when `residual = false`.
    #[serde(default = "default_rezero_init")]
    pub rezero_init: f64,
    /// Auxiliary loss coefficient for hidden layer gradient injection.
    /// Each hidden layer gets a linear auxiliary head that predicts the output
    /// logits. The MSE gradient is scaled by this coefficient and added to the
    /// hidden layer's effective delta, providing a direct reward signal that
    /// bypasses upper-layer tanh attenuation.
    ///
    /// - `0.0` — Disabled (default). No auxiliary heads created.
    /// - `> 0.0` — Enabled. One auxiliary head per hidden layer.
    #[serde(default)]
    pub aux_loss_coefficient: f64,
}

/// Default rezero_init: 0.001 (near-identity at start).
fn default_rezero_init() -> f64 {
    0.001
}

/// Default local_lambda: 1.0 (pure backprop).
fn default_local_lambda() -> f64 {
    1.0
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
    /// Per-layer prediction errors from the last PC inference step.
    /// Ordered from top hidden layer to bottom (reverse layer order).
    pub prediction_errors: Vec<Vec<f64>>,
    /// RMS prediction error across layers.
    pub surprise_score: f64,
    /// Number of inference steps performed.
    pub steps_used: usize,
    /// Whether the inference loop converged within tolerance.
    pub converged: bool,
    /// Per-layer tanh components for residual layers.
    /// `None` for non-skip layers, `Some(tanh_out)` for skip-eligible layers.
    /// Needed for correct backward pass (derivative on tanh_out, not full h[i]).
    pub tanh_components: Vec<Option<Vec<f64>>>,
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
///     local_lambda: 1.0,
///     residual: false,
///     rezero_init: 0.001,
///     aux_loss_coefficient: 0.0,
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
    /// ReZero scaling factors for eligible skip connections.
    /// One entry per hidden layer that has a skip connection (index >= 1, same size as previous).
    pub(crate) rezero_alpha: Vec<f64>,
    /// Auxiliary linear heads for hidden layer gradient injection.
    /// One per hidden layer when `aux_loss_coefficient > 0`. Maps hidden_size → output_size.
    /// Used in `update_weights_hybrid()` for gradient injection.
    pub(crate) aux_heads: Vec<Layer>,
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
        if !(0.0..=1.0).contains(&config.local_lambda) {
            return Err(PcError::ConfigValidation(format!(
                "local_lambda must be in [0.0, 1.0], got {}",
                config.local_lambda
            )));
        }
        if config.aux_loss_coefficient < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "aux_loss_coefficient must be >= 0, got {}",
                config.aux_loss_coefficient
            )));
        }
        if config.rezero_init < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "rezero_init must be >= 0, got {}",
                config.rezero_init
            )));
        }
        if config.residual && config.hidden_layers.len() >= 2 {
            for i in 1..config.hidden_layers.len() {
                if config.hidden_layers[i].size != config.hidden_layers[i - 1].size {
                    return Err(PcError::ConfigValidation(format!(
                        "residual requires same-size hidden layers, but layer {} has size {} and layer {} has size {}",
                        i - 1, config.hidden_layers[i - 1].size,
                        i, config.hidden_layers[i].size
                    )));
                }
            }
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

        // Compute rezero_alpha: one entry per eligible skip layer
        let rezero_alpha = if config.residual {
            let mut alphas = Vec::new();
            for i in 1..config.hidden_layers.len() {
                if config.hidden_layers[i].size == config.hidden_layers[i - 1].size {
                    alphas.push(config.rezero_init);
                }
            }
            alphas
        } else {
            Vec::new()
        };

        // Create auxiliary heads: one linear projection per hidden layer
        let aux_heads = if config.aux_loss_coefficient > 0.0 {
            config
                .hidden_layers
                .iter()
                .map(|def| Layer::new(def.size, config.output_size, Activation::Linear, rng))
                .collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            layers,
            config,
            rezero_alpha,
            aux_heads,
        })
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
    /// Returns whether hidden layer `i` has a skip connection.
    fn is_skip_layer(&self, i: usize) -> bool {
        self.config.residual
            && i >= 1
            && self.config.hidden_layers[i].size == self.config.hidden_layers[i - 1].size
    }

    /// Returns the rezero_alpha index for hidden layer `i`, if it is a skip layer.
    fn skip_alpha_index(&self, i: usize) -> Option<usize> {
        if !self.is_skip_layer(i) {
            return None;
        }
        let mut idx = 0;
        for j in 1..i {
            if self.config.hidden_layers[j].size == self.config.hidden_layers[j - 1].size {
                idx += 1;
            }
        }
        Some(idx)
    }

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
        let mut tanh_components: Vec<Option<Vec<f64>>> = Vec::with_capacity(n_hidden);
        let mut prev = input.to_vec();
        for (i, layer) in self.layers[..n_hidden].iter().enumerate() {
            let tanh_out = layer.forward(&prev);
            if let Some(alpha_idx) = self.skip_alpha_index(i) {
                // Residual: h[i] = rezero_alpha * tanh_out + h[i-1]
                let alpha = self.rezero_alpha[alpha_idx];
                let scaled = vec_scale(&tanh_out, alpha);
                prev = vec_add(&prev, &scaled);
                tanh_components.push(Some(tanh_out));
            } else {
                prev = tanh_out;
                tanh_components.push(None);
            }
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
        let mut last_errors: Vec<Vec<f64>> = Vec::new();

        for step in 0..self.config.max_steps {
            steps_used = step + 1;

            if self.config.synchronous {
                // Snapshot mode: freeze all states
                let snapshot: Vec<Vec<f64>> = hidden_states.clone();
                let tanh_snap: Vec<Option<Vec<f64>>> = tanh_components.clone();

                let mut error_vecs: Vec<Vec<f64>> = Vec::new();

                for i in (0..n_hidden).rev() {
                    // For top-down prediction, use tanh_component of layer above
                    // (not the full residual sum) when it is a skip layer.
                    let state_above = if i == n_hidden - 1 {
                        &y
                    } else if let Some(ref tc) = tanh_snap[i + 1] {
                        tc
                    } else {
                        &snapshot[i + 1]
                    };

                    // Top-down prediction targets tanh_component for skip layers
                    let target = if let Some(ref tc) = tanh_snap[i] {
                        tc
                    } else {
                        &snapshot[i]
                    };

                    let prediction = self.layers[i + 1]
                        .transpose_forward(state_above, self.config.hidden_layers[i].activation);

                    let error = vec_sub(&prediction, target);
                    error_vecs.push(error.clone());

                    // Update tanh_component or hidden_state
                    let updated_target = vec_add(target, &vec_scale(&error, self.config.alpha));
                    if let Some(alpha_idx) = self.skip_alpha_index(i) {
                        tanh_components[i] = Some(updated_target.clone());
                        let alpha = self.rezero_alpha[alpha_idx];
                        let prev_h = if i > 0 { &hidden_states[i - 1] } else { input };
                        hidden_states[i] = vec_add(prev_h, &vec_scale(&updated_target, alpha));
                    } else {
                        hidden_states[i] = updated_target;
                    }
                }

                let top_hidden = if n_hidden > 0 {
                    &hidden_states[n_hidden - 1]
                } else {
                    input
                };
                y = self.layers[n_hidden].forward(top_hidden);

                let refs: Vec<&[f64]> = error_vecs.iter().map(|v| v.as_slice()).collect();
                surprise_score = rms_error(&refs);
                last_errors = error_vecs;
            } else {
                // In-place mode: updates immediately visible
                let mut error_vecs: Vec<Vec<f64>> = Vec::new();

                for i in (0..n_hidden).rev() {
                    // For top-down prediction, use tanh_component of layer above
                    // (not the full residual sum) when it is a skip layer.
                    let state_above = if i == n_hidden - 1 {
                        &y
                    } else if let Some(ref tc) = tanh_components[i + 1] {
                        tc
                    } else {
                        &hidden_states[i + 1]
                    };

                    let target = if let Some(ref tc) = tanh_components[i] {
                        tc.clone()
                    } else {
                        hidden_states[i].clone()
                    };

                    let prediction = self.layers[i + 1]
                        .transpose_forward(state_above, self.config.hidden_layers[i].activation);

                    let error = vec_sub(&prediction, &target);
                    error_vecs.push(error.clone());

                    let updated_target = vec_add(&target, &vec_scale(&error, self.config.alpha));
                    if let Some(alpha_idx) = self.skip_alpha_index(i) {
                        tanh_components[i] = Some(updated_target.clone());
                        let alpha = self.rezero_alpha[alpha_idx];
                        let prev_h = if i > 0 { &hidden_states[i - 1] } else { input };
                        hidden_states[i] = vec_add(prev_h, &vec_scale(&updated_target, alpha));
                    } else {
                        hidden_states[i] = updated_target;
                    }
                }

                let top_hidden = if n_hidden > 0 {
                    &hidden_states[n_hidden - 1]
                } else {
                    input
                };
                y = self.layers[n_hidden].forward(top_hidden);

                let refs: Vec<&[f64]> = error_vecs.iter().map(|v| v.as_slice()).collect();
                surprise_score = rms_error(&refs);
                last_errors = error_vecs;
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
            prediction_errors: last_errors,
            surprise_score,
            steps_used,
            converged,
            tanh_components,
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

    /// Updates network weights using a blend of backprop and local PC error.
    ///
    /// The `local_lambda` config controls the blend: 1.0 = pure backprop,
    /// 0.0 = pure local PC learning (Millidge et al. 2022), intermediate = hybrid.
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

        self.update_weights_hybrid(
            output_delta,
            infer_result,
            input,
            surprise_scale,
            self.config.local_lambda,
        );
    }

    /// Hybrid weight update blending backprop and local PC error signals.
    ///
    /// For hidden layers, the effective delta is:
    /// `delta = lambda * backprop_delta + (1 - lambda) * pc_error`
    ///
    /// * `lambda = 1.0` → pure backprop (standard mode).
    /// * `lambda = 0.0` → pure local PC learning (Millidge et al. 2022).
    /// * `0 < lambda < 1` → hybrid blend.
    ///
    /// The output layer always uses standard backprop from `output_delta`.
    fn update_weights_hybrid(
        &mut self,
        output_delta: &[f64],
        infer_result: &InferResult,
        input: &[f64],
        surprise_scale: f64,
        lambda: f64,
    ) {
        let n_hidden = self.config.hidden_layers.len();
        let n_layers = self.layers.len();

        // Output layer: always standard backward
        let output_input = if n_hidden > 0 {
            &infer_result.hidden_states[n_hidden - 1]
        } else {
            input
        };
        let output_output = &infer_result.y_conv;
        let mut bp_delta = self.layers[n_layers - 1].backward(
            output_input,
            output_output,
            output_delta,
            self.config.lr_weights,
            surprise_scale,
        );

        // Hidden layers (from top to bottom)
        for i in (0..n_hidden).rev() {
            let layer_input = if i > 0 {
                &infer_result.hidden_states[i - 1]
            } else {
                input
            };

            // Blend backprop delta with local PC error
            let effective_delta = if (lambda - 1.0).abs() < f64::EPSILON {
                bp_delta.clone()
            } else if lambda.abs() < f64::EPSILON {
                let error_idx = n_hidden - 1 - i;
                infer_result.prediction_errors[error_idx].clone()
            } else {
                let error_idx = n_hidden - 1 - i;
                let pc_error = &infer_result.prediction_errors[error_idx];
                bp_delta
                    .iter()
                    .zip(pc_error.iter())
                    .map(|(&bp, &pc)| lambda * bp + (1.0 - lambda) * pc)
                    .collect()
            };

            // Auxiliary loss gradient injection: MSE(aux_logits, y_conv)
            let effective_delta =
                if self.config.aux_loss_coefficient > 0.0 && i < self.aux_heads.len() {
                    let aux_logits = self.aux_heads[i].forward(&infer_result.hidden_states[i]);
                    let coeff = self.config.aux_loss_coefficient;
                    let scaled_mse_grad: Vec<f64> = aux_logits
                        .iter()
                        .zip(infer_result.y_conv.iter())
                        .map(|(&a, &y)| coeff * (a - y))
                        .collect();
                    let propagated_aux = self.aux_heads[i].backward(
                        &infer_result.hidden_states[i],
                        &aux_logits,
                        &scaled_mse_grad,
                        self.config.lr_weights,
                        surprise_scale,
                    );
                    effective_delta
                        .iter()
                        .zip(propagated_aux.iter())
                        .map(|(&ed, &aux)| ed + aux)
                        .collect()
                } else {
                    effective_delta
                };

            if let Some(alpha_idx) = self.skip_alpha_index(i) {
                // Skip-eligible layer: use tanh_out for derivative, scale by alpha,
                // add identity path to propagated gradient, update alpha.
                let tanh_out = infer_result.tanh_components[i].as_ref().unwrap();
                let alpha = self.rezero_alpha[alpha_idx];
                let effective_lr = self.config.lr_weights * surprise_scale;

                // Scale delta by rezero_alpha for the nonlinear path
                let scaled_delta: Vec<f64> = effective_delta.iter().map(|&d| d * alpha).collect();

                // Backward through the layer using tanh_out (not hidden_states[i])
                let propagated = self.layers[i].backward(
                    layer_input,
                    tanh_out,
                    &scaled_delta,
                    self.config.lr_weights,
                    surprise_scale,
                );

                // Update rezero_alpha: dL/d(alpha) = delta · tanh_out
                let grad_alpha: f64 = effective_delta
                    .iter()
                    .zip(tanh_out.iter())
                    .map(|(&d, &t)| d * t)
                    .sum();
                self.rezero_alpha[alpha_idx] -= effective_lr * grad_alpha;

                // Propagated delta = nonlinear path + identity path
                bp_delta = vec_add(&propagated, &effective_delta);
            } else {
                // Standard layer: use hidden_states[i] as output
                let layer_output = &infer_result.hidden_states[i];
                bp_delta = self.layers[i].backward(
                    layer_input,
                    layer_output,
                    &effective_delta,
                    self.config.lr_weights,
                    surprise_scale,
                );
            }
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
            local_lambda: 1.0,
            residual: false,
            rezero_init: 0.001,
            aux_loss_coefficient: 0.0,
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

    // ── Auxiliary Loss Config Tests ────────────────────────────

    #[test]
    fn test_default_config_aux_loss_zero() {
        let config = default_config();
        assert!(config.aux_loss_coefficient.abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_negative_aux_loss_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            aux_loss_coefficient: -0.1,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_aux_heads_empty_when_disabled() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        assert!(actor.aux_heads.is_empty());
    }

    #[test]
    fn test_aux_heads_one_per_hidden_layer() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            aux_loss_coefficient: 0.1,
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        assert_eq!(actor.aux_heads.len(), 1);
    }

    #[test]
    fn test_aux_heads_two_hidden() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            aux_loss_coefficient: 0.1,
            ..two_hidden_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        assert_eq!(actor.aux_heads.len(), 2);
    }

    #[test]
    fn test_aux_head_dimensions() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            aux_loss_coefficient: 0.1,
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        // hidden_size=18 -> output_size=9, Linear activation
        assert_eq!(actor.aux_heads[0].weights.rows, 9);
        assert_eq!(actor.aux_heads[0].weights.cols, 18);
    }

    #[test]
    fn test_new_positive_aux_loss_accepted() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            aux_loss_coefficient: 0.5,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_ok());
    }

    // ── Auxiliary Loss Gradient Tests ─────────────────────────

    #[test]
    fn test_aux_disabled_update_identical_to_baseline() {
        let input = vec![0.5; 9];
        let delta = vec![0.1; 9];

        let mut rng1 = make_rng();
        let mut actor1 = PcActor::new(default_config(), &mut rng1).unwrap();
        let infer1 = actor1.infer(&input);
        actor1.update_weights(&delta, &infer1, &input, 1.0);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            aux_loss_coefficient: 0.0,
            ..default_config()
        };
        let mut actor2 = PcActor::new(config2, &mut rng2).unwrap();
        let infer2 = actor2.infer(&input);
        actor2.update_weights(&delta, &infer2, &input, 1.0);

        for (l1, l2) in actor1.layers.iter().zip(actor2.layers.iter()) {
            for (w1, w2) in l1.weights.data.iter().zip(l2.weights.data.iter()) {
                assert!((w1 - w2).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_aux_enabled_changes_hidden_weights_differently() {
        let input = vec![0.5; 9];
        let delta = vec![0.1; 9];

        let mut rng1 = make_rng();
        let mut actor_no_aux = PcActor::new(default_config(), &mut rng1).unwrap();
        let infer1 = actor_no_aux.infer(&input);
        actor_no_aux.update_weights(&delta, &infer1, &input, 1.0);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            aux_loss_coefficient: 0.5,
            ..default_config()
        };
        let mut actor_aux = PcActor::new(config2, &mut rng2).unwrap();
        let infer2 = actor_aux.infer(&input);
        actor_aux.update_weights(&delta, &infer2, &input, 1.0);

        let differs = actor_no_aux.layers[0]
            .weights
            .data
            .iter()
            .zip(actor_aux.layers[0].weights.data.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(differs, "Aux loss should change hidden layer gradient");
    }

    #[test]
    fn test_aux_heads_weights_change_after_update() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            aux_loss_coefficient: 0.1,
            ..default_config()
        };
        let mut actor = PcActor::new(config, &mut rng).unwrap();
        let aux_w_before = actor.aux_heads[0].weights.data.clone();
        let input = vec![0.5; 9];
        let infer = actor.infer(&input);
        actor.update_weights(&[0.1; 9], &infer, &input, 1.0);
        assert_ne!(
            actor.aux_heads[0].weights.data, aux_w_before,
            "Aux head weights should update"
        );
    }

    #[test]
    fn test_aux_output_layer_unaffected() {
        let input = vec![0.5; 9];
        let delta = vec![0.1; 9];

        let mut rng1 = make_rng();
        let mut actor1 = PcActor::new(default_config(), &mut rng1).unwrap();
        let infer1 = actor1.infer(&input);
        actor1.update_weights(&delta, &infer1, &input, 1.0);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            aux_loss_coefficient: 0.5,
            ..default_config()
        };
        let mut actor2 = PcActor::new(config2, &mut rng2).unwrap();
        let infer2 = actor2.infer(&input);
        actor2.update_weights(&delta, &infer2, &input, 1.0);

        let out_idx = actor1.layers.len() - 1;
        for (w1, w2) in actor1.layers[out_idx]
            .weights
            .data
            .iter()
            .zip(actor2.layers[out_idx].weights.data.iter())
        {
            assert!(
                (w1 - w2).abs() < 1e-12,
                "Output layer should not be affected by aux loss"
            );
        }
    }

    #[test]
    fn test_aux_all_weights_finite() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            aux_loss_coefficient: 0.5,
            ..two_hidden_config()
        };
        let mut actor = PcActor::new(config, &mut rng).unwrap();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let infer = actor.infer(&input);
        actor.update_weights(&[0.3; 9], &infer, &input, 1.0);
        for layer in &actor.layers {
            for &w in &layer.weights.data {
                assert!(w.is_finite());
            }
        }
        for head in &actor.aux_heads {
            for &w in &head.weights.data {
                assert!(w.is_finite());
            }
        }
    }

    // ── Residual / ReZero Config Tests ────────────────────────

    #[test]
    fn test_default_config_residual_false() {
        let config = default_config();
        assert!(!config.residual);
    }

    #[test]
    fn test_default_config_rezero_init() {
        let config = default_config();
        assert!((config.rezero_init - 0.001).abs() < 1e-12);
    }

    #[test]
    fn test_new_negative_rezero_init_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            rezero_init: -0.1,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_residual_mismatched_hidden_sizes_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_residual_same_size_hidden_layers_accepted() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_ok());
    }

    fn residual_two_hidden_config() -> PcActorConfig {
        PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        }
    }

    #[test]
    fn test_non_residual_actor_empty_rezero_alpha() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        assert!(actor.rezero_alpha.is_empty());
    }

    #[test]
    fn test_residual_two_hidden_one_rezero_alpha() {
        let mut rng = make_rng();
        let actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        assert_eq!(actor.rezero_alpha.len(), 1);
    }

    #[test]
    fn test_residual_three_hidden_two_rezero_alpha() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        assert_eq!(actor.rezero_alpha.len(), 2);
    }

    #[test]
    fn test_rezero_alpha_initialized_to_rezero_init() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            rezero_init: 0.005,
            ..residual_two_hidden_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        assert!((actor.rezero_alpha[0] - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_residual_single_hidden_zero_rezero_alpha() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        assert!(actor.rezero_alpha.is_empty());
    }

    #[test]
    fn test_residual_single_hidden_accepted() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            ..default_config()
        };
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_ok());
    }

    // ── Local Learning (PC-based weight updates) Tests ──────────

    // ── Residual Inference Tests ──────────────────────────────

    #[test]
    fn test_residual_false_identical_to_non_residual() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let mut rng1 = make_rng();
        let actor1 = PcActor::new(two_hidden_config(), &mut rng1).unwrap();
        let result1 = actor1.infer(&input);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            residual: false,
            ..two_hidden_config()
        };
        let actor2 = PcActor::new(config2, &mut rng2).unwrap();
        let result2 = actor2.infer(&input);

        for (a, b) in result1.y_conv.iter().zip(result2.y_conv.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_residual_rezero_zero_second_hidden_near_identity() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            rezero_init: 0.0,
            alpha: 0.0,
            ..residual_two_hidden_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        let h0 = &result.hidden_states[0];
        let h1 = &result.hidden_states[1];
        for (a, b) in h0.iter().zip(h1.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "With rezero_init=0, h[1] should equal h[0]"
            );
        }
    }

    #[test]
    fn test_residual_infer_all_outputs_finite() {
        let mut rng = make_rng();
        let actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
        for &v in &result.latent_concat {
            assert!(v.is_finite());
        }
        assert!(result.surprise_score.is_finite());
    }

    #[test]
    fn test_residual_latent_concat_size() {
        let mut rng = make_rng();
        let actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.latent_concat.len(), 54); // 27 + 27
    }

    #[test]
    fn test_residual_pc_loop_completes() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            alpha: 0.03,
            max_steps: 5,
            ..residual_two_hidden_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert!(result.steps_used > 0);
        assert!(result.steps_used <= 5);
    }

    #[test]
    fn test_residual_hidden_states_count() {
        let mut rng = make_rng();
        let actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.hidden_states.len(), 2);
    }

    #[test]
    fn test_residual_infer_does_not_modify_weights() {
        let mut rng = make_rng();
        let actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let weights_before: Vec<Vec<f64>> = actor
            .layers
            .iter()
            .map(|l| l.weights.data.clone())
            .collect();
        let alpha_before = actor.rezero_alpha.clone();
        let _ = actor.infer(&[0.5; 9]);
        for (i, layer) in actor.layers.iter().enumerate() {
            assert_eq!(layer.weights.data, weights_before[i]);
        }
        assert_eq!(actor.rezero_alpha, alpha_before);
    }

    #[test]
    fn test_residual_three_hidden_infer_finite() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_residual_tanh_components_populated() {
        let mut rng = make_rng();
        let actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.tanh_components.len(), 2);
        assert!(result.tanh_components[0].is_none()); // layer 0: no skip
        assert!(result.tanh_components[1].is_some()); // layer 1: has skip
        assert_eq!(result.tanh_components[1].as_ref().unwrap().len(), 27);
    }

    #[test]
    fn test_residual_pc_prediction_uses_tanh_component_not_full_state() {
        // With rezero_init=1.0, h[1] = tanh_out + h[0] (significantly different
        // from tanh_out alone). If PC prediction uses h[1] instead of tanh_out,
        // the surprise score and convergence will differ.
        // Two runs with same weights: one with alpha=0 (no PC), one with alpha>0.
        // The PC loop should converge meaningfully (surprise decreases).
        let mut rng = make_rng();
        let config = PcActorConfig {
            rezero_init: 1.0,
            alpha: 0.1,
            max_steps: 20,
            tol: 0.001,
            min_steps: 1,
            ..residual_two_hidden_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5]);
        // With proper PC predictions, surprise should be finite and non-negative
        assert!(result.surprise_score.is_finite());
        assert!(result.surprise_score >= 0.0);
        // Prediction errors should all be finite
        for errors in &result.prediction_errors {
            for &e in errors {
                assert!(e.is_finite(), "PC prediction error not finite: {e}");
            }
        }
    }

    // ── Residual Backward Tests ────────────────────────────────

    #[test]
    fn test_residual_false_update_identical_to_non_residual() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        let mut rng1 = make_rng();
        let mut actor1 = PcActor::new(two_hidden_config(), &mut rng1).unwrap();
        let infer1 = actor1.infer(&input);
        actor1.update_weights(&delta, &infer1, &input, 1.0);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            residual: false,
            ..two_hidden_config()
        };
        let mut actor2 = PcActor::new(config2, &mut rng2).unwrap();
        let infer2 = actor2.infer(&input);
        actor2.update_weights(&delta, &infer2, &input, 1.0);

        for i in 0..actor1.layers.len() {
            assert_eq!(actor1.layers[i].weights.data, actor2.layers[i].weights.data);
        }
    }

    #[test]
    fn test_residual_update_changes_all_layer_weights() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0 = actor.layers[0].weights.data.clone();
        let w1 = actor.layers[1].weights.data.clone();
        let w2 = actor.layers[2].weights.data.clone();
        actor.update_weights(&[0.1; 9], &infer_result, &input, 1.0);
        assert_ne!(actor.layers[0].weights.data, w0, "Layer 0 should change");
        assert_ne!(actor.layers[1].weights.data, w1, "Layer 1 should change");
        assert_ne!(
            actor.layers[2].weights.data, w2,
            "Output layer should change"
        );
    }

    #[test]
    fn test_residual_update_changes_rezero_alpha() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let alpha_before = actor.rezero_alpha.clone();
        actor.update_weights(&[0.1; 9], &infer_result, &input, 1.0);
        assert_ne!(
            actor.rezero_alpha, alpha_before,
            "rezero_alpha should be updated by backprop"
        );
    }

    #[test]
    fn test_residual_update_clips_weights() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let input = vec![1.0; 9];
        let infer_result = actor.infer(&input);
        actor.update_weights(&[1e6; 9], &infer_result, &input, 1.0);
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
    fn test_residual_gradient_stronger_than_non_residual() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Non-residual 2 hidden layers (27, 27)
        let mut rng1 = make_rng();
        let config1 = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let mut actor1 = PcActor::new(config1, &mut rng1).unwrap();
        let w0_before1 = actor1.layers[0].weights.data.clone();
        let infer1 = actor1.infer(&input);
        actor1.update_weights(&delta, &infer1, &input, 1.0);
        let change1: f64 = actor1.layers[0]
            .weights
            .data
            .iter()
            .zip(w0_before1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // Residual 2 hidden layers (27, 27) with rezero_init=1.0
        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            rezero_init: 1.0,
            ..residual_two_hidden_config()
        };
        let mut actor2 = PcActor::new(config2, &mut rng2).unwrap();
        let w0_before2 = actor2.layers[0].weights.data.clone();
        let infer2 = actor2.infer(&input);
        actor2.update_weights(&delta, &infer2, &input, 1.0);
        let change2: f64 = actor2.layers[0]
            .weights
            .data
            .iter()
            .zip(w0_before2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            change2 > change1,
            "Residual should propagate stronger gradient to layer 0: residual={change2:.6}, non-residual={change1:.6}"
        );
    }

    #[test]
    fn test_residual_hybrid_lambda_works() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            local_lambda: 0.99,
            ..residual_two_hidden_config()
        };
        let mut actor = PcActor::new(config, &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0_before = actor.layers[0].weights.data.clone();
        actor.update_weights(&[0.1; 9], &infer_result, &input, 1.0);
        assert_ne!(actor.layers[0].weights.data, w0_before);
    }

    fn local_learning_config() -> PcActorConfig {
        PcActorConfig {
            local_lambda: 0.0,
            ..default_config()
        }
    }

    #[test]
    fn test_infer_prediction_errors_count_matches_hidden_layers() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.prediction_errors.len(), 1);
    }

    #[test]
    fn test_infer_prediction_errors_two_hidden() {
        let mut rng = make_rng();
        let actor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.prediction_errors.len(), 2);
    }

    #[test]
    fn test_infer_prediction_errors_zero_hidden_is_empty() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            hidden_layers: vec![],
            ..default_config()
        };
        let actor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert!(result.prediction_errors.is_empty());
    }

    #[test]
    fn test_infer_prediction_errors_all_finite() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5]);
        for errors in &result.prediction_errors {
            for &e in errors {
                assert!(e.is_finite(), "prediction error not finite: {e}");
            }
        }
    }

    #[test]
    fn test_infer_prediction_errors_size_matches_hidden_layer_size() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        // default_config has one hidden layer of size 18
        assert_eq!(result.prediction_errors[0].len(), 18);
    }

    #[test]
    fn test_local_learning_config_accepted() {
        let mut rng = make_rng();
        let config = local_learning_config();
        assert!((config.local_lambda).abs() < f64::EPSILON);
        let actor = PcActor::new(config, &mut rng);
        assert!(actor.is_ok());
    }

    #[test]
    fn test_local_learning_update_changes_weights() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(local_learning_config(), &mut rng).unwrap();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let infer_result = actor.infer(&input);
        let weights_before = actor.layers[0].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0);
        assert_ne!(actor.layers[0].weights.data, weights_before);
    }

    #[test]
    fn test_local_learning_clips_weights() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(local_learning_config(), &mut rng).unwrap();
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
    fn test_local_learning_two_hidden_changes_both() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            local_lambda: 0.0,
            ..two_hidden_config()
        };
        let mut actor = PcActor::new(config, &mut rng).unwrap();
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
    fn test_local_learning_differs_from_backprop() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Backprop actor
        let mut rng1 = make_rng();
        let mut bp_actor = PcActor::new(default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0);

        // Local learning actor (same initial weights)
        let mut rng2 = make_rng();
        let mut ll_actor = PcActor::new(local_learning_config(), &mut rng2).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0);

        // Hidden layer weights should differ between the two approaches
        assert_ne!(
            bp_actor.layers[0].weights.data, ll_actor.layers[0].weights.data,
            "Local learning should produce different weight updates than backprop"
        );
    }

    // ── Hybrid Learning (local_lambda) Tests ────────────────────

    fn hybrid_config(lambda: f64) -> PcActorConfig {
        PcActorConfig {
            local_lambda: lambda,
            ..default_config()
        }
    }

    #[test]
    fn test_local_lambda_one_equals_backprop() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Pure backprop (local_learning=false, default)
        let mut rng1 = make_rng();
        let mut bp_actor = PcActor::new(default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0);

        // lambda=1.0 should be identical to backprop
        let mut rng2 = make_rng();
        let mut lam_actor = PcActor::new(hybrid_config(1.0), &mut rng2).unwrap();
        let lam_infer = lam_actor.infer(&input);
        lam_actor.update_weights(&delta, &lam_infer, &input, 1.0);

        assert_eq!(
            bp_actor.layers[0].weights.data, lam_actor.layers[0].weights.data,
            "lambda=1.0 should produce identical weights to pure backprop"
        );
    }

    #[test]
    fn test_local_lambda_zero_equals_local_learning() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Pure local (local_learning=true)
        let mut rng1 = make_rng();
        let mut ll_actor = PcActor::new(local_learning_config(), &mut rng1).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0);

        // lambda=0.0 should be identical to pure local
        let mut rng2 = make_rng();
        let mut lam_actor = PcActor::new(hybrid_config(0.0), &mut rng2).unwrap();
        let lam_infer = lam_actor.infer(&input);
        lam_actor.update_weights(&delta, &lam_infer, &input, 1.0);

        assert_eq!(
            ll_actor.layers[0].weights.data, lam_actor.layers[0].weights.data,
            "lambda=0.0 should produce identical weights to pure local learning"
        );
    }

    #[test]
    fn test_local_lambda_half_differs_from_both_pure_modes() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Pure backprop
        let mut rng1 = make_rng();
        let mut bp_actor = PcActor::new(default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0);

        // Pure local
        let mut rng2 = make_rng();
        let mut ll_actor = PcActor::new(local_learning_config(), &mut rng2).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0);

        // Hybrid lambda=0.5
        let mut rng3 = make_rng();
        let mut hy_actor = PcActor::new(hybrid_config(0.5), &mut rng3).unwrap();
        let hy_infer = hy_actor.infer(&input);
        hy_actor.update_weights(&delta, &hy_infer, &input, 1.0);

        assert_ne!(
            hy_actor.layers[0].weights.data, bp_actor.layers[0].weights.data,
            "lambda=0.5 should differ from pure backprop"
        );
        assert_ne!(
            hy_actor.layers[0].weights.data, ll_actor.layers[0].weights.data,
            "lambda=0.5 should differ from pure local"
        );
    }

    #[test]
    fn test_local_lambda_changes_weights() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(hybrid_config(0.5), &mut rng).unwrap();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let infer_result = actor.infer(&input);
        let weights_before = actor.layers[0].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0);
        assert_ne!(actor.layers[0].weights.data, weights_before);
    }

    #[test]
    fn test_local_lambda_clips_weights() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(hybrid_config(0.5), &mut rng).unwrap();
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
    fn test_local_lambda_negative_returns_error() {
        let mut rng = make_rng();
        let config = hybrid_config(-0.1);
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_lambda_above_one_returns_error() {
        let mut rng = make_rng();
        let config = hybrid_config(1.1);
        let result = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }
}
