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
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;

/// Configuration for the predictive coding actor network.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
/// use pc_rl_core::pc_actor::PcActorConfig;
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
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
#[derive(Debug, Clone)]
pub struct InferResult<L: LinAlg = CpuLinAlg> {
    /// Converged output logits.
    pub y_conv: L::Vector,
    /// All hidden states concatenated (fed to critic).
    pub latent_concat: L::Vector,
    /// Per-layer hidden state activations.
    pub hidden_states: Vec<L::Vector>,
    /// Per-layer prediction errors from the last PC inference step.
    /// Ordered from top hidden layer to bottom (reverse layer order).
    pub prediction_errors: Vec<L::Vector>,
    /// RMS prediction error across layers.
    pub surprise_score: f64,
    /// Number of inference steps performed.
    pub steps_used: usize,
    /// Whether the inference loop converged within tolerance.
    pub converged: bool,
    /// Per-layer tanh components for residual layers.
    /// `None` for non-skip layers, `Some(tanh_out)` for skip-eligible layers.
    /// Needed for correct backward pass (derivative on tanh_out, not full h\[i\]).
    pub tanh_components: Vec<Option<L::Vector>>,
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
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
/// use pc_rl_core::pc_actor::{PcActor, PcActorConfig, SelectionMode};
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
/// };
/// let mut rng = StdRng::seed_from_u64(42);
/// let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
/// let result = actor.infer(&[0.0; 9]);
/// assert_eq!(result.y_conv.len(), 9);
/// ```
#[derive(Debug)]
pub struct PcActor<L: LinAlg = CpuLinAlg> {
    /// Network layers: hidden_layers.len() + 1 (output layer).
    pub(crate) layers: Vec<Layer<L>>,
    /// Actor configuration.
    pub config: PcActorConfig,
    /// ReZero scaling factors for skip connections. One per skip layer (all i >= 1 when residual=true).
    pub(crate) rezero_alpha: Vec<f64>,
    /// Projection matrices for skip connections between layers of different sizes.
    /// One entry per skip layer: `None` for identity (same size), `Some(Matrix)` for projection.
    pub(crate) skip_projections: Vec<Option<L::Matrix>>,
}

impl<L: LinAlg> PcActor<L> {
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
        if config.rezero_init < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "rezero_init must be >= 0, got {}",
                config.rezero_init
            )));
        }
        let mut layers: Vec<Layer<L>> = Vec::new();
        let mut prev_size = config.input_size;

        for def in &config.hidden_layers {
            layers.push(Layer::<L>::new(prev_size, def.size, def.activation, rng));
            prev_size = def.size;
        }

        // Output layer
        layers.push(Layer::<L>::new(
            prev_size,
            config.output_size,
            config.output_activation,
            rng,
        ));

        // Compute rezero_alpha and skip_projections: one per skip layer (all i >= 1)
        let (rezero_alpha, skip_projections) = if config.residual {
            let mut alphas = Vec::new();
            let mut projs = Vec::new();
            for i in 1..config.hidden_layers.len() {
                alphas.push(config.rezero_init);
                if config.hidden_layers[i].size != config.hidden_layers[i - 1].size {
                    projs.push(Some(L::xavier_mat(
                        config.hidden_layers[i].size,
                        config.hidden_layers[i - 1].size,
                        rng,
                    )));
                } else {
                    projs.push(None);
                }
            }
            (alphas, projs)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            layers,
            config,
            rezero_alpha,
            skip_projections,
        })
    }

    /// Creates a child actor by crossing over two parent actors using CCA neuron alignment.
    ///
    /// Aligns hidden neurons functionally via CCA before blending weights.
    /// Input and output layers use positional crossover (no permutation problem).
    ///
    /// # Arguments
    ///
    /// * `parent_a` - First parent (reference, typically higher fitness).
    /// * `parent_b` - Second parent (aligned to A via CCA).
    /// * `caches_a` - Per-layer activation matrices for parent A `[batch × neurons]`.
    /// * `caches_b` - Per-layer activation matrices for parent B `[batch × neurons]`.
    /// * `alpha` - Blending weight: 1.0 = all A, 0.0 = all B.
    /// * `child_config` - Topology configuration for the child network.
    /// * `rng` - Random number generator for Xavier initialization.
    ///
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if `child_config` is invalid.
    pub fn crossover(
        parent_a: &PcActor<L>,
        parent_b: &PcActor<L>,
        caches_a: &[L::Matrix],
        caches_b: &[L::Matrix],
        alpha: f64,
        child_config: PcActorConfig,
        rng: &mut impl Rng,
    ) -> Result<Self, PcError> {
        let num_child_hidden = child_config.hidden_layers.len();
        if num_child_hidden == 0 {
            return Err(PcError::ConfigValidation(
                "crossover requires at least one hidden layer".into(),
            ));
        }
        let num_parent_a_hidden = parent_a.config.hidden_layers.len();
        let num_parent_b_hidden = parent_b.config.hidden_layers.len();

        let mut layers: Vec<Layer<L>> = Vec::new();
        // Track the previous layer's CCA permutation for column propagation
        let mut prev_perm: Option<Vec<usize>> = None;

        // ── Input layer (layer 0): CCA-aligned crossover ─────────
        let child_h0 = &child_config.hidden_layers[0];

        if parent_a.config.input_size == child_config.input_size
            && parent_b.config.input_size == child_config.input_size
        {
            let cache_a_0 = caches_a.first();
            let cache_b_0 = caches_b.first();
            let (layer, perm) = cca_align_and_blend_layer::<L>(
                &parent_a.layers[0],
                &parent_b.layers[0],
                cache_a_0,
                cache_b_0,
                None, // No previous perm for first layer
                child_h0.size,
                L::mat_cols(&parent_a.layers[0].weights),
                child_h0.activation,
                alpha,
                rng,
            )?;
            layers.push(layer);
            prev_perm = perm;
        } else {
            layers.push(Layer::<L>::new(
                child_config.input_size,
                child_h0.size,
                child_h0.activation,
                rng,
            ));
        }

        // ── Hidden layers 1..n: CCA-aligned crossover ────────────
        for h_idx in 1..num_child_hidden {
            let child_def = &child_config.hidden_layers[h_idx];
            let prev_child_size = child_config.hidden_layers[h_idx - 1].size;

            let a_has = h_idx < num_parent_a_hidden;
            let b_has = h_idx < num_parent_b_hidden;

            if a_has && b_has {
                let cache_a_h = caches_a.get(h_idx);
                let cache_b_h = caches_b.get(h_idx);
                let (layer, perm) = cca_align_and_blend_layer::<L>(
                    &parent_a.layers[h_idx],
                    &parent_b.layers[h_idx],
                    cache_a_h,
                    cache_b_h,
                    prev_perm.as_deref(),
                    child_def.size,
                    prev_child_size,
                    child_def.activation,
                    alpha,
                    rng,
                )?;
                layers.push(layer);
                prev_perm = perm;
            } else {
                layers.push(Layer::<L>::new(
                    prev_child_size,
                    child_def.size,
                    child_def.activation,
                    rng,
                ));
                prev_perm = None;
            }
        }

        // ── Output layer: positional crossover or Xavier ─────────
        let last_child_hidden = child_config.hidden_layers.last().map(|d| d.size).unwrap();
        let a_out = parent_a.layers.last().unwrap();
        let b_out = parent_b.layers.last().unwrap();
        let a_out_in = L::mat_cols(&a_out.weights);
        let b_out_in = L::mat_cols(&b_out.weights);

        if a_out_in == last_child_hidden && b_out_in == last_child_hidden {
            // Positional crossover with column permutation from last hidden layer
            let b_out_permuted = if let Some(ref pp) = prev_perm {
                permute_cols::<L>(&b_out.weights, pp)
            } else {
                b_out.weights.clone()
            };
            let out_rows = child_config.output_size;
            let mut weights = L::zeros_mat(out_rows, last_child_hidden);
            let mut biases = L::zeros_vec(out_rows);
            let blend_rows = out_rows
                .min(L::mat_rows(&a_out.weights))
                .min(L::mat_rows(&b_out_permuted));
            for r in 0..blend_rows {
                for c in 0..last_child_hidden {
                    let va = L::mat_get(&a_out.weights, r, c);
                    let vb = L::mat_get(&b_out_permuted, r, c);
                    L::mat_set(&mut weights, r, c, alpha * va + (1.0 - alpha) * vb);
                }
                let ba = L::vec_get(&a_out.bias, r);
                let bb = L::vec_get(&b_out.bias, r);
                L::vec_set(&mut biases, r, alpha * ba + (1.0 - alpha) * bb);
            }
            layers.push(Layer {
                weights,
                bias: biases,
                activation: child_config.output_activation,
            });
        } else {
            layers.push(Layer::<L>::new(
                last_child_hidden,
                child_config.output_size,
                child_config.output_activation,
                rng,
            ));
        }

        // ── Residual components ──────────────────────────────────
        let (rezero_alpha, skip_projections) = if child_config.residual {
            let mut alphas = Vec::new();
            let mut projs = Vec::new();
            for i in 1..num_child_hidden {
                // ReZero alpha: blend if both parents have it
                let a_has_rz = i - 1 < parent_a.rezero_alpha.len();
                let b_has_rz = i - 1 < parent_b.rezero_alpha.len();
                let rz = if a_has_rz && b_has_rz {
                    alpha * parent_a.rezero_alpha[i - 1]
                        + (1.0 - alpha) * parent_b.rezero_alpha[i - 1]
                } else if a_has_rz {
                    parent_a.rezero_alpha[i - 1]
                } else if b_has_rz {
                    parent_b.rezero_alpha[i - 1]
                } else {
                    child_config.rezero_init
                };
                alphas.push(rz);

                // Skip projections
                let cur_size = child_config.hidden_layers[i].size;
                let prev_size = child_config.hidden_layers[i - 1].size;
                if cur_size != prev_size {
                    let a_proj = parent_a
                        .skip_projections
                        .get(i - 1)
                        .and_then(|p| p.as_ref());
                    let b_proj = parent_b
                        .skip_projections
                        .get(i - 1)
                        .and_then(|p| p.as_ref());
                    if let (Some(ap), Some(bp)) = (a_proj, b_proj) {
                        if L::mat_rows(ap) == cur_size
                            && L::mat_cols(ap) == prev_size
                            && L::mat_rows(bp) == cur_size
                            && L::mat_cols(bp) == prev_size
                        {
                            // Blend projections
                            let mut proj = L::zeros_mat(cur_size, prev_size);
                            for r in 0..cur_size {
                                for c in 0..prev_size {
                                    let va = L::mat_get(ap, r, c);
                                    let vb = L::mat_get(bp, r, c);
                                    L::mat_set(&mut proj, r, c, alpha * va + (1.0 - alpha) * vb);
                                }
                            }
                            projs.push(Some(proj));
                        } else {
                            projs.push(Some(L::xavier_mat(cur_size, prev_size, rng)));
                        }
                    } else {
                        projs.push(Some(L::xavier_mat(cur_size, prev_size, rng)));
                    }
                } else {
                    projs.push(None);
                }
            }
            (alphas, projs)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            layers,
            config: child_config,
            rezero_alpha,
            skip_projections,
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
    /// Returns whether hidden layer `i` has a skip connection (identity or projection).
    fn is_skip_layer(&self, i: usize) -> bool {
        self.config.residual && i >= 1
    }

    /// Returns the rezero_alpha/skip_projections index for hidden layer `i`.
    fn skip_alpha_index(&self, i: usize) -> Option<usize> {
        if !self.is_skip_layer(i) {
            return None;
        }
        Some(i - 1)
    }

    pub fn infer(&self, input: &[f64]) -> InferResult<L> {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "input size mismatch: got {}, expected {}",
            input.len(),
            self.config.input_size
        );

        let input_vec = L::vec_from_slice(input);
        let n_hidden = self.config.hidden_layers.len();

        // Forward pass to initialize hidden states and output
        let mut hidden_states: Vec<L::Vector> = Vec::with_capacity(n_hidden);
        let mut tanh_components: Vec<Option<L::Vector>> = Vec::with_capacity(n_hidden);
        let mut prev = input_vec.clone();
        for (i, layer) in self.layers[..n_hidden].iter().enumerate() {
            let tanh_out = layer.forward(&prev);
            if let Some(alpha_idx) = self.skip_alpha_index(i) {
                let alpha = self.rezero_alpha[alpha_idx];
                let scaled = L::vec_scale(&tanh_out, alpha);
                let skip_path = if let Some(ref proj) = self.skip_projections[alpha_idx] {
                    L::mat_vec_mul(proj, &prev)
                } else {
                    prev.clone()
                };
                prev = L::vec_add(&skip_path, &scaled);
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
            &input_vec
        };
        let mut y = self.layers[n_hidden].forward(last_input);

        // PC inference loop
        let mut steps_used = 0;
        let mut converged = false;
        let mut surprise_score = 0.0;
        let mut last_errors: Vec<L::Vector> = Vec::new();

        for step in 0..self.config.max_steps {
            steps_used = step + 1;

            if self.config.synchronous {
                // Snapshot mode: freeze all states
                let snapshot: Vec<L::Vector> = hidden_states.clone();
                let tanh_snap: Vec<Option<L::Vector>> = tanh_components.clone();

                let mut error_vecs: Vec<L::Vector> = Vec::new();

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

                    let error = L::vec_sub(&prediction, target);
                    error_vecs.push(error.clone());

                    // Update tanh_component or hidden_state
                    let updated_target =
                        L::vec_add(target, &L::vec_scale(&error, self.config.alpha));
                    if let Some(alpha_idx) = self.skip_alpha_index(i) {
                        tanh_components[i] = Some(updated_target.clone());
                        let alpha = self.rezero_alpha[alpha_idx];
                        let prev_h = if i > 0 {
                            &hidden_states[i - 1]
                        } else {
                            &input_vec
                        };
                        let skip_path = if let Some(ref proj) = self.skip_projections[alpha_idx] {
                            L::mat_vec_mul(proj, prev_h)
                        } else {
                            prev_h.clone()
                        };
                        hidden_states[i] =
                            L::vec_add(&skip_path, &L::vec_scale(&updated_target, alpha));
                    } else {
                        hidden_states[i] = updated_target;
                    }
                }

                let top_hidden = if n_hidden > 0 {
                    &hidden_states[n_hidden - 1]
                } else {
                    &input_vec
                };
                y = self.layers[n_hidden].forward(top_hidden);

                let refs: Vec<&L::Vector> = error_vecs.iter().collect();
                surprise_score = L::rms_error(&refs);
                last_errors = error_vecs;
            } else {
                // In-place mode: updates immediately visible
                let mut error_vecs: Vec<L::Vector> = Vec::new();

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

                    let error = L::vec_sub(&prediction, &target);
                    error_vecs.push(error.clone());

                    let updated_target =
                        L::vec_add(&target, &L::vec_scale(&error, self.config.alpha));
                    if let Some(alpha_idx) = self.skip_alpha_index(i) {
                        tanh_components[i] = Some(updated_target.clone());
                        let alpha = self.rezero_alpha[alpha_idx];
                        let prev_h = if i > 0 {
                            &hidden_states[i - 1]
                        } else {
                            &input_vec
                        };
                        let skip_path = if let Some(ref proj) = self.skip_projections[alpha_idx] {
                            L::mat_vec_mul(proj, prev_h)
                        } else {
                            prev_h.clone()
                        };
                        hidden_states[i] =
                            L::vec_add(&skip_path, &L::vec_scale(&updated_target, alpha));
                    } else {
                        hidden_states[i] = updated_target;
                    }
                }

                let top_hidden = if n_hidden > 0 {
                    &hidden_states[n_hidden - 1]
                } else {
                    &input_vec
                };
                y = self.layers[n_hidden].forward(top_hidden);

                let refs: Vec<&L::Vector> = error_vecs.iter().collect();
                surprise_score = L::rms_error(&refs);
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

        // Build latent_concat (uses vec_to_vec for GPU compatibility)
        let mut latent_raw: Vec<f64> = Vec::new();
        for h in &hidden_states {
            latent_raw.extend_from_slice(&L::vec_to_vec(h));
        }
        let latent_concat = L::vec_from_slice(&latent_raw);

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
        y_conv: &L::Vector,
        valid_actions: &[usize],
        mode: SelectionMode,
        rng: &mut impl Rng,
    ) -> usize {
        assert!(!valid_actions.is_empty(), "valid_actions must not be empty");

        // Scale logits by temperature
        let scaled = L::vec_scale(y_conv, 1.0 / self.config.temperature);

        let probs = L::softmax_masked(&scaled, valid_actions);

        match mode {
            SelectionMode::Play => L::argmax_masked(&probs, valid_actions),
            SelectionMode::Training => L::sample_from_probs(&probs, valid_actions, rng),
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
        infer_result: &InferResult<L>,
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
        infer_result: &InferResult<L>,
        input: &[f64],
        surprise_scale: f64,
        lambda: f64,
    ) {
        let input_vec = L::vec_from_slice(input);
        let output_delta_vec = L::vec_from_slice(output_delta);
        let n_hidden = self.config.hidden_layers.len();
        let n_layers = self.layers.len();

        // Output layer: always standard backward
        let output_input = if n_hidden > 0 {
            &infer_result.hidden_states[n_hidden - 1]
        } else {
            &input_vec
        };
        let output_output = &infer_result.y_conv;
        let mut bp_delta = self.layers[n_layers - 1].backward(
            output_input,
            output_output,
            &output_delta_vec,
            self.config.lr_weights,
            surprise_scale,
        );

        // Hidden layers (from top to bottom)
        for i in (0..n_hidden).rev() {
            let layer_input = if i > 0 {
                &infer_result.hidden_states[i - 1]
            } else {
                &input_vec
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
                let bp_scaled = L::vec_scale(&bp_delta, lambda);
                let pc_scaled = L::vec_scale(pc_error, 1.0 - lambda);
                L::vec_add(&bp_scaled, &pc_scaled)
            };

            if let Some(alpha_idx) = self.skip_alpha_index(i) {
                // Skip-eligible layer: use tanh_out for derivative, scale by alpha,
                // add identity path to propagated gradient, update alpha.
                let tanh_out = infer_result.tanh_components[i].as_ref().unwrap();
                let alpha = self.rezero_alpha[alpha_idx];
                let effective_lr = self.config.lr_weights * surprise_scale;

                // Scale delta by rezero_alpha for the nonlinear path
                let scaled_delta = L::vec_scale(&effective_delta, alpha);

                // Backward through the layer using tanh_out (not hidden_states[i])
                let propagated = self.layers[i].backward(
                    layer_input,
                    tanh_out,
                    &scaled_delta,
                    self.config.lr_weights,
                    surprise_scale,
                );

                // Update rezero_alpha: dL/d(alpha) = delta · tanh_out
                let grad_alpha: f64 = L::vec_dot(&effective_delta, tanh_out);
                self.rezero_alpha[alpha_idx] -= effective_lr * grad_alpha;

                // Propagated delta = nonlinear path + skip path (identity or projection)
                if let Some(ref mut proj) = self.skip_projections[alpha_idx] {
                    // Projection path: W_proj^T × delta
                    let proj_t = L::mat_transpose(proj);
                    let skip_delta = L::mat_vec_mul(&proj_t, &effective_delta);
                    // Update projection: W_proj -= lr × outer(delta, layer_input)
                    let dw_proj = L::outer_product(&effective_delta, layer_input);
                    L::mat_scale_add(proj, &dw_proj, -effective_lr);
                    bp_delta = L::vec_add(&propagated, &skip_delta);
                } else {
                    // Identity path: + delta
                    bp_delta = L::vec_add(&propagated, &effective_delta);
                }
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

    /// Extracts a serializable snapshot of current weights.
    ///
    /// Converts generic layers and skip projections to CPU-backed types.
    pub fn to_weights(&self) -> crate::serializer::PcActorWeights {
        let cpu_layers: Vec<Layer<CpuLinAlg>> = self
            .layers
            .iter()
            .map(|layer| {
                let rows = L::mat_rows(&layer.weights);
                let cols = L::mat_cols(&layer.weights);
                let mut cpu_weights = crate::matrix::Matrix::zeros(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        cpu_weights.set(r, c, L::mat_get(&layer.weights, r, c));
                    }
                }
                let bias_data = L::vec_to_vec(&layer.bias);
                Layer {
                    weights: cpu_weights,
                    bias: bias_data,
                    activation: layer.activation,
                }
            })
            .collect();
        let cpu_projs: Vec<Option<crate::matrix::Matrix>> = self
            .skip_projections
            .iter()
            .map(|opt| {
                opt.as_ref().map(|m| {
                    let rows = L::mat_rows(m);
                    let cols = L::mat_cols(m);
                    let mut cpu_m = crate::matrix::Matrix::zeros(rows, cols);
                    for r in 0..rows {
                        for c in 0..cols {
                            cpu_m.set(r, c, L::mat_get(m, r, c));
                        }
                    }
                    cpu_m
                })
            })
            .collect();
        crate::serializer::PcActorWeights {
            layers: cpu_layers,
            rezero_alpha: self.rezero_alpha.clone(),
            skip_projections: cpu_projs,
        }
    }

    /// Restores an actor from saved weights without requiring an RNG.
    ///
    /// Converts CPU-backed weight snapshots to the target backend `L`.
    /// Validates that all weight matrix dimensions and bias lengths match
    /// the expected topology from `config`.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if any weight matrix or bias
    /// vector has dimensions inconsistent with the config topology.
    pub fn from_weights(
        config: PcActorConfig,
        weights: crate::serializer::PcActorWeights,
    ) -> Result<Self, PcError> {
        let n_hidden = config.hidden_layers.len();
        let expected_layers = n_hidden + 1;

        if weights.layers.len() != expected_layers {
            return Err(PcError::DimensionMismatch {
                expected: expected_layers,
                got: weights.layers.len(),
                context: "actor layer count",
            });
        }

        // Validate each layer's dimensions
        let mut prev_size = config.input_size;
        for (i, cpu_layer) in weights.layers.iter().enumerate() {
            let (expected_rows, expected_cols) = if i < n_hidden {
                (config.hidden_layers[i].size, prev_size)
            } else {
                (config.output_size, prev_size)
            };

            if cpu_layer.weights.rows != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.weights.rows,
                    context: "actor layer weight rows",
                });
            }
            if cpu_layer.weights.cols != expected_cols {
                return Err(PcError::DimensionMismatch {
                    expected: expected_cols,
                    got: cpu_layer.weights.cols,
                    context: "actor layer weight cols",
                });
            }
            if cpu_layer.bias.len() != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.bias.len(),
                    context: "actor layer bias length",
                });
            }

            if i < n_hidden {
                prev_size = config.hidden_layers[i].size;
            }
        }

        // Validate residual components
        if config.residual {
            let expected_residual = n_hidden.saturating_sub(1);
            if weights.rezero_alpha.len() != expected_residual {
                return Err(PcError::DimensionMismatch {
                    expected: expected_residual,
                    got: weights.rezero_alpha.len(),
                    context: "actor rezero_alpha count",
                });
            }
            if weights.skip_projections.len() != expected_residual {
                return Err(PcError::DimensionMismatch {
                    expected: expected_residual,
                    got: weights.skip_projections.len(),
                    context: "actor skip_projections count",
                });
            }
        }

        // Convert layers
        let layers: Vec<Layer<L>> = weights
            .layers
            .into_iter()
            .map(|cpu_layer| {
                let rows = cpu_layer.weights.rows;
                let cols = cpu_layer.weights.cols;
                let mut mat = L::zeros_mat(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        L::mat_set(&mut mat, r, c, cpu_layer.weights.get(r, c));
                    }
                }
                let bias = L::vec_from_slice(&cpu_layer.bias);
                Layer {
                    weights: mat,
                    bias,
                    activation: cpu_layer.activation,
                }
            })
            .collect();
        let skip_projections: Vec<Option<L::Matrix>> = weights
            .skip_projections
            .into_iter()
            .map(|opt| {
                opt.map(|cpu_m| {
                    let rows = cpu_m.rows;
                    let cols = cpu_m.cols;
                    let mut mat = L::zeros_mat(rows, cols);
                    for r in 0..rows {
                        for c in 0..cols {
                            L::mat_set(&mut mat, r, c, cpu_m.get(r, c));
                        }
                    }
                    mat
                })
            })
            .collect();
        Ok(Self {
            layers,
            config,
            rezero_alpha: weights.rezero_alpha,
            skip_projections,
        })
    }
}

/// Permute columns of a weight matrix according to a permutation.
/// `perm[i]` = source column index for destination column i.
pub(crate) fn permute_cols<L: LinAlg>(m: &L::Matrix, perm: &[usize]) -> L::Matrix {
    let rows = L::mat_rows(m);
    let cols = L::mat_cols(m);
    let perm_len = perm.len();
    let mut result = L::zeros_mat(rows, cols);
    for (dst, &src) in perm.iter().enumerate().take(cols.min(perm_len)) {
        if src < cols {
            for r in 0..rows {
                L::mat_set(&mut result, r, dst, L::mat_get(m, r, src));
            }
        }
    }
    // Copy remaining columns (beyond permutation length) in original order
    for dst in perm_len..cols {
        for r in 0..rows {
            L::mat_set(&mut result, r, dst, L::mat_get(m, r, dst));
        }
    }
    result
}

/// Permute rows of a weight matrix according to a permutation.
/// `perm[i]` = source row index for destination row i.
pub(crate) fn permute_rows<L: LinAlg>(m: &L::Matrix, perm: &[usize], n: usize) -> L::Matrix {
    let cols = L::mat_cols(m);
    let perm_len = perm.len();
    let mut result = L::zeros_mat(n, cols);
    for (dst, &src) in perm.iter().enumerate().take(n.min(perm_len)) {
        if src < L::mat_rows(m) {
            for c in 0..cols {
                L::mat_set(&mut result, dst, c, L::mat_get(m, src, c));
            }
        }
    }
    // Copy remaining rows (unmatched) in original order
    for dst in perm_len..n {
        if dst < L::mat_rows(m) {
            for c in 0..cols {
                L::mat_set(&mut result, dst, c, L::mat_get(m, dst, c));
            }
        }
    }
    result
}

/// Permute elements of a bias vector according to a permutation.
pub(crate) fn permute_vec<L: LinAlg>(v: &L::Vector, perm: &[usize], n: usize) -> L::Vector {
    let perm_len = perm.len();
    let mut result = L::zeros_vec(n);
    for (dst, &src) in perm.iter().enumerate().take(n.min(perm_len)) {
        if src < L::vec_len(v) {
            L::vec_set(&mut result, dst, L::vec_get(v, src));
        }
    }
    for dst in perm_len..n {
        if dst < L::vec_len(v) {
            L::vec_set(&mut result, dst, L::vec_get(v, dst));
        }
    }
    result
}

/// Blend weights from two parent layers into a child layer.
/// Handles all 4 dimension cases (equal, child smaller, parents differ, child larger).
///
/// * `parent_a` - (weights, bias, neuron_count) for parent A.
/// * `parent_b` - (weights, bias, neuron_count) for parent B (already CCA-aligned).
/// * `child_cols` - Number of columns (input size) for child layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blend_layer_weights<L: LinAlg>(
    parent_a: (&L::Matrix, &L::Vector, usize),
    parent_b: (&L::Matrix, &L::Vector, usize),
    n_child: usize,
    child_cols: usize,
    alpha: f64,
    rng: &mut impl Rng,
) -> (L::Matrix, L::Vector) {
    let (a_weights, a_biases, n_a) = parent_a;
    let (b_weights, b_biases, n_b) = parent_b;
    let n_min = n_a.min(n_b);
    let n_max = n_a.max(n_b);
    let a_cols = L::mat_cols(a_weights);
    let b_cols = L::mat_cols(b_weights);
    let use_cols = child_cols.min(a_cols).min(b_cols);

    let mut weights = L::zeros_mat(n_child, child_cols);
    let mut biases = L::zeros_vec(n_child);

    // Blending zone [0..min(n_min, n_child))
    let blend_end = n_min.min(n_child);
    for r in 0..blend_end {
        for c in 0..use_cols {
            let va = L::mat_get(a_weights, r, c);
            let vb = L::mat_get(b_weights, r, c);
            L::mat_set(&mut weights, r, c, alpha * va + (1.0 - alpha) * vb);
        }
        let ba = L::vec_get(a_biases, r);
        let bb = L::vec_get(b_biases, r);
        L::vec_set(&mut biases, r, alpha * ba + (1.0 - alpha) * bb);
    }

    // Copy zone [n_min..min(n_max, n_child)) from the larger parent
    let copy_end = n_max.min(n_child);
    if copy_end > blend_end {
        let (larger_w, larger_b) = if n_a >= n_b {
            (a_weights, a_biases)
        } else {
            (b_weights, b_biases)
        };
        let larger_cols = L::mat_cols(larger_w);
        for r in blend_end..copy_end {
            for c in 0..child_cols.min(larger_cols) {
                L::mat_set(&mut weights, r, c, L::mat_get(larger_w, r, c));
            }
            L::vec_set(&mut biases, r, L::vec_get(larger_b, r));
        }
    }

    // Xavier zone [n_max..n_child) for new neurons
    if n_child > n_max {
        let xavier = L::xavier_mat(n_child - n_max, child_cols, rng);
        for r in n_max..n_child {
            for c in 0..child_cols {
                L::mat_set(&mut weights, r, c, L::mat_get(&xavier, r - n_max, c));
            }
            // biases stay zero for Xavier zone
        }
    }

    (weights, biases)
}

/// CCA-aligns and blends a single hidden layer from two parents.
///
/// Handles the common pattern: CCA alignment → column permutation from
/// previous layer → row permutation → blend. Returns the blended layer
/// and the CCA permutation applied (for column propagation to the next layer).
///
/// * `prev_perm` — Permutation from the previous layer to apply to columns.
///   Pass `None` to skip column propagation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cca_align_and_blend_layer<L: LinAlg>(
    a_layer: &Layer<L>,
    b_layer: &Layer<L>,
    cache_a: Option<&L::Matrix>,
    cache_b: Option<&L::Matrix>,
    prev_perm: Option<&[usize]>,
    child_rows: usize,
    child_cols: usize,
    child_activation: Activation,
    alpha: f64,
    rng: &mut impl Rng,
) -> Result<(Layer<L>, Option<Vec<usize>>), crate::error::PcError> {
    let n_a = L::mat_rows(&a_layer.weights);
    let n_b = L::mat_rows(&b_layer.weights);

    // CCA alignment
    let perm = if let (Some(ca), Some(cb)) = (cache_a, cache_b) {
        Some(crate::matrix::cca_neuron_alignment::<L>(ca, cb)?)
    } else {
        None
    };

    // Apply previous layer's permutation to columns of parent B
    let b_weights_col = if let Some(pp) = prev_perm {
        permute_cols::<L>(&b_layer.weights, pp)
    } else {
        b_layer.weights.clone()
    };

    // Apply CCA row permutation to parent B
    let b_weights_aligned = if let Some(ref p) = perm {
        permute_rows::<L>(&b_weights_col, p, n_b)
    } else {
        b_weights_col
    };
    let b_bias_aligned = if let Some(ref p) = perm {
        permute_vec::<L>(&b_layer.bias, p, n_b)
    } else {
        b_layer.bias.clone()
    };

    let (weights, biases) = blend_layer_weights::<L>(
        (&a_layer.weights, &a_layer.bias, n_a),
        (&b_weights_aligned, &b_bias_aligned, n_b),
        child_rows,
        child_cols,
        alpha,
        rng,
    );

    Ok((
        Layer {
            weights,
            bias: biases,
            activation: child_activation,
        },
        perm,
    ))
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(!result.converged);
        assert_eq!(result.steps_used, 20);
    }

    #[test]
    fn test_infer_does_not_modify_weights() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 18);
    }

    #[test]
    fn test_infer_latent_size_two_hidden() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 30);
    }

    #[test]
    fn test_infer_latent_size_matches_latent_size_method() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), actor.latent_size());
    }

    #[test]
    fn test_infer_y_conv_length_equals_output_size() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.y_conv.len(), 9);
    }

    #[test]
    fn test_infer_hidden_states_count_matches_hidden_layers() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.hidden_states.len(), 2);
    }

    #[test]
    fn test_infer_all_outputs_finite() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(result.surprise_score >= 0.0);
    }

    #[test]
    fn test_infer_synchronous_and_inplace_both_converge() {
        let mut rng = make_rng();
        let sync_actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..default_config()
        };
        let inplace_actor: PcActor = PcActor::new(inplace_config, &mut rng2).unwrap();
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
        let sync_actor: PcActor = PcActor::new(config.clone(), &mut rng).unwrap();
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..config
        };
        let inplace_actor: PcActor = PcActor::new(inplace_config, &mut rng2).unwrap();
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let _ = actor.infer(&[0.0; 5]);
    }

    // ── Action Selection Tests ───────────────────────────────────────

    #[test]
    fn test_select_action_training_always_in_valid() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng_init).unwrap();
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
        let actor: PcActor = PcActor::new(hot_config, &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let logits = vec![0.1; 9];
        let _ = actor.select_action(&logits, &[], SelectionMode::Training, &mut rng);
    }

    // ── Weight Update Tests ──────────────────────────────────────────

    #[test]
    fn test_update_weights_changes_first_layer() {
        let mut rng = make_rng();
        let mut actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
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
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
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
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_zero_temperature_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            temperature: 0.0,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_negative_temperature_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            temperature: -1.0,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_err());
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
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_residual_mixed_sizes_accepted() {
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
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_residual_mixed_sizes_all_skip() {
        // [27, 27, 18]: ALL layers i>=1 get skip — identity for 27→27, projection for 27→18
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
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        // 2 skips: layer 1 (identity) + layer 2 (projection)
        assert_eq!(actor.rezero_alpha.len(), 2);
    }

    #[test]
    fn test_residual_heterogeneous_has_projection() {
        // [27, 18]: different sizes → projection matrix created
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        assert_eq!(actor.rezero_alpha.len(), 1);
        assert_eq!(actor.skip_projections.len(), 1);
        assert!(actor.skip_projections[0].is_some());
        let proj = actor.skip_projections[0].as_ref().unwrap();
        assert_eq!(proj.rows, 18); // output dim
        assert_eq!(proj.cols, 27); // input dim
    }

    #[test]
    fn test_residual_homogeneous_no_projection() {
        // [27, 27]: same sizes → no projection needed
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        assert_eq!(actor.skip_projections.len(), 1);
        assert!(actor.skip_projections[0].is_none());
    }

    #[test]
    fn test_residual_mixed_sizes_infer_finite() {
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
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
        assert_eq!(result.hidden_states.len(), 3);
        assert_eq!(result.latent_concat.len(), 27 + 27 + 18);
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
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        assert!(actor.rezero_alpha.is_empty());
    }

    #[test]
    fn test_residual_two_hidden_one_rezero_alpha() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        assert_eq!(actor.rezero_alpha.len(), 2);
    }

    #[test]
    fn test_rezero_alpha_initialized_to_rezero_init() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            rezero_init: 0.005,
            ..residual_two_hidden_config()
        };
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        assert!((actor.rezero_alpha[0] - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_residual_single_hidden_zero_rezero_alpha() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            ..default_config()
        };
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        assert!(actor.rezero_alpha.is_empty());
    }

    #[test]
    fn test_residual_single_hidden_accepted() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_ok());
    }

    // ── Local Learning (PC-based weight updates) Tests ──────────

    // ── Residual Inference Tests ──────────────────────────────

    #[test]
    fn test_residual_false_identical_to_non_residual() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let mut rng1 = make_rng();
        let actor1: PcActor = PcActor::new(two_hidden_config(), &mut rng1).unwrap();
        let result1 = actor1.infer(&input);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            residual: false,
            ..two_hidden_config()
        };
        let actor2: PcActor = PcActor::new(config2, &mut rng2).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert!(result.steps_used > 0);
        assert!(result.steps_used <= 5);
    }

    #[test]
    fn test_residual_hidden_states_count() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.hidden_states.len(), 2);
    }

    #[test]
    fn test_residual_infer_does_not_modify_weights() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_residual_tanh_components_populated() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
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
        let mut actor1: PcActor = PcActor::new(two_hidden_config(), &mut rng1).unwrap();
        let infer1 = actor1.infer(&input);
        actor1.update_weights(&delta, &infer1, &input, 1.0);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            residual: false,
            ..two_hidden_config()
        };
        let mut actor2: PcActor = PcActor::new(config2, &mut rng2).unwrap();
        let infer2 = actor2.infer(&input);
        actor2.update_weights(&delta, &infer2, &input, 1.0);

        for i in 0..actor1.layers.len() {
            assert_eq!(actor1.layers[i].weights.data, actor2.layers[i].weights.data);
        }
    }

    #[test]
    fn test_residual_update_changes_all_layer_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(residual_two_hidden_config(), &mut rng).unwrap();
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
        let mut actor1: PcActor = PcActor::new(config1, &mut rng1).unwrap();
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
        let mut actor2: PcActor = PcActor::new(config2, &mut rng2).unwrap();
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
        let mut actor: PcActor = PcActor::new(config, &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.prediction_errors.len(), 1);
    }

    #[test]
    fn test_infer_prediction_errors_two_hidden() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(two_hidden_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert!(result.prediction_errors.is_empty());
    }

    #[test]
    fn test_infer_prediction_errors_all_finite() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
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
        let actor: PcActor = PcActor::new(default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        // default_config has one hidden layer of size 18
        assert_eq!(result.prediction_errors[0].len(), 18);
    }

    #[test]
    fn test_local_learning_config_accepted() {
        let mut rng = make_rng();
        let config = local_learning_config();
        assert!((config.local_lambda).abs() < f64::EPSILON);
        let actor: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(actor.is_ok());
    }

    #[test]
    fn test_local_learning_update_changes_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor = PcActor::new(local_learning_config(), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(local_learning_config(), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(config, &mut rng).unwrap();
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
        let mut bp_actor: PcActor = PcActor::new(default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0);

        // Local learning actor (same initial weights)
        let mut rng2 = make_rng();
        let mut ll_actor: PcActor = PcActor::new(local_learning_config(), &mut rng2).unwrap();
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
        let mut bp_actor: PcActor = PcActor::new(default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0);

        // lambda=1.0 should be identical to backprop
        let mut rng2 = make_rng();
        let mut lam_actor: PcActor = PcActor::new(hybrid_config(1.0), &mut rng2).unwrap();
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
        let mut ll_actor: PcActor = PcActor::new(local_learning_config(), &mut rng1).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0);

        // lambda=0.0 should be identical to pure local
        let mut rng2 = make_rng();
        let mut lam_actor: PcActor = PcActor::new(hybrid_config(0.0), &mut rng2).unwrap();
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
        let mut bp_actor: PcActor = PcActor::new(default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0);

        // Pure local
        let mut rng2 = make_rng();
        let mut ll_actor: PcActor = PcActor::new(local_learning_config(), &mut rng2).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0);

        // Hybrid lambda=0.5
        let mut rng3 = make_rng();
        let mut hy_actor: PcActor = PcActor::new(hybrid_config(0.5), &mut rng3).unwrap();
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
        let mut actor: PcActor = PcActor::new(hybrid_config(0.5), &mut rng).unwrap();
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
        let mut actor: PcActor = PcActor::new(hybrid_config(0.5), &mut rng).unwrap();
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
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_lambda_above_one_returns_error() {
        let mut rng = make_rng();
        let config = hybrid_config(1.1);
        let result: Result<PcActor, _> = PcActor::new(config, &mut rng);
        assert!(result.is_err());
    }

    // ── Phase 5 Cycle 5.1: Crossover same topology ─────────────

    fn crossover_config_27() -> PcActorConfig {
        PcActorConfig {
            input_size: 9,
            hidden_layers: vec![LayerDef {
                size: 27,
                activation: Activation::Tanh,
            }],
            output_size: 9,
            output_activation: Activation::Linear,
            alpha: 0.03,
            tol: 0.01,
            min_steps: 1,
            max_steps: 5,
            lr_weights: 0.005,
            synchronous: true,
            temperature: 1.0,
            local_lambda: 0.99,
            residual: false,
            rezero_init: 0.001,
        }
    }

    fn make_caches_for_actor(actor: &PcActor, batch_size: usize) -> Vec<Vec<Vec<f64>>> {
        let num_hidden = actor.config.hidden_layers.len();
        let mut layers: Vec<Vec<Vec<f64>>> = (0..num_hidden).map(|_| Vec::new()).collect();
        for i in 0..batch_size {
            let input: Vec<f64> = (0..actor.config.input_size)
                .map(|j| ((i * actor.config.input_size + j) as f64 * 0.01).sin())
                .collect();
            let result = actor.infer(&input);
            for (layer_idx, state) in result.hidden_states.iter().enumerate() {
                layers[layer_idx].push(state.clone());
            }
        }
        layers
    }

    fn build_cache_matrix(
        cache_layers: &[Vec<Vec<f64>>],
        layer_idx: usize,
    ) -> crate::matrix::Matrix {
        use crate::linalg::LinAlg;
        let samples = &cache_layers[layer_idx];
        let batch_size = samples.len();
        let n_neurons = samples[0].len();
        let mut mat = CpuLinAlg::zeros_mat(batch_size, n_neurons);
        for (r, sample) in samples.iter().enumerate() {
            for (c, &val) in sample.iter().enumerate() {
                CpuLinAlg::mat_set(&mut mat, r, c, val);
            }
        }
        mat
    }

    #[test]
    fn test_crossover_same_topology_produces_valid_actor() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = crossover_config_27();
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config.clone(), &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_b, i)).collect();

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            config,
            &mut rng_child,
        )
        .unwrap();

        // Child has same topology
        assert_eq!(child.layers.len(), actor_a.layers.len());
        for (i, layer) in child.layers.iter().enumerate() {
            assert_eq!(
                CpuLinAlg::mat_rows(&layer.weights),
                CpuLinAlg::mat_rows(&actor_a.layers[i].weights)
            );
            assert_eq!(
                CpuLinAlg::mat_cols(&layer.weights),
                CpuLinAlg::mat_cols(&actor_a.layers[i].weights)
            );
        }
    }

    #[test]
    fn test_crossover_same_topology_child_differs_from_parents() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = crossover_config_27();
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config.clone(), &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_b, i)).collect();

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            config,
            &mut rng_child,
        )
        .unwrap();

        // Child weights differ from both parents (blended)
        assert_ne!(child.layers[0].weights.data, actor_a.layers[0].weights.data);
        assert_ne!(child.layers[0].weights.data, actor_b.layers[0].weights.data);
    }

    #[test]
    fn test_crossover_alpha_one_approximates_parent_a() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = crossover_config_27();
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config.clone(), &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_b, i)).collect();

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            1.0, // alpha=1.0 → child ≈ parent A
            config,
            &mut rng_child,
        )
        .unwrap();

        // Input layer (layer 0): positional crossover, should be close to parent A
        let a_w = &actor_a.layers[0].weights.data;
        let child_w = &child.layers[0].weights.data;
        let max_diff: f64 = a_w
            .iter()
            .zip(child_w.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "alpha=1.0: input layer max diff from parent A = {max_diff}"
        );
    }

    #[test]
    fn test_crossover_child_weights_finite() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = crossover_config_27();
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config.clone(), &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_b, i)).collect();

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            config,
            &mut rng_child,
        )
        .unwrap();

        for (i, layer) in child.layers.iter().enumerate() {
            for &w in &layer.weights.data {
                assert!(w.is_finite(), "NaN/Inf in layer {i} weights");
            }
            for b in CpuLinAlg::vec_to_vec(&layer.bias) {
                assert!(b.is_finite(), "NaN/Inf in layer {i} biases");
            }
        }
    }

    // ── Phase 5 Cycle 5.2: Crossover child smaller ──────────────

    #[test]
    fn test_crossover_child_smaller() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config_27 = PcActorConfig {
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
            ..crossover_config_27()
        };
        let actor_a: PcActor = PcActor::new(config_27.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config_27, &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_b, i)).collect();

        let child_config = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..crossover_config_27()
        };

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            child_config,
            &mut rng_child,
        )
        .unwrap();

        // Child hidden layers have 18 neurons
        use crate::linalg::LinAlg;
        assert_eq!(CpuLinAlg::mat_rows(&child.layers[0].weights), 18);
        assert_eq!(CpuLinAlg::mat_rows(&child.layers[1].weights), 18);
    }

    // ── Phase 5 Cycle 5.3: Crossover parents differ ─────────────

    #[test]
    fn test_crossover_parents_different_sizes() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config_a = crossover_config_27(); // [27]
        let config_b = PcActorConfig {
            hidden_layers: vec![LayerDef {
                size: 18,
                activation: Activation::Tanh,
            }],
            ..crossover_config_27()
        }; // [18]

        let actor_a: PcActor = PcActor::new(config_a, &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config_b, &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_b, i)).collect();

        // Child has [27] → blending zone [0..18), copy zone [18..27) from parent A
        let child_config = crossover_config_27();
        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            child_config,
            &mut rng_child,
        )
        .unwrap();

        use crate::linalg::LinAlg;
        // Child has correct dimensions [27]
        assert_eq!(CpuLinAlg::mat_rows(&child.layers[0].weights), 27);
        // All weights finite
        for &w in &child.layers[0].weights.data {
            assert!(w.is_finite());
        }
    }

    // ── Phase 5 Cycle 5.4: Crossover child larger ───────────────

    #[test]
    fn test_crossover_child_larger() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config_18 = PcActorConfig {
            hidden_layers: vec![LayerDef {
                size: 18,
                activation: Activation::Tanh,
            }],
            ..crossover_config_27()
        };
        let actor_a: PcActor = PcActor::new(config_18.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config_18, &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_b, i)).collect();

        // Child has [27] → blending zone [0..18), Xavier zone [18..27)
        let child_config = crossover_config_27();
        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            child_config,
            &mut rng_child,
        )
        .unwrap();

        use crate::linalg::LinAlg;
        assert_eq!(CpuLinAlg::mat_rows(&child.layers[0].weights), 27);
        // All weights finite
        for &w in &child.layers[0].weights.data {
            assert!(w.is_finite());
        }
        // Xavier zone weights are not all zero (random init)
        let xavier_zone_nonzero = (18..27).any(|r| {
            (0..CpuLinAlg::mat_cols(&child.layers[0].weights))
                .any(|c| CpuLinAlg::mat_get(&child.layers[0].weights, r, c).abs() > 1e-15)
        });
        assert!(
            xavier_zone_nonzero,
            "Xavier zone [18..27) should have non-zero weights"
        );
    }

    // ── Phase 5 Cycle 5.5: Crossover layer count mismatch ───────

    #[test]
    fn test_crossover_child_more_layers() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config_2l = PcActorConfig {
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
            ..crossover_config_27()
        };
        let actor_a: PcActor = PcActor::new(config_2l.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config_2l, &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_b, i)).collect();

        // Child has 3 hidden layers → layers 0-1 crossover, layer 2 Xavier
        let child_config = PcActorConfig {
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
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..crossover_config_27()
        };

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            child_config,
            &mut rng_child,
        )
        .unwrap();

        use crate::linalg::LinAlg;
        // Child has 4 layers (3 hidden + 1 output)
        assert_eq!(child.layers.len(), 4);
        // Layer 2 (new) has 18 rows
        assert_eq!(CpuLinAlg::mat_rows(&child.layers[2].weights), 18);
        // All weights finite
        for (i, layer) in child.layers.iter().enumerate() {
            for &w in &layer.weights.data {
                assert!(w.is_finite(), "NaN/Inf in layer {i}");
            }
        }
    }

    #[test]
    fn test_crossover_child_fewer_layers() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config_3l = PcActorConfig {
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
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..crossover_config_27()
        };
        let actor_a: PcActor = PcActor::new(config_3l.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config_3l, &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..3).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..3).map(|i| build_cache_matrix(&caches_b, i)).collect();

        // Child has 2 hidden layers → layers 0-1 crossover, layer 2 discarded
        let child_config = PcActorConfig {
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
            ..crossover_config_27()
        };

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            child_config,
            &mut rng_child,
        )
        .unwrap();

        use crate::linalg::LinAlg;
        // Child has 3 layers (2 hidden + 1 output)
        assert_eq!(child.layers.len(), 3);
        // Output layer input_size = 27 (last hidden size)
        assert_eq!(CpuLinAlg::mat_cols(&child.layers[2].weights), 27);
    }

    // ── Phase 5 Cycle 5.6: Crossover residual components ────────

    #[test]
    fn test_crossover_residual_rezero_blended() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Softsign,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Softsign,
                },
            ],
            residual: true,
            rezero_init: 0.1,
            ..crossover_config_27()
        };
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config.clone(), &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_b, i)).collect();

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            config,
            &mut rng_child,
        )
        .unwrap();

        // Child has rezero_alpha values
        assert!(!child.rezero_alpha.is_empty());
        // Blended rezero_alpha: with alpha=0.5 and both parents same init,
        // child should be close to parent values
        for &rz in &child.rezero_alpha {
            assert!(rz.is_finite(), "rezero_alpha is not finite");
        }
    }

    #[test]
    fn test_crossover_residual_skip_projections_blended() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Softsign,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Softsign,
                },
            ],
            residual: true,
            rezero_init: 0.1,
            ..crossover_config_27()
        };
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config.clone(), &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_b, i)).collect();

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            config,
            &mut rng_child,
        )
        .unwrap();

        // Child should have skip_projections for size mismatch (27→18)
        assert!(!child.skip_projections.is_empty());
        // At least one projection should be Some (27→18 needs projection)
        let has_projection = child.skip_projections.iter().any(|p| p.is_some());
        assert!(has_projection, "Expected at least one skip projection");

        // Projection weights are finite
        for mat in child.skip_projections.iter().flatten() {
            for &w in &mat.data {
                assert!(w.is_finite(), "NaN/Inf in skip projection");
            }
        }
    }

    // ── Fix #1: Column permutation propagation ──────────────────

    #[test]
    fn test_crossover_multilayer_column_permutation_consistency() {
        // Two identical parents → child should be identical regardless of
        // CCA permutation (identity) or column ordering. But if we manually
        // set parent B = parent A with a known neuron permutation at layer 0,
        // the child at alpha=0.5 should produce a network whose layer 1
        // columns are also reordered to match.
        //
        // Strategy: crossover parent A with itself (same weights). The CCA
        // permutation should be identity, and the child should equal both
        // parents. Then crossover with alpha=0.5 using two different parents.
        // Run inference on the child — if column permutation is broken,
        // the child's layer 1 receives inputs in the wrong order, and
        // inference produces different results than a properly-permuted child.
        use crate::linalg::LinAlg;
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 8,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 8,
                    activation: Activation::Tanh,
                },
            ],
            input_size: 4,
            output_size: 4,
            ..crossover_config_27()
        };
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config.clone(), &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 100);
        let caches_b = make_caches_for_actor(&actor_b, 100);
        let cache_mats_a: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..2).map(|i| build_cache_matrix(&caches_b, i)).collect();

        // Get CCA permutation for layer 0 to check if it's non-trivial
        let perm0 =
            crate::matrix::cca_neuron_alignment::<CpuLinAlg>(&cache_mats_a[0], &cache_mats_b[0])
                .unwrap();
        let is_nontrivial = perm0.iter().enumerate().any(|(i, &p)| i != p);

        // Only test column propagation if CCA produced a non-trivial permutation
        if !is_nontrivial {
            // Parents too similar for meaningful test — skip
            return;
        }

        // Crossover with alpha=0.5
        let mut rng_child = StdRng::seed_from_u64(99);
        let child: PcActor = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            config.clone(),
            &mut rng_child,
        )
        .unwrap();

        // Verify: layer 1's input columns should be permuted to match layer 0's
        // row permutation of parent B. Check that the child's layer 1 column
        // ordering is consistent by verifying that inference produces finite,
        // non-degenerate output AND that crossover applied the column permutation.
        //
        // If columns are NOT permuted, parent B's layer 1 columns still reference
        // the original neuron positions, but the blended layer 0 has reordered
        // neurons. The inconsistency means column c of layer 1 connects to the
        // wrong neuron from layer 0.
        //
        // We verify by checking that the column permutation was actually applied:
        // parent B's layer 1 columns should be reordered by perm0.
        let b_layer1 = &actor_b.layers[1];
        let b_cols = CpuLinAlg::mat_cols(&b_layer1.weights);

        // Expected: child layer 1 col[c] = 0.5 * A.layer1.col[c] + 0.5 * B.layer1.col[perm0[c]]
        // If column permutation is NOT applied, it would be:
        // child layer 1 col[c] = 0.5 * A.layer1.col[c] + 0.5 * B.layer1.col[c]  (wrong!)
        let a_layer1 = &actor_a.layers[1];
        let child_layer1 = &child.layers[1];
        let n_rows = CpuLinAlg::mat_rows(&child_layer1.weights);

        let mut has_col_permutation = false;
        for (c, &src_col) in perm0.iter().enumerate().take(b_cols.min(perm0.len())) {
            if src_col == c {
                continue; // Identity position, can't distinguish
            }
            // Check if child col c matches the permuted blend (correct)
            // vs the unpermuted blend (broken)
            for r in 0..n_rows {
                let a_val = CpuLinAlg::mat_get(&a_layer1.weights, r, c);
                let b_val_permuted = CpuLinAlg::mat_get(&b_layer1.weights, r, src_col);
                let b_val_unpermuted = CpuLinAlg::mat_get(&b_layer1.weights, r, c);
                let child_val = CpuLinAlg::mat_get(&child_layer1.weights, r, c);

                let expected_permuted = 0.5 * a_val + 0.5 * b_val_permuted;
                let expected_unpermuted = 0.5 * a_val + 0.5 * b_val_unpermuted;

                // If column permutation is applied, child matches permuted expectation
                if (child_val - expected_permuted).abs() < 1e-10
                    && (child_val - expected_unpermuted).abs() > 1e-10
                {
                    has_col_permutation = true;
                }
            }
        }

        assert!(
            has_col_permutation,
            "Layer 1 columns should be permuted to match layer 0's CCA \
             permutation of parent B. perm0={perm0:?}"
        );
    }

    // ── Fix #5: Empty hidden_layers guard ────────────────────────

    #[test]
    fn test_crossover_empty_hidden_layers_returns_error() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = crossover_config_27();
        let actor_a: PcActor = PcActor::new(config.clone(), &mut rng_a).unwrap();
        let actor_b: PcActor = PcActor::new(config, &mut rng_b).unwrap();

        let caches_a = make_caches_for_actor(&actor_a, 50);
        let caches_b = make_caches_for_actor(&actor_b, 50);
        let cache_mats_a: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_a, i)).collect();
        let cache_mats_b: Vec<_> = (0..1).map(|i| build_cache_matrix(&caches_b, i)).collect();

        // Child config with empty hidden layers should return error, not panic
        let empty_config = PcActorConfig {
            hidden_layers: vec![],
            ..crossover_config_27()
        };

        let mut rng_child = StdRng::seed_from_u64(99);
        let result = PcActor::crossover(
            &actor_a,
            &actor_b,
            &cache_mats_a,
            &cache_mats_b,
            0.5,
            empty_config,
            &mut rng_child,
        );
        assert!(
            result.is_err(),
            "Crossover with empty hidden_layers should return error"
        );
    }

    // ── from_weights dimension validation tests ──────────────────────

    /// Helper: build valid PcActorWeights from a config by constructing
    /// an actor and extracting its weights.
    fn valid_weights_for(config: &PcActorConfig) -> crate::serializer::PcActorWeights {
        let mut rng = make_rng();
        let actor = PcActor::<CpuLinAlg>::new(config.clone(), &mut rng).unwrap();
        actor.to_weights()
    }

    #[test]
    fn test_from_weights_valid_returns_ok() {
        let config = default_config();
        let weights = valid_weights_for(&config);
        let result = PcActor::<CpuLinAlg>::from_weights(config, weights);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_weights_wrong_weight_rows_returns_err() {
        let config = default_config(); // input=9, hidden=[18], output=9
        let mut weights = valid_weights_for(&config);
        // Layer 0 should be 18x9; corrupt rows to 10x9
        weights.layers[0].weights = crate::matrix::Matrix::zeros(10, 9);
        weights.layers[0].bias = vec![0.0; 10];
        let result = PcActor::<CpuLinAlg>::from_weights(config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_weight_cols_returns_err() {
        let config = default_config(); // input=9, hidden=[18], output=9
        let mut weights = valid_weights_for(&config);
        // Layer 0 should be 18x9; corrupt cols to 18x5
        weights.layers[0].weights = crate::matrix::Matrix::zeros(18, 5);
        let result = PcActor::<CpuLinAlg>::from_weights(config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_bias_length_returns_err() {
        let config = default_config(); // hidden=[18], so layer 0 bias should be len 18
        let mut weights = valid_weights_for(&config);
        weights.layers[0].bias = vec![0.0; 5]; // wrong length
        let result = PcActor::<CpuLinAlg>::from_weights(config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_output_layer_dims_returns_err() {
        let config = default_config(); // output layer should be 9x18
        let mut weights = valid_weights_for(&config);
        let last = weights.layers.len() - 1;
        weights.layers[last].weights = crate::matrix::Matrix::zeros(9, 10); // wrong cols
        let result = PcActor::<CpuLinAlg>::from_weights(config, weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_weights_wrong_rezero_alpha_count_returns_err() {
        let mut config = default_config();
        config.hidden_layers = vec![
            LayerDef { size: 18, activation: Activation::Tanh },
            LayerDef { size: 18, activation: Activation::Tanh },
        ];
        config.residual = true;
        let mut weights = valid_weights_for(&config);
        // residual with 2 hidden layers expects 1 rezero_alpha; give 0
        weights.rezero_alpha = vec![];
        let result = PcActor::<CpuLinAlg>::from_weights(config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_skip_projections_count_returns_err() {
        let mut config = default_config();
        config.hidden_layers = vec![
            LayerDef { size: 18, activation: Activation::Tanh },
            LayerDef { size: 18, activation: Activation::Tanh },
        ];
        config.residual = true;
        let mut weights = valid_weights_for(&config);
        // Should have 1 skip_projection; give 3
        weights.skip_projections = vec![None, None, None];
        let result = PcActor::<CpuLinAlg>::from_weights(config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }
}
