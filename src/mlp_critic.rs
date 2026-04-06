// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Standard MLP critic (value function) for the PC Actor-Critic agent.
//!
//! Receives the concatenation of board state and actor latent representation,
//! outputs a scalar value estimate. Learns via MSE loss backpropagation.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::error::PcError;
use crate::layer::{Layer, LayerDef};
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;

/// Default critic learning rate.
fn default_critic_lr() -> f64 {
    0.005
}

/// Configuration for the MLP critic network.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
/// use pc_rl_core::mlp_critic::MlpCriticConfig;
///
/// let config = MlpCriticConfig {
///     input_size: 27,
///     hidden_layers: vec![LayerDef { size: 36, activation: Activation::Tanh }],
///     output_activation: Activation::Linear,
///     lr: 0.005,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpCriticConfig {
    /// Dimensionality of the input vector (board state + latent concat).
    pub input_size: usize,
    /// Hidden layer definitions (sizes and activations).
    pub hidden_layers: Vec<LayerDef>,
    /// Activation for the single-neuron output layer.
    pub output_activation: Activation,
    /// Learning rate for weight updates. Default: 0.005.
    #[serde(default = "default_critic_lr")]
    pub lr: f64,
}

/// Serializable snapshot of critic weights.
///
/// Used by the serializer module to persist and restore the critic
/// without requiring an RNG.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MlpCriticWeights {
    /// Layer weight snapshots in order (hidden layers + output layer).
    pub layers: Vec<Layer>,
}

/// Standard MLP value function (critic).
///
/// Estimates V(s) given the concatenation of board state and actor latent
/// activations. Trained via MSE loss backpropagation through dense layers.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`] for
/// backward compatibility.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
/// use pc_rl_core::mlp_critic::{MlpCritic, MlpCriticConfig};
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
///
/// let config = MlpCriticConfig {
///     input_size: 27,
///     hidden_layers: vec![LayerDef { size: 36, activation: Activation::Tanh }],
///     output_activation: Activation::Linear,
///     lr: 0.005,
/// };
/// let mut rng = StdRng::seed_from_u64(42);
/// let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config, &mut rng).unwrap();
/// let value = critic.forward(&vec![0.0; 27]);
/// assert!(value.is_finite());
/// ```
#[derive(Debug)]
pub struct MlpCritic<L: LinAlg = CpuLinAlg> {
    /// Dense layers: hidden layers followed by the output layer (1 neuron).
    pub(crate) layers: Vec<Layer<L>>,
    /// Configuration used to build this critic.
    pub config: MlpCriticConfig,
    /// Backend used for linear algebra operations.
    pub(crate) backend: L,
}

impl<L: LinAlg> MlpCritic<L> {
    /// Builds the layer chain from the given configuration.
    ///
    /// The output layer always has exactly 1 neuron with the configured
    /// `output_activation`.
    ///
    /// # Arguments
    ///
    /// * `config` - Critic topology and hyperparameters.
    /// * `rng` - Random number generator for Xavier weight initialization.
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if `input_size` is zero.
    pub fn new(backend: L, config: MlpCriticConfig, rng: &mut impl Rng) -> Result<Self, PcError> {
        if config.input_size == 0 {
            return Err(PcError::ConfigValidation(
                "critic input_size must be > 0".into(),
            ));
        }

        let mut layers: Vec<Layer<L>> = Vec::with_capacity(config.hidden_layers.len() + 1);
        let mut prev_size = config.input_size;

        for def in &config.hidden_layers {
            layers.push(Layer::<L>::new(prev_size, def.size, def.activation, &backend, rng));
            prev_size = def.size;
        }

        // Output layer: 1 neuron
        layers.push(Layer::<L>::new(prev_size, 1, config.output_activation, &backend, rng));

        Ok(Self { layers, config, backend })
    }

    /// Creates a child critic by crossing over two parent critics using CCA neuron alignment.
    ///
    /// Same logic as `PcActor::crossover` but simpler: no residual components.
    /// Input and output layers use positional crossover, hidden layers use CCA.
    ///
    /// # Arguments
    ///
    /// * `parent_a` - First parent critic.
    /// * `parent_b` - Second parent critic.
    /// * `caches_a` - Per-layer activation matrices for parent A.
    /// * `caches_b` - Per-layer activation matrices for parent B.
    /// * `alpha` - Blending weight: 1.0 = all A, 0.0 = all B.
    /// * `child_config` - Topology configuration for the child.
    /// * `rng` - Random number generator for Xavier initialization.
    ///
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if `child_config` is invalid.
    pub fn crossover(
        parent_a: &MlpCritic<L>,
        parent_b: &MlpCritic<L>,
        caches_a: &[L::Matrix],
        caches_b: &[L::Matrix],
        alpha: f64,
        child_config: MlpCriticConfig,
        rng: &mut impl Rng,
    ) -> Result<Self, PcError> {
        use crate::pc_actor::{cca_align_and_blend_layer, permute_cols};

        let num_child_hidden = child_config.hidden_layers.len();
        if num_child_hidden == 0 {
            return Err(PcError::ConfigValidation(
                "crossover requires at least one hidden layer".into(),
            ));
        }
        let num_a_hidden = parent_a.config.hidden_layers.len();
        let num_b_hidden = parent_b.config.hidden_layers.len();

        let mut layers: Vec<Layer<L>> = Vec::new();
        let mut prev_perm: Option<Vec<usize>> = None;

        // Input layer (layer 0): CCA-aligned crossover
        let child_h0 = &child_config.hidden_layers[0];

        if parent_a.config.input_size == child_config.input_size
            && parent_b.config.input_size == child_config.input_size
        {
            let (layer, perm) = cca_align_and_blend_layer(&parent_a.backend,
                &parent_a.layers[0],
                &parent_b.layers[0],
                caches_a.first(),
                caches_b.first(),
                None,
                child_h0.size,
                parent_a.backend.mat_cols(&parent_a.layers[0].weights),
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
                &parent_a.backend,
                rng,
            ));
        }

        // Hidden layers 1..n
        for h_idx in 1..num_child_hidden {
            let child_def = &child_config.hidden_layers[h_idx];
            let prev_child_size = child_config.hidden_layers[h_idx - 1].size;

            if h_idx < num_a_hidden && h_idx < num_b_hidden {
                let (layer, perm) = cca_align_and_blend_layer(&parent_a.backend,
                    &parent_a.layers[h_idx],
                    &parent_b.layers[h_idx],
                    caches_a.get(h_idx),
                    caches_b.get(h_idx),
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
                    &parent_a.backend,
                    rng,
                ));
                prev_perm = None;
            }
        }

        // Output layer (1 neuron): positional crossover with column propagation
        let last_child_hidden = child_config.hidden_layers.last().map(|d| d.size).unwrap();
        let a_out = parent_a.layers.last().unwrap();
        let b_out = parent_b.layers.last().unwrap();

        if parent_a.backend.mat_cols(&a_out.weights) == last_child_hidden
            && parent_a.backend.mat_cols(&b_out.weights) == last_child_hidden
        {
            let b_out_permuted = if let Some(ref pp) = prev_perm {
                permute_cols(&parent_a.backend, &b_out.weights, pp)
            } else {
                b_out.weights.clone()
            };
            let mut weights = parent_a.backend.zeros_mat(1, last_child_hidden);
            let mut biases = parent_a.backend.zeros_vec(1);
            for c in 0..last_child_hidden {
                let va = parent_a.backend.mat_get(&a_out.weights, 0, c);
                let vb = parent_a.backend.mat_get(&b_out_permuted, 0, c);
                parent_a.backend.mat_set(&mut weights, 0, c, alpha * va + (1.0 - alpha) * vb);
            }
            let ba = parent_a.backend.vec_get(&a_out.bias, 0);
            let bb = parent_a.backend.vec_get(&b_out.bias, 0);
            parent_a.backend.vec_set(&mut biases, 0, alpha * ba + (1.0 - alpha) * bb);
            layers.push(Layer {
                weights,
                bias: biases,
                activation: child_config.output_activation,
                backend: parent_a.backend.clone(),
            });
        } else {
            layers.push(Layer::<L>::new(
                last_child_hidden,
                1,
                child_config.output_activation,
                &parent_a.backend,
                rng,
            ));
        }

        Ok(Self {
            layers,
            config: child_config,
            backend: parent_a.backend.clone(),
        })
    }

    /// Computes the scalar value estimate V(s).
    ///
    /// Sequentially forwards through all layers and returns the single
    /// output neuron's activation.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    pub fn forward(&self, input: &[f64]) -> f64 {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "MlpCritic::forward: expected input size {}, got {}",
            self.config.input_size,
            input.len()
        );
        let mut current = self.backend.vec_from_slice(input);
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        self.backend.vec_get(&current, 0)
    }

    /// Computes V(s) and returns both the value and hidden layer activations.
    ///
    /// Identical to [`forward`](Self::forward) but also captures intermediate
    /// hidden layer activations for use in CCA neuron alignment during crossover.
    ///
    /// # Returns
    ///
    /// `(value, hidden_states)` where `hidden_states[i]` is the activation
    /// vector of hidden layer `i` (excludes the output layer).
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    pub fn forward_with_hidden(&self, input: &[f64]) -> (f64, Vec<L::Vector>) {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "MlpCritic::forward_with_hidden: expected input size {}, got {}",
            self.config.input_size,
            input.len()
        );
        let num_hidden = self.config.hidden_layers.len();
        let mut hidden_states = Vec::with_capacity(num_hidden);
        let mut current = self.backend.vec_from_slice(input);
        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current);
            if i < num_hidden {
                hidden_states.push(current.clone());
            }
        }
        (self.backend.vec_get(&current, 0), hidden_states)
    }

    /// Performs one MSE-loss update and returns the loss.
    ///
    /// 1. Forward pass storing each layer's input and output.
    /// 2. Loss = (target - predicted)^2.
    /// 3. Output gradient: delta = [-2.0 * (target - predicted)].
    /// 4. Backprop through layers in reverse via `layer.backward(...)`.
    /// 5. Returns loss.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated board state + latent activations.
    /// * `target` - Target value (e.g., discounted return).
    pub fn update(&mut self, input: &[f64], target: f64) -> f64 {
        // Forward pass, storing intermediate inputs and outputs
        let mut inputs: Vec<L::Vector> = Vec::with_capacity(self.layers.len());
        let mut outputs: Vec<L::Vector> = Vec::with_capacity(self.layers.len());

        let mut current = self.backend.vec_from_slice(input);
        for layer in &self.layers {
            inputs.push(current.clone());
            current = layer.forward(&current);
            outputs.push(current.clone());
        }

        let predicted = self.backend.vec_get(&current, 0);
        let error = target - predicted;
        let loss = error * error;

        // Output gradient: d(loss)/d(predicted) = -2*(target - predicted)
        let mut delta = self.backend.vec_from_slice(&[-2.0 * error]);

        // Backprop through layers in reverse
        for i in (0..self.layers.len()).rev() {
            delta = self.layers[i].backward(
                &inputs[i],
                &outputs[i],
                &delta,
                self.config.lr,
                1.0, // surprise_scale = 1.0 for critic
            );
        }

        loss
    }

    /// Extracts a serializable snapshot of current weights.
    ///
    /// Converts generic layers to CPU layers element-by-element for
    /// backend-agnostic serialization.
    pub fn to_weights(&self) -> MlpCriticWeights {
        let cpu_layers: Vec<Layer> = self
            .layers
            .iter()
            .map(|layer| {
                let rows = self.backend.mat_rows(&layer.weights);
                let cols = self.backend.mat_cols(&layer.weights);
                let cpu = CpuLinAlg::new();
                let mut cpu_weights = cpu.zeros_mat(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        cpu.mat_set(
                            &mut cpu_weights,
                            r,
                            c,
                            self.backend.mat_get(&layer.weights, r, c),
                        );
                    }
                }
                let cpu_bias = self.backend.vec_to_vec(&layer.bias);
                Layer {
                    weights: cpu_weights,
                    bias: cpu_bias,
                    activation: layer.activation,
                    backend: CpuLinAlg::new(),
                }
            })
            .collect();
        MlpCriticWeights { layers: cpu_layers }
    }

    /// Restores a critic from saved weights without requiring an RNG.
    ///
    /// Converts CPU layers to generic layers element-by-element for
    /// backend-agnostic restoration. Validates that all weight matrix
    /// dimensions and bias lengths match the expected topology.
    ///
    /// # Arguments
    ///
    /// * `config` - Must match the topology used when weights were saved.
    /// * `weights` - Previously saved weight snapshot.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if any weight matrix or bias
    /// vector has dimensions inconsistent with the config topology.
    pub fn from_weights(
        backend: L,
        config: MlpCriticConfig,
        weights: MlpCriticWeights,
    ) -> Result<Self, PcError> {
        let n_hidden = config.hidden_layers.len();
        let expected_layers = n_hidden + 1;

        if weights.layers.len() != expected_layers {
            return Err(PcError::DimensionMismatch {
                expected: expected_layers,
                got: weights.layers.len(),
                context: "critic layer count",
            });
        }

        let mut prev_size = config.input_size;
        for (i, cpu_layer) in weights.layers.iter().enumerate() {
            let (expected_rows, expected_cols) = if i < n_hidden {
                (config.hidden_layers[i].size, prev_size)
            } else {
                (1, prev_size) // output layer: 1 neuron
            };

            if cpu_layer.weights.rows != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.weights.rows,
                    context: "critic layer weight rows",
                });
            }
            if cpu_layer.weights.cols != expected_cols {
                return Err(PcError::DimensionMismatch {
                    expected: expected_cols,
                    got: cpu_layer.weights.cols,
                    context: "critic layer weight cols",
                });
            }
            if cpu_layer.bias.len() != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.bias.len(),
                    context: "critic layer bias length",
                });
            }

            if i < n_hidden {
                prev_size = config.hidden_layers[i].size;
            }
        }

        let generic_layers: Vec<Layer<L>> = weights
            .layers
            .into_iter()
            .map(|cpu_layer| {
                let cpu = CpuLinAlg::new();
                let rows = cpu.mat_rows(&cpu_layer.weights);
                let cols = cpu.mat_cols(&cpu_layer.weights);
                let mut generic_weights = backend.zeros_mat(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        backend.mat_set(
                            &mut generic_weights,
                            r,
                            c,
                            cpu.mat_get(&cpu_layer.weights, r, c),
                        );
                    }
                }
                let generic_bias = backend.vec_from_slice(&cpu_layer.bias);
                Layer {
                    weights: generic_weights,
                    bias: generic_bias,
                    activation: cpu_layer.activation,
                    backend: backend.clone(),
                }
            })
            .collect();
        Ok(Self {
            layers: generic_layers,
            config,
            backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::Activation;
    use crate::layer::LayerDef;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn default_config() -> MlpCriticConfig {
        MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![LayerDef {
                size: 36,
                activation: Activation::Tanh,
            }],
            output_activation: Activation::Linear,
            lr: 0.005,
        }
    }

    // ── forward tests ──────────────────────────────────────────────

    #[test]
    fn test_forward_returns_finite_scalar() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.0; 27];
        let v = critic.forward(&input);
        assert!(v.is_finite(), "forward output {v} is not finite");
    }

    #[test]
    fn test_forward_different_inputs_give_different_outputs() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let a = critic.forward(&vec![0.0; 27]);
        let mut input_b = vec![0.0; 27];
        input_b[0] = 1.0;
        input_b[5] = -1.0;
        let b = critic.forward(&input_b);
        assert!(
            (a - b).abs() > 1e-12,
            "Different inputs should give different outputs: {a} vs {b}"
        );
    }

    #[test]
    fn test_forward_deep_topology_returns_finite() {
        let mut rng = make_rng();
        let config = MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![
                LayerDef {
                    size: 36,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            output_activation: Activation::Linear,
            lr: 0.005,
        };
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let v = critic.forward(&vec![0.5; 27]);
        assert!(v.is_finite(), "Deep topology output {v} is not finite");
    }

    #[test]
    fn test_forward_extreme_input_still_finite() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input: Vec<f64> = (0..27)
            .map(|i| if i % 2 == 0 { 1e6 } else { -1e6 })
            .collect();
        let v = critic.forward(&input);
        assert!(v.is_finite(), "Extreme input output {v} is not finite");
    }

    #[test]
    #[should_panic]
    fn test_forward_panics_wrong_input_size() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let _ = critic.forward(&[0.0; 10]); // wrong size
    }

    // ── update tests ───────────────────────────────────────────────

    #[test]
    fn test_update_loss_decreases_over_30_iterations() {
        let mut rng = make_rng();
        let mut critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.1; 27];
        let target = 0.5;
        let initial_loss = critic.update(&input, target);
        let mut final_loss = initial_loss;
        for _ in 0..29 {
            final_loss = critic.update(&input, target);
        }
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={initial_loss}, final={final_loss}"
        );
    }

    #[test]
    fn test_update_returns_finite_nonneg_loss() {
        let mut rng = make_rng();
        let mut critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let loss = critic.update(&vec![0.0; 27], 1.0);
        assert!(loss.is_finite(), "Loss {loss} is not finite");
        assert!(loss >= 0.0, "Loss {loss} is negative");
    }

    #[test]
    fn test_update_changes_weights() {
        let mut rng = make_rng();
        let mut critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let w_before = critic.layers[0].weights.get(0, 0);
        let _ = critic.update(&vec![0.1; 27], 1.0);
        let w_after = critic.layers[0].weights.get(0, 0);
        assert!(
            (w_before - w_after).abs() > 1e-15,
            "Weights should change after update"
        );
    }

    #[test]
    fn test_update_clips_weights() {
        let mut rng = make_rng();
        let mut critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        // Extreme update to force clipping
        for _ in 0..100 {
            let _ = critic.update(&vec![10.0; 27], 1e6);
        }
        for layer in &critic.layers {
            for r in 0..layer.weights.rows {
                for c in 0..layer.weights.cols {
                    let w = layer.weights.get(r, c);
                    assert!(
                        w.abs() <= crate::matrix::WEIGHT_CLIP + 1e-12,
                        "Weight {w} exceeds WEIGHT_CLIP"
                    );
                }
            }
        }
    }

    // ── serde test ─────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip_preserves_weights() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.3; 27];
        let original_output = critic.forward(&input);

        let weights = critic.to_weights();
        let restored: MlpCritic = MlpCritic::from_weights(CpuLinAlg::new(), default_config(), weights).unwrap();
        let restored_output = restored.forward(&input);

        assert!(
            (original_output - restored_output).abs() < 1e-12,
            "Serde roundtrip changed output: {original_output} vs {restored_output}"
        );
    }

    // ── Phase 6 Cycle 6.1: MlpCritic crossover same topology ───

    fn make_critic_cache(critic_input_size: usize, batch_size: usize) -> crate::matrix::Matrix {
        use crate::linalg::LinAlg;
        // Dummy cache: just random-ish activations
        let backend = CpuLinAlg::new();
        let mut mat = backend.zeros_mat(batch_size, critic_input_size);
        for r in 0..batch_size {
            for c in 0..critic_input_size {
                let val = ((r * critic_input_size + c) as f64 * 0.037).sin();
                backend.mat_set(&mut mat, r, c, val);
            }
        }
        mat
    }

    #[test]
    fn test_critic_crossover_same_config_produces_valid() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = default_config();
        let critic_a: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config.clone(), &mut rng_b).unwrap();

        let cache_a = vec![make_critic_cache(36, 50)];
        let cache_b = vec![make_critic_cache(36, 50)];

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: MlpCritic = MlpCritic::crossover(
            &critic_a,
            &critic_b,
            &cache_a,
            &cache_b,
            0.5,
            config,
            &mut rng_child,
        )
        .unwrap();

        // Same layer count
        assert_eq!(child.layers.len(), critic_a.layers.len());
        // Forward produces finite output
        let input = vec![0.3; 27];
        let v = child.forward(&input);
        assert!(v.is_finite(), "child forward not finite: {v}");
    }

    #[test]
    fn test_critic_crossover_child_blended() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = default_config();
        let critic_a: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config.clone(), &mut rng_b).unwrap();

        let cache_a = vec![make_critic_cache(36, 50)];
        let cache_b = vec![make_critic_cache(36, 50)];

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: MlpCritic = MlpCritic::crossover(
            &critic_a,
            &critic_b,
            &cache_a,
            &cache_b,
            0.5,
            config,
            &mut rng_child,
        )
        .unwrap();

        // Weights differ from both parents
        assert_ne!(
            child.layers[0].weights.data,
            critic_a.layers[0].weights.data
        );
        assert_ne!(
            child.layers[0].weights.data,
            critic_b.layers[0].weights.data
        );
    }

    #[test]
    fn test_critic_crossover_alpha_one_equals_parent_a() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = default_config();
        let critic_a: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config.clone(), &mut rng_b).unwrap();

        let cache_a = vec![make_critic_cache(36, 50)];
        let cache_b = vec![make_critic_cache(36, 50)];

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: MlpCritic = MlpCritic::crossover(
            &critic_a,
            &critic_b,
            &cache_a,
            &cache_b,
            1.0, // alpha=1.0 → child ≈ parent A
            config,
            &mut rng_child,
        )
        .unwrap();

        // Input layer (positional) should match parent A exactly
        let max_diff: f64 = critic_a.layers[0]
            .weights
            .data
            .iter()
            .zip(child.layers[0].weights.data.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "alpha=1.0: input layer max diff = {max_diff}"
        );
    }

    // ── Phase 6 Cycle 6.2: MlpCritic crossover dimension mismatch ──

    #[test]
    fn test_critic_crossover_child_larger() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config_36 = default_config(); // hidden [36]
        let critic_a: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config_36.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config_36, &mut rng_b).unwrap();

        let cache_a = vec![make_critic_cache(36, 50)];
        let cache_b = vec![make_critic_cache(36, 50)];

        // Child has [48] hidden
        let child_config = MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![LayerDef {
                size: 48,
                activation: Activation::Tanh,
            }],
            output_activation: Activation::Linear,
            lr: 0.005,
        };

        let mut rng_child = StdRng::seed_from_u64(99);
        let child: MlpCritic = MlpCritic::crossover(
            &critic_a,
            &critic_b,
            &cache_a,
            &cache_b,
            0.5,
            child_config,
            &mut rng_child,
        )
        .unwrap();

        use crate::linalg::LinAlg;
        assert_eq!(CpuLinAlg::new().mat_rows(&child.layers[0].weights), 48);
        let v = child.forward(&vec![0.3; 27]);
        assert!(v.is_finite());
    }

    #[test]
    fn test_critic_crossover_child_smaller() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config_48 = MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![LayerDef {
                size: 48,
                activation: Activation::Tanh,
            }],
            output_activation: Activation::Linear,
            lr: 0.005,
        };
        let critic_a: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config_48.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config_48, &mut rng_b).unwrap();

        let cache_a = vec![make_critic_cache(48, 50)];
        let cache_b = vec![make_critic_cache(48, 50)];

        // Child has [36] hidden → truncation
        let child_config = default_config();
        let mut rng_child = StdRng::seed_from_u64(99);
        let child: MlpCritic = MlpCritic::crossover(
            &critic_a,
            &critic_b,
            &cache_a,
            &cache_b,
            0.5,
            child_config,
            &mut rng_child,
        )
        .unwrap();

        use crate::linalg::LinAlg;
        assert_eq!(CpuLinAlg::new().mat_rows(&child.layers[0].weights), 36);
        let v = child.forward(&vec![0.3; 27]);
        assert!(v.is_finite());
    }

    // ── Fix #5: Empty hidden_layers guard ────────────────────────

    // ── Fix #2: forward_with_hidden ────────────────────────────

    #[test]
    fn test_forward_with_hidden_returns_value_and_states() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.3; 27];
        let (value, hidden_states) = critic.forward_with_hidden(&input);

        assert!(value.is_finite(), "value not finite: {value}");
        // 1 hidden layer → 1 entry in hidden_states
        assert_eq!(hidden_states.len(), 1);
        // Hidden layer has 36 neurons
        assert_eq!(hidden_states[0].len(), 36);
    }

    #[test]
    fn test_forward_with_hidden_matches_forward() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.3; 27];
        let value_plain = critic.forward(&input);
        let (value_hidden, _) = critic.forward_with_hidden(&input);

        assert!(
            (value_plain - value_hidden).abs() < 1e-12,
            "forward and forward_with_hidden should return same value: {value_plain} vs {value_hidden}"
        );
    }

    #[test]
    fn test_forward_with_hidden_two_layers() {
        let config = MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![
                LayerDef {
                    size: 36,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 24,
                    activation: Activation::Tanh,
                },
            ],
            output_activation: Activation::Linear,
            lr: 0.005,
        };
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let input = vec![0.3; 27];
        let (value, hidden_states) = critic.forward_with_hidden(&input);

        assert!(value.is_finite());
        assert_eq!(hidden_states.len(), 2);
        assert_eq!(hidden_states[0].len(), 36);
        assert_eq!(hidden_states[1].len(), 24);
    }

    // ── Fix #5: Empty hidden_layers guard ────────────────────────

    #[test]
    fn test_critic_crossover_empty_hidden_layers_returns_error() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = default_config();
        let critic_a: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config, &mut rng_b).unwrap();

        let cache_a = vec![make_critic_cache(36, 50)];
        let cache_b = vec![make_critic_cache(36, 50)];

        let empty_config = MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![],
            output_activation: Activation::Linear,
            lr: 0.005,
        };

        let mut rng_child = StdRng::seed_from_u64(99);
        let result = MlpCritic::crossover(
            &critic_a,
            &critic_b,
            &cache_a,
            &cache_b,
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

    /// Helper: build valid MlpCriticWeights from a config.
    fn valid_weights_for(config: &MlpCriticConfig) -> MlpCriticWeights {
        let mut rng = make_rng();
        let critic = MlpCritic::<CpuLinAlg>::new(CpuLinAlg::new(), config.clone(), &mut rng).unwrap();
        critic.to_weights()
    }

    #[test]
    fn test_from_weights_valid_returns_ok() {
        let config = default_config();
        let weights = valid_weights_for(&config);
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_weights_wrong_weight_rows_returns_err() {
        let config = default_config(); // input=27, hidden=[36], output=1
        let mut weights = valid_weights_for(&config);
        // Layer 0 should be 36x27; corrupt rows to 20x27
        weights.layers[0].weights = crate::matrix::Matrix::zeros(20, 27);
        weights.layers[0].bias = vec![0.0; 20];
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_weight_cols_returns_err() {
        let config = default_config(); // layer 0 should be 36x27
        let mut weights = valid_weights_for(&config);
        weights.layers[0].weights = crate::matrix::Matrix::zeros(36, 10); // wrong cols
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_bias_length_returns_err() {
        let config = default_config(); // layer 0 bias should be len 36
        let mut weights = valid_weights_for(&config);
        weights.layers[0].bias = vec![0.0; 5];
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_output_layer_dims_returns_err() {
        let config = default_config(); // output layer should be 1x36
        let mut weights = valid_weights_for(&config);
        let last = weights.layers.len() - 1;
        weights.layers[last].weights = crate::matrix::Matrix::zeros(1, 10); // wrong cols
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
    }
}
