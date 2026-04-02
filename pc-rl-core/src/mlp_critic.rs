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
    /// Learning rate for weight updates.
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
/// let critic: MlpCritic = MlpCritic::new(config, &mut rng).unwrap();
/// let value = critic.forward(&vec![0.0; 27]);
/// assert!(value.is_finite());
/// ```
#[derive(Debug)]
pub struct MlpCritic<L: LinAlg = CpuLinAlg> {
    /// Dense layers: hidden layers followed by the output layer (1 neuron).
    pub(crate) layers: Vec<Layer<L>>,
    /// Configuration used to build this critic.
    pub config: MlpCriticConfig,
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
    pub fn new(config: MlpCriticConfig, rng: &mut impl Rng) -> Result<Self, PcError> {
        if config.input_size == 0 {
            return Err(PcError::ConfigValidation(
                "critic input_size must be > 0".into(),
            ));
        }

        let mut layers: Vec<Layer<L>> = Vec::with_capacity(config.hidden_layers.len() + 1);
        let mut prev_size = config.input_size;

        for def in &config.hidden_layers {
            layers.push(Layer::<L>::new(prev_size, def.size, def.activation, rng));
            prev_size = def.size;
        }

        // Output layer: 1 neuron
        layers.push(Layer::<L>::new(prev_size, 1, config.output_activation, rng));

        Ok(Self { layers, config })
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
        use crate::pc_actor::{blend_layer_weights, permute_rows, permute_vec};

        let num_child_hidden = child_config.hidden_layers.len();
        let num_a_hidden = parent_a.config.hidden_layers.len();
        let num_b_hidden = parent_b.config.hidden_layers.len();

        let mut layers: Vec<Layer<L>> = Vec::new();

        // Input layer (layer 0): CCA-aligned crossover
        let child_h0_size = child_config.hidden_layers[0].size;
        let child_h0_act = child_config.hidden_layers[0].activation;

        if parent_a.config.input_size == child_config.input_size
            && parent_b.config.input_size == child_config.input_size
        {
            let a_layer = &parent_a.layers[0];
            let b_layer = &parent_b.layers[0];
            let n_a = L::mat_rows(&a_layer.weights);
            let n_b = L::mat_rows(&b_layer.weights);

            let perm = if !caches_a.is_empty() && !caches_b.is_empty() {
                Some(crate::matrix::cca_neuron_alignment::<L>(
                    &caches_a[0],
                    &caches_b[0],
                ))
            } else {
                None
            };

            let b_weights_aligned = if let Some(ref p) = perm {
                permute_rows::<L>(&b_layer.weights, p, n_b)
            } else {
                b_layer.weights.clone()
            };
            let b_bias_aligned = if let Some(ref p) = perm {
                permute_vec::<L>(&b_layer.bias, p, n_b)
            } else {
                b_layer.bias.clone()
            };

            let (weights, biases) = blend_layer_weights::<L>(
                (&a_layer.weights, &a_layer.bias, n_a),
                (&b_weights_aligned, &b_bias_aligned, n_b),
                child_h0_size,
                L::mat_cols(&a_layer.weights),
                alpha,
                rng,
            );
            layers.push(Layer {
                weights,
                bias: biases,
                activation: child_h0_act,
            });
        } else {
            layers.push(Layer::<L>::new(
                child_config.input_size,
                child_h0_size,
                child_h0_act,
                rng,
            ));
        }

        // Hidden layers 1..n
        for h_idx in 1..num_child_hidden {
            let child_size = child_config.hidden_layers[h_idx].size;
            let child_act = child_config.hidden_layers[h_idx].activation;
            let prev_child_size = child_config.hidden_layers[h_idx - 1].size;

            if h_idx < num_a_hidden && h_idx < num_b_hidden {
                let a_layer = &parent_a.layers[h_idx];
                let b_layer = &parent_b.layers[h_idx];
                let n_a = L::mat_rows(&a_layer.weights);
                let n_b = L::mat_rows(&b_layer.weights);

                let perm = if h_idx < caches_a.len() && h_idx < caches_b.len() {
                    Some(crate::matrix::cca_neuron_alignment::<L>(
                        &caches_a[h_idx],
                        &caches_b[h_idx],
                    ))
                } else {
                    None
                };

                let b_w = if let Some(ref p) = perm {
                    permute_rows::<L>(&b_layer.weights, p, n_b)
                } else {
                    b_layer.weights.clone()
                };
                let b_b = if let Some(ref p) = perm {
                    permute_vec::<L>(&b_layer.bias, p, n_b)
                } else {
                    b_layer.bias.clone()
                };

                let (weights, biases) = blend_layer_weights::<L>(
                    (&a_layer.weights, &a_layer.bias, n_a),
                    (&b_w, &b_b, n_b),
                    child_size,
                    prev_child_size,
                    alpha,
                    rng,
                );
                layers.push(Layer {
                    weights,
                    bias: biases,
                    activation: child_act,
                });
            } else {
                layers.push(Layer::<L>::new(prev_child_size, child_size, child_act, rng));
            }
        }

        // Output layer (1 neuron): positional crossover or Xavier
        let last_child_hidden = child_config.hidden_layers.last().map(|d| d.size).unwrap();
        let a_out = parent_a.layers.last().unwrap();
        let b_out = parent_b.layers.last().unwrap();

        if L::mat_cols(&a_out.weights) == last_child_hidden
            && L::mat_cols(&b_out.weights) == last_child_hidden
        {
            let mut weights = L::zeros_mat(1, last_child_hidden);
            let mut biases = L::zeros_vec(1);
            for c in 0..last_child_hidden {
                let va = L::mat_get(&a_out.weights, 0, c);
                let vb = L::mat_get(&b_out.weights, 0, c);
                L::mat_set(&mut weights, 0, c, alpha * va + (1.0 - alpha) * vb);
            }
            let ba = L::vec_get(&a_out.bias, 0);
            let bb = L::vec_get(&b_out.bias, 0);
            L::vec_set(&mut biases, 0, alpha * ba + (1.0 - alpha) * bb);
            layers.push(Layer {
                weights,
                bias: biases,
                activation: child_config.output_activation,
            });
        } else {
            layers.push(Layer::<L>::new(
                last_child_hidden,
                1,
                child_config.output_activation,
                rng,
            ));
        }

        Ok(Self {
            layers,
            config: child_config,
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
        let mut current = L::vec_from_slice(input);
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        L::vec_get(&current, 0)
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

        let mut current = L::vec_from_slice(input);
        for layer in &self.layers {
            inputs.push(current.clone());
            current = layer.forward(&current);
            outputs.push(current.clone());
        }

        let predicted = L::vec_get(&current, 0);
        let error = target - predicted;
        let loss = error * error;

        // Output gradient: d(loss)/d(predicted) = -2*(target - predicted)
        let mut delta = L::vec_from_slice(&[-2.0 * error]);

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
                let rows = L::mat_rows(&layer.weights);
                let cols = L::mat_cols(&layer.weights);
                let mut cpu_weights = CpuLinAlg::zeros_mat(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        CpuLinAlg::mat_set(
                            &mut cpu_weights,
                            r,
                            c,
                            L::mat_get(&layer.weights, r, c),
                        );
                    }
                }
                let cpu_bias = L::vec_to_vec(&layer.bias);
                Layer {
                    weights: cpu_weights,
                    bias: cpu_bias,
                    activation: layer.activation,
                }
            })
            .collect();
        MlpCriticWeights { layers: cpu_layers }
    }

    /// Restores a critic from saved weights without requiring an RNG.
    ///
    /// Converts CPU layers to generic layers element-by-element for
    /// backend-agnostic restoration.
    ///
    /// # Arguments
    ///
    /// * `config` - Must match the topology used when weights were saved.
    /// * `weights` - Previously saved weight snapshot.
    pub fn from_weights(config: MlpCriticConfig, weights: MlpCriticWeights) -> Self {
        let generic_layers: Vec<Layer<L>> = weights
            .layers
            .into_iter()
            .map(|cpu_layer| {
                let rows = CpuLinAlg::mat_rows(&cpu_layer.weights);
                let cols = CpuLinAlg::mat_cols(&cpu_layer.weights);
                let mut generic_weights = L::zeros_mat(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        L::mat_set(
                            &mut generic_weights,
                            r,
                            c,
                            CpuLinAlg::mat_get(&cpu_layer.weights, r, c),
                        );
                    }
                }
                let generic_bias = L::vec_from_slice(&cpu_layer.bias);
                Layer {
                    weights: generic_weights,
                    bias: generic_bias,
                    activation: cpu_layer.activation,
                }
            })
            .collect();
        Self {
            layers: generic_layers,
            config,
        }
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
        let critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
        let input = vec![0.0; 27];
        let v = critic.forward(&input);
        assert!(v.is_finite(), "forward output {v} is not finite");
    }

    #[test]
    fn test_forward_different_inputs_give_different_outputs() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
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
        let critic: MlpCritic = MlpCritic::new(config, &mut rng).unwrap();
        let v = critic.forward(&vec![0.5; 27]);
        assert!(v.is_finite(), "Deep topology output {v} is not finite");
    }

    #[test]
    fn test_forward_extreme_input_still_finite() {
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
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
        let critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
        let _ = critic.forward(&[0.0; 10]); // wrong size
    }

    // ── update tests ───────────────────────────────────────────────

    #[test]
    fn test_update_loss_decreases_over_30_iterations() {
        let mut rng = make_rng();
        let mut critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
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
        let mut critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
        let loss = critic.update(&vec![0.0; 27], 1.0);
        assert!(loss.is_finite(), "Loss {loss} is not finite");
        assert!(loss >= 0.0, "Loss {loss} is negative");
    }

    #[test]
    fn test_update_changes_weights() {
        let mut rng = make_rng();
        let mut critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
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
        let mut critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
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
        let critic: MlpCritic = MlpCritic::new(default_config(), &mut rng).unwrap();
        let input = vec![0.3; 27];
        let original_output = critic.forward(&input);

        let weights = critic.to_weights();
        let restored: MlpCritic = MlpCritic::from_weights(default_config(), weights);
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
        let mut mat = CpuLinAlg::zeros_mat(batch_size, critic_input_size);
        for r in 0..batch_size {
            for c in 0..critic_input_size {
                let val = ((r * critic_input_size + c) as f64 * 0.037).sin();
                CpuLinAlg::mat_set(&mut mat, r, c, val);
            }
        }
        mat
    }

    #[test]
    fn test_critic_crossover_same_config_produces_valid() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(123);
        let config = default_config();
        let critic_a: MlpCritic = MlpCritic::new(config.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(config.clone(), &mut rng_b).unwrap();

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
        let critic_a: MlpCritic = MlpCritic::new(config.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(config.clone(), &mut rng_b).unwrap();

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
        let critic_a: MlpCritic = MlpCritic::new(config.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(config.clone(), &mut rng_b).unwrap();

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
        let critic_a: MlpCritic = MlpCritic::new(config_36.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(config_36, &mut rng_b).unwrap();

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
        assert_eq!(CpuLinAlg::mat_rows(&child.layers[0].weights), 48);
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
        let critic_a: MlpCritic = MlpCritic::new(config_48.clone(), &mut rng_a).unwrap();
        let critic_b: MlpCritic = MlpCritic::new(config_48, &mut rng_b).unwrap();

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
        assert_eq!(CpuLinAlg::mat_rows(&child.layers[0].weights), 36);
        let v = child.forward(&vec![0.3; 27]);
        assert!(v.is_finite());
    }
}
