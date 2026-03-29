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
/// use pc_core::activation::Activation;
/// use pc_core::layer::LayerDef;
/// use pc_core::mlp_critic::MlpCriticConfig;
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
/// use pc_core::activation::Activation;
/// use pc_core::layer::LayerDef;
/// use pc_core::mlp_critic::{MlpCritic, MlpCriticConfig};
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
}
