// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Dense neural network layer.
//!
//! Provides forward propagation, transpose forward (PC top-down pass),
//! and backward propagation with gradient/weight clipping. Building
//! block for both [`crate::PcActor`] and [`crate::MlpCritic`].

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;
use crate::matrix::{GRAD_CLIP, WEIGHT_CLIP};

/// Definition of a layer's shape and activation, used for topology configuration.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
///
/// let def = LayerDef { size: 64, activation: Activation::Tanh };
/// assert_eq!(def.size, 64);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDef {
    /// Number of neurons in this layer.
    pub size: usize,
    /// Activation function applied after the linear transform.
    pub activation: Activation,
}

/// A single dense layer with weights, bias, and activation function.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`] for
/// backward compatibility.
///
/// Weights have shape `[output_size × input_size]`. Bias has length `output_size`.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::Layer;
/// use pc_rl_core::linalg::cpu::CpuLinAlg;
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
///
/// let backend = CpuLinAlg::new();
/// let mut rng = StdRng::seed_from_u64(42);
/// let layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
/// let output: Vec<f64> = layer.forward(&vec![1.0, 0.0, -1.0, 0.5]);
/// assert_eq!(output.len(), 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "L::Matrix: Serialize, L::Vector: Serialize",
    deserialize = "L::Matrix: for<'a> Deserialize<'a>, L::Vector: for<'a> Deserialize<'a>, L: Default"
))]
pub struct Layer<L: LinAlg = CpuLinAlg> {
    /// Weight matrix of shape `[output_size × input_size]`.
    pub weights: L::Matrix,
    /// Bias vector of length `output_size`.
    pub bias: L::Vector,
    /// Activation function applied element-wise after the linear transform.
    pub activation: Activation,
    /// Backend used for linear algebra operations.
    #[serde(skip, default)]
    pub(crate) backend: L,
}

impl<L: LinAlg> Layer<L> {
    /// Creates a new layer with Xavier-initialized weights and zero bias.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of inputs to this layer.
    /// * `output_size` - Number of neurons (outputs) in this layer.
    /// * `activation` - Activation function to apply after the linear transform.
    /// * `rng` - Random number generator for weight initialization.
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        backend: &L,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            weights: backend.xavier_mat(output_size, input_size, rng),
            bias: backend.zeros_vec(output_size),
            activation,
            backend: backend.clone(),
        }
    }

    /// Computes `activation(W * input + bias)`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != input_size` (number of columns in weights).
    pub fn forward(&self, input: &L::Vector) -> L::Vector {
        let linear = self.backend.mat_vec_mul(&self.weights, input);
        let biased = self.backend.vec_add(&linear, &self.bias);
        self.backend.apply_activation(&biased, self.activation)
    }

    /// Computes `activation(W^T * input)` (no bias).
    ///
    /// Used for PC top-down predictions. The `activation` parameter is
    /// separate from `self.activation` because at the output→last-hidden
    /// boundary, different activations may apply.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != output_size` (number of rows in weights).
    pub fn transpose_forward(&self, input: &L::Vector, activation: Activation) -> L::Vector {
        let wt = self.backend.mat_transpose(&self.weights);
        let linear = self.backend.mat_vec_mul(&wt, input);
        self.backend.apply_activation(&linear, activation)
    }

    /// Backpropagation with gradient and weight clipping.
    ///
    /// Returns the propagated delta for the layer below (length = input_size).
    ///
    /// # Arguments
    ///
    /// * `input` - Input that was fed to this layer during forward pass.
    /// * `output` - Output of this layer from the forward pass (post-activation).
    /// * `delta` - Error signal from the layer above.
    /// * `lr` - Base learning rate.
    /// * `surprise_scale` - Multiplier on `lr` based on surprise score.
    ///
    /// # Panics
    ///
    /// Panics on dimension mismatches.
    pub fn backward(
        &mut self,
        input: &L::Vector,
        output: &L::Vector,
        delta: &L::Vector,
        lr: f64,
        surprise_scale: f64,
    ) -> L::Vector {
        // 1. Activation derivative
        let deriv = self.backend.apply_derivative(output, self.activation);

        // 2. Local gradient = delta * deriv (element-wise Hadamard product)
        let mut grad = self.backend.vec_hadamard(delta, &deriv);

        // 3. Clip gradient
        self.backend.clip_vec(&mut grad, GRAD_CLIP);

        // 4. Effective learning rate
        let effective_lr = lr * surprise_scale;

        // 5. Weight gradient: dW = outer(grad, input)
        let dw = self.backend.outer_product(&grad, input);

        // 6. Update weights (scale_add includes WEIGHT_CLIP clamping)
        self.backend
            .mat_scale_add(&mut self.weights, &dw, -effective_lr);

        // 7. Update bias with clamping
        let bias_update = self.backend.vec_scale(&grad, effective_lr);
        let new_bias = self.backend.vec_sub(&self.bias, &bias_update);
        self.bias = new_bias;
        self.backend.clip_vec(&mut self.bias, WEIGHT_CLIP);

        // 8. Propagated delta: W^T * grad
        let wt = self.backend.mat_transpose(&self.weights);
        self.backend.mat_vec_mul(&wt, &grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn make_backend() -> CpuLinAlg {
        CpuLinAlg::new()
    }

    // ── forward tests ──────────────────────────────────────────────

    #[test]
    fn test_forward_output_length_equals_output_size() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Linear, &backend, &mut rng);
        let out = layer.forward(&vec![1.0, 0.0, -1.0, 0.5]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_forward_linear_known_value() {
        let mut rng = make_rng();
        let backend = make_backend();
        let mut layer: Layer = Layer::new(2, 1, Activation::Linear, &backend, &mut rng);
        // Set known weights and bias
        layer.weights.set(0, 0, 2.0);
        layer.weights.set(0, 1, 3.0);
        layer.bias[0] = 1.0;
        // output = 2*1 + 3*2 + 1 = 9
        let out = layer.forward(&vec![1.0, 2.0]);
        assert!((out[0] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_forward_tanh_output_bounded() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 5, Activation::Tanh, &backend, &mut rng);
        let out = layer.forward(&vec![10.0, -10.0, 5.0, -5.0]);
        for &v in &out {
            assert!(v > -1.0 && v < 1.0, "Tanh output {v} not in (-1,1)");
        }
    }

    #[test]
    fn test_forward_sigmoid_output_bounded() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 5, Activation::Sigmoid, &backend, &mut rng);
        let out = layer.forward(&vec![10.0, -10.0, 5.0, -5.0]);
        for &v in &out {
            assert!(v > 0.0 && v < 1.0, "Sigmoid output {v} not in (0,1)");
        }
    }

    #[test]
    fn test_forward_relu_no_negative_outputs() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 5, Activation::Relu, &backend, &mut rng);
        let out = layer.forward(&vec![10.0, -10.0, 5.0, -5.0]);
        for &v in &out {
            assert!(v >= 0.0, "ReLU output {v} is negative");
        }
    }

    #[test]
    fn test_forward_all_outputs_finite() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let out = layer.forward(&vec![1e6, -1e6, 1e3, -1e3]);
        for &v in &out {
            assert!(v.is_finite(), "Output {v} is not finite");
        }
    }

    #[test]
    #[should_panic]
    fn test_forward_panics_wrong_input_length() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Linear, &backend, &mut rng);
        let _ = layer.forward(&vec![1.0, 2.0]); // wrong length
    }

    // ── transpose_forward tests ────────────────────────────────────

    #[test]
    fn test_transpose_forward_output_length_equals_input_size() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        // transpose_forward takes output_size input, returns input_size
        let out = layer.transpose_forward(&vec![0.5, -0.5, 0.0], Activation::Tanh);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_transpose_forward_all_finite() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let out = layer.transpose_forward(&vec![1e3, -1e3, 0.0], Activation::Tanh);
        for &v in &out {
            assert!(v.is_finite(), "transpose_forward output {v} is not finite");
        }
    }

    #[test]
    fn test_transpose_forward_different_activation_changes_output() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let input = vec![0.5, -0.5, 0.3];
        let out_tanh = layer.transpose_forward(&input, Activation::Tanh);
        let out_linear = layer.transpose_forward(&input, Activation::Linear);
        // At least one element should differ
        let differs = out_tanh
            .iter()
            .zip(out_linear.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(
            differs,
            "Different activations should produce different outputs"
        );
    }

    #[test]
    #[should_panic]
    fn test_transpose_forward_panics_wrong_input_length() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let _ = layer.transpose_forward(&vec![0.5, -0.5], Activation::Tanh); // wrong length
    }

    // ── backward tests ─────────────────────────────────────────────

    #[test]
    fn test_backward_changes_weights() {
        let mut rng = make_rng();
        let backend = make_backend();
        let mut layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = layer.forward(&input);
        let delta = vec![0.1, -0.2, 0.3];
        let weights_before = layer.weights.clone();
        let _ = layer.backward(&input, &output, &delta, 0.01, 1.0);
        // At least one weight should change
        let changed = (0..3).any(|r| {
            (0..4).any(|c| (layer.weights.get(r, c) - weights_before.get(r, c)).abs() > 1e-15)
        });
        assert!(changed, "Weights should change after backward");
    }

    #[test]
    fn test_backward_changes_bias() {
        let mut rng = make_rng();
        let backend = make_backend();
        let mut layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = layer.forward(&input);
        let delta = vec![0.1, -0.2, 0.3];
        let bias_before = layer.bias.clone();
        let _ = layer.backward(&input, &output, &delta, 0.01, 1.0);
        let changed = layer
            .bias
            .iter()
            .zip(bias_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(changed, "Bias should change after backward");
    }

    #[test]
    fn test_backward_returns_delta_of_correct_length() {
        let mut rng = make_rng();
        let backend = make_backend();
        let mut layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = layer.forward(&input);
        let delta = vec![0.1, -0.2, 0.3];
        let prop_delta = layer.backward(&input, &output, &delta, 0.01, 1.0);
        assert_eq!(prop_delta.len(), 4);
    }

    #[test]
    fn test_backward_clips_weights_to_weight_clip() {
        let mut rng = make_rng();
        let backend = make_backend();
        let mut layer: Layer = Layer::new(4, 3, Activation::Linear, &backend, &mut rng);
        let input = vec![100.0, 100.0, 100.0, 100.0];
        let output = layer.forward(&input);
        let delta = vec![1e6, 1e6, 1e6];
        let _ = layer.backward(&input, &output, &delta, 1.0, 1.0);
        for r in 0..3 {
            for c in 0..4 {
                let w = layer.weights.get(r, c);
                assert!(
                    w.abs() <= WEIGHT_CLIP + 1e-12,
                    "Weight {w} exceeds WEIGHT_CLIP"
                );
            }
        }
        for &b in &layer.bias {
            assert!(
                b.abs() <= WEIGHT_CLIP + 1e-12,
                "Bias {b} exceeds WEIGHT_CLIP"
            );
        }
    }

    #[test]
    fn test_backward_returns_finite_delta() {
        let mut rng = make_rng();
        let backend = make_backend();
        let mut layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = layer.forward(&input);
        let delta = vec![0.1, -0.2, 0.3];
        let prop_delta = layer.backward(&input, &output, &delta, 0.01, 1.0);
        for &v in &prop_delta {
            assert!(v.is_finite(), "Propagated delta {v} is not finite");
        }
    }

    #[test]
    fn test_backward_zero_lr_does_not_change_weights() {
        let mut rng = make_rng();
        let backend = make_backend();
        let mut layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = layer.forward(&input);
        let delta = vec![0.1, -0.2, 0.3];
        let weights_before = layer.weights.clone();
        let bias_before = layer.bias.clone();
        let _ = layer.backward(&input, &output, &delta, 0.0, 1.0);
        for r in 0..3 {
            for c in 0..4 {
                assert!(
                    (layer.weights.get(r, c) - weights_before.get(r, c)).abs() < 1e-15,
                    "Weights changed with zero lr"
                );
            }
        }
        for (a, b) in layer.bias.iter().zip(bias_before.iter()) {
            assert!((a - b).abs() < 1e-15, "Bias changed with zero lr");
        }
    }

    // ── serde test ─────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip_preserves_weights_and_activation() {
        let mut rng = make_rng();
        let backend = make_backend();
        let layer: Layer = Layer::new(4, 3, Activation::Tanh, &backend, &mut rng);
        let json = serde_json::to_string(&layer).unwrap();
        let restored: Layer = serde_json::from_str(&json).unwrap();
        assert_eq!(layer.bias, restored.bias);
        assert_eq!(layer.activation, restored.activation);
        for r in 0..3 {
            for c in 0..4 {
                assert!(
                    (layer.weights.get(r, c) - restored.weights.get(r, c)).abs() < 1e-15,
                    "Weights not preserved in serde roundtrip"
                );
            }
        }
    }
}
