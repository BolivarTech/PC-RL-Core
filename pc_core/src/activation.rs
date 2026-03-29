// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Activation functions for neural network layers.
//!
//! Provides an enum of common activation functions with element-wise
//! `apply` and `derivative` operations. Used by layers, the PC actor,
//! and the MLP critic.

use serde::{Deserialize, Serialize};

/// Supported activation function variants.
///
/// Each variant implements `apply(x)` for the forward pass and
/// `derivative(fx)` which takes the **post-activation** value.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
///
/// let act = Activation::Tanh;
/// let y = act.apply(0.5);
/// let dy = act.derivative(y);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// Hyperbolic tangent: output in (-1, 1).
    Tanh,
    /// Rectified linear unit: max(0, x).
    Relu,
    /// Logistic sigmoid: output in (0, 1).
    Sigmoid,
    /// Exponential linear unit: smooth in negatives, avoids dying neurons.
    Elu,
    /// Softsign: bounded in (-1, 1) with slower saturation than tanh.
    /// Preserves more gradient in high-saturation regions.
    Softsign,
    /// Identity function: output equals input.
    Linear,
}

impl Activation {
    /// Applies the activation function to a single scalar value.
    ///
    /// # Parameters
    ///
    /// * `x` - Pre-activation input value.
    ///
    /// # Returns
    ///
    /// The activated output value.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Activation::Softsign => x / (1.0 + x.abs()),
            Activation::Linear => x,
        }
    }

    /// Computes the derivative given the post-activation value.
    ///
    /// # Parameters
    ///
    /// * `fx` - The post-activation value (output of `apply`).
    ///
    /// # Returns
    ///
    /// The derivative at `fx`.
    pub fn derivative(&self, fx: f64) -> f64 {
        match self {
            Activation::Tanh => 1.0 - fx * fx,
            Activation::Relu => {
                if fx > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => fx * (1.0 - fx),
            Activation::Elu => {
                if fx > 0.0 {
                    1.0
                } else {
                    fx + 1.0
                }
            }
            Activation::Softsign => {
                // fx = x/(1+|x|), so (1-|fx|) = 1/(1+|x|), derivative = (1-|fx|)^2
                let t = 1.0 - fx.abs();
                t * t
            }
            Activation::Linear => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── apply tests ──────────────────────────────────────────────

    #[test]
    fn test_tanh_apply_zero() {
        assert_eq!(Activation::Tanh.apply(0.0), 0.0);
    }

    #[test]
    fn test_tanh_apply_known() {
        let expected = 1.0_f64.tanh();
        let result = Activation::Tanh.apply(1.0);
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_tanh_apply_negative() {
        let expected = (-2.0_f64).tanh();
        let result = Activation::Tanh.apply(-2.0);
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_relu_apply_negative_is_zero() {
        assert_eq!(Activation::Relu.apply(-5.0), 0.0);
    }

    #[test]
    fn test_relu_apply_zero_is_zero() {
        assert_eq!(Activation::Relu.apply(0.0), 0.0);
    }

    #[test]
    fn test_relu_apply_positive_is_identity() {
        assert_eq!(Activation::Relu.apply(3.7), 3.7);
    }

    #[test]
    fn test_sigmoid_apply_zero_is_half() {
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_sigmoid_apply_large_stays_below_one() {
        // At x=30, exp(-30) ≈ 9.4e-14 which is representable in f64.
        // At x=100, f64 rounds sigmoid to exactly 1.0.
        let result = Activation::Sigmoid.apply(30.0);
        assert!(result < 1.0);
        assert!(result > 0.99);
    }

    #[test]
    fn test_sigmoid_apply_very_negative_stays_above_zero() {
        let result = Activation::Sigmoid.apply(-100.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_elu_apply_positive_is_identity() {
        assert_eq!(Activation::Elu.apply(3.0), 3.0);
    }

    #[test]
    fn test_elu_apply_zero_is_zero() {
        assert!((Activation::Elu.apply(0.0)).abs() < 1e-12);
    }

    #[test]
    fn test_elu_apply_negative_is_exp_minus_one() {
        let expected = (-1.0_f64).exp() - 1.0;
        let result = Activation::Elu.apply(-1.0);
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_elu_apply_large_negative_approaches_minus_one() {
        let result = Activation::Elu.apply(-100.0);
        assert!((result - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_softsign_apply_positive() {
        // softsign(2.0) = 2.0 / (1 + 2.0) = 2/3
        let result = Activation::Softsign.apply(2.0);
        assert!((result - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_apply_zero() {
        assert!((Activation::Softsign.apply(0.0)).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_apply_negative() {
        // softsign(-3.0) = -3.0 / (1 + 3.0) = -0.75
        let result = Activation::Softsign.apply(-3.0);
        assert!((result - (-0.75)).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_apply_bounded() {
        // Output must be in (-1, 1) for any input
        assert!(Activation::Softsign.apply(100.0) < 1.0);
        assert!(Activation::Softsign.apply(-100.0) > -1.0);
    }

    #[test]
    fn test_linear_apply_is_identity() {
        assert_eq!(Activation::Linear.apply(42.0), 42.0);
    }

    // ── derivative tests ─────────────────────────────────────────

    #[test]
    fn test_tanh_derivative_formula() {
        // derivative(fx) = 1 - fx^2, for fx = 0.5 => 0.75
        let result = Activation::Tanh.derivative(0.5);
        assert!((result - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_tanh_derivative_at_zero_is_one() {
        assert!((Activation::Tanh.derivative(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_relu_derivative_zero_output_is_zero() {
        assert_eq!(Activation::Relu.derivative(0.0), 0.0);
    }

    #[test]
    fn test_relu_derivative_positive_output_is_one() {
        assert_eq!(Activation::Relu.derivative(2.0), 1.0);
    }

    #[test]
    fn test_sigmoid_derivative_formula() {
        // derivative(fx) = fx * (1 - fx), for fx = 0.7 => 0.21
        let result = Activation::Sigmoid.derivative(0.7);
        assert!((result - 0.21).abs() < 1e-12);
    }

    #[test]
    fn test_sigmoid_derivative_at_half() {
        // derivative(0.5) = 0.5 * 0.5 = 0.25
        assert!((Activation::Sigmoid.derivative(0.5) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_elu_derivative_positive_is_one() {
        assert_eq!(Activation::Elu.derivative(2.0), 1.0);
    }

    #[test]
    fn test_elu_derivative_negative_is_fx_plus_one() {
        // fx = -0.6, derivative = -0.6 + 1.0 = 0.4
        let result = Activation::Elu.derivative(-0.6);
        assert!((result - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_elu_derivative_at_minus_one_is_zero() {
        // ELU floor is -1.0, derivative there = -1.0 + 1.0 = 0.0
        assert!((Activation::Elu.derivative(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_derivative_at_zero() {
        // derivative(softsign(0)) = 1 / (1 + 0)^2 = 1.0
        assert!((Activation::Softsign.derivative(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_derivative_positive() {
        // softsign(2) = 2/3 ≈ 0.6667, |x| = 2, derivative = 1/(1+2)^2 = 1/9
        // But derivative takes fx (post-activation), so we need to recover |x|
        // fx = x/(1+|x|), so |x| = |fx|/(1-|fx|)
        // For fx=0.5: |x| = 0.5/0.5 = 1.0, derivative = 1/(1+1)^2 = 0.25
        let result = Activation::Softsign.derivative(0.5);
        assert!((result - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_derivative_negative() {
        // For fx=-0.5: |x| = 0.5/0.5 = 1.0, derivative = 1/(1+1)^2 = 0.25
        let result = Activation::Softsign.derivative(-0.5);
        assert!((result - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_derivative_high_saturation() {
        // For fx=0.9: |x| = 0.9/0.1 = 9, derivative = 1/(1+9)^2 = 0.01
        let result = Activation::Softsign.derivative(0.9);
        assert!((result - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_softsign_derivative_always_positive() {
        for &fx in &[-0.9, -0.5, 0.0, 0.5, 0.9] {
            assert!(Activation::Softsign.derivative(fx) > 0.0);
        }
    }

    #[test]
    fn test_linear_derivative_always_one() {
        assert_eq!(Activation::Linear.derivative(999.0), 1.0);
        assert_eq!(Activation::Linear.derivative(-42.0), 1.0);
        assert_eq!(Activation::Linear.derivative(0.0), 1.0);
    }

    // ── robustness tests ─────────────────────────────────────────

    #[test]
    fn test_all_activations_produce_finite_output_for_extreme_inputs() {
        let variants = [
            Activation::Tanh,
            Activation::Relu,
            Activation::Sigmoid,
            Activation::Elu,
            Activation::Softsign,
            Activation::Linear,
        ];
        for act in &variants {
            for &x in &[-100.0, 100.0] {
                let y = act.apply(x);
                assert!(y.is_finite(), "{:?}.apply({}) was not finite", act, x);
            }
        }
    }

    #[test]
    fn test_all_derivatives_finite_for_typical_post_activation_values() {
        let cases: [(Activation, f64); 6] = [
            (Activation::Tanh, 0.5),
            (Activation::Relu, 1.0),
            (Activation::Sigmoid, 0.5),
            (Activation::Elu, -0.5),
            (Activation::Softsign, 0.5),
            (Activation::Linear, 0.0),
        ];
        for (act, fx) in &cases {
            let d = act.derivative(*fx);
            assert!(d.is_finite(), "{:?}.derivative({}) was not finite", act, fx);
        }
    }

    // ── serde tests ──────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip_all_variants() {
        let variants = [
            Activation::Tanh,
            Activation::Relu,
            Activation::Sigmoid,
            Activation::Elu,
            Activation::Softsign,
            Activation::Linear,
        ];
        for act in &variants {
            let json = serde_json::to_string(act).unwrap();
            let back: Activation = serde_json::from_str(&json).unwrap();
            assert_eq!(*act, back);
        }
    }

    #[test]
    fn test_serde_unknown_variant_returns_error() {
        let result = serde_json::from_str::<Activation>("\"Softmax\"");
        assert!(result.is_err());
    }
}
