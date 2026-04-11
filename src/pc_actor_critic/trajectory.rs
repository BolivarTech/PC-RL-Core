// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-09

//! Trajectory step and activation cache types.
//!
//! [`TrajectoryStep<L>`] records a single (state, action, reward) transition
//! during episodic learning. [`ActivationCache<L>`] captures per-layer
//! activations for use by the CCA crossover operator.

use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;

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

// ============================================================================
// Helper functions
// ============================================================================

/// Converts an `ActivationCache` into a vector of matrices `[batch × neurons]`,
/// one per hidden layer, suitable for CCA alignment.
pub(crate) fn cache_to_matrices<L: LinAlg>(
    backend: &L,
    cache: &ActivationCache<L>,
) -> Vec<L::Matrix> {
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

    #[test]
    fn test_activation_cache_new_creates_empty() {
        let cache: ActivationCache = ActivationCache::new(3);
        assert_eq!(cache.batch_size(), 0);
    }
}
