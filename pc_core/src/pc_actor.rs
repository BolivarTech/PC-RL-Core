// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Predictive Coding Actor Network.
//!
//! Implements an actor that uses iterative top-down/bottom-up predictive coding
//! inference loops instead of standard feedforward passes. The prediction error
//! (surprise score) drives learning rate modulation in the actor-critic agent.

// TODO: implement

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
        let actor = PcActor::new(default_config(), &mut rng);
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
        let actor = PcActor::new(config, &mut rng);
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
        let actor = PcActor::new(config, &mut rng);
        let result = actor.infer(&[0.0; 9]);
        assert!(!result.converged);
        assert_eq!(result.steps_used, 20);
    }

    #[test]
    fn test_infer_does_not_modify_weights() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng);
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
        let actor = PcActor::new(default_config(), &mut rng);
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 18);
    }

    #[test]
    fn test_infer_latent_size_two_hidden() {
        let mut rng = make_rng();
        let actor = PcActor::new(two_hidden_config(), &mut rng);
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 30);
    }

    #[test]
    fn test_infer_latent_size_matches_latent_size_method() {
        let mut rng = make_rng();
        let actor = PcActor::new(two_hidden_config(), &mut rng);
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), actor.latent_size());
    }

    #[test]
    fn test_infer_y_conv_length_equals_output_size() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng);
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.y_conv.len(), 9);
    }

    #[test]
    fn test_infer_hidden_states_count_matches_hidden_layers() {
        let mut rng = make_rng();
        let actor = PcActor::new(two_hidden_config(), &mut rng);
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.hidden_states.len(), 2);
    }

    #[test]
    fn test_infer_all_outputs_finite() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng);
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
        let actor = PcActor::new(default_config(), &mut rng);
        let result = actor.infer(&[0.0; 9]);
        assert!(result.surprise_score >= 0.0);
    }

    #[test]
    fn test_infer_synchronous_and_inplace_both_converge() {
        let mut rng = make_rng();
        let sync_actor = PcActor::new(default_config(), &mut rng);
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..default_config()
        };
        let inplace_actor = PcActor::new(inplace_config, &mut rng2);
        let sync_result = sync_actor.infer(&[0.0; 9]);
        let inplace_result = inplace_actor.infer(&[0.0; 9]);
        // Both should complete without panic; at least one should converge or use all steps
        assert!(sync_result.steps_used > 0);
        assert!(inplace_result.steps_used > 0);
    }

    #[test]
    fn test_infer_synchronous_produces_different_result_than_inplace() {
        let mut rng = make_rng();
        let sync_actor = PcActor::new(default_config(), &mut rng);
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..default_config()
        };
        let inplace_actor = PcActor::new(inplace_config, &mut rng2);
        let sync_result = sync_actor.infer(&[0.0; 9]);
        let inplace_result = inplace_actor.infer(&[0.0; 9]);
        // Different update orders should usually produce different results
        let differs = sync_result
            .y_conv
            .iter()
            .zip(inplace_result.y_conv.iter())
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
        let actor = PcActor::new(default_config(), &mut rng);
        let _ = actor.infer(&[0.0; 5]);
    }

    // ── Action Selection Tests ───────────────────────────────────────

    #[test]
    fn test_select_action_training_always_in_valid() {
        let mut rng = make_rng();
        let actor = PcActor::new(default_config(), &mut rng);
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
        let actor = PcActor::new(default_config(), &mut rng_init);
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
        let actor = PcActor::new(hot_config, &mut rng);
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
        let actor = PcActor::new(default_config(), &mut rng);
        let logits = vec![0.1; 9];
        let _ = actor.select_action(&logits, &[], SelectionMode::Training, &mut rng);
    }

    // ── Weight Update Tests ──────────────────────────────────────────

    #[test]
    fn test_update_weights_changes_first_layer() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(default_config(), &mut rng);
        let input = vec![0.0; 9];
        let infer_result = actor.infer(&input);
        let weights_before = actor.layers[0].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0);
        assert_ne!(actor.layers[0].weights.data, weights_before);
    }

    #[test]
    fn test_update_weights_clips_all_layers() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(default_config(), &mut rng);
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
        let mut actor = PcActor::new(two_hidden_config(), &mut rng);
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0_before = actor.layers[0].weights.data.clone();
        let w1_before = actor.layers[1].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0);
        assert_ne!(actor.layers[0].weights.data, w0_before, "Layer 0 should change");
        assert_ne!(actor.layers[1].weights.data, w1_before, "Layer 1 should change");
    }

    #[test]
    #[should_panic(expected = "input size")]
    fn test_update_weights_panics_wrong_x_size() {
        let mut rng = make_rng();
        let mut actor = PcActor::new(default_config(), &mut rng);
        let input = vec![0.0; 9];
        let infer_result = actor.infer(&input);
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &[0.0; 5], 1.0);
    }
}
