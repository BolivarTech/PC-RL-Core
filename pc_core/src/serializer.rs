// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! JSON-based weight persistence for the PC-Actor-Critic agent.
//!
//! Provides save/load for complete agent state (weights, config, metadata)
//! and checkpoint support with auto-named files.

use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::PcError;
use crate::layer::Layer;
use crate::mlp_critic::MlpCritic;
use crate::pc_actor::PcActor;
use crate::pc_actor_critic::{PcActorCritic, PcActorCriticConfig};

/// Metadata embedded in every save file.
///
/// Tracks version, creation timestamp, episode count, and optional
/// training metrics for provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Crate version string.
    pub version: String,
    /// UTC timestamp of when the file was created.
    pub created: String,
    /// Episode number at time of save.
    pub episode: usize,
    /// Optional training statistics snapshot.
    pub metrics: Option<TrainingMetrics>,
}

/// Training statistics snapshot for inclusion in save files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Fraction of games won.
    pub win_rate: f64,
    /// Fraction of games lost.
    pub loss_rate: f64,
    /// Fraction of games drawn.
    pub draw_rate: f64,
    /// Average surprise score over recent episodes.
    pub avg_surprise: f64,
    /// Current curriculum depth level.
    pub curriculum_depth: usize,
}

/// Serializable weight snapshot for the PC actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcActorWeights {
    /// Layer snapshots in order (hidden layers + output layer).
    pub layers: Vec<Layer>,
    /// ReZero scaling factors for residual skip connections.
    #[serde(default)]
    pub rezero_alpha: Vec<f64>,
    /// Auxiliary linear heads for hidden layer gradient injection.
    #[serde(default)]
    pub aux_heads: Vec<Layer>,
}

/// Complete save file containing agent state and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveFile {
    /// File metadata (version, timestamp, episode).
    pub metadata: AgentMetadata,
    /// Agent configuration.
    pub config: PcActorCriticConfig,
    /// Actor network weights.
    pub actor_weights: PcActorWeights,
    /// Critic network weights.
    pub critic_weights: crate::mlp_critic::MlpCriticWeights,
}

/// Saves the agent's full state to a JSON file.
///
/// Creates parent directories if they don't exist. Extracts weights
/// from both actor and critic, bundles with config and metadata,
/// and writes as pretty-printed JSON.
///
/// # Arguments
///
/// * `agent` - The agent to save.
/// * `path` - File path for the JSON output.
/// * `episode` - Current episode number.
/// * `metrics` - Optional training metrics snapshot.
///
/// # Errors
///
/// Returns `PcError::Io` on file system errors, `PcError::Serialization`
/// on JSON encoding errors.
pub fn save_agent(
    agent: &PcActorCritic,
    path: &str,
    episode: usize,
    metrics: Option<TrainingMetrics>,
) -> Result<(), PcError> {
    let save_file = SaveFile {
        metadata: AgentMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            created: Utc::now().to_rfc3339(),
            episode,
            metrics,
        },
        config: agent.config.clone(),
        actor_weights: PcActorWeights {
            layers: agent.actor.layers.clone(),
            rezero_alpha: agent.actor.rezero_alpha.clone(),
            aux_heads: agent.actor.aux_heads.clone(),
        },
        critic_weights: crate::mlp_critic::MlpCriticWeights {
            layers: agent.critic.layers.clone(),
        },
    };

    let json = serde_json::to_string_pretty(&save_file)?;

    // Create parent directories if needed
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    std::fs::write(path, json)?;
    Ok(())
}

/// Loads an agent from a JSON save file.
///
/// Reads the file, deserializes the `SaveFile`, validates that the
/// topology matches the config, then reconstructs the agent.
///
/// # Arguments
///
/// * `path` - Path to the JSON save file.
///
/// # Errors
///
/// Returns `PcError::Io` if the file doesn't exist, `PcError::Serialization`
/// for invalid JSON, or `PcError::DimensionMismatch` if the saved weights
/// don't match the config topology.
pub fn load_agent(path: &str) -> Result<(PcActorCritic, AgentMetadata), PcError> {
    let json = std::fs::read_to_string(path)?;
    let save_file: SaveFile = serde_json::from_str(&json)?;

    // Validate actor layer count
    let expected_actor_layers = save_file.config.actor.hidden_layers.len() + 1;
    if save_file.actor_weights.layers.len() != expected_actor_layers {
        return Err(PcError::DimensionMismatch {
            expected: expected_actor_layers,
            got: save_file.actor_weights.layers.len(),
            context: "actor layer count",
        });
    }

    // Validate critic layer count
    let expected_critic_layers = save_file.config.critic.hidden_layers.len() + 1;
    if save_file.critic_weights.layers.len() != expected_critic_layers {
        return Err(PcError::DimensionMismatch {
            expected: expected_critic_layers,
            got: save_file.critic_weights.layers.len(),
            context: "critic layer count",
        });
    }

    let actor = PcActor {
        layers: save_file.actor_weights.layers,
        config: save_file.config.actor.clone(),
        rezero_alpha: save_file.actor_weights.rezero_alpha,
        skip_projections: Vec::new(), // Reconstructed from save in serialization cycle
        aux_heads: save_file.actor_weights.aux_heads,
    };

    let critic = MlpCritic::from_weights(
        save_file.config.critic.clone(),
        crate::mlp_critic::MlpCriticWeights {
            layers: save_file.critic_weights.layers,
        },
    );

    use rand::SeedableRng;
    let rng = rand::rngs::StdRng::from_entropy();

    let agent = PcActorCritic::from_parts(save_file.config, actor, critic, rng);

    Ok((agent, save_file.metadata))
}

/// Generates a checkpoint filename with no colons (filesystem-safe).
///
/// Format: `checkpoint_ep{N}_{YYYYMMDD_HHMMSS}.json`
///
/// # Arguments
///
/// * `episode` - Episode number to embed in the filename.
///
/// # Examples
///
/// ```
/// use pc_core::serializer::checkpoint_filename;
///
/// let name = checkpoint_filename(100);
/// assert!(name.starts_with("checkpoint_ep100_"));
/// assert!(name.ends_with(".json"));
/// assert!(!name.contains(':'));
/// ```
pub fn checkpoint_filename(episode: usize) -> String {
    let now = Utc::now().format("%Y%m%d_%H%M%S");
    format!("checkpoint_ep{episode}_{now}.json")
}

/// Saves a checkpoint to a directory with an auto-generated filename.
///
/// # Arguments
///
/// * `agent` - The agent to checkpoint.
/// * `dir` - Directory where the checkpoint file will be created.
/// * `episode` - Current episode number.
/// * `metrics` - Optional training metrics snapshot.
///
/// # Returns
///
/// The full path to the created checkpoint file.
///
/// # Errors
///
/// Returns `PcError` on I/O or serialization failures.
pub fn save_checkpoint(
    agent: &PcActorCritic,
    dir: &str,
    episode: usize,
    metrics: Option<TrainingMetrics>,
) -> Result<PathBuf, PcError> {
    let filename = checkpoint_filename(episode);
    let path = Path::new(dir).join(filename);
    let path_str = path.to_string_lossy().to_string();
    save_agent(agent, &path_str, episode, metrics)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::LayerDef;
    use crate::mlp_critic::MlpCriticConfig;
    use crate::pc_actor::PcActorConfig;
    use std::fs;

    fn default_config() -> PcActorCriticConfig {
        PcActorCriticConfig {
            actor: PcActorConfig {
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
            },
            critic: MlpCriticConfig {
                input_size: 27,
                hidden_layers: vec![LayerDef {
                    size: 36,
                    activation: Activation::Tanh,
                }],
                output_activation: Activation::Linear,
                lr: 0.005,
            },
            gamma: 0.95,
            surprise_low: 0.02,
            surprise_high: 0.15,
            adaptive_surprise: false,
            entropy_coeff: 0.01,
        }
    }

    fn make_agent() -> PcActorCritic {
        PcActorCritic::new(default_config(), 42).unwrap()
    }

    fn temp_path(name: &str) -> String {
        let dir = std::env::temp_dir().join("pc_core_tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_string_lossy().to_string()
    }

    /// Asserts two f64 slices are approximately equal (within 1e-15).
    fn assert_vecs_approx_eq(a: &[f64], b: &[f64]) {
        assert_eq!(
            a.len(),
            b.len(),
            "Lengths differ: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((va - vb).abs() < 1e-15, "Element {i} differs: {va} vs {vb}");
        }
    }

    #[test]
    fn test_roundtrip_preserves_actor_weights() {
        let agent = make_agent();
        let path = temp_path("test_actor_roundtrip.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();
        for (orig, loaded_layer) in agent.actor.layers.iter().zip(loaded.actor.layers.iter()) {
            assert_vecs_approx_eq(&orig.weights.data, &loaded_layer.weights.data);
            assert_vecs_approx_eq(&orig.bias, &loaded_layer.bias);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_roundtrip_preserves_critic_weights() {
        let agent = make_agent();
        let path = temp_path("test_critic_roundtrip.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();
        for (orig, loaded_layer) in agent.critic.layers.iter().zip(loaded.critic.layers.iter()) {
            assert_vecs_approx_eq(&orig.weights.data, &loaded_layer.weights.data);
            assert_vecs_approx_eq(&orig.bias, &loaded_layer.bias);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_roundtrip_preserves_config() {
        let agent = make_agent();
        let path = temp_path("test_config_roundtrip.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();
        assert_eq!(loaded.config.gamma, agent.config.gamma);
        assert_eq!(
            loaded.config.actor.input_size,
            agent.config.actor.input_size
        );
        assert_eq!(
            loaded.config.critic.input_size,
            agent.config.critic.input_size
        );
        assert_eq!(loaded.config.entropy_coeff, agent.config.entropy_coeff);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_metadata_includes_version_and_episode() {
        let agent = make_agent();
        let path = temp_path("test_metadata.json");
        save_agent(&agent, &path, 42, None).unwrap();
        let (_, metadata) = load_agent(&path).unwrap();
        assert!(!metadata.version.is_empty());
        assert_eq!(metadata.episode, 42);
        assert!(!metadata.created.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_filename_no_colons() {
        let name = checkpoint_filename(100);
        assert!(!name.contains(':'), "Filename contains colons: {name}");
    }

    #[test]
    fn test_checkpoint_filename_contains_episode_number() {
        let name = checkpoint_filename(42);
        assert!(
            name.contains("ep42"),
            "Filename doesn't contain episode number: {name}"
        );
        assert!(name.ends_with(".json"));
    }

    #[test]
    fn test_load_nonexistent_returns_error() {
        let result = load_agent("/nonexistent/path/agent.json");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            matches!(err, PcError::Io(_)),
            "Expected PcError::Io, got: {err}"
        );
    }

    #[test]
    fn test_load_invalid_json_returns_error() {
        let path = temp_path("test_invalid.json");
        fs::write(&path, "not valid json {{{").unwrap();
        let result = load_agent(&path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            matches!(err, PcError::Serialization(_)),
            "Expected PcError::Serialization, got: {err}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_load_topology_mismatch_returns_error() {
        let agent = make_agent();
        let path = temp_path("test_mismatch.json");
        save_agent(&agent, &path, 0, None).unwrap();

        // Tamper: read JSON, change actor layer count in config
        let json = fs::read_to_string(&path).unwrap();
        let mut save_file: SaveFile = serde_json::from_str(&json).unwrap();
        // Add an extra hidden layer to config (but not weights)
        save_file.config.actor.hidden_layers.push(LayerDef {
            size: 10,
            activation: Activation::Relu,
        });
        let tampered = serde_json::to_string_pretty(&save_file).unwrap();
        fs::write(&path, tampered).unwrap();

        let result = load_agent(&path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected PcError::DimensionMismatch, got: {err}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_load_agent_uses_entropy_seed_not_fixed() {
        let agent = make_agent();
        let path = temp_path("test_seed_entropy.json");
        save_agent(&agent, &path, 10, None).unwrap();

        let (mut loaded1, _) = load_agent(&path).unwrap();
        let (mut loaded2, _) = load_agent(&path).unwrap();

        // Both agents should produce different action sequences
        // because they use entropy-based RNG seeding
        let input = vec![0.5; 9];
        let valid: Vec<usize> = (0..9).collect();

        let mut actions1 = Vec::new();
        let mut actions2 = Vec::new();
        for _ in 0..20 {
            let (a1, _) = loaded1.act(&input, &valid, crate::pc_actor::SelectionMode::Training);
            let (a2, _) = loaded2.act(&input, &valid, crate::pc_actor::SelectionMode::Training);
            actions1.push(a1);
            actions2.push(a2);
        }

        assert_ne!(
            actions1, actions2,
            "Two loaded agents should have different exploration due to entropy seeding"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_loaded_agent_produces_identical_inference() {
        let agent = make_agent();
        let path = temp_path("test_identical_infer.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();

        let input = vec![0.5, -0.5, 1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0];
        let orig_result = agent.infer(&input);
        let loaded_result = loaded.infer(&input);

        // y_conv must be identical
        assert_eq!(orig_result.y_conv.len(), loaded_result.y_conv.len());
        for (a, b) in orig_result.y_conv.iter().zip(loaded_result.y_conv.iter()) {
            assert!((a - b).abs() < 1e-12, "y_conv differs: {a} vs {b}");
        }
        // latent_concat must be identical
        for (a, b) in orig_result
            .latent_concat
            .iter()
            .zip(loaded_result.latent_concat.iter())
        {
            assert!((a - b).abs() < 1e-12, "latent_concat differs: {a} vs {b}");
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_save_creates_parent_directories() {
        let dir = std::env::temp_dir()
            .join("pc_core_tests")
            .join("nested")
            .join("deep");
        let path = dir.join("agent.json").to_string_lossy().to_string();

        // Remove if exists from prior run
        let _ = fs::remove_dir_all(&dir);

        let agent = make_agent();
        save_agent(&agent, &path, 0, None).unwrap();
        assert!(Path::new(&path).exists());

        // Cleanup
        let _ = fs::remove_dir_all(std::env::temp_dir().join("pc_core_tests").join("nested"));
    }

    #[test]
    fn test_roundtrip_preserves_modified_rezero_alpha() {
        use crate::pc_actor::SelectionMode;
        let config = PcActorCriticConfig {
            actor: PcActorConfig {
                residual: true,
                rezero_init: 0.005,
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
                ..default_config().actor
            },
            critic: MlpCriticConfig {
                input_size: 63,
                ..default_config().critic
            },
            ..default_config()
        };
        let mut agent = PcActorCritic::new(config, 42).unwrap();
        // Train one step to modify rezero_alpha
        let input = vec![0.5; 9];
        let valid: Vec<usize> = (0..9).collect();
        let (action, infer) = agent.act(&input, &valid, SelectionMode::Training);
        let trajectory = vec![crate::pc_actor_critic::TrajectoryStep {
            input: input.clone(),
            latent_concat: infer.latent_concat,
            y_conv: infer.y_conv,
            hidden_states: infer.hidden_states,
            prediction_errors: infer.prediction_errors,
            tanh_components: infer.tanh_components,
            action,
            valid_actions: valid,
            reward: 1.0,
            surprise_score: infer.surprise_score,
            steps_used: infer.steps_used,
        }];
        agent.learn(&trajectory);
        let alpha_after_train = agent.actor.rezero_alpha.clone();
        // Alpha should have changed from init
        assert_ne!(alpha_after_train, vec![0.005]);

        let path = temp_path("test_rezero_roundtrip.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();
        assert_eq!(
            loaded.actor.rezero_alpha, alpha_after_train,
            "Loaded rezero_alpha should match trained value, not rezero_init"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_roundtrip_non_residual_backward_compat() {
        let agent = make_agent();
        assert!(agent.actor.rezero_alpha.is_empty());

        let path = temp_path("test_nonresidual_compat.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();
        assert!(loaded.actor.rezero_alpha.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_roundtrip_preserves_aux_heads() {
        let config = PcActorCriticConfig {
            actor: PcActorConfig {
                aux_loss_coefficient: 0.1,
                ..default_config().actor
            },
            ..default_config()
        };
        let agent = PcActorCritic::new(config, 42).unwrap();
        assert_eq!(agent.actor.aux_heads.len(), 1);

        let path = temp_path("test_aux_roundtrip.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();
        assert_eq!(loaded.actor.aux_heads.len(), 1);
        for (a, b) in agent.actor.aux_heads[0]
            .weights
            .data
            .iter()
            .zip(loaded.actor.aux_heads[0].weights.data.iter())
        {
            assert!(
                (a - b).abs() < 1e-12,
                "Aux head weight mismatch: {a} vs {b}"
            );
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_roundtrip_no_aux_backward_compat() {
        let agent = make_agent();
        assert!(agent.actor.aux_heads.is_empty());
        let path = temp_path("test_no_aux_compat.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path).unwrap();
        assert!(loaded.actor.aux_heads.is_empty());
        let _ = fs::remove_file(&path);
    }
}
