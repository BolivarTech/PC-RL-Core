// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! JSON-based weight persistence for the PC-Actor-Critic agent.
//!
//! Provides save/load for complete agent state (weights, config, metadata)
//! and checkpoint support with auto-named files.
//!
//! Serialization always goes through CPU types (`CpuLinAlg`). Generic agents
//! convert to/from CPU weights via `to_weights()` / `from_weights()`.

use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::PcError;
use crate::layer::Layer;
use crate::linalg::LinAlg;
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
    /// Projection matrices for heterogeneous skip connections.
    #[serde(default)]
    pub skip_projections: Vec<Option<crate::matrix::Matrix>>,
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
/// from both actor and critic via `to_weights()`, bundles with config
/// and metadata, and writes as pretty-printed JSON.
///
/// # Arguments
///
/// * `agent` - The agent to save (any `LinAlg` backend).
/// * `path` - File path for the JSON output.
/// * `episode` - Current episode number.
/// * `metrics` - Optional training metrics snapshot.
///
/// # Errors
///
/// Returns `PcError::Io` on file system errors, `PcError::Serialization`
/// on JSON encoding errors.
pub fn save_agent<L: LinAlg>(
    agent: &PcActorCritic<L>,
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
        actor_weights: agent.actor.to_weights(),
        critic_weights: agent.critic.to_weights(),
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

/// Loads an agent from a JSON save file (CPU backend).
///
/// Reads the file, deserializes the `SaveFile`, validates that the
/// topology matches the config, then reconstructs the agent using
/// `CpuLinAlg` (the default backend).
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
pub fn load_agent(
    path: &str,
    backend: crate::linalg::cpu::CpuLinAlg,
) -> Result<(PcActorCritic, AgentMetadata), PcError> {
    load_agent_generic(path, backend)
}

/// Loads an agent from a JSON save file with a specific `LinAlg` backend.
///
/// Same as [`load_agent`] but reconstructs the agent using the specified
/// backend type `L`. Weights are deserialized as CPU types and then
/// converted via `PcActor::<L>::from_weights()` and
/// `MlpCritic::<L>::from_weights()`.
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
pub fn load_agent_generic<L: LinAlg>(
    path: &str,
    backend: L,
) -> Result<(PcActorCritic<L>, AgentMetadata), PcError> {
    let json = std::fs::read_to_string(path)?;
    let save_file: SaveFile = serde_json::from_str(&json)?;

    let actor = PcActor::<L>::from_weights(
        backend.clone(),
        save_file.config.actor.clone(),
        save_file.actor_weights,
    )?;
    let critic = MlpCritic::<L>::from_weights(
        backend.clone(),
        save_file.config.critic.clone(),
        save_file.critic_weights,
    )?;

    use rand::SeedableRng;
    let rng = rand::rngs::StdRng::from_entropy();

    let agent = PcActorCritic::from_parts(save_file.config, actor, critic, rng, backend);

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
/// use pc_rl_core::serializer::checkpoint_filename;
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
/// * `agent` - The agent to checkpoint (any `LinAlg` backend).
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
pub fn save_checkpoint<L: LinAlg>(
    agent: &PcActorCritic<L>,
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
            surprise_buffer_size: 100,
            entropy_coeff: 0.01,
            scale_floor: 0.1,
            scale_ceil: 2.0,
            actor_hysteresis: false,
            actor_fast_window: 20,
            actor_slow_window: 100,
            actor_wake_fraction: 0.5,
            actor_sleep_fraction: 0.3,
            critic_hysteresis: false,
            critic_fast_window: 20,
            critic_slow_window: 100,
            critic_wake_fraction: 0.5,
            critic_sleep_fraction: 0.3,
            actor_wakes_critic: false,
            actor_wakes_critic_threshold: 1000,
        }
    }

    fn make_agent() -> PcActorCritic {
        use crate::linalg::cpu::CpuLinAlg;
        let agent: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), default_config(), 42).unwrap();
        agent
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
        let (loaded, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
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
        let (loaded, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
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
        let (loaded, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
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
        let (_, metadata) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
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
        let result = load_agent(
            "/nonexistent/path/agent.json",
            crate::linalg::cpu::CpuLinAlg::new(),
        );
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
        let result = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new());
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

        let result = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new());
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

        let (mut loaded1, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
        let (mut loaded2, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();

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
        let (loaded, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();

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
    #[allow(deprecated)]
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
        let mut agent: PcActorCritic =
            PcActorCritic::new(crate::linalg::cpu::CpuLinAlg::new(), config, 42).unwrap();
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
        let (loaded, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
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
        let (loaded, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
        assert!(loaded.actor.rezero_alpha.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_load_agent_generic_matches_load_agent() {
        let agent = make_agent();
        let path = temp_path("test_generic_load.json");
        save_agent(&agent, &path, 10, None).unwrap();

        let (loaded_default, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();
        let (loaded_generic, _) = load_agent_generic::<crate::linalg::cpu::CpuLinAlg>(
            &path,
            crate::linalg::cpu::CpuLinAlg::new(),
        )
        .unwrap();

        let input = vec![0.5, -0.5, 1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0];
        let r1 = loaded_default.infer(&input);
        let r2 = loaded_generic.infer(&input);

        for (a, b) in r1.y_conv.iter().zip(r2.y_conv.iter()) {
            assert!((a - b).abs() < 1e-15, "y_conv differs: {a} vs {b}");
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[allow(deprecated)]
    fn test_roundtrip_preserves_skip_projections_directly() {
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
                        size: 18,
                        activation: Activation::Tanh,
                    },
                ],
                ..default_config().actor
            },
            critic: MlpCriticConfig {
                input_size: 54,
                ..default_config().critic
            },
            ..default_config()
        };
        let mut agent: PcActorCritic =
            PcActorCritic::new(crate::linalg::cpu::CpuLinAlg::new(), config, 42).unwrap();
        // Train to modify projection weights
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

        // Verify projection exists (27→18 requires projection)
        assert!(agent.actor.skip_projections[0].is_some());
        let orig_proj = agent.actor.skip_projections[0].as_ref().unwrap();
        let orig_data = orig_proj.data.clone();

        let path = temp_path("test_skip_proj_roundtrip.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (loaded, _) = load_agent(&path, crate::linalg::cpu::CpuLinAlg::new()).unwrap();

        let loaded_proj = loaded.actor.skip_projections[0].as_ref().unwrap();
        assert_eq!(orig_data.len(), loaded_proj.data.len());
        for (i, (a, b)) in orig_data.iter().zip(loaded_proj.data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "skip_projection element {i} differs: {a} vs {b}"
            );
        }
        let _ = fs::remove_file(&path);
    }

    /// Escenario 10: v1.2.3 JSON loads in v2.0 (backward compat).
    #[test]
    fn test_v1_fixture_loads_in_v2() {
        let backend = crate::linalg::cpu::CpuLinAlg::new();
        let (mut agent, metadata) = load_agent("tests/fixtures/v1_model.json", backend).unwrap();
        // Verify metadata is valid
        assert!(!metadata.version.is_empty());
        assert_eq!(metadata.episode, 100);
        // Verify agent produces valid inference
        let state = vec![0.5; 9];
        let valid: Vec<usize> = (0..9).collect();
        let (action, _) = agent.act(&state, &valid, crate::pc_actor::SelectionMode::Play);
        assert!(action < 9, "Action must be in valid range");
    }
}
