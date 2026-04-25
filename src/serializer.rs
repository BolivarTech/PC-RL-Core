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
use crate::matrix::Matrix;
use crate::mlp_critic::MlpCritic;
use crate::pc_actor::PcActor;
use crate::pc_actor_critic::{PcActorCritic, PcActorCriticConfig, PlasticityState};

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

/// Serializable per-layer Fisher information state.
///
/// Stores accumulated Fisher (`f_total`), current-phase EMA (`f_ema`),
/// and optional weight snapshots as CPU-side `Matrix`/`Vec<f64>`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FisherStateSerialized {
    /// Accumulated Fisher information for weights.
    pub f_total_weights: Matrix,
    /// Accumulated Fisher information for biases.
    pub f_total_bias: Vec<f64>,
    /// Current-phase running EMA of squared gradients for weights.
    pub f_ema_weights: Matrix,
    /// Current-phase running EMA of squared gradients for biases.
    pub f_ema_bias: Vec<f64>,
    /// Snapshot of weights at last PLASTIC→FROZEN transition.
    #[serde(default)]
    pub theta_snapshot_weights: Option<Matrix>,
    /// Snapshot of biases at last PLASTIC→FROZEN transition.
    #[serde(default)]
    pub theta_snapshot_bias: Option<Vec<f64>>,
    /// Snapshot of rezero alpha (for residual layers).
    #[serde(default)]
    pub theta_snapshot_rezero_alpha: Option<f64>,
    /// Snapshot of skip projection matrix (for heterogeneous residual layers).
    #[serde(default)]
    pub theta_snapshot_skip_proj: Option<Matrix>,
}

/// Serializable EWMA tracker state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EwmaTrackerSerialized {
    /// Current EWMA value.
    pub value: f64,
    /// Step counter.
    pub k: u64,
    /// Window size.
    pub window: usize,
}

/// Serializable hysteresis state machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HysteresisStateSerialized {
    /// Fast EWMA tracker.
    pub fast: EwmaTrackerSerialized,
    /// Slow EWMA tracker.
    pub slow: EwmaTrackerSerialized,
    /// Current plasticity state.
    pub state: PlasticityState,
    /// Wake fraction threshold.
    pub wake_fraction: f64,
    /// Sleep fraction threshold.
    pub sleep_fraction: f64,
    /// Minimum fast EWMA steps before sleep is allowed.
    pub min_initial_plastic: u64,
}

/// Top-level container for all continuous learning state.
///
/// Persisted in `SaveFile` as `Option<ClState>`. Legacy JSON files
/// without this field load as `None`, which means clean PLASTIC defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClState {
    /// Actor hysteresis state (None when disabled).
    #[serde(default)]
    pub actor_hysteresis: Option<HysteresisStateSerialized>,
    /// Critic hysteresis state (None when disabled).
    #[serde(default)]
    pub critic_hysteresis: Option<HysteresisStateSerialized>,
    /// Steps the actor has been in PLASTIC state.
    #[serde(default)]
    pub actor_plastic_step_counter: u64,
    /// Steps the critic has been in PLASTIC state.
    #[serde(default)]
    pub critic_plastic_step_counter: u64,
    /// Consecutive steps the critic has been FROZEN.
    #[serde(default)]
    pub critic_frozen_steps: u64,
    /// Consecutive steps the actor has been FROZEN.
    #[serde(default)]
    pub actor_frozen_steps: u64,
    /// Per-layer Fisher state for actor.
    #[serde(default)]
    pub actor_fisher: Vec<FisherStateSerialized>,
    /// Per-layer Fisher state for critic.
    #[serde(default)]
    pub critic_fisher: Vec<FisherStateSerialized>,
    /// Whether the last actor PLASTIC phase was reliable.
    #[serde(default)]
    pub actor_last_phase_reliable: bool,
    /// Whether the last critic PLASTIC phase was reliable.
    #[serde(default)]
    pub critic_last_phase_reliable: bool,
    /// Per-layer prediction error EMA for adaptive consolidation (M3b).
    #[serde(default)]
    pub layer_error_ema: Vec<f64>,
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
    /// Continuous learning state (None for legacy/v2.0.0 files).
    #[serde(default)]
    pub cl_state: Option<ClState>,
    /// Polyak-averaged target actor weights (None when lambda == 0 or legacy file).
    #[serde(default)]
    pub polyak_target_weights: Option<PcActorWeights>,
    /// Frozen champion actor weights (None when lambda == 0 or legacy file).
    #[serde(default)]
    pub frozen_champion_weights: Option<PcActorWeights>,
    /// Dual-compartment replay buffer state (None when
    /// `replay_training_capacity == 0` or legacy file).
    #[serde(default)]
    pub replay_buffer: Option<crate::pc_actor_critic::replay::ReplayBuffer>,
    /// Monotonic count of replay_learn saturation events (legacy files
    /// default to 0).
    #[serde(default)]
    pub replay_clamp_count: u64,
    /// Number of learn steps elapsed since the last `rollback_hard()`.
    /// Legacy files default to `u64::MAX` (the "unlocked" bootstrap
    /// sentinel), which preserves the pre-W2-fix behaviour where a
    /// freshly-loaded agent can always invoke `rollback_hard()` at
    /// least once. New files persist the actual counter so a
    /// save-reload cycle cannot silently bypass the cooldown.
    #[serde(default = "default_steps_since_last_rollback_hard")]
    pub steps_since_last_rollback_hard: u64,
    /// User-configurable cooldown window (defaults to
    /// [`DEFAULT_ROLLBACK_HARD_COOLDOWN`](crate::pc_actor_critic::DEFAULT_ROLLBACK_HARD_COOLDOWN)
    /// when absent, so legacy files and any user override via
    /// `set_rollback_hard_cooldown` both round-trip cleanly).
    #[serde(default = "default_rollback_hard_cooldown_steps")]
    pub rollback_hard_cooldown_steps: u64,
}

fn default_steps_since_last_rollback_hard() -> u64 {
    u64::MAX
}

fn default_rollback_hard_cooldown_steps() -> u64 {
    crate::pc_actor_critic::DEFAULT_ROLLBACK_HARD_COOLDOWN
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
        cl_state: agent.to_cl_state(),
        polyak_target_weights: agent.polyak_target.as_ref().map(|a| a.to_weights()),
        frozen_champion_weights: agent.frozen_champion.as_ref().map(|a| a.to_weights()),
        replay_buffer: agent.replay_buffer.clone(),
        replay_clamp_count: agent.replay_clamp_count,
        steps_since_last_rollback_hard: agent.steps_since_last_rollback_hard,
        rollback_hard_cooldown_steps: agent.rollback_hard_cooldown_steps,
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

    let mut agent = PcActorCritic::from_parts(
        save_file.config.clone(),
        actor,
        critic,
        rng,
        backend.clone(),
    );

    // Restore Polyak target: saved weights > legacy clone > None
    if save_file.config.distillation_lambda_polyak > 0.0 {
        if let Some(polyak_weights) = save_file.polyak_target_weights {
            let polyak = PcActor::<L>::from_weights(
                backend.clone(),
                save_file.config.actor.clone(),
                polyak_weights,
            )?;
            agent.polyak_target = Some(polyak);
        }
        // else: from_parts already cloned actor (legacy compat)
    } else {
        agent.polyak_target = None;
    }

    // Restore frozen champion: saved weights > legacy clone > None
    if save_file.config.distillation_lambda_frozen > 0.0 {
        if let Some(frozen_weights) = save_file.frozen_champion_weights {
            let frozen = PcActor::<L>::from_weights(
                backend,
                save_file.config.actor.clone(),
                frozen_weights,
            )?;
            agent.frozen_champion = Some(frozen);
        }
        // else: from_parts already cloned actor (legacy compat)
    } else {
        agent.frozen_champion = None;
    }

    if let Some(cl_state) = save_file.cl_state {
        agent.restore_cl_state(cl_state);
    }

    // Restore replay buffer:
    //   * If the SaveFile carries a `Some(buf)`, use it directly.
    //   * Else if the effective config's `replay_training_capacity > 0`, allocate
    //     a fresh empty buffer (legacy save-file compat — Phase 1 files lack the
    //     `replay_buffer` key).
    //   * Else, no buffer.
    agent.replay_buffer = if let Some(buf) = save_file.replay_buffer {
        Some(buf)
    } else if save_file.config.replay_training_capacity > 0 {
        Some(crate::pc_actor_critic::replay::ReplayBuffer::new(
            save_file.config.replay_training_capacity,
            save_file.config.replay_recent_capacity,
            save_file.config.replay_positive_only,
        ))
    } else {
        None
    };

    // Restore replay telemetry and rollback cooldown state so dashboards
    // and the cooldown gate survive save/load cycles. Legacy files that
    // pre-date these fields deserialize with sensible defaults via
    // `#[serde(default = ...)]`: 0 for the clamp counter, `u64::MAX`
    // (unlocked bootstrap) for the elapsed counter, and
    // `DEFAULT_ROLLBACK_HARD_COOLDOWN` for the cooldown window.
    agent.replay_clamp_count = save_file.replay_clamp_count;
    agent.steps_since_last_rollback_hard = save_file.steps_since_last_rollback_hard;
    agent.rollback_hard_cooldown_steps = save_file.rollback_hard_cooldown_steps;

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
            actor_wakes_critic: true,
            actor_wakes_critic_threshold: 1000,
            critic_wakes_actor: true,
            critic_wakes_actor_threshold: 1000,
            consolidation_decay: 1.0,
            critic_consolidation_decay: 1.0,
            adaptive_consolidation: false,
            consolidation_ema_beta: 0.99,
            consolidation_sigmoid_k: 10.0,
            consolidation_error_threshold: 0.05,
            ewc_lambda: 0.0,
            fisher_decay: 0.9,
            fisher_ema_beta: 0.99,
            logits_reversal: false,
            td_steps: 0,
            gae_lambda: None,
            distillation_lambda_polyak: 0.0,
            polyak_tau: 0.005,
            distillation_lambda_frozen: 0.0,
            replay_training_capacity: 0,
            replay_recent_capacity: 0,
            replay_positive_only: true,
            replay_batch_size: 64,
            scale_floor_replay: -1.0,
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

    // ── Section 07: Serialization with CL State ─────────────────

    #[test]
    fn test_cl_agent_save_load_roundtrip() {
        use crate::linalg::cpu::CpuLinAlg;

        // Create CL agent with EWC + hysteresis enabled
        let mut config = default_config();
        config.ewc_lambda = 1.0;
        config.fisher_decay = 0.9;
        config.fisher_ema_beta = 0.99;
        config.actor_hysteresis = true;
        config.actor_fast_window = 5;
        config.actor_slow_window = 20;
        config.actor_wake_fraction = 0.5;
        config.actor_sleep_fraction = 0.3;
        config.critic_hysteresis = true;
        config.critic_fast_window = 5;
        config.critic_slow_window = 20;
        config.critic_wake_fraction = 0.5;
        config.critic_sleep_fraction = 0.3;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Train to accumulate non-zero Fisher and EWMA state
        let state1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let state2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        for _ in 0..30 {
            agent.step(&state1, 0.0, false);
            agent.step(&state2, 1.0, true);
        }

        // Save the JSON and verify cl_state is present with EWMA and Fisher
        let path = temp_path("test_cl_roundtrip.json");
        save_agent(&agent, &path, 500, None).unwrap();

        // Verify JSON has CL fields
        let json_str = fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let cl = &json["cl_state"];
        assert!(!cl.is_null(), "cl_state must be present");
        assert!(
            cl["actor_hysteresis"].is_object(),
            "actor_hysteresis must be present"
        );
        assert!(
            cl["actor_fisher"].is_array(),
            "actor_fisher must be present"
        );

        // Verify EWMA k > 0 in saved JSON
        let fast_k = cl["actor_hysteresis"]["fast"]["k"].as_u64().unwrap();
        assert!(fast_k > 0, "fast EWMA k must be > 0 after training");

        // Load and verify inference matches original
        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        let test_input = vec![0.3, -0.7, 0.1, 0.5, -0.2, 0.8, -0.4, 0.6, -0.9];
        let orig_infer = agent.infer(&test_input);
        let loaded_infer = loaded.infer(&test_input);
        for (a, b) in orig_infer.y_conv.iter().zip(loaded_infer.y_conv.iter()) {
            assert!((a - b).abs() < 1e-12, "y_conv differs: {a} vs {b}");
        }

        // Load the saved JSON again and verify CL fields round-trip
        let json_str2 = fs::read_to_string(&path).unwrap();
        let json2: serde_json::Value = serde_json::from_str(&json_str2).unwrap();
        let cl2 = &json2["cl_state"];

        // EWMA k should match
        assert_eq!(
            cl["actor_hysteresis"]["fast"]["k"],
            cl2["actor_hysteresis"]["fast"]["k"]
        );
        // Fisher arrays should match length
        assert_eq!(
            cl["actor_fisher"].as_array().unwrap().len(),
            cl2["actor_fisher"].as_array().unwrap().len()
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_legacy_json_loads_as_clean_plastic() {
        use crate::linalg::cpu::CpuLinAlg;
        use crate::pc_actor::SelectionMode;

        // v2.0.0 fixture has no CL fields → should load with clean PLASTIC defaults
        // Agent should be fully functional
        let (mut agent, _) = load_agent("tests/fixtures/v1_model.json", CpuLinAlg::new()).unwrap();

        // Agent should work normally (no panic from missing CL state)
        let state = vec![0.5; 9];
        let valid: Vec<usize> = (0..9).collect();
        let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
        assert!(action < 9);

        // Step should work (no CL processing since all disabled by defaults)
        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        let _a1 = agent.step(&s1, 0.0, false);
        let _a2 = agent.step(&s2, 1.0, true);
    }

    #[test]
    fn test_step_state_not_serialized() {
        use crate::linalg::cpu::CpuLinAlg;

        let mut agent = make_agent();
        // Start a mid-episode step (state_prev will be set internally)
        let state = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        agent.step(&state, 0.0, false);

        let path = temp_path("test_step_state_not_serialized.json");
        save_agent(&agent, &path, 10, None).unwrap();
        let (mut loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        // Loaded agent's first step() should behave as a "first call" (no prior state)
        // which means no learning happens on the very first call
        let state2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        let weights_before = loaded.actor.layers[0].weights.data.clone();
        loaded.step(&state2, 1.0, false);
        // First step after load should NOT learn (no prev state)
        assert_eq!(
            loaded.actor.layers[0].weights.data, weights_before,
            "First step after load should not modify weights (no prior state)"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_v2_json_fixture_loads_successfully() {
        use crate::linalg::cpu::CpuLinAlg;
        use crate::pc_actor::SelectionMode;

        // Existing v1 fixture loads without error and agent is functional
        let (mut agent, metadata) =
            load_agent("tests/fixtures/v1_model.json", CpuLinAlg::new()).unwrap();
        assert!(!metadata.version.is_empty());
        assert_eq!(metadata.episode, 100);

        let state = vec![0.5; 9];
        let valid: Vec<usize> = (0..9).collect();
        let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
        assert!(action < 9);
    }

    #[test]
    fn test_cl_json_has_fisher_fields() {
        use crate::linalg::cpu::CpuLinAlg;

        let mut config = default_config();
        config.ewc_lambda = 1.0;
        config.fisher_ema_beta = 0.99;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Train to populate Fisher
        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        agent.step(&s1, 0.0, false);
        agent.step(&s2, 1.0, true);

        let path = temp_path("test_cl_fisher_fields.json");
        save_agent(&agent, &path, 10, None).unwrap();

        // Parse raw JSON and check for Fisher fields
        let json_str = fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let cl = &json["cl_state"];
        assert!(!cl.is_null(), "cl_state should be present in saved JSON");
        assert!(
            cl["actor_fisher"].is_array(),
            "actor_fisher should be an array"
        );
        assert!(
            cl["critic_fisher"].is_array(),
            "critic_fisher should be an array"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_cl_json_has_ewma_fields() {
        use crate::linalg::cpu::CpuLinAlg;

        let mut config = default_config();
        config.actor_hysteresis = true;
        config.actor_fast_window = 5;
        config.actor_slow_window = 20;
        config.actor_wake_fraction = 0.5;
        config.actor_sleep_fraction = 0.3;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Train a few steps to populate EWMAs
        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        for _ in 0..5 {
            agent.step(&s1, 0.0, false);
            agent.step(&s2, 1.0, true);
        }

        let path = temp_path("test_cl_ewma_fields.json");
        save_agent(&agent, &path, 10, None).unwrap();

        // Parse raw JSON and check for EWMA fields
        let json_str = fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let cl = &json["cl_state"];
        assert!(!cl.is_null(), "cl_state should be present in saved JSON");
        assert!(
            cl["actor_hysteresis"].is_object(),
            "actor_hysteresis should be an object"
        );
        let ah = &cl["actor_hysteresis"];
        assert!(ah["fast"].is_object(), "fast EWMA should be present");
        assert!(ah["slow"].is_object(), "slow EWMA should be present");

        let _ = fs::remove_file(&path);
    }

    /// v2.0.0 fixture must load with fully inert CL state.
    /// Load → verify to_cl_state() is None → step() → save → reload → verify
    /// round-trip preserves weights and CL stays inert.
    #[test]
    fn test_v2_fixture_roundtrip_inert_cl_state() {
        use crate::linalg::cpu::CpuLinAlg;

        // Load the v2.0.0 (pre-CL) fixture
        let (mut loaded_agent, _) =
            load_agent("tests/fixtures/v1_model.json", CpuLinAlg::new()).unwrap();

        // CL state must be None (all defaults — no Fisher, no hysteresis, no EMA)
        assert!(
            loaded_agent.to_cl_state().is_none(),
            "v2.0.0 fixture must have no CL state (all defaults)"
        );

        // step() on a legacy agent should work with zero CL overhead
        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        let _a1 = loaded_agent.step(&s1, 0.0, false);
        let _a2 = loaded_agent.step(&s2, 1.0, true);

        // After step, CL state should still be None (no CL features enabled)
        assert!(
            loaded_agent.to_cl_state().is_none(),
            "step() on default-config agent must not create CL state"
        );

        // Save, reload, verify round-trip fidelity
        let rt_path = temp_path("test_v2_roundtrip.json");
        save_agent(&loaded_agent, &rt_path, 100, None).unwrap();
        let (reloaded, _) = load_agent(&rt_path, CpuLinAlg::new()).unwrap();

        // Round-tripped agent must also have no CL state
        assert!(
            reloaded.to_cl_state().is_none(),
            "Round-tripped v2.0.0 agent must still have no CL state"
        );

        // Verify actor weights survived the round-trip (f64 JSON roundtrip tolerance)
        for (i, (a, b)) in loaded_agent.actor.layers[0]
            .weights
            .data
            .iter()
            .zip(reloaded.actor.layers[0].weights.data.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-14,
                "Actor weight[{i}] drift after round-trip: {a} vs {b}"
            );
        }

        let _ = fs::remove_file(&rt_path);
    }

    // ── Section 08: Distillation anchor serialization ──────────────

    #[test]

    fn test_save_load_preserves_polyak_target() {
        use crate::linalg::cpu::CpuLinAlg;

        let mut config = default_config();
        config.distillation_lambda_polyak = 0.1;
        config.polyak_tau = 0.005;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Drift live actor for 100 steps so Polyak diverges from live
        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        for _ in 0..50 {
            agent.step(&s1, 0.0, false);
            agent.step(&s2, 1.0, true);
        }

        // Polyak target should have diverged from live actor
        let polyak_w = agent.polyak_target.as_ref().unwrap().layers[0]
            .weights
            .data
            .clone();
        let live_w = agent.actor.layers[0].weights.data.clone();
        assert_ne!(
            polyak_w, live_w,
            "Polyak must differ from live after training"
        );

        let path = temp_path("test_polyak_roundtrip.json");
        save_agent(&agent, &path, 100, None).unwrap();
        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        // Loaded polyak_target must be byte-equal to the saved state
        let loaded_polyak = loaded
            .polyak_target
            .as_ref()
            .expect("polyak_target must be Some");
        for (i, (a, b)) in polyak_w
            .iter()
            .zip(loaded_polyak.layers[0].weights.data.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-15,
                "polyak_target weight[{i}] differs: {a} vs {b}"
            );
        }

        let _ = fs::remove_file(&path);
    }

    #[test]

    fn test_save_load_preserves_frozen_champion_after_update() {
        use crate::linalg::cpu::CpuLinAlg;

        let mut config = default_config();
        config.distillation_lambda_frozen = 0.1;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Drift live actor so frozen will differ from initial
        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        for _ in 0..30 {
            agent.step(&s1, 0.0, false);
            agent.step(&s2, 1.0, true);
        }

        // Call champion_update to snapshot the trained live actor
        agent.champion_update().unwrap();

        let frozen_w = agent.frozen_champion.as_ref().unwrap().layers[0]
            .weights
            .data
            .clone();

        let path = temp_path("test_frozen_champion_roundtrip.json");
        save_agent(&agent, &path, 200, None).unwrap();
        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        // Loaded frozen_champion must match the post-update state
        let loaded_frozen = loaded
            .frozen_champion
            .as_ref()
            .expect("frozen_champion must be Some");
        for (i, (a, b)) in frozen_w
            .iter()
            .zip(loaded_frozen.layers[0].weights.data.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-15,
                "frozen_champion weight[{i}] differs: {a} vs {b}"
            );
        }

        let _ = fs::remove_file(&path);
    }

    #[test]

    fn test_legacy_save_file_initializes_anchors_from_loaded_actor() {
        use crate::linalg::cpu::CpuLinAlg;

        // Create an agent without distillation, save it (mimics pre-distillation save)
        let agent = make_agent();
        let path = temp_path("test_legacy_anchor_init.json");
        save_agent(&agent, &path, 50, None).unwrap();

        // Tamper: set both lambdas > 0 in the saved config
        let json_str = fs::read_to_string(&path).unwrap();
        let mut save_file: SaveFile = serde_json::from_str(&json_str).unwrap();
        save_file.config.distillation_lambda_polyak = 0.05;
        save_file.config.distillation_lambda_frozen = 0.05;
        let tampered = serde_json::to_string_pretty(&save_file).unwrap();
        fs::write(&path, tampered).unwrap();

        // Load — anchors must auto-initialize from the loaded actor
        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        let loaded_actor_w = &loaded.actor.layers[0].weights.data;
        let polyak = loaded
            .polyak_target
            .as_ref()
            .expect("polyak_target must be Some for lambda > 0");
        let frozen = loaded
            .frozen_champion
            .as_ref()
            .expect("frozen_champion must be Some for lambda > 0");

        // Both anchors must be copies of the loaded actor
        assert_vecs_approx_eq(loaded_actor_w, &polyak.layers[0].weights.data);
        assert_vecs_approx_eq(loaded_actor_w, &frozen.layers[0].weights.data);

        let _ = fs::remove_file(&path);
    }

    #[test]

    fn test_load_drops_orphan_anchor_weights_when_lambda_zero() {
        use crate::linalg::cpu::CpuLinAlg;

        // Create agent with both lambdas > 0, train to populate non-trivial anchors
        let mut config = default_config();
        config.distillation_lambda_polyak = 0.1;
        config.polyak_tau = 0.005;
        config.distillation_lambda_frozen = 0.1;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let s1 = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5];
        let s2 = vec![0.5, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        for _ in 0..50 {
            agent.step(&s1, 0.0, false);
            agent.step(&s2, 1.0, true);
        }

        // Save with anchors populated
        let path = temp_path("test_orphan_anchor_drop.json");
        save_agent(&agent, &path, 300, None).unwrap();

        // Tamper: set both lambdas to 0 in the saved JSON
        let json_str = fs::read_to_string(&path).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        json["config"]["distillation_lambda_polyak"] = serde_json::json!(0.0);
        json["config"]["distillation_lambda_frozen"] = serde_json::json!(0.0);
        fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        // Load — anchors must be None despite anchor weights in file
        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();
        assert!(
            loaded.polyak_target.is_none(),
            "polyak_target must be None when lambda == 0"
        );
        assert!(
            loaded.frozen_champion.is_none(),
            "frozen_champion must be None when lambda == 0"
        );

        // Re-save and verify no anchor weights are serialized
        let resave_path = temp_path("test_orphan_anchor_resave.json");
        save_agent(&loaded, &resave_path, 301, None).unwrap();
        let resave_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&resave_path).unwrap()).unwrap();
        assert!(
            resave_json.get("polyak_target_weights").is_none()
                || resave_json["polyak_target_weights"].is_null(),
            "Re-saved file must not contain polyak_target_weights"
        );
        assert!(
            resave_json.get("frozen_champion_weights").is_none()
                || resave_json["frozen_champion_weights"].is_null(),
            "Re-saved file must not contain frozen_champion_weights"
        );

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&resave_path);
    }

    // ── Phase 2 Section: Replay buffer serialization ───────────────

    /// Red test — commit 18 must persist both replay compartments plus the
    /// `training_phase` flag across a save/load round-trip.
    #[test]
    fn test_save_load_preserves_replay_buffer() {
        use crate::linalg::cpu::CpuLinAlg;
        use crate::pc_actor_critic::replay::ReplayTransition;

        // Agent configured with a replay buffer of both compartments.
        let mut config = default_config();
        config.replay_training_capacity = 100;
        config.replay_recent_capacity = 50;
        // Disable positive_only so synthetic transitions with reward=0.0 are retained
        // (content equality is what we care about, not reward filtering).
        config.replay_positive_only = false;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        assert!(
            agent.replay_buffer.is_some(),
            "replay_training_capacity > 0 must allocate a buffer at construction"
        );

        // Build 30 deterministic training transitions and push them directly
        // into the buffer (bypasses step_masked — serde layer is the subject
        // under test, push/seal semantics are already covered by replay.rs).
        let make_tx = |marker: f64, reward: f64| -> ReplayTransition {
            let mut state = vec![0.0; 9];
            state[0] = marker;
            let mut next_state = vec![0.0; 9];
            next_state[1] = marker;
            ReplayTransition {
                state,
                action: (marker as usize) % 9,
                reward,
                next_state,
                done: false,
                valid_actions: (0..9).collect(),
            }
        };

        let mut training_originals: Vec<ReplayTransition> = Vec::with_capacity(30);
        {
            let buf = agent.replay_buffer.as_mut().unwrap();
            for i in 0..30 {
                let tx = make_tx(i as f64, 1.0);
                training_originals.push(tx.clone());
                buf.push(tx);
            }
        }
        assert_eq!(
            agent
                .replay_buffer
                .as_ref()
                .unwrap()
                .training_memories
                .len(),
            30
        );

        // Seal — subsequent pushes must route to the recent compartment.
        agent.seal_replay_training_memories().unwrap();
        assert!(!agent.replay_buffer.as_ref().unwrap().training_phase);

        let mut recent_originals: Vec<ReplayTransition> = Vec::with_capacity(20);
        {
            let buf = agent.replay_buffer.as_mut().unwrap();
            for i in 0..20 {
                let tx = make_tx(100.0 + i as f64, 0.5);
                recent_originals.push(tx.clone());
                buf.push(tx);
            }
        }
        assert_eq!(
            agent.replay_buffer.as_ref().unwrap().recent_memories.len(),
            20
        );

        // Save and reload.
        let path = temp_path("test_save_load_preserves_replay_buffer.json");
        save_agent(&agent, &path, 123, None).unwrap();
        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        // Buffer must be present on the loaded agent.
        let loaded_buf = loaded
            .replay_buffer
            .as_ref()
            .expect("loaded agent must have a replay buffer after round-trip");

        // Compartment lengths must match exactly.
        assert_eq!(
            loaded_buf.training_memories.len(),
            30,
            "training compartment size must survive round-trip"
        );
        assert_eq!(
            loaded_buf.recent_memories.len(),
            20,
            "recent compartment size must survive round-trip"
        );

        // Seal flag must survive.
        assert!(
            !loaded_buf.training_phase,
            "training_phase must remain false after round-trip (seal was called before save)"
        );

        // Content equality for each compartment, element-wise.
        for (i, (expected, actual)) in training_originals
            .iter()
            .zip(loaded_buf.training_memories.iter())
            .enumerate()
        {
            assert_eq!(
                expected, actual,
                "training_memories[{i}] must match original after round-trip"
            );
        }
        for (i, (expected, actual)) in recent_originals
            .iter()
            .zip(loaded_buf.recent_memories.iter())
            .enumerate()
        {
            assert_eq!(
                expected, actual,
                "recent_memories[{i}] must match original after round-trip"
            );
        }

        let _ = fs::remove_file(&path);
    }

    /// Red test — a pre-Phase-2 save file (no `replay_buffer` field in the
    /// JSON) must load as a freshly-allocated empty buffer when the loaded
    /// config has `replay_training_capacity > 0`. This preserves the
    /// invariant established by `PcActorCritic::new`: capacity > 0 ⇒
    /// `replay_buffer.is_some()`.
    #[test]
    fn test_legacy_save_file_replay_buffer_none() {
        use crate::linalg::cpu::CpuLinAlg;

        // Build an agent with a configured (but empty) replay buffer, save
        // it with the current code, strip any `replay_buffer` key to mimic
        // a pre-Phase-2 save file, then load and verify the buffer was
        // allocated fresh from the config.
        let mut config = default_config();
        config.replay_training_capacity = 100;
        config.replay_recent_capacity = 50;

        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();
        assert!(
            agent.replay_buffer.is_some(),
            "construction must allocate a buffer when capacity > 0"
        );
        assert_eq!(
            agent.replay_buffer.as_ref().unwrap().total_len(),
            0,
            "newly-constructed buffer must be empty"
        );

        let path = temp_path("test_legacy_save_file_replay_buffer_none.json");
        save_agent(&agent, &path, 0, None).unwrap();

        // Strip any `replay_buffer` key at the top level so the file looks
        // exactly like a Phase-1 save (no replay_buffer field at all).
        let json_str = fs::read_to_string(&path).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        if let Some(obj) = json.as_object_mut() {
            obj.remove("replay_buffer");
        }
        fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        // Load — the deserializer must tolerate the missing field (legacy
        // compat) and `load_agent` must allocate a fresh empty buffer from
        // the config's replay_training_capacity > 0.
        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        assert!(
            loaded.replay_buffer.is_some(),
            "legacy save file must still yield a buffer when config has replay_training_capacity > 0"
        );
        let buf = loaded.replay_buffer.as_ref().unwrap();
        assert_eq!(
            buf.total_len(),
            0,
            "legacy load must produce an empty buffer (no transitions in file)"
        );
        assert!(
            buf.training_phase,
            "fresh buffer must start in training_phase = true"
        );

        let _ = fs::remove_file(&path);
    }

    /// Locks the MAGI Caspar fix for serialized telemetry + cooldown
    /// state. A save/load cycle must preserve `replay_clamp_count`,
    /// `steps_since_last_rollback_hard`, and `rollback_hard_cooldown_steps`
    /// exactly — dashboards cannot see an artificial counter reset and
    /// the cooldown gate cannot be silently bypassed by reloading a file
    /// taken mid-window.
    #[test]
    fn test_save_load_preserves_clamp_count_and_cooldown_state() {
        use crate::linalg::cpu::CpuLinAlg;

        let mut config = default_config();
        config.replay_training_capacity = 100;
        config.replay_recent_capacity = 0;

        let mut agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        // Set a custom cooldown window, consume most of it, and tick
        // the clamp counter so none of the three fields match their
        // construction defaults (u64::MAX, 100, 0).
        agent.rollback_hard_cooldown_steps = 777;
        agent.steps_since_last_rollback_hard = 42;
        agent.replay_clamp_count = 9;

        let path = temp_path("test_save_load_preserves_clamp_count_and_cooldown_state.json");
        save_agent(&agent, &path, 0, None).unwrap();

        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        assert_eq!(
            loaded.replay_clamp_count, 9,
            "replay_clamp_count must survive save/load"
        );
        assert_eq!(
            loaded.steps_since_last_rollback_hard, 42,
            "steps_since_last_rollback_hard must survive save/load — \
             otherwise a mid-cooldown save/reload silently bypasses the gate"
        );
        assert_eq!(
            loaded.rollback_hard_cooldown_steps, 777,
            "rollback_hard_cooldown_steps must survive save/load — \
             otherwise set_rollback_hard_cooldown is silently reverted"
        );

        let _ = fs::remove_file(&path);
    }

    /// Legacy compat for the three newly-persisted fields: a save file
    /// lacking any of them must still load with the pre-W2-fix
    /// bootstrap defaults (clamp_count = 0, cooldown unlocked via
    /// `u64::MAX`, window = `DEFAULT_ROLLBACK_HARD_COOLDOWN`).
    #[test]
    fn test_legacy_save_file_defaults_clamp_count_and_cooldown_state() {
        use crate::linalg::cpu::CpuLinAlg;

        let config = default_config();
        let agent: PcActorCritic = PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

        let path = temp_path("test_legacy_save_file_defaults_clamp_count_and_cooldown_state.json");
        save_agent(&agent, &path, 0, None).unwrap();

        // Strip the three newly-added fields to mimic a pre-W2-fix save.
        let json_str = fs::read_to_string(&path).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        if let Some(obj) = json.as_object_mut() {
            obj.remove("replay_clamp_count");
            obj.remove("steps_since_last_rollback_hard");
            obj.remove("rollback_hard_cooldown_steps");
        }
        fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let (loaded, _) = load_agent(&path, CpuLinAlg::new()).unwrap();

        assert_eq!(loaded.replay_clamp_count, 0);
        assert_eq!(loaded.steps_since_last_rollback_hard, u64::MAX);
        assert_eq!(
            loaded.rollback_hard_cooldown_steps,
            crate::pc_actor_critic::DEFAULT_ROLLBACK_HARD_COOLDOWN
        );

        let _ = fs::remove_file(&path);
    }
}
