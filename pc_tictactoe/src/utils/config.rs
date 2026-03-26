// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! TOML-based application configuration.
//!
//! Parses a TOML config file into [`AppConfig`], validates topology
//! consistency (critic input size must match actor input + latent size),
//! and converts to pc_core agent configuration types.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

use pc_core::activation::Activation;
use pc_core::layer::LayerDef;
use pc_core::mlp_critic::MlpCriticConfig;
use pc_core::pc_actor::PcActorConfig;
use pc_core::pc_actor_critic::PcActorCriticConfig;
use serde::Deserialize;

/// Top-level application configuration parsed from TOML.
///
/// Contains all settings for the agent, training loop, curriculum,
/// continuous learning, and logging.
///
/// # Examples
///
/// ```
/// use pc_tictactoe::utils::config::AppConfig;
///
/// let config = AppConfig::default();
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AppConfig {
    /// Agent architecture configuration.
    #[serde(default)]
    pub agent: AgentSection,
    /// Training loop configuration.
    #[serde(default)]
    pub training: TrainingSection,
    /// Curriculum schedule configuration.
    #[serde(default)]
    pub curriculum: CurriculumSection,
    /// Continuous learning configuration.
    #[serde(default)]
    pub continuous: ContinuousSection,
    /// Logger configuration.
    #[serde(default)]
    pub logger: LoggerSection,
}

/// Agent architecture: actor, critic, and shared hyperparameters.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentSection {
    /// Actor sub-configuration.
    #[serde(default)]
    pub actor: ActorSection,
    /// Critic sub-configuration.
    #[serde(default)]
    pub critic: CriticSection,
    /// Discount factor for returns.
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Surprise threshold below which learning rate is reduced.
    #[serde(default = "default_surprise_low")]
    pub surprise_low: f64,
    /// Surprise threshold above which learning rate is increased.
    #[serde(default = "default_surprise_high")]
    pub surprise_high: f64,
    /// Whether to adaptively recalibrate surprise thresholds.
    #[serde(default)]
    pub adaptive_surprise: bool,
    /// Entropy regularization coefficient.
    #[serde(default = "default_entropy_coeff")]
    pub entropy_coeff: f64,
}

/// Actor network configuration section.
#[derive(Debug, Clone, Deserialize)]
pub struct ActorSection {
    /// Number of input features.
    #[serde(default = "default_input_size")]
    pub input_size: usize,
    /// Hidden layer definitions.
    #[serde(default = "default_actor_hidden")]
    pub hidden_layers: Vec<HiddenLayerDef>,
    /// Number of output actions.
    #[serde(default = "default_output_size")]
    pub output_size: usize,
    /// Activation for the output layer.
    #[serde(default = "default_activation_tanh")]
    pub output_activation: String,
    /// PC inference update rate.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// Convergence tolerance.
    #[serde(default = "default_tol")]
    pub tol: f64,
    /// Minimum inference steps.
    #[serde(default = "default_min_steps")]
    pub min_steps: usize,
    /// Maximum inference steps.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// Weight learning rate.
    #[serde(default = "default_lr_weights")]
    pub lr_weights: f64,
    /// Whether to use synchronous PC updates.
    #[serde(default = "default_true")]
    pub synchronous: bool,
    /// Softmax temperature for action selection.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Use local PC prediction errors for weight updates (Millidge et al. 2022).
    #[serde(default)]
    pub local_learning: bool,
}

/// Critic network configuration section.
#[derive(Debug, Clone, Deserialize)]
pub struct CriticSection {
    /// Dimensionality of critic input (board state + actor latent).
    #[serde(default = "default_critic_input_size")]
    pub input_size: usize,
    /// Hidden layer definitions.
    #[serde(default = "default_critic_hidden")]
    pub hidden_layers: Vec<HiddenLayerDef>,
    /// Activation for the output layer.
    #[serde(default = "default_activation_linear")]
    pub output_activation: String,
    /// Learning rate.
    #[serde(default = "default_critic_lr")]
    pub lr: f64,
}

/// Definition of a single hidden layer in TOML format.
#[derive(Debug, Clone, Deserialize)]
pub struct HiddenLayerDef {
    /// Number of neurons.
    pub size: usize,
    /// Activation function name (case-insensitive).
    pub activation: String,
}

/// Training loop configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingSection {
    /// Number of episodes to train.
    #[serde(default = "default_episodes")]
    pub episodes: usize,
    /// Checkpoint interval (episodes).
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,
    /// How often to print progress stats (every N episodes).
    #[serde(default = "default_log_interval")]
    pub log_interval: usize,
    /// Random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

/// Curriculum schedule configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct CurriculumSection {
    /// Win rate threshold to advance difficulty.
    #[serde(default = "default_advance_threshold")]
    pub advance_threshold: f64,
    /// Window size for measuring win rate.
    #[serde(default = "default_window_size")]
    pub window_size: usize,
}

/// Continuous learning configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ContinuousSection {
    /// Maximum episodes for continuous mode.
    #[serde(default = "default_max_episodes")]
    pub max_episodes: usize,
    /// Surprise threshold for immediate updates.
    #[serde(default = "default_surprise_threshold")]
    pub surprise_threshold: f64,
}

/// Logger configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LoggerSection {
    /// Minimum log level: "debug", "info", "training", "warning", "error".
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Path to log file (optional).
    #[serde(default)]
    pub file: Option<String>,
    /// Path to CSV metrics file (optional).
    #[serde(default)]
    pub csv_file: Option<String>,
    /// Maximum number of log file backups for rotation.
    #[serde(default = "default_max_backups")]
    pub max_backups: usize,
    /// Maximum log file size in bytes before rotation.
    #[serde(default = "default_max_size")]
    pub max_size: u64,
}

/// Configuration validation error.
#[derive(Debug)]
pub struct ConfigError {
    /// Human-readable description of the validation failure.
    pub message: String,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config validation error: {}", self.message)
    }
}

impl Error for ConfigError {}

// ─── Default value functions ────────────────────────────────────────────────

fn default_input_size() -> usize {
    9
}
fn default_output_size() -> usize {
    9
}
fn default_critic_input_size() -> usize {
    27
}
fn default_gamma() -> f64 {
    0.95
}
fn default_surprise_low() -> f64 {
    0.02
}
fn default_surprise_high() -> f64 {
    0.15
}
fn default_entropy_coeff() -> f64 {
    0.01
}
fn default_alpha() -> f64 {
    0.1
}
fn default_tol() -> f64 {
    0.01
}
fn default_min_steps() -> usize {
    1
}
fn default_max_steps() -> usize {
    20
}
fn default_lr_weights() -> f64 {
    0.01
}
fn default_true() -> bool {
    true
}
fn default_temperature() -> f64 {
    1.0
}
fn default_critic_lr() -> f64 {
    0.005
}
fn default_episodes() -> usize {
    10000
}
fn default_checkpoint_interval() -> usize {
    1000
}
fn default_log_interval() -> usize {
    500
}
fn default_seed() -> u64 {
    42
}
fn default_advance_threshold() -> f64 {
    0.6
}
fn default_window_size() -> usize {
    100
}
fn default_max_episodes() -> usize {
    50000
}
fn default_surprise_threshold() -> f64 {
    0.1
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_max_backups() -> usize {
    3
}
fn default_max_size() -> u64 {
    10_485_760
}
fn default_activation_tanh() -> String {
    "tanh".to_string()
}
fn default_activation_linear() -> String {
    "linear".to_string()
}

fn default_actor_hidden() -> Vec<HiddenLayerDef> {
    vec![HiddenLayerDef {
        size: 18,
        activation: "tanh".to_string(),
    }]
}

fn default_critic_hidden() -> Vec<HiddenLayerDef> {
    vec![HiddenLayerDef {
        size: 36,
        activation: "tanh".to_string(),
    }]
}

// ─── Default trait impls ────────────────────────────────────────────────────

impl Default for AgentSection {
    fn default() -> Self {
        Self {
            actor: ActorSection::default(),
            critic: CriticSection::default(),
            gamma: default_gamma(),
            surprise_low: default_surprise_low(),
            surprise_high: default_surprise_high(),
            adaptive_surprise: false,
            entropy_coeff: default_entropy_coeff(),
        }
    }
}

impl Default for ActorSection {
    fn default() -> Self {
        Self {
            input_size: default_input_size(),
            hidden_layers: default_actor_hidden(),
            output_size: default_output_size(),
            output_activation: default_activation_tanh(),
            alpha: default_alpha(),
            tol: default_tol(),
            min_steps: default_min_steps(),
            max_steps: default_max_steps(),
            lr_weights: default_lr_weights(),
            synchronous: default_true(),
            temperature: default_temperature(),
            local_learning: false,
        }
    }
}

impl Default for CriticSection {
    fn default() -> Self {
        Self {
            input_size: default_critic_input_size(),
            hidden_layers: default_critic_hidden(),
            output_activation: default_activation_linear(),
            lr: default_critic_lr(),
        }
    }
}

impl Default for TrainingSection {
    fn default() -> Self {
        Self {
            episodes: default_episodes(),
            checkpoint_interval: default_checkpoint_interval(),
            log_interval: default_log_interval(),
            seed: default_seed(),
        }
    }
}

impl Default for CurriculumSection {
    fn default() -> Self {
        Self {
            advance_threshold: default_advance_threshold(),
            window_size: default_window_size(),
        }
    }
}

impl Default for ContinuousSection {
    fn default() -> Self {
        Self {
            max_episodes: default_max_episodes(),
            surprise_threshold: default_surprise_threshold(),
        }
    }
}

impl Default for LoggerSection {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
            csv_file: None,
            max_backups: default_max_backups(),
            max_size: default_max_size(),
        }
    }
}

// ─── Parsing helpers ────────────────────────────────────────────────────────

/// Parses an activation name string (case-insensitive) into an [`Activation`].
///
/// # Errors
///
/// Returns a [`ConfigError`] if the string does not match a known activation.
///
/// # Parameters
///
/// * `s` - Activation name (e.g. "tanh", "relu", "sigmoid", "elu", "linear").
fn parse_activation(s: &str) -> Result<Activation, ConfigError> {
    match s.to_lowercase().as_str() {
        "tanh" => Ok(Activation::Tanh),
        "relu" => Ok(Activation::Relu),
        "sigmoid" => Ok(Activation::Sigmoid),
        "elu" => Ok(Activation::Elu),
        "linear" => Ok(Activation::Linear),
        other => Err(ConfigError {
            message: format!(
                "unknown activation '{other}'; expected tanh, relu, sigmoid, elu, or linear"
            ),
        }),
    }
}

/// Converts a slice of [`HiddenLayerDef`] to pc_core [`LayerDef`] values.
///
/// # Errors
///
/// Returns a [`ConfigError`] if any activation string is invalid.
fn convert_hidden(defs: &[HiddenLayerDef]) -> Result<Vec<LayerDef>, ConfigError> {
    defs.iter()
        .map(|h| {
            Ok(LayerDef {
                size: h.size,
                activation: parse_activation(&h.activation)?,
            })
        })
        .collect()
}

// ─── AppConfig methods ──────────────────────────────────────────────────────

impl AppConfig {
    /// Loads configuration from a TOML file, falling back to defaults if the
    /// file does not exist.
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the TOML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be parsed.
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&text)?;
        Ok(config)
    }

    /// Validates topology consistency.
    ///
    /// Checks that `critic.input_size == actor.input_size + sum(actor.hidden_layers[i].size)`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] describing the mismatch.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let latent_sum: usize = self.agent.actor.hidden_layers.iter().map(|h| h.size).sum();
        let expected = self.agent.actor.input_size + latent_sum;
        let actual = self.agent.critic.input_size;
        if actual != expected {
            return Err(ConfigError {
                message: format!(
                    "critic.input_size ({actual}) != actor.input_size ({}) + sum(hidden sizes) ({latent_sum}) = {expected}",
                    self.agent.actor.input_size
                ),
            });
        }
        Ok(())
    }

    /// Converts the TOML config to a pc_core [`PcActorCriticConfig`].
    ///
    /// Calls [`validate`](AppConfig::validate) first, then parses activation
    /// strings into enum variants.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails or an activation string is unknown.
    pub fn to_agent_config(&self) -> Result<PcActorCriticConfig, Box<dyn Error>> {
        self.validate()?;
        let actor_hidden = convert_hidden(&self.agent.actor.hidden_layers)?;
        let critic_hidden = convert_hidden(&self.agent.critic.hidden_layers)?;
        let output_act = parse_activation(&self.agent.actor.output_activation)?;
        let critic_output_act = parse_activation(&self.agent.critic.output_activation)?;

        Ok(PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: self.agent.actor.input_size,
                hidden_layers: actor_hidden,
                output_size: self.agent.actor.output_size,
                output_activation: output_act,
                alpha: self.agent.actor.alpha,
                tol: self.agent.actor.tol,
                min_steps: self.agent.actor.min_steps,
                max_steps: self.agent.actor.max_steps,
                lr_weights: self.agent.actor.lr_weights,
                synchronous: self.agent.actor.synchronous,
                temperature: self.agent.actor.temperature,
                local_learning: self.agent.actor.local_learning,
            },
            critic: MlpCriticConfig {
                input_size: self.agent.critic.input_size,
                hidden_layers: critic_hidden,
                output_activation: critic_output_act,
                lr: self.agent.critic.lr,
            },
            gamma: self.agent.gamma,
            surprise_low: self.agent.surprise_low,
            surprise_high: self.agent.surprise_high,
            adaptive_surprise: self.agent.adaptive_surprise,
            entropy_coeff: self.agent.entropy_coeff,
        })
    }

    /// Applies CLI overrides to the configuration.
    ///
    /// Only overrides fields for which a `Some` value is provided.
    ///
    /// # Parameters
    ///
    /// * `episodes` - Override for `training.episodes`.
    /// * `seed` - Override for `training.seed`.
    pub fn apply_cli_overrides(&mut self, episodes: Option<usize>, seed: Option<u64>) {
        if let Some(ep) = episodes {
            self.training.episodes = ep;
        }
        if let Some(s) = seed {
            self.training.seed = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_passes_validation() {
        let config = AppConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_inconsistent_critic_input_fails_validation() {
        let mut config = AppConfig::default();
        config.agent.critic.input_size = 999;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_to_agent_config_succeeds_for_valid_config() {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config();
        assert!(agent_config.is_ok());
        let ac = agent_config.unwrap();
        assert_eq!(ac.actor.input_size, 9);
        assert_eq!(ac.actor.output_size, 9);
        assert_eq!(ac.critic.input_size, 27);
    }

    #[test]
    fn test_to_agent_config_fails_for_invalid_config() {
        let mut config = AppConfig::default();
        config.agent.critic.input_size = 10;
        assert!(config.to_agent_config().is_err());
    }

    #[test]
    fn test_toml_deserialization_default_config_succeeds() {
        let toml_str = r#"
[agent]
gamma = 0.99

[agent.actor]
input_size = 9
output_size = 9
output_activation = "tanh"

[[agent.actor.hidden_layers]]
size = 18
activation = "tanh"

[agent.critic]
input_size = 27

[[agent.critic.hidden_layers]]
size = 36
activation = "tanh"

[training]
episodes = 5000
"#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.agent.gamma, 0.99);
        assert_eq!(config.training.episodes, 5000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_two_hidden_layers_validates_correctly() {
        let mut config = AppConfig::default();
        config.agent.actor.hidden_layers = vec![
            HiddenLayerDef {
                size: 18,
                activation: "tanh".to_string(),
            },
            HiddenLayerDef {
                size: 12,
                activation: "relu".to_string(),
            },
        ];
        // critic input must be input_size + 18 + 12 = 39
        config.agent.critic.input_size = 39;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_unknown_activation_string_returns_error() {
        let result = parse_activation("swish");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file_returns_default() {
        let config = AppConfig::load(Path::new("nonexistent_config_file.toml")).unwrap();
        assert!(config.validate().is_ok());
        assert_eq!(config.agent.actor.input_size, 9);
    }

    #[test]
    fn test_cli_override_episodes_wins_over_toml() {
        let mut config = AppConfig::default();
        assert_eq!(config.training.episodes, 10000);
        config.apply_cli_overrides(Some(500), None);
        assert_eq!(config.training.episodes, 500);
    }

    #[test]
    fn test_cli_override_none_does_not_change_value() {
        let mut config = AppConfig::default();
        let original = config.training.episodes;
        config.apply_cli_overrides(None, None);
        assert_eq!(config.training.episodes, original);
    }
}
