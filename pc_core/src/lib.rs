// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Predictive Coding Actor-Critic framework.
//!
//! A publishable reinforcement learning library implementing a novel architecture where the
//! actor uses iterative top-down/bottom-up predictive coding inference loops instead of
//! standard feedforward passes, combined with a standard MLP critic for value estimation.
//!
//! # Key Components
//!
//! - [`PcActor`] — Predictive coding network with variable hidden topology, iterative
//!   inference loop, and surprise scoring.
//! - [`MlpCritic`] — Standard MLP value function with MSE loss backpropagation.
//! - [`PcActorCritic`] — Integrated agent: act, learn (episodic/continuous), surprise-based
//!   scheduling, save/load.
//! - [`serializer`] — JSON weight persistence with checkpointing support.

pub mod activation;
pub mod error;
pub mod layer;
pub mod matrix;
pub mod mlp_critic;
pub mod pc_actor;
pub mod pc_actor_critic;
pub mod serializer;

pub use activation::Activation;
pub use error::PcError;
pub use layer::{Layer, LayerDef};
pub use matrix::{
    argmax_masked, rms_error, sample_from_probs, softmax_masked, Matrix, GRAD_CLIP, WEIGHT_CLIP,
};
pub use mlp_critic::{MlpCritic, MlpCriticConfig, MlpCriticWeights};
pub use pc_actor::{InferResult, PcActor, PcActorConfig, SelectionMode};
pub use pc_actor_critic::{PcActorCritic, PcActorCriticConfig, TrajectoryStep};
pub use serializer::{
    checkpoint_filename, load_agent, save_agent, save_checkpoint, AgentMetadata, PcActorWeights,
    SaveFile, TrainingMetrics,
};
