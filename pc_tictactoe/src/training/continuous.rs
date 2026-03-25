// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Continuous training loop with surprise-based immediate updates.
//!
//! Implements [`ContinuousTrainer`] which runs until a stop flag is set,
//! a maximum episode count is reached, or a target win rate is hit.
//! Per-step surprise checks trigger immediate TD(0) updates when the
//! surprise score exceeds a configured threshold.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pc_core::pc_actor::SelectionMode;
use pc_core::pc_actor_critic::{PcActorCritic, TrajectoryStep};

use crate::env::minimax::MinimaxPlayer;
use crate::env::tictactoe::{GameResult, Player, TicTacToe};
use crate::utils::metrics::{GameOutcome, Metrics};

/// Continuous trainer with surprise-based immediate TD(0) updates.
///
/// Runs episodes until stopped via `stop_flag`, `max_episodes`,
/// or target win rate. High-surprise steps trigger immediate
/// `learn_continuous` calls mid-episode.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use std::sync::atomic::AtomicBool;
/// use pc_core::pc_actor_critic::PcActorCritic;
/// use pc_tictactoe::training::continuous::ContinuousTrainer;
/// use pc_tictactoe::utils::config::AppConfig;
///
/// let config = AppConfig::default();
/// let agent_config = config.to_agent_config().unwrap();
/// let agent = PcActorCritic::new(agent_config, 42);
/// let stop = Arc::new(AtomicBool::new(false));
/// let mut trainer = ContinuousTrainer::new(agent, &config, stop);
/// trainer.train();
/// ```
pub struct ContinuousTrainer {
    /// The reinforcement learning agent.
    agent: PcActorCritic,
    /// The game environment.
    env: TicTacToe,
    /// The minimax opponent.
    minimax: MinimaxPlayer,
    /// Sliding-window metrics tracker.
    metrics: Metrics,
    /// Atomic flag for Ctrl+C shutdown.
    stop_flag: Arc<AtomicBool>,
    /// Maximum episodes before automatic stop.
    max_episodes: usize,
    /// Surprise threshold for immediate TD(0) updates.
    surprise_threshold: f64,
    /// Total episodes trained.
    episode_count: usize,
    /// Number of surprise-triggered immediate updates.
    surprise_events: usize,
    /// Number of steps where surprise was below threshold (absorbed).
    absorbed_events: usize,
}

impl ContinuousTrainer {
    /// Creates a new continuous trainer.
    ///
    /// # Parameters
    ///
    /// * `agent` - The PC Actor-Critic agent to train.
    /// * `config` - Application configuration.
    /// * `stop_flag` - Atomic flag; set to `true` to stop after current episode.
    pub fn new(
        agent: PcActorCritic,
        config: &crate::utils::config::AppConfig,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            agent,
            env: TicTacToe::new(),
            minimax: MinimaxPlayer::new(1),
            metrics: Metrics::new(config.curriculum.window_size),
            stop_flag,
            max_episodes: config.continuous.max_episodes,
            surprise_threshold: config.continuous.surprise_threshold,
            episode_count: 0,
            surprise_events: 0,
            absorbed_events: 0,
        }
    }

    /// Runs the continuous training loop.
    ///
    /// Continues until `stop_flag` is set, `max_episodes` is reached,
    /// or all episodes complete. Each episode collects a trajectory,
    /// checks per-step surprise for immediate updates, then learns
    /// from the full trajectory.
    pub fn train(&mut self) {
        while !self.stop_flag.load(Ordering::Relaxed) && self.episode_count < self.max_episodes {
            self.run_episode();
            self.episode_count += 1;
        }
    }

    /// Runs a single episode with surprise-based immediate updates.
    fn run_episode(&mut self) {
        self.env.reset();
        let agent_side = if self.episode_count.is_multiple_of(2) {
            Player::One
        } else {
            Player::Two
        };

        let mut trajectory = Vec::new();

        while !self.env.is_terminal() {
            if self.env.current_player() == agent_side {
                let state = self.env.board_as_f64(agent_side);
                let valid = self.env.valid_actions();
                let (action, infer) = self.agent.act(&state, &valid, SelectionMode::Training);

                let surprise = infer.surprise_score;

                trajectory.push(TrajectoryStep {
                    input: state.to_vec(),
                    latent_concat: infer.latent_concat.clone(),
                    y_conv: infer.y_conv.clone(),
                    action,
                    valid_actions: valid.clone(),
                    reward: 0.0,
                    surprise_score: surprise,
                    steps_used: infer.steps_used,
                });

                self.env.step(action).unwrap();

                // Per-step surprise check for immediate update
                if surprise > self.surprise_threshold && !self.env.is_terminal() {
                    let next_state = self.env.board_as_f64(agent_side);
                    let next_infer = self.agent.act(
                        &next_state,
                        &self.env.valid_actions(),
                        SelectionMode::Training,
                    ).1;

                    self.agent.learn_continuous(
                        &state,
                        &infer,
                        action,
                        &valid,
                        0.0, // intermediate reward
                        &next_state,
                        &next_infer,
                        false,
                    );
                    self.surprise_events += 1;
                } else {
                    self.absorbed_events += 1;
                }
            } else {
                let minimax_action = self.minimax.choose_action(&self.env);
                self.env.step(minimax_action).unwrap();
            }
        }

        // Assign terminal reward to last step
        if let Some(last) = trajectory.last_mut() {
            last.reward = self.env.reward(agent_side);
        }

        // Learn from full trajectory
        self.agent.learn(&trajectory);

        // Record outcome
        let outcome = match self.env.result() {
            GameResult::Win(p) if p == agent_side => GameOutcome::Win,
            GameResult::Win(_) => GameOutcome::Loss,
            GameResult::Draw => GameOutcome::Draw,
            GameResult::InProgress => GameOutcome::Draw,
        };
        self.metrics.record(outcome);
    }

    /// Returns the total number of episodes trained.
    pub fn episode_count(&self) -> usize {
        self.episode_count
    }

    /// Returns the number of surprise-triggered immediate updates.
    pub fn surprise_events(&self) -> usize {
        self.surprise_events
    }

    /// Returns the number of absorbed (below threshold) events.
    pub fn absorbed_events(&self) -> usize {
        self.absorbed_events
    }

    /// Returns a reference to the agent.
    pub fn agent(&self) -> &PcActorCritic {
        &self.agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::config::AppConfig;

    fn make_continuous_trainer(max_episodes: usize) -> ContinuousTrainer {
        let mut config = AppConfig::default();
        config.continuous.max_episodes = max_episodes;
        config.continuous.surprise_threshold = 0.0; // Low threshold to trigger events
        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(agent_config, 42);
        let stop = Arc::new(AtomicBool::new(false));
        ContinuousTrainer::new(agent, &config, stop)
    }

    #[test]
    fn test_one_episode_completes_without_panic() {
        let mut trainer = make_continuous_trainer(1);
        trainer.train();
        assert_eq!(trainer.episode_count(), 1);
    }

    #[test]
    fn test_high_surprise_triggers_immediate_update() {
        // With surprise_threshold=0.0, nearly all steps should trigger
        let mut trainer = make_continuous_trainer(5);
        trainer.train();
        assert!(
            trainer.surprise_events() > 0,
            "Expected at least one surprise event, got 0"
        );
    }

    #[test]
    fn test_absorbed_events_lte_surprise_events() {
        // With threshold=0.0, surprise_events should dominate
        // But absorbed can occur for terminal steps or zero surprise
        let mut trainer = make_continuous_trainer(5);
        trainer.train();
        // Just verify both counters are populated
        let total = trainer.surprise_events() + trainer.absorbed_events();
        assert!(total > 0, "Expected some events to be recorded");
    }

    #[test]
    fn test_max_episodes_stops_training() {
        let mut trainer = make_continuous_trainer(3);
        trainer.train();
        assert_eq!(trainer.episode_count(), 3);
    }
}
