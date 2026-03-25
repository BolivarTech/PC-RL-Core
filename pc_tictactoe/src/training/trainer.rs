// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Episode-based training loop with curriculum learning.
//!
//! Implements [`Trainer`] which runs a configurable number of episodes,
//! alternates the agent between Player One and Player Two, collects
//! trajectories, and advances minimax depth when the agent's win rate
//! exceeds a configurable threshold.

use pc_core::pc_actor::SelectionMode;
use pc_core::pc_actor_critic::{PcActorCritic, TrajectoryStep};

use crate::env::minimax::MinimaxPlayer;
use crate::env::tictactoe::{GameResult, Player, TicTacToe};
use crate::utils::metrics::{GameOutcome, Metrics};

/// Episode-based trainer with curriculum learning.
///
/// Trains the agent by playing episodes against a [`MinimaxPlayer`]
/// whose depth increases as the agent's win rate crosses a threshold.
///
/// # Examples
///
/// ```no_run
/// use pc_core::pc_actor_critic::PcActorCritic;
/// use pc_tictactoe::training::trainer::Trainer;
/// use pc_tictactoe::utils::config::AppConfig;
///
/// let config = AppConfig::default();
/// let agent_config = config.to_agent_config().unwrap();
/// let agent = PcActorCritic::new(agent_config, 42).unwrap();
/// let mut trainer = Trainer::new(agent, &config);
/// trainer.train(10);
/// ```
pub struct Trainer {
    /// The reinforcement learning agent.
    agent: PcActorCritic,
    /// The game environment.
    env: TicTacToe,
    /// The minimax opponent.
    minimax: MinimaxPlayer,
    /// Sliding-window metrics tracker.
    metrics: Metrics,
    /// Current minimax search depth (curriculum level).
    current_depth: usize,
    /// Win rate threshold to advance curriculum depth.
    advance_threshold: f64,
    /// Total episodes trained so far.
    episode_count: usize,
}

impl Trainer {
    /// Creates a new trainer from an agent and application config.
    ///
    /// # Parameters
    ///
    /// * `agent` - The PC Actor-Critic agent to train.
    /// * `config` - Application configuration with curriculum and training settings.
    pub fn new(agent: PcActorCritic, config: &crate::utils::config::AppConfig) -> Self {
        Self {
            agent,
            env: TicTacToe::new(),
            minimax: MinimaxPlayer::new(1),
            metrics: Metrics::new(config.curriculum.window_size),
            current_depth: 1,
            advance_threshold: config.curriculum.advance_threshold,
            episode_count: 0,
        }
    }

    /// Trains the agent for `num_episodes` episodes.
    ///
    /// Each episode alternates the agent's side (Player One on even
    /// episodes, Player Two on odd). After each episode, metrics are
    /// updated and curriculum advancement is checked.
    ///
    /// # Parameters
    ///
    /// * `num_episodes` - Number of episodes to run.
    pub fn train(&mut self, num_episodes: usize) {
        for _ in 0..num_episodes {
            let trajectory = self.run_episode();
            self.agent.learn(&trajectory);

            // Record outcome
            let outcome = self.episode_outcome();
            self.metrics.record(outcome);

            // Check curriculum advancement
            if self.metrics.win_rate() > self.advance_threshold && self.current_depth < 9 {
                self.current_depth += 1;
                self.minimax = MinimaxPlayer::new(self.current_depth);
            }

            self.episode_count += 1;
        }
    }

    /// Runs a single episode and returns the agent's trajectory.
    ///
    /// # Returns
    ///
    /// Vector of [`TrajectoryStep`] for the agent's actions in this episode.
    fn run_episode(&mut self) -> Vec<TrajectoryStep> {
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

                trajectory.push(TrajectoryStep {
                    input: state.to_vec(),
                    latent_concat: infer.latent_concat,
                    y_conv: infer.y_conv,
                    hidden_states: infer.hidden_states,
                    action,
                    valid_actions: valid,
                    reward: 0.0,
                    surprise_score: infer.surprise_score,
                    steps_used: infer.steps_used,
                });

                self.env.step(action).unwrap();
            } else {
                let minimax_action = self.minimax.choose_action(&self.env);
                self.env.step(minimax_action).unwrap();
            }
        }

        // Assign terminal reward to last step
        if let Some(last) = trajectory.last_mut() {
            last.reward = self.env.reward(agent_side);
        }

        trajectory
    }

    /// Determines the game outcome from the agent's perspective.
    fn episode_outcome(&self) -> GameOutcome {
        let agent_side = if self.episode_count.is_multiple_of(2) {
            Player::One
        } else {
            Player::Two
        };
        match self.env.result() {
            GameResult::Win(p) if p == agent_side => GameOutcome::Win,
            GameResult::Win(_) => GameOutcome::Loss,
            GameResult::Draw => GameOutcome::Draw,
            GameResult::InProgress => GameOutcome::Draw,
        }
    }

    /// Returns a reference to the agent.
    pub fn agent(&self) -> &PcActorCritic {
        &self.agent
    }

    /// Returns the current curriculum depth.
    pub fn current_depth(&self) -> usize {
        self.current_depth
    }

    /// Returns the total number of episodes trained.
    pub fn episode_count(&self) -> usize {
        self.episode_count
    }

    /// Returns a reference to the metrics tracker.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::config::AppConfig;

    fn make_trainer(num_episodes: usize) -> Trainer {
        let mut config = AppConfig::default();
        config.training.episodes = num_episodes;
        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(agent_config, 42).unwrap();
        Trainer::new(agent, &config)
    }

    #[test]
    fn test_training_one_episode_completes_without_panic() {
        let mut trainer = make_trainer(1);
        trainer.train(1);
        assert_eq!(trainer.episode_count(), 1);
    }

    #[test]
    fn test_episode_count_is_exact() {
        let mut trainer = make_trainer(5);
        trainer.train(5);
        assert_eq!(trainer.episode_count(), 5);
    }

    #[test]
    fn test_weights_differ_after_training() {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config().unwrap();

        // Train an agent
        let trained_agent = PcActorCritic::new(agent_config.clone(), 42).unwrap();
        let mut trainer = Trainer::new(trained_agent, &config);
        trainer.train(20);

        // Compare serialized weights of fresh vs trained agent
        use pc_core::serializer::save_agent;
        let dir = std::env::temp_dir().join(format!("pc_weight_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let fresh_path = dir.join("fresh.json");
        let trained_path = dir.join("trained.json");

        let fresh2 = PcActorCritic::new(config.to_agent_config().unwrap(), 42).unwrap();
        save_agent(&fresh2, &fresh_path.to_string_lossy(), 0, None).unwrap();
        save_agent(trainer.agent(), &trained_path.to_string_lossy(), 20, None).unwrap();

        let fresh_data = std::fs::read_to_string(&fresh_path).unwrap();
        let trained_data = std::fs::read_to_string(&trained_path).unwrap();
        assert_ne!(
            fresh_data, trained_data,
            "Weights should differ after training"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_trajectory_steps_contain_valid_actions_only() {
        let mut trainer = make_trainer(1);
        // Run one episode manually to inspect trajectory
        let trajectory = trainer.run_episode();
        for step in &trajectory {
            assert!(
                step.valid_actions.contains(&step.action),
                "Action {} not in valid actions {:?}",
                step.action,
                step.valid_actions
            );
        }
    }

    #[test]
    fn test_agent_sees_both_sides_over_episodes() {
        let mut trainer = make_trainer(4);
        // Episode 0: agent is Player::One (even)
        // Episode 1: agent is Player::Two (odd)
        // Episode 2: agent is Player::One (even)
        // Episode 3: agent is Player::Two (odd)
        trainer.train(4);
        // If we got here without panic, both sides worked
        assert_eq!(trainer.episode_count(), 4);
    }

    #[test]
    fn test_curriculum_advances_when_win_rate_exceeds_threshold() {
        let mut config = AppConfig::default();
        config.curriculum.advance_threshold = 0.0; // Always advance
        config.curriculum.window_size = 1;
        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(agent_config, 42).unwrap();
        let mut trainer = Trainer::new(agent, &config);

        // With threshold=0.0 and window=1, any win advances depth
        // Depth 1 minimax is weak; run a few episodes
        trainer.train(10);

        assert!(
            trainer.current_depth() > 1,
            "Depth should have advanced from 1, got {}",
            trainer.current_depth()
        );
    }

    #[test]
    fn test_metadata_wins_losses_draws_sum_to_episodes() {
        let mut trainer = make_trainer(20);
        trainer.train(20);
        let wr = trainer.metrics().win_rate();
        let lr = trainer.metrics().loss_rate();
        let dr = trainer.metrics().draw_rate();
        let sum = wr + lr + dr;
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Rates should sum to 1.0, got {sum}"
        );
    }
}
