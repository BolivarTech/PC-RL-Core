// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Cross-crate integration tests (CHECKPOINT 4).
//!
//! Verifies that `AppConfig` → `PcActorCritic` → game loop → save/load
//! works end-to-end.

#[cfg(test)]
mod tests {
    use pc_core::pc_actor::SelectionMode;
    use pc_core::pc_actor_critic::PcActorCritic;
    use pc_core::serializer::{load_agent, save_agent};

    use crate::env::tictactoe::TicTacToe;
    use crate::utils::config::{AppConfig, HiddenLayerDef};

    /// Creates an agent from default AppConfig.
    fn agent_from_default_config() -> PcActorCritic {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config().unwrap();
        PcActorCritic::new(agent_config, 42).unwrap()
    }

    #[test]
    fn test_agent_from_config_plays_complete_game_without_panic() {
        let mut agent = agent_from_default_config();
        let mut env = TicTacToe::new();

        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
            env.step(action).unwrap();
        }

        assert!(env.is_terminal());
    }

    #[test]
    fn test_latent_size_matches_critic_input_size() {
        let config = AppConfig::default();
        let latent_sum: usize = config
            .agent
            .actor
            .hidden_layers
            .iter()
            .map(|h| h.size)
            .sum();
        let expected_critic_input = config.agent.actor.input_size + latent_sum;
        assert_eq!(
            config.agent.critic.input_size,
            expected_critic_input,
            "critic.input_size ({}) != actor.input_size ({}) + latent_sum ({}) = {}",
            config.agent.critic.input_size,
            config.agent.actor.input_size,
            latent_sum,
            expected_critic_input,
        );
    }

    #[test]
    fn test_config_validation_catches_topology_inconsistency() {
        let mut config = AppConfig::default();
        config.agent.critic.input_size = 999;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_save_load_survives_full_session() {
        let mut agent = agent_from_default_config();
        let mut env = TicTacToe::new();

        // Play a complete game
        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Training);
            env.step(action).unwrap();
        }

        // Save
        let dir = std::env::temp_dir().join(format!("pc_integ_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("session.json");
        let path_str = path.to_string_lossy().to_string();
        save_agent(&agent, &path_str, 1, None).unwrap();

        // Load
        let (loaded, metadata) = load_agent(&path_str).unwrap();
        assert_eq!(metadata.episode, 1);
        assert_eq!(
            loaded.config.actor.input_size,
            agent.config.actor.input_size
        );
        assert_eq!(
            loaded.config.critic.input_size,
            agent.config.critic.input_size
        );

        // Loaded agent can still play
        let mut env2 = TicTacToe::new();
        let mut loaded_agent = loaded;
        while !env2.is_terminal() {
            let state = env2.board_as_f64(env2.current_player());
            let valid = env2.valid_actions();
            let (action, _) = loaded_agent.act(&state, &valid, SelectionMode::Play);
            env2.step(action).unwrap();
        }
        assert!(env2.is_terminal());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_toml_two_hidden_layers_correct_critic_input_fails_if_wrong() {
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
        // Wrong critic input: still 27 instead of 9 + 18 + 12 = 39
        config.agent.critic.input_size = 27;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_two_hidden_layers_correct_input_passes() {
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
        // Correct critic input: 9 + 18 + 12 = 39
        config.agent.critic.input_size = 39;
        assert!(config.validate().is_ok());

        // Also verify we can create an agent from this config
        let agent_config = config.to_agent_config().unwrap();
        let mut agent: PcActorCritic = PcActorCritic::new(agent_config, 42).unwrap();

        // Play a game to verify topology works end-to-end
        let mut env = TicTacToe::new();
        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
            env.step(action).unwrap();
        }
        assert!(env.is_terminal());
    }
}
