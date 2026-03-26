// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Clap-based CLI with subcommands for training, playing, evaluating, and
//! benchmarking the PC Actor-Critic agent on Tic-Tac-Toe.
//!
//! # Subcommands
//!
//! - **train** — Run episodic or continuous training.
//! - **play** — Interactive text-based game against the agent.
//! - **evaluate** — Win/draw/loss statistics vs minimax at a given depth.
//! - **benchmark** — Timing and throughput metrics for training.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};

use pc_core::pc_actor::SelectionMode;
use pc_core::pc_actor_critic::PcActorCritic;
use pc_core::serializer::{load_agent, save_agent};

use crate::env::minimax::MinimaxPlayer;
use crate::env::tictactoe::{GameResult, Player, TicTacToe};
use crate::training::continuous::ContinuousTrainer;
use crate::training::trainer::Trainer;
use crate::utils::config::AppConfig;

/// PC-TicTacToe: Predictive Coding Actor-Critic for Tic-Tac-Toe.
#[derive(Parser)]
#[command(name = "pc_tictactoe", version, about)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Command {
    /// Train the agent against minimax opponents.
    Train(TrainArgs),
    /// Play an interactive game against the agent.
    Play(PlayArgs),
    /// Evaluate the agent vs minimax and print statistics.
    Evaluate(EvaluateArgs),
    /// Benchmark training throughput.
    Benchmark(BenchmarkArgs),
}

/// Arguments for the train subcommand.
#[derive(Parser)]
pub struct TrainArgs {
    /// Number of training episodes.
    #[arg(long, short)]
    pub episodes: Option<usize>,
    /// Path to TOML configuration file.
    #[arg(long, short, default_value = "config.toml")]
    pub config: String,
    /// Use continuous learning mode instead of episodic.
    #[arg(long)]
    pub continuous: bool,
    /// Maximum episodes for continuous mode.
    #[arg(long)]
    pub max_episodes: Option<usize>,
    /// Target win rate for curriculum advancement.
    #[arg(long)]
    pub target_winrate: Option<f64>,
    /// Blend factor: 1.0 = pure backprop, 0.0 = pure local PC, intermediate = hybrid.
    #[arg(long)]
    pub local_lambda: Option<f64>,
}

/// Arguments for the play subcommand.
#[derive(Parser)]
pub struct PlayArgs {
    /// Path to a saved model file.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Play as first player (agent goes second).
    #[arg(long)]
    pub first: bool,
}

/// Arguments for the evaluate subcommand.
#[derive(Parser)]
pub struct EvaluateArgs {
    /// Path to a saved model file.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Number of evaluation games.
    #[arg(long, short, default_value = "100")]
    pub games: usize,
    /// Minimax search depth for the opponent.
    #[arg(long, short, default_value = "9")]
    pub depth: usize,
}

/// Arguments for the benchmark subcommand.
#[derive(Parser)]
pub struct BenchmarkArgs {
    /// Path to a saved model file.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Number of training episodes for the benchmark.
    #[arg(long, short, default_value = "100")]
    pub episodes: usize,
}

/// Runs the train subcommand.
///
/// Loads config, creates agent + trainer, trains, and saves the final model.
///
/// # Arguments
///
/// * `args` - Training arguments from CLI.
///
/// # Errors
///
/// Returns an error on config/IO failures.
pub fn run_train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = AppConfig::load(Path::new(&args.config))?;
    config.apply_cli_overrides(args.episodes, None);

    if let Some(wr) = args.target_winrate {
        config.curriculum.advance_threshold = wr;
    }

    if let Some(lambda) = args.local_lambda {
        config.agent.actor.local_lambda = lambda;
    }

    config.validate()?;
    let agent_config = config.to_agent_config()?;
    let agent = PcActorCritic::new(agent_config, config.training.seed)?;

    if args.continuous {
        if let Some(max_ep) = args.max_episodes {
            config.continuous.max_episodes = max_ep;
        }
        let stop_flag = Arc::new(AtomicBool::new(false));
        let flag = stop_flag.clone();
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        });
        let mut trainer = ContinuousTrainer::new(agent, &config, stop_flag);
        trainer.train();
        save_agent(
            trainer.agent(),
            "model.json",
            config.continuous.max_episodes,
            None,
        )?;
    } else {
        let episodes = config.training.episodes;
        let mut trainer = Trainer::new(agent, &config);
        trainer.train(episodes);
        save_agent(trainer.agent(), "model.json", episodes, None)?;
    }

    println!("Training complete. Model saved to model.json");
    Ok(())
}

/// Runs the play subcommand.
///
/// Loads a model (or creates a fresh agent) and plays an interactive game.
///
/// # Arguments
///
/// * `args` - Play arguments from CLI.
///
/// # Errors
///
/// Returns an error on IO/model failures.
pub fn run_play(args: PlayArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = if let Some(path) = &args.model {
        let (agent, _) = load_agent(path)?;
        agent
    } else {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config()?;
        PcActorCritic::new(agent_config, 42)?
    };

    let mut env = TicTacToe::new();
    let human_side = if args.first { Player::One } else { Player::Two };

    println!("You are {human_side:?}. Board positions 0-8:");
    println!(" 0 | 1 | 2 ");
    println!(" ---------  ");
    println!(" 3 | 4 | 5 ");
    println!(" ---------  ");
    println!(" 6 | 7 | 8 ");
    println!();

    let stdin = io::stdin();

    while !env.is_terminal() {
        if env.current_player() == human_side {
            print_board(&env);
            print!("Your move (0-8): ");
            io::stdout().flush()?;
            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;
            let action: usize = match line.trim().parse() {
                Ok(a) => a,
                Err(_) => {
                    println!("Invalid input. Enter a number 0-8.");
                    continue;
                }
            };
            if let Err(e) = env.step(action) {
                println!("Invalid move: {e}. Try again.");
                continue;
            }
        } else {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
            println!("Agent plays: {action}");
            env.step(action).unwrap();
        }
    }

    print_board(&env);
    match env.result() {
        GameResult::Win(p) if p == human_side => println!("You win!"),
        GameResult::Win(_) => println!("Agent wins!"),
        GameResult::Draw => println!("Draw!"),
        GameResult::InProgress => unreachable!(),
    }

    Ok(())
}

/// Prints the current board state to stdout.
///
/// # Arguments
///
/// * `env` - The TicTacToe environment.
fn print_board(env: &TicTacToe) {
    let board = env.board_as_f64(Player::One);
    for row in 0..3 {
        for col in 0..3 {
            let idx = row * 3 + col;
            let ch = if board[idx] > 0.5 {
                "X"
            } else if board[idx] < -0.5 {
                "O"
            } else {
                "."
            };
            if col < 2 {
                print!(" {ch} |");
            } else {
                println!(" {ch} ");
            }
        }
        if row < 2 {
            println!("-----------");
        }
    }
    println!();
}

/// Runs the evaluate subcommand.
///
/// Plays N games of agent vs minimax and prints win/draw/loss statistics.
///
/// # Arguments
///
/// * `args` - Evaluate arguments from CLI.
///
/// # Errors
///
/// Returns an error on model loading failures.
pub fn run_evaluate(args: EvaluateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = if let Some(path) = &args.model {
        let (agent, _) = load_agent(path)?;
        agent
    } else {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config()?;
        PcActorCritic::new(agent_config, 42)?
    };

    let mut minimax = MinimaxPlayer::new(args.depth);
    let mut wins = 0usize;
    let mut draws = 0usize;
    let mut losses = 0usize;

    for game_idx in 0..args.games {
        let mut env = TicTacToe::new();
        let agent_side = if game_idx.is_multiple_of(2) {
            Player::One
        } else {
            Player::Two
        };

        while !env.is_terminal() {
            if env.current_player() == agent_side {
                let state = env.board_as_f64(agent_side);
                let valid = env.valid_actions();
                let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
                env.step(action).unwrap();
            } else {
                let action = minimax.choose_action(&env);
                env.step(action).unwrap();
            }
        }

        match env.result() {
            GameResult::Win(p) if p == agent_side => wins += 1,
            GameResult::Win(_) => losses += 1,
            GameResult::Draw => draws += 1,
            GameResult::InProgress => {}
        }
    }

    println!(
        "Evaluation: {games} games vs minimax depth {depth}",
        games = args.games,
        depth = args.depth
    );
    println!(
        "  Wins:   {wins} ({:.1}%)",
        100.0 * wins as f64 / args.games as f64
    );
    println!(
        "  Draws:  {draws} ({:.1}%)",
        100.0 * draws as f64 / args.games as f64
    );
    println!(
        "  Losses: {losses} ({:.1}%)",
        100.0 * losses as f64 / args.games as f64
    );

    Ok(())
}

/// Runs the benchmark subcommand.
///
/// Times training for a given number of episodes and reports throughput.
///
/// # Arguments
///
/// * `args` - Benchmark arguments from CLI.
///
/// # Errors
///
/// Returns an error on config/model failures.
pub fn run_benchmark(args: BenchmarkArgs) -> Result<(), Box<dyn std::error::Error>> {
    let agent = if let Some(path) = &args.model {
        let (agent, _) = load_agent(path)?;
        agent
    } else {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config()?;
        PcActorCritic::new(agent_config, 42)?
    };

    let config = AppConfig::default();
    let mut trainer = Trainer::new(agent, &config);

    let start = Instant::now();
    trainer.train(args.episodes);
    let elapsed = start.elapsed();

    let eps_per_sec = args.episodes as f64 / elapsed.as_secs_f64();
    println!(
        "Benchmark: {ep} episodes in {elapsed:.2?} ({eps_per_sec:.1} episodes/sec)",
        ep = args.episodes
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_help_parses() {
        // Verify CLI struct can be constructed — clap derive is valid
        use clap::CommandFactory;
        let cmd = Cli::command();
        assert!(cmd.get_name() == "pc_tictactoe");
    }

    #[test]
    fn test_all_subcommands_have_help() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let subs: Vec<&str> = cmd.get_subcommands().map(|s| s.get_name()).collect();
        assert!(subs.contains(&"train"));
        assert!(subs.contains(&"play"));
        assert!(subs.contains(&"evaluate"));
        assert!(subs.contains(&"benchmark"));
    }
}
