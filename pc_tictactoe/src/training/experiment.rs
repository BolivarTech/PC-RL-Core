// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-26

//! Parameter sweep experiment runner.
//!
//! Runs training across a range of hyperparameter values with random seeds,
//! collecting max depth and final metrics for each combination.
//! Supports sweeping `local_lambda` or `aux_loss_coefficient`.

use std::fmt;
use std::io::Write;

use crate::training::trainer::Trainer;
use crate::utils::config::AppConfig;
use pc_core::pc_actor_critic::PcActorCritic;

/// Which hyperparameter to sweep in the experiment.
#[derive(Debug, Clone, Copy)]
pub enum SweepParam {
    /// Sweep local_lambda [0.95, 0.96, ..., 1.00] (6 values).
    Lambda,
    /// Sweep aux_loss_coefficient [0.05, 0.10, ..., 0.50] (10 values).
    AuxLoss,
}

impl SweepParam {
    /// Returns the sweep values for this parameter.
    pub fn values(&self) -> Vec<f64> {
        match self {
            SweepParam::Lambda => vec![0.95, 0.96, 0.97, 0.98, 0.99, 1.00],
            SweepParam::AuxLoss => (1..=10).map(|i| i as f64 * 0.05).collect(),
        }
    }

    /// Returns the parameter name for display.
    pub fn name(&self) -> &'static str {
        match self {
            SweepParam::Lambda => "lambda",
            SweepParam::AuxLoss => "aux",
        }
    }
}

/// Result of a single training run within an experiment.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Random seed used for this run.
    pub seed: u64,
    /// local_lambda value used.
    pub lambda: f64,
    /// aux_loss_coefficient value used.
    pub aux_coefficient: f64,
    /// Maximum curriculum depth reached.
    pub max_depth: usize,
    /// Final win rate at end of training.
    pub win_rate: f64,
    /// Final loss rate at end of training.
    pub loss_rate: f64,
    /// Final draw rate at end of training.
    pub draw_rate: f64,
    /// Training log lines collected during the run.
    pub log_lines: Vec<String>,
}

impl fmt::Display for RunResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "============")?;
        writeln!(f, "seed={}", self.seed)?;
        writeln!(f, "lambda={:.6}", self.lambda)?;
        writeln!(f, "aux={:.2}", self.aux_coefficient)?;
        for line in &self.log_lines {
            writeln!(f, "{line}")?;
        }
        writeln!(f, "------------")
    }
}

/// Runs a single training cycle with the given seed and lambda.
///
/// Returns the [`RunResult`] with collected metrics and log lines.
///
/// # Arguments
///
/// * `base_config` - Base configuration (episodes, curriculum, etc.).
/// * `seed` - Random seed for weight initialization.
/// * `lambda` - local_lambda value for the actor.
///
/// # Errors
///
/// Returns an error if agent creation or config conversion fails.
pub fn run_single(
    base_config: &AppConfig,
    seed: u64,
    lambda: f64,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let mut config = base_config.clone();
    config.training.seed = seed;
    config.agent.actor.local_lambda = lambda;

    let agent_config = config.to_agent_config()?;
    let agent = PcActorCritic::new(agent_config, seed)?;

    let episodes = config.training.episodes;
    let log_interval = config.training.log_interval;
    let mut trainer = Trainer::new(agent, &config);

    let mut log_lines = Vec::new();
    let mut prev_depth = 1;

    for _ in 0..episodes {
        let trajectory = trainer.run_episode_pub();
        trainer.agent_mut().learn(&trajectory);
        trainer.record_and_advance();

        if log_interval > 0 && trainer.episode_count().is_multiple_of(log_interval) {
            let line = format!(
                "[ep {:>6}/{total}] win={win:.1}% loss={loss:.1}% draw={draw:.1}% | depth={depth}",
                trainer.episode_count(),
                total = episodes,
                win = trainer.metrics().win_rate() * 100.0,
                loss = trainer.metrics().loss_rate() * 100.0,
                draw = trainer.metrics().draw_rate() * 100.0,
                depth = trainer.current_depth(),
            );
            log_lines.push(line);
        }

        let cur_depth = trainer.current_depth();
        if prev_depth != cur_depth {
            log_lines.push(format!(
                "  >> Curriculum advanced: depth {prev_depth} -> {cur_depth}"
            ));
            prev_depth = cur_depth;
        }
    }

    Ok(RunResult {
        seed,
        lambda,
        aux_coefficient: config.agent.actor.aux_loss_coefficient,
        max_depth: trainer.current_depth(),
        win_rate: trainer.metrics().win_rate(),
        loss_rate: trainer.metrics().loss_rate(),
        draw_rate: trainer.metrics().draw_rate(),
        log_lines,
    })
}

/// Runs a single training cycle with the given seed and sweep parameter.
///
/// # Arguments
///
/// * `base_config` - Base configuration.
/// * `seed` - Random seed.
/// * `sweep` - Which parameter to sweep.
/// * `value` - The sweep value to use.
///
/// # Errors
///
/// Returns an error if agent creation or config conversion fails.
pub fn run_single_with_sweep(
    base_config: &AppConfig,
    seed: u64,
    sweep: SweepParam,
    value: f64,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let mut config = base_config.clone();
    config.training.seed = seed;
    match sweep {
        SweepParam::Lambda => config.agent.actor.local_lambda = value,
        SweepParam::AuxLoss => config.agent.actor.aux_loss_coefficient = value,
    }

    let agent_config = config.to_agent_config()?;
    let agent = PcActorCritic::new(agent_config, seed)?;

    let episodes = config.training.episodes;
    let log_interval = config.training.log_interval;
    let mut trainer = Trainer::new(agent, &config);

    let mut log_lines = Vec::new();
    let mut prev_depth = 1;

    for _ in 0..episodes {
        let trajectory = trainer.run_episode_pub();
        trainer.agent_mut().learn(&trajectory);
        trainer.record_and_advance();

        if log_interval > 0 && trainer.episode_count().is_multiple_of(log_interval) {
            let line = format!(
                "[ep {:>6}/{total}] win={win:.1}% loss={loss:.1}% draw={draw:.1}% | depth={depth}",
                trainer.episode_count(),
                total = episodes,
                win = trainer.metrics().win_rate() * 100.0,
                loss = trainer.metrics().loss_rate() * 100.0,
                draw = trainer.metrics().draw_rate() * 100.0,
                depth = trainer.current_depth(),
            );
            log_lines.push(line);
        }

        let cur_depth = trainer.current_depth();
        if prev_depth != cur_depth {
            log_lines.push(format!(
                "  >> Curriculum advanced: depth {prev_depth} -> {cur_depth}"
            ));
            prev_depth = cur_depth;
        }
    }

    Ok(RunResult {
        seed,
        lambda: config.agent.actor.local_lambda,
        aux_coefficient: config.agent.actor.aux_loss_coefficient,
        max_depth: trainer.current_depth(),
        win_rate: trainer.metrics().win_rate(),
        loss_rate: trainer.metrics().loss_rate(),
        draw_rate: trainer.metrics().draw_rate(),
        log_lines,
    })
}

/// Runs a full experiment: N repetitions × parameter sweep.
///
/// For each repetition, generates a random seed and runs training for each
/// value of the swept parameter.
///
/// # Arguments
///
/// * `base_config` - Base configuration.
/// * `n` - Number of repetitions (random seeds).
/// * `sweep` - Which parameter to sweep.
/// * `output` - Writer for results.
///
/// # Errors
///
/// Returns an error on training or I/O failures.
pub fn run_experiment_sweep<W: Write>(
    base_config: &AppConfig,
    n: usize,
    sweep: SweepParam,
    output: &mut W,
) -> Result<Vec<RunResult>, Box<dyn std::error::Error>> {
    let values = sweep.values();
    let mut all_results = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let seed: u64 = rand::Rng::gen(&mut rng);

        for &value in &values {
            let result = run_single_with_sweep(base_config, seed, sweep, value)?;
            write!(output, "{result}")?;
            output.flush()?;
            all_results.push(result);
        }
    }

    Ok(all_results)
}

/// Runs a full experiment: N repetitions × lambda sweep.
///
/// For each repetition, generates a random seed and runs training for each
/// lambda value in the sweep range [0.95, 0.96, ..., 1.00].
///
/// # Arguments
///
/// * `base_config` - Base configuration.
/// * `n` - Number of repetitions (random seeds).
/// * `output` - Writer for results (file + stdout via [`MultiWriter`]).
///
/// # Errors
///
/// Returns an error on training or I/O failures.
pub fn run_experiment<W: Write>(
    base_config: &AppConfig,
    n: usize,
    output: &mut W,
) -> Result<Vec<RunResult>, Box<dyn std::error::Error>> {
    let lambdas = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00];
    let mut all_results = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let seed: u64 = rand::Rng::gen(&mut rng);

        for &lambda in &lambdas {
            let result = run_single(base_config, seed, lambda)?;
            write!(output, "{result}")?;
            output.flush()?;
            all_results.push(result);
        }
    }

    Ok(all_results)
}

/// Runs N training runs with fixed config, varying only the seed.
///
/// Uses the config as-is (no parameter overrides). Each run gets a
/// unique random seed. Use this to test statistical stability of a
/// specific configuration across different weight initializations.
///
/// # Arguments
///
/// * `base_config` - Configuration to use for all runs.
/// * `n` - Number of runs (random seeds).
/// * `output` - Writer for results.
///
/// # Errors
///
/// Returns an error on training or I/O failures.
pub fn run_seed_test<W: Write>(
    base_config: &AppConfig,
    n: usize,
    output: &mut W,
) -> Result<Vec<RunResult>, Box<dyn std::error::Error>> {
    let mut all_results = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let seed: u64 = rand::Rng::gen(&mut rng);

        let mut config = base_config.clone();
        config.training.seed = seed;

        let agent_config = config.to_agent_config()?;
        let agent = PcActorCritic::new(agent_config, seed)?;

        let episodes = config.training.episodes;
        let log_interval = config.training.log_interval;
        let mut trainer = Trainer::new(agent, &config);

        let mut log_lines = Vec::new();
        let mut prev_depth = 1;

        for _ in 0..episodes {
            let trajectory = trainer.run_episode_pub();
            trainer.agent_mut().learn(&trajectory);
            trainer.record_and_advance();

            if log_interval > 0 && trainer.episode_count().is_multiple_of(log_interval) {
                let line = format!(
                    "[ep {:>6}/{total}] win={win:.1}% loss={loss:.1}% draw={draw:.1}% | depth={depth}",
                    trainer.episode_count(),
                    total = episodes,
                    win = trainer.metrics().win_rate() * 100.0,
                    loss = trainer.metrics().loss_rate() * 100.0,
                    draw = trainer.metrics().draw_rate() * 100.0,
                    depth = trainer.current_depth(),
                );
                log_lines.push(line);
            }

            let cur_depth = trainer.current_depth();
            if prev_depth != cur_depth {
                log_lines.push(format!(
                    "  >> Curriculum advanced: depth {prev_depth} -> {cur_depth}"
                ));
                prev_depth = cur_depth;
            }
        }

        let result = RunResult {
            seed,
            lambda: config.agent.actor.local_lambda,
            aux_coefficient: config.agent.actor.aux_loss_coefficient,
            max_depth: trainer.current_depth(),
            win_rate: trainer.metrics().win_rate(),
            loss_rate: trainer.metrics().loss_rate(),
            draw_rate: trainer.metrics().draw_rate(),
            log_lines,
        };
        write!(output, "{result}")?;
        output.flush()?;
        all_results.push(result);
    }

    Ok(all_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::config::AppConfig;
    use std::path::Path;

    fn test_config() -> AppConfig {
        let mut config = AppConfig::load(Path::new("pc_tictactoe/config.toml"))
            .unwrap_or_else(|_| AppConfig::load(Path::new("config.toml")).unwrap());
        config.training.episodes = 100;
        config.training.log_interval = 50;
        config
    }

    #[test]
    fn test_run_single_returns_valid_result() {
        let config = test_config();
        let result = run_single(&config, 42, 1.0).unwrap();
        assert_eq!(result.seed, 42);
        assert!((result.lambda - 1.0).abs() < f64::EPSILON);
        assert!(result.max_depth >= 1);
        assert!(!result.log_lines.is_empty());
    }

    #[test]
    fn test_run_single_different_seeds_differ() {
        let config = test_config();
        let r1 = run_single(&config, 42, 1.0).unwrap();
        let r2 = run_single(&config, 123, 1.0).unwrap();
        // With different seeds, log lines should differ
        assert_ne!(r1.log_lines, r2.log_lines);
    }

    #[test]
    fn test_run_single_same_seed_deterministic() {
        let config = test_config();
        let r1 = run_single(&config, 42, 1.0).unwrap();
        let r2 = run_single(&config, 42, 1.0).unwrap();
        assert_eq!(r1.log_lines, r2.log_lines);
        assert_eq!(r1.max_depth, r2.max_depth);
    }

    #[test]
    fn test_run_single_respects_lambda() {
        let config = test_config();
        let r1 = run_single(&config, 42, 1.0).unwrap();
        let r2 = run_single(&config, 42, 0.95).unwrap();
        assert!((r1.lambda - 1.0).abs() < f64::EPSILON);
        assert!((r2.lambda - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_display_format() {
        let result = RunResult {
            seed: 42,
            lambda: 0.99,
            aux_coefficient: 0.0,
            max_depth: 8,
            win_rate: 0.0,
            loss_rate: 0.5,
            draw_rate: 0.5,
            log_lines: vec!["[ep    50/100] win=50.0% loss=50.0% draw=0.0% | depth=1".into()],
        };
        let output = format!("{result}");
        assert!(output.contains("============"));
        assert!(output.contains("seed=42"));
        assert!(output.contains("lambda=0.99"));
        assert!(output.contains("[ep    50/100]"));
        assert!(output.contains("------------"));
    }

    #[test]
    fn test_run_experiment_produces_n_times_6_results() {
        let config = test_config();
        let mut buf = Vec::new();
        let results = run_experiment(&config, 1, &mut buf).unwrap();
        assert_eq!(results.len(), 6); // 1 seed × 6 lambdas
    }

    #[test]
    fn test_sweep_param_aux_values() {
        let values = SweepParam::AuxLoss.values();
        assert_eq!(values.len(), 10);
        assert!((values[0] - 0.05).abs() < 1e-12);
        assert!((values[9] - 0.50).abs() < 1e-12);
    }

    #[test]
    fn test_sweep_param_lambda_values() {
        let values = SweepParam::Lambda.values();
        assert_eq!(values.len(), 6);
        assert!((values[0] - 0.95).abs() < 1e-12);
        assert!((values[5] - 1.00).abs() < 1e-12);
    }

    #[test]
    fn test_run_result_has_aux_coefficient() {
        let result = RunResult {
            seed: 42,
            lambda: 0.99,
            aux_coefficient: 0.1,
            max_depth: 8,
            win_rate: 0.0,
            loss_rate: 0.5,
            draw_rate: 0.5,
            log_lines: vec![],
        };
        assert!((result.aux_coefficient - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_run_experiment_aux_sweep_produces_n_times_10() {
        let config = test_config();
        let mut buf = Vec::new();
        let results = run_experiment_sweep(&config, 1, SweepParam::AuxLoss, &mut buf).unwrap();
        assert_eq!(results.len(), 10); // 1 seed × 10 aux values
    }

    #[test]
    fn test_run_experiment_lambda_sweep_backward_compat() {
        let config = test_config();
        let mut buf = Vec::new();
        let results = run_experiment_sweep(&config, 1, SweepParam::Lambda, &mut buf).unwrap();
        assert_eq!(results.len(), 6); // 1 seed × 6 lambdas
    }

    #[test]
    fn test_run_seed_test_produces_n_results() {
        let config = test_config();
        let mut buf = Vec::new();
        let results = run_seed_test(&config, 3, &mut buf).unwrap();
        assert_eq!(results.len(), 3); // 3 seeds × 1 config
    }

    #[test]
    fn test_run_seed_test_uses_config_lambda() {
        let mut config = test_config();
        config.agent.actor.local_lambda = 0.9999;
        let mut buf = Vec::new();
        let results = run_seed_test(&config, 2, &mut buf).unwrap();
        for r in &results {
            assert!(
                (r.lambda - 0.9999).abs() < 1e-12,
                "Should use config lambda"
            );
        }
    }

    #[test]
    fn test_run_seed_test_different_seeds() {
        let config = test_config();
        let mut buf = Vec::new();
        let results = run_seed_test(&config, 3, &mut buf).unwrap();
        let seeds: Vec<u64> = results.iter().map(|r| r.seed).collect();
        assert_ne!(seeds[0], seeds[1], "Seeds should differ");
        assert_ne!(seeds[1], seeds[2], "Seeds should differ");
    }

    #[test]
    fn test_run_seed_test_writes_output() {
        let config = test_config();
        let mut buf = Vec::new();
        let _ = run_seed_test(&config, 1, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("============"));
        assert!(output.contains("seed="));
        assert!(output.contains("------------"));
    }

    #[test]
    fn test_run_experiment_writes_output() {
        let config = test_config();
        let mut buf = Vec::new();
        let _ = run_experiment(&config, 1, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("============"));
        assert!(output.contains("seed="));
        assert!(output.contains("lambda="));
        assert!(output.contains("------------"));
    }
}
