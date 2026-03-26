// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

pub mod env;
#[cfg(test)]
mod integration_tests;
pub mod training;
pub mod ui;
pub mod utils;

use clap::Parser;

fn main() {
    let cli = ui::cli::Cli::parse();
    let result = match cli.command {
        ui::cli::Command::Train(args) => ui::cli::run_train(args),
        ui::cli::Command::Play(args) => ui::cli::run_play(args),
        ui::cli::Command::Evaluate(args) => ui::cli::run_evaluate(args),
        ui::cli::Command::Benchmark(args) => ui::cli::run_benchmark(args),
        ui::cli::Command::Experiment(args) => ui::cli::run_experiment(args),
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
