// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Sliding-window game outcome metrics.
//!
//! Tracks win/loss/draw rates over a configurable window of recent games.

use std::collections::VecDeque;

/// Possible outcomes of a single game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOutcome {
    /// Agent won the game.
    Win,
    /// Agent lost the game.
    Loss,
    /// Game ended in a draw.
    Draw,
}

/// Sliding-window metrics tracker for game outcomes.
///
/// Maintains a fixed-size window of the most recent outcomes and computes
/// win/loss/draw rates from that window.
///
/// # Examples
///
/// ```
/// use pc_tictactoe::utils::metrics::{Metrics, GameOutcome};
///
/// let mut m = Metrics::new(100);
/// m.record(GameOutcome::Win);
/// m.record(GameOutcome::Loss);
/// assert!((m.win_rate() - 0.5).abs() < 1e-9);
/// ```
pub struct Metrics {
    /// Maximum window size.
    window_size: usize,
    /// Recent outcomes stored in insertion order.
    outcomes: VecDeque<GameOutcome>,
    /// Running surprise sum for average computation.
    surprise_sum: f64,
    /// Number of surprise values recorded.
    surprise_count: usize,
}

impl Metrics {
    /// Creates a new metrics tracker with the given sliding window size.
    ///
    /// # Parameters
    ///
    /// * `window_size` - Maximum number of outcomes to retain.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            outcomes: VecDeque::with_capacity(window_size),
            surprise_sum: 0.0,
            surprise_count: 0,
        }
    }

    /// Records a game outcome, dropping the oldest if the window is full.
    ///
    /// # Parameters
    ///
    /// * `outcome` - The result of the most recent game.
    pub fn record(&mut self, outcome: GameOutcome) {
        if self.window_size == 0 {
            return;
        }
        if self.outcomes.len() == self.window_size {
            self.outcomes.pop_front();
        }
        self.outcomes.push_back(outcome);
    }

    /// Records a surprise value for running average computation.
    ///
    /// # Parameters
    ///
    /// * `surprise` - The surprise score from the most recent inference.
    pub fn record_surprise(&mut self, surprise: f64) {
        self.surprise_sum += surprise;
        self.surprise_count += 1;
    }

    /// Returns the fraction of outcomes that are wins.
    ///
    /// Returns `0.0` if no outcomes have been recorded.
    pub fn win_rate(&self) -> f64 {
        self.rate_of(GameOutcome::Win)
    }

    /// Returns the fraction of outcomes that are losses.
    ///
    /// Returns `0.0` if no outcomes have been recorded.
    pub fn loss_rate(&self) -> f64 {
        self.rate_of(GameOutcome::Loss)
    }

    /// Returns the fraction of outcomes that are draws.
    ///
    /// Returns `0.0` if no outcomes have been recorded.
    pub fn draw_rate(&self) -> f64 {
        self.rate_of(GameOutcome::Draw)
    }

    /// Returns the average surprise score, or `0.0` if none recorded.
    pub fn surprise_avg(&self) -> f64 {
        if self.surprise_count == 0 {
            0.0
        } else {
            self.surprise_sum / self.surprise_count as f64
        }
    }

    /// Computes the rate of a specific outcome in the window.
    ///
    /// # Parameters
    ///
    /// * `target` - The outcome type to count.
    ///
    /// # Returns
    ///
    /// Fraction of outcomes matching `target`, or `0.0` if window is empty.
    fn rate_of(&self, target: GameOutcome) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        let count = self.outcomes.iter().filter(|o| **o == target).count();
        count as f64 / self.outcomes.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_metrics_all_zero() {
        let m = Metrics::new(100);
        assert_eq!(m.win_rate(), 0.0);
        assert_eq!(m.loss_rate(), 0.0);
        assert_eq!(m.draw_rate(), 0.0);
        assert_eq!(m.surprise_avg(), 0.0);
    }

    #[test]
    fn test_win_rate_correct() {
        let mut m = Metrics::new(100);
        m.record(GameOutcome::Win);
        m.record(GameOutcome::Win);
        m.record(GameOutcome::Loss);
        m.record(GameOutcome::Draw);
        assert!((m.win_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_rates_sum_to_one() {
        let mut m = Metrics::new(100);
        m.record(GameOutcome::Win);
        m.record(GameOutcome::Loss);
        m.record(GameOutcome::Draw);
        m.record(GameOutcome::Win);
        m.record(GameOutcome::Win);
        let sum = m.win_rate() + m.loss_rate() + m.draw_rate();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sliding_window_drops_oldest() {
        let mut m = Metrics::new(3);
        m.record(GameOutcome::Loss);
        m.record(GameOutcome::Loss);
        m.record(GameOutcome::Loss);
        // Window full of losses; now add wins to push out losses.
        m.record(GameOutcome::Win);
        m.record(GameOutcome::Win);
        m.record(GameOutcome::Win);
        assert!((m.win_rate() - 1.0).abs() < 1e-9);
        assert_eq!(m.loss_rate(), 0.0);
    }

    #[test]
    fn test_surprise_avg_correct() {
        let mut m = Metrics::new(100);
        m.record_surprise(0.1);
        m.record_surprise(0.3);
        m.record_surprise(0.5);
        let avg = m.surprise_avg();
        assert!((avg - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_zero_window_size_still_safe() {
        let mut m = Metrics::new(0);
        m.record(GameOutcome::Win);
        // Should not panic; win_rate on empty window returns 0.0
        assert_eq!(m.win_rate(), 0.0);
    }
}
