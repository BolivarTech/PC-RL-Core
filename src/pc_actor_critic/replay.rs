// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-17

//! Dual-compartment replay buffer for off-policy learning from stored
//! positive-reward trajectories. See docs/pc_rl_core_self_recovery_spec.md
//! Phase 2 for the design rationale.
//!
//! The buffer maintains two FIFO compartments:
//! * `training_memories` — sealed after an initial learning phase; retains
//!   successful trajectories collected during bootstrapping.
//! * `recent_memories` — continuously updated with new transitions after
//!   the training compartment is sealed.
//!
//! Sampling blends both compartments with a target 50/50 split, falling
//! back to the non-empty compartment when one is empty.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Single transition stored in a [`ReplayBuffer`].
///
/// Captures the canonical MDP tuple `(s, a, r, s', done)` plus the mask of
/// valid actions in `state` required to recompute masked log-probabilities
/// and critic targets during off-policy replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTransition {
    /// State vector at the moment the action was chosen.
    pub state: Vec<f64>,
    /// Action index selected by the policy.
    pub action: usize,
    /// Scalar reward observed after executing `action`.
    pub reward: f64,
    /// Successor state vector.
    pub next_state: Vec<f64>,
    /// `true` if the transition terminated the episode.
    pub done: bool,
    /// Valid action indices available from `state` (action mask).
    pub valid_actions: Vec<usize>,
}

/// Dual-compartment FIFO replay buffer with optional positive-reward filter.
///
/// # Compartments
///
/// * During the **training phase** (`training_phase == true`), pushes land
///   in `training_memories` up to `training_capacity`.
/// * [`ReplayBuffer::seal_training_memories`] flips `training_phase` to
///   `false` and freezes `training_memories`; all subsequent pushes route
///   to `recent_memories` (up to `recent_capacity`).
///
/// Both compartments use FIFO eviction when capacity is exceeded.
///
/// # Positive-only filter
///
/// When `positive_only == true`, transitions with `reward <= 0` are
/// silently dropped at push time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    /// Sealed successful trajectories collected during the training phase.
    pub training_memories: VecDeque<ReplayTransition>,
    /// Maximum size of `training_memories`.
    training_capacity: usize,
    /// Rolling window of post-seal transitions.
    pub recent_memories: VecDeque<ReplayTransition>,
    /// Maximum size of `recent_memories`.
    recent_capacity: usize,
    /// `true` until [`ReplayBuffer::seal_training_memories`] is called.
    pub training_phase: bool,
    /// When `true`, drops transitions with `reward <= 0` at push time.
    pub positive_only: bool,
}

impl ReplayBuffer {
    /// Construct an empty buffer.
    ///
    /// # Parameters
    ///
    /// * `training_capacity` — maximum entries kept in `training_memories`.
    /// * `recent_capacity` — maximum entries kept in `recent_memories`.
    /// * `positive_only` — if `true`, pushes with `reward <= 0` are dropped.
    ///
    /// # Returns
    ///
    /// A buffer with `training_phase = true` and both compartments empty.
    pub fn new(training_capacity: usize, recent_capacity: usize, positive_only: bool) -> Self {
        Self {
            training_memories: VecDeque::with_capacity(training_capacity),
            training_capacity,
            recent_memories: VecDeque::with_capacity(recent_capacity),
            recent_capacity,
            training_phase: true,
            positive_only,
        }
    }

    /// Insert a transition into the active compartment.
    ///
    /// Routing depends on `training_phase`: `true` → `training_memories`,
    /// `false` → `recent_memories`. Applies the `positive_only` filter and
    /// evicts the oldest entry when the target compartment is at capacity.
    pub fn push(&mut self, transition: ReplayTransition) {
        // positive_only filter: silently drop non-positive rewards.
        if self.positive_only && transition.reward <= 0.0 {
            return;
        }
        if self.training_phase {
            // Training compartment (A): accept up to capacity, then drop.
            // Training memories are immutable once accumulated — no eviction.
            if self.training_memories.len() >= self.training_capacity {
                return;
            }
            self.training_memories.push_back(transition);
        } else {
            // Recent compartment (B): FIFO eviction when at capacity.
            if self.recent_capacity == 0 {
                return;
            }
            if self.recent_memories.len() >= self.recent_capacity {
                self.recent_memories.pop_front();
            }
            self.recent_memories.push_back(transition);
        }
    }

    /// Transition from the training phase to the recent-memory phase.
    ///
    /// Flips `training_phase` to `false` so subsequent [`ReplayBuffer::push`]
    /// calls land in `recent_memories`. `training_memories` is preserved
    /// read-only.
    pub fn seal_training_memories(&mut self) {
        self.training_phase = false;
    }

    /// Draw up to `batch_size` transitions from both compartments.
    ///
    /// Targets a 50/50 split between `training_memories` and
    /// `recent_memories`. When one compartment is empty, the full batch is
    /// drawn from the non-empty compartment. Samples without replacement
    /// per compartment.
    ///
    /// # Parameters
    ///
    /// * `batch_size` — desired number of transitions to return.
    /// * `rng` — RNG driving the sampling.
    ///
    /// # Returns
    ///
    /// A `Vec<ReplayTransition>` of length `min(batch_size, total_len())`
    /// or empty if the buffer is empty.
    pub fn sample(&self, batch_size: usize, rng: &mut StdRng) -> Vec<ReplayTransition> {
        let a_len = self.training_memories.len();
        let b_len = self.recent_memories.len();
        if (a_len == 0 && b_len == 0) || batch_size == 0 {
            return Vec::new();
        }
        let target_a = batch_size / 2;
        let target_b = batch_size - target_a;
        // Fallback: if one compartment is empty, take the full batch from the other.
        let (actual_a, actual_b) = match (a_len, b_len) {
            (0, _) => (0, batch_size),
            (_, 0) => (batch_size, 0),
            _ => (target_a, target_b),
        };
        // Respect actual compartment sizes (cannot draw more than len).
        let actual_a = actual_a.min(a_len);
        let actual_b = actual_b.min(b_len);
        let mut out = Vec::with_capacity(actual_a + actual_b);
        // Draw from A (training) without replacement.
        if actual_a > 0 {
            let mut indices: Vec<usize> = (0..a_len).collect();
            indices.shuffle(rng);
            for &i in indices.iter().take(actual_a) {
                out.push(self.training_memories[i].clone());
            }
        }
        // Draw from B (recent) without replacement.
        if actual_b > 0 {
            let mut indices: Vec<usize> = (0..b_len).collect();
            indices.shuffle(rng);
            for &i in indices.iter().take(actual_b) {
                out.push(self.recent_memories[i].clone());
            }
        }
        out
    }

    /// Total number of stored transitions across both compartments.
    pub fn total_len(&self) -> usize {
        self.training_memories.len() + self.recent_memories.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Build a minimal transition with a configurable `reward` and the
    /// scalar `marker` placed in `state[0]` for content comparisons.
    fn make_transition(reward: f64, marker: f64) -> ReplayTransition {
        ReplayTransition {
            state: vec![marker, 0.0, 0.0],
            action: 0,
            reward,
            next_state: vec![0.0, 0.0, 0.0],
            done: false,
            valid_actions: vec![0, 1, 2],
        }
    }

    #[test]
    fn test_replay_buffer_empty_sample_returns_empty() {
        let buf = ReplayBuffer::new(100, 50, true);
        let mut rng = StdRng::seed_from_u64(42);
        let batch = buf.sample(10, &mut rng);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_replay_buffer_push_respects_training_phase() {
        let mut buf = ReplayBuffer::new(100, 50, false);
        assert!(buf.training_phase);

        buf.push(make_transition(1.0, 0.0));
        assert_eq!(buf.training_memories.len(), 1);
        assert!(buf.recent_memories.is_empty());

        buf.seal_training_memories();
        buf.push(make_transition(1.0, 1.0));
        assert_eq!(
            buf.training_memories.len(),
            1,
            "training compartment unchanged after seal"
        );
        assert_eq!(buf.recent_memories.len(), 1);
    }

    #[test]
    fn test_replay_buffer_positive_only_filter() {
        let mut buf = ReplayBuffer::new(100, 50, true);

        buf.push(make_transition(-0.5, 0.0));
        assert_eq!(buf.total_len(), 0, "negative reward dropped by filter");

        buf.push(make_transition(0.5, 0.0));
        assert_eq!(buf.total_len(), 1, "positive reward retained");
    }

    #[test]
    fn test_replay_buffer_fifo_eviction_recent_compartment() {
        let mut buf = ReplayBuffer::new(100, 3, false);
        buf.seal_training_memories();

        buf.push(make_transition(1.0, 0.0));
        buf.push(make_transition(1.0, 1.0));
        buf.push(make_transition(1.0, 2.0));
        buf.push(make_transition(1.0, 3.0));

        assert_eq!(buf.recent_memories.len(), 3);
        let front = buf
            .recent_memories
            .front()
            .expect("non-empty recent compartment");
        assert_eq!(
            front.state[0], 1.0,
            "oldest entry evicted; front now holds marker=1.0"
        );
    }

    #[test]
    fn test_replay_buffer_seal_routes_new_pushes_to_recent() {
        let mut buf = ReplayBuffer::new(100, 50, false);

        buf.push(make_transition(1.0, 0.0));
        assert_eq!(buf.training_memories.len(), 1);
        assert!(buf.recent_memories.is_empty());

        buf.seal_training_memories();
        assert!(!buf.training_phase, "training_phase flipped by seal");

        buf.push(make_transition(1.0, 1.0));
        assert_eq!(buf.training_memories.len(), 1);
        assert_eq!(buf.recent_memories.len(), 1);
    }

    #[test]
    fn test_replay_buffer_sample_50_50_split() {
        let mut buf = ReplayBuffer::new(100, 100, false);

        for _ in 0..100 {
            buf.push(make_transition(1.0, -1.0));
        }
        buf.seal_training_memories();
        for _ in 0..100 {
            buf.push(make_transition(1.0, 1.0));
        }

        let mut rng = StdRng::seed_from_u64(42);
        let batch = buf.sample(50, &mut rng);
        assert_eq!(batch.len(), 50);

        let a_count = batch.iter().filter(|t| t.state[0] < 0.0).count() as i32;
        let b_count = batch.iter().filter(|t| t.state[0] > 0.0).count() as i32;

        assert!(
            (a_count - 25).abs() <= 2,
            "A count {} deviates from 25 by more than 2",
            a_count
        );
        assert!(
            (b_count - 25).abs() <= 2,
            "B count {} deviates from 25 by more than 2",
            b_count
        );
    }

    #[test]
    fn test_replay_buffer_sample_fallback_when_one_empty() {
        // A-only: training compartment populated, recent empty.
        let mut buf_a = ReplayBuffer::new(200, 200, false);
        for _ in 0..100 {
            buf_a.push(make_transition(1.0, -1.0));
        }
        let mut rng = StdRng::seed_from_u64(42);
        let batch_a = buf_a.sample(50, &mut rng);
        assert_eq!(batch_a.len(), 50);
        assert!(
            batch_a.iter().all(|t| t.state[0] < 0.0),
            "all samples drawn from A (training) compartment"
        );

        // B-only: training sealed empty, recent populated.
        let mut buf_b = ReplayBuffer::new(200, 200, false);
        buf_b.seal_training_memories();
        for _ in 0..100 {
            buf_b.push(make_transition(1.0, 1.0));
        }
        let mut rng_b = StdRng::seed_from_u64(42);
        let batch_b = buf_b.sample(50, &mut rng_b);
        assert_eq!(batch_b.len(), 50);
        assert!(
            batch_b.iter().all(|t| t.state[0] > 0.0),
            "all samples drawn from B (recent) compartment"
        );
    }

    #[test]
    fn test_replay_buffer_serialization_round_trip() {
        let mut buf = ReplayBuffer::new(10, 5, true);

        // Populate training compartment with three distinct transitions.
        buf.push(make_transition(1.0, -1.0));
        buf.push(make_transition(2.0, -2.0));
        buf.push(make_transition(3.0, -3.0));

        buf.seal_training_memories();

        // Populate recent compartment with two distinct transitions.
        buf.push(make_transition(0.5, 1.0));
        buf.push(make_transition(0.75, 2.0));

        let json = serde_json::to_string(&buf).expect("serialize ReplayBuffer");
        let restored: ReplayBuffer = serde_json::from_str(&json).expect("deserialize ReplayBuffer");

        assert_eq!(restored.training_capacity, 10);
        assert_eq!(restored.recent_capacity, 5);
        assert!(!restored.training_phase);
        assert!(restored.positive_only);

        assert_eq!(
            restored.training_memories.len(),
            buf.training_memories.len()
        );
        assert_eq!(restored.recent_memories.len(), buf.recent_memories.len());

        for (expected, actual) in buf
            .training_memories
            .iter()
            .zip(restored.training_memories.iter())
        {
            assert_eq!(expected, actual);
        }
        for (expected, actual) in buf
            .recent_memories
            .iter()
            .zip(restored.recent_memories.iter())
        {
            assert_eq!(expected, actual);
        }
    }
}
