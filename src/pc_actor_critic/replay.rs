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

/// v4.0.0 — generic action variant. Replaces the v3.x `action: usize`
/// field of `ReplayTransition`.
///
/// **Brainstorm Q1:** `#[serde(untagged)]` is the binding representation.
/// JSON token-type discriminates: integer → `Discrete`, array → `Continuous`.
/// v3.x bare integer `"action": 5` deserializes automatically as
/// `Action::Discrete(5)` — no custom Deserialize needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Action {
    /// Discrete action index. JSON form: bare integer, e.g. `5`.
    Discrete(usize),
    /// Continuous action vector of length = `actor.output_size`.
    /// JSON form: bare array of f64, e.g. `[0.1, 0.7, -0.3]`.
    Continuous(Vec<f64>),
}

impl Action {
    /// Returns `true` if this variant matches the supplied `ActionSpace`.
    ///
    /// Used by `ReplayBuffer::push` for cross-mode contamination check.
    ///
    /// # Parameters
    ///
    /// * `space` — The `ActionSpace` to match against.
    ///
    /// # Returns
    ///
    /// `true` if the action variant is consistent with the given space.
    pub fn matches_space(&self, space: crate::pc_actor_critic::ActionSpace) -> bool {
        use crate::pc_actor_critic::ActionSpace;
        matches!(
            (self, space),
            (Action::Discrete(_), ActionSpace::Discrete)
                | (Action::Continuous(_), ActionSpace::Continuous)
        )
    }
}

/// v4.0.0 schema. Captures the canonical MDP tuple `(s, a, r, s', done)`
/// plus the optional mask of valid actions in `state` (Discrete only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTransition {
    /// State vector at the moment the action was chosen.
    pub state: Vec<f64>,
    /// Action variant (Discrete or Continuous). Brainstorm Q1.
    pub action: Action,
    /// Scalar reward observed after executing `action`.
    pub reward: f64,
    /// Successor state vector.
    pub next_state: Vec<f64>,
    /// `true` if the transition terminated the episode.
    pub done: bool,
    /// Valid action indices available from `state`. `Some(mask)` for
    /// Discrete; `None` for Continuous (mask concept N/A).
    ///
    /// **Brainstorm Q8:** `#[serde(default)]` covers all v3→v4 forms:
    /// legacy bare `[0,1,2]` → `Some([0,1,2])` via Some-elision;
    /// explicit null → `None`; absent → `None` via default.
    /// No custom `deserialize_with` needed.
    #[serde(default)]
    pub valid_actions: Option<Vec<usize>>,
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
    /// Action space discriminator. Set at construction from the agent's
    /// config; immutable afterward. Used by `push` to reject cross-mode
    /// contamination (Brainstorm Q2).
    ///
    /// **Brainstorm Q8 / v3.x compat:** `#[serde(default)]` allows pre-v4
    /// JSON that lacks this field to deserialize as `ActionSpace::Discrete`.
    #[serde(default)]
    pub(crate) action_space: crate::pc_actor_critic::ActionSpace,
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
    /// * `action_space` — action space discriminator stored for cross-mode
    ///   contamination checks in `push` (Phase 2.2).
    ///
    /// # Returns
    ///
    /// A buffer with `training_phase = true` and both compartments empty.
    pub fn new(
        training_capacity: usize,
        recent_capacity: usize,
        positive_only: bool,
        action_space: crate::pc_actor_critic::ActionSpace,
    ) -> Self {
        Self {
            action_space,
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
    ///
    /// # Validation (Phase 2.2)
    ///
    /// Two invariants are enforced before any routing logic:
    ///
    /// 1. **Cross-mode contamination** (Brainstorm Q2): the `action` variant
    ///    must match the buffer's `action_space`. A `Continuous` action pushed
    ///    into a `Discrete` buffer (and vice-versa) is rejected.
    /// 2. **valid_actions ↔ Action variant** (Brainstorm Q8):
    ///    `valid_actions` must be `Some(_)` for `Action::Discrete` and
    ///    `None` for `Action::Continuous`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::PcError::ConfigValidation`] when either
    /// invariant is violated.
    pub fn push(&mut self, transition: ReplayTransition) -> Result<(), crate::error::PcError> {
        // Brainstorm Q2: cross-mode contamination check.
        if !transition.action.matches_space(self.action_space) {
            return Err(crate::error::PcError::ConfigValidation(format!(
                "transition action variant does not match buffer's \
                 action_space {:?}. Cross-mode buffer contamination is rejected.",
                self.action_space
            )));
        }
        // valid_actions ↔ Action variant invariant (Brainstorm Q8).
        let consistency_ok = matches!(
            (&transition.action, &transition.valid_actions),
            (Action::Discrete(_), Some(_)) | (Action::Continuous(_), None)
        );
        if !consistency_ok {
            return Err(crate::error::PcError::ConfigValidation(
                "valid_actions must be Some(_) for Action::Discrete and None \
                 for Action::Continuous"
                    .into(),
            ));
        }
        // positive_only filter: silently drop non-positive rewards.
        if self.positive_only && transition.reward <= 0.0 {
            return Ok(());
        }
        if self.training_phase {
            // Training compartment (A): accept up to capacity, then drop.
            // Training memories are immutable once accumulated — no eviction.
            if self.training_memories.len() >= self.training_capacity {
                return Ok(());
            }
            self.training_memories.push_back(transition);
        } else {
            // Recent compartment (B): FIFO eviction when at capacity.
            if self.recent_capacity == 0 {
                return Ok(());
            }
            if self.recent_memories.len() >= self.recent_capacity {
                self.recent_memories.pop_front();
            }
            self.recent_memories.push_back(transition);
        }
        Ok(())
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
            action: Action::Discrete(0),
            reward,
            next_state: vec![0.0, 0.0, 0.0],
            done: false,
            valid_actions: Some(vec![0, 1, 2]),
        }
    }

    #[test]
    fn test_replay_buffer_empty_sample_returns_empty() {
        let buf = ReplayBuffer::new(100, 50, true, crate::pc_actor_critic::ActionSpace::Discrete);
        let mut rng = StdRng::seed_from_u64(42);
        let batch = buf.sample(10, &mut rng);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_replay_buffer_push_respects_training_phase() {
        let mut buf = ReplayBuffer::new(
            100,
            50,
            false,
            crate::pc_actor_critic::ActionSpace::Discrete,
        );
        assert!(buf.training_phase);

        buf.push(make_transition(1.0, 0.0)).unwrap();
        assert_eq!(buf.training_memories.len(), 1);
        assert!(buf.recent_memories.is_empty());

        buf.seal_training_memories();
        buf.push(make_transition(1.0, 1.0)).unwrap();
        assert_eq!(
            buf.training_memories.len(),
            1,
            "training compartment unchanged after seal"
        );
        assert_eq!(buf.recent_memories.len(), 1);
    }

    #[test]
    fn test_replay_buffer_positive_only_filter() {
        let mut buf =
            ReplayBuffer::new(100, 50, true, crate::pc_actor_critic::ActionSpace::Discrete);

        buf.push(make_transition(-0.5, 0.0)).unwrap();
        assert_eq!(buf.total_len(), 0, "negative reward dropped by filter");

        buf.push(make_transition(0.5, 0.0)).unwrap();
        assert_eq!(buf.total_len(), 1, "positive reward retained");
    }

    #[test]
    fn test_replay_buffer_fifo_eviction_recent_compartment() {
        let mut buf =
            ReplayBuffer::new(100, 3, false, crate::pc_actor_critic::ActionSpace::Discrete);
        buf.seal_training_memories();

        buf.push(make_transition(1.0, 0.0)).unwrap();
        buf.push(make_transition(1.0, 1.0)).unwrap();
        buf.push(make_transition(1.0, 2.0)).unwrap();
        buf.push(make_transition(1.0, 3.0)).unwrap();

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
        let mut buf = ReplayBuffer::new(
            100,
            50,
            false,
            crate::pc_actor_critic::ActionSpace::Discrete,
        );

        buf.push(make_transition(1.0, 0.0)).unwrap();
        assert_eq!(buf.training_memories.len(), 1);
        assert!(buf.recent_memories.is_empty());

        buf.seal_training_memories();
        assert!(!buf.training_phase, "training_phase flipped by seal");

        buf.push(make_transition(1.0, 1.0)).unwrap();
        assert_eq!(buf.training_memories.len(), 1);
        assert_eq!(buf.recent_memories.len(), 1);
    }

    #[test]
    fn test_replay_buffer_sample_50_50_split() {
        let mut buf = ReplayBuffer::new(
            100,
            100,
            false,
            crate::pc_actor_critic::ActionSpace::Discrete,
        );

        for _ in 0..100 {
            buf.push(make_transition(1.0, -1.0)).unwrap();
        }
        buf.seal_training_memories();
        for _ in 0..100 {
            buf.push(make_transition(1.0, 1.0)).unwrap();
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
        let mut buf_a = ReplayBuffer::new(
            200,
            200,
            false,
            crate::pc_actor_critic::ActionSpace::Discrete,
        );
        for _ in 0..100 {
            buf_a.push(make_transition(1.0, -1.0)).unwrap();
        }
        let mut rng = StdRng::seed_from_u64(42);
        let batch_a = buf_a.sample(50, &mut rng);
        assert_eq!(batch_a.len(), 50);
        assert!(
            batch_a.iter().all(|t| t.state[0] < 0.0),
            "all samples drawn from A (training) compartment"
        );

        // B-only: training sealed empty, recent populated.
        let mut buf_b = ReplayBuffer::new(
            200,
            200,
            false,
            crate::pc_actor_critic::ActionSpace::Discrete,
        );
        buf_b.seal_training_memories();
        for _ in 0..100 {
            buf_b.push(make_transition(1.0, 1.0)).unwrap();
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
        let mut buf = ReplayBuffer::new(10, 5, true, crate::pc_actor_critic::ActionSpace::Discrete);

        // Populate training compartment with three distinct transitions.
        buf.push(make_transition(1.0, -1.0)).unwrap();
        buf.push(make_transition(2.0, -2.0)).unwrap();
        buf.push(make_transition(3.0, -3.0)).unwrap();

        buf.seal_training_memories();

        // Populate recent compartment with two distinct transitions.
        buf.push(make_transition(0.5, 1.0)).unwrap();
        buf.push(make_transition(0.75, 2.0)).unwrap();

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

    // ── v4.0.0 Action enum + cross-mode validation tests ────────────────

    #[test]
    fn test_push_rejects_cross_mode_continuous_into_discrete_buffer() {
        use crate::pc_actor_critic::ActionSpace;
        let mut buffer = ReplayBuffer::new(10, 0, false, ActionSpace::Discrete);
        let transition = ReplayTransition {
            state: vec![0.0; 4],
            action: Action::Continuous(vec![0.5, 0.3]),
            reward: 1.0,
            next_state: vec![0.0; 4],
            done: false,
            valid_actions: None,
        };
        let result = buffer.push(transition);
        assert!(
            result.is_err(),
            "Continuous transition into Discrete buffer must reject"
        );
    }

    #[test]
    fn test_push_rejects_discrete_into_continuous_buffer() {
        use crate::pc_actor_critic::ActionSpace;
        let mut buffer = ReplayBuffer::new(10, 0, false, ActionSpace::Continuous);
        let transition = ReplayTransition {
            state: vec![0.0; 4],
            action: Action::Discrete(2),
            reward: 1.0,
            next_state: vec![0.0; 4],
            done: false,
            valid_actions: Some(vec![0, 1, 2, 3]),
        };
        let result = buffer.push(transition);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_rejects_valid_actions_mismatch() {
        // Discrete transition with valid_actions = None must reject.
        use crate::pc_actor_critic::ActionSpace;
        let mut buffer = ReplayBuffer::new(10, 0, false, ActionSpace::Discrete);
        let transition = ReplayTransition {
            state: vec![0.0; 4],
            action: Action::Discrete(0),
            reward: 1.0,
            next_state: vec![0.0; 4],
            done: false,
            valid_actions: None, // <-- mismatch with Discrete action
        };
        let result = buffer.push(transition);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_accepts_consistent_discrete() {
        use crate::pc_actor_critic::ActionSpace;
        let mut buffer = ReplayBuffer::new(10, 0, false, ActionSpace::Discrete);
        let transition = ReplayTransition {
            state: vec![0.0; 4],
            action: Action::Discrete(0),
            reward: 1.0,
            next_state: vec![0.0; 4],
            done: false,
            valid_actions: Some(vec![0, 1, 2]),
        };
        assert!(buffer.push(transition).is_ok());
    }

    #[test]
    fn test_push_accepts_consistent_continuous() {
        use crate::pc_actor_critic::ActionSpace;
        let mut buffer = ReplayBuffer::new(10, 0, false, ActionSpace::Continuous);
        let transition = ReplayTransition {
            state: vec![0.0; 4],
            action: Action::Continuous(vec![0.5]),
            reward: 1.0,
            next_state: vec![0.0; 4],
            done: false,
            valid_actions: None,
        };
        assert!(buffer.push(transition).is_ok());
    }
}
