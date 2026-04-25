// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-17

//! Phase 2 commit 14 soft-checkpoint smoke test (MAGI R3 W4).
//!
//! Gated `#[ignore]` — not invoked by `cargo nextest run`. Invoked
//! explicitly by the SDD soft-checkpoint gate at the Phase 2 boundary.

use pc_rl_core::pc_actor_critic::replay::{Action, ReplayBuffer, ReplayTransition};
use pc_rl_core::pc_actor_critic::ActionSpace;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Build a minimal transition with a configurable reward and a scalar
/// `marker` placed in `state[0]`. Used to disambiguate A-drawn vs
/// B-drawn samples in the batch.
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
#[ignore = "Smoke gate: invoked explicitly by Phase 2 soft checkpoint"]
fn replay_buffer_standalone_smoke() {
    let start = std::time::Instant::now();

    // --- Construction -----------------------------------------------------
    let mut buf = ReplayBuffer::new(100, 50, true, ActionSpace::Discrete);
    assert!(buf.training_phase, "new buffer starts in training phase");
    assert!(buf.positive_only, "positive_only flag is propagated");

    // --- Populate training compartment (A) --------------------------------
    // 80 positive transitions with A-markers (state[0] negative).
    for i in 0..80 {
        let marker = -1.0 - (i as f64);
        buf.push(make_transition(1.0, marker)).unwrap();
    }
    assert_eq!(
        buf.training_memories.len(),
        80,
        "80 positive pushes land in training compartment"
    );
    assert!(
        buf.recent_memories.is_empty(),
        "recent compartment still empty before seal"
    );

    // --- Seal and populate recent compartment (B) -------------------------
    buf.seal_training_memories();
    assert!(!buf.training_phase, "seal flips training_phase to false");

    // Push 60 transitions: 50 positive (B-markers, state[0] positive) and
    // 10 non-positive (reward = -1.0) that the positive_only filter drops.
    let mut positives_pushed = 0;
    for i in 0..60 {
        if i % 6 == 5 {
            // 10 of the 60 are non-positive and must be dropped.
            buf.push(make_transition(-1.0, 999.0)).unwrap();
        } else {
            positives_pushed += 1;
            let marker = 1.0 + (i as f64);
            buf.push(make_transition(1.0, marker)).unwrap();
        }
    }
    assert_eq!(
        positives_pushed, 50,
        "test scenario: 50 positive pushes out of 60"
    );
    assert_eq!(
        buf.recent_memories.len(),
        50,
        "50 positives land in recent compartment (cap=50, filter dropped 10)"
    );
    assert_eq!(
        buf.training_memories.len(),
        80,
        "training compartment unchanged after seal"
    );
    assert_eq!(buf.total_len(), 130, "total = A(80) + B(50)");

    // --- Sample a batch of 64 with seeded RNG -----------------------------
    let mut rng = StdRng::seed_from_u64(42);
    let batch = buf.sample(64, &mut rng);
    assert_eq!(batch.len(), 64, "batch returns requested 64 transitions");

    // Count A vs B by marker sign.
    let a_count = batch.iter().filter(|t| t.state[0] < 0.0).count() as i32;
    let b_count = batch.iter().filter(|t| t.state[0] > 0.0).count() as i32;
    assert_eq!(a_count + b_count, 64, "every sampled transition is marked");
    assert!(
        (a_count - 32).abs() <= 4,
        "A count {a_count} deviates from 32 by more than 4"
    );
    assert!(
        (b_count - 32).abs() <= 4,
        "B count {b_count} deviates from 32 by more than 4"
    );

    // --- Serde round-trip -------------------------------------------------
    let json = serde_json::to_string(&buf).expect("serialize ReplayBuffer");
    let restored: ReplayBuffer = serde_json::from_str(&json).expect("deserialize ReplayBuffer");
    assert_eq!(
        restored.training_memories.len(),
        buf.training_memories.len()
    );
    assert_eq!(restored.recent_memories.len(), buf.recent_memories.len());
    assert_eq!(restored.training_phase, buf.training_phase);
    assert_eq!(restored.positive_only, buf.positive_only);
    for (expected, actual) in buf
        .training_memories
        .iter()
        .zip(restored.training_memories.iter())
    {
        assert_eq!(expected, actual, "training compartment entries round-trip");
    }
    for (expected, actual) in buf
        .recent_memories
        .iter()
        .zip(restored.recent_memories.iter())
    {
        assert_eq!(expected, actual, "recent compartment entries round-trip");
    }

    // --- Runtime bound (plan requires < 2s) -------------------------------
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "smoke test must complete in < 2s, took {:.3}s",
        elapsed.as_secs_f64()
    );
}
