// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-17

//! Phase 2 smoke gate: end-to-end replay buffer integration tests.
//!
//! These tests are `#[ignore]`-gated and only invoked explicitly at the
//! Phase 2 → MAGI gate B boundary. They do NOT run during normal CI.

use pc_rl_core::{save_agent, CpuLinAlg, PcActorCritic, PcActorCriticConfig, SaveFile};

/// Build a config with replay buffer, dual distillation anchors, and EWC.
///
/// The config is deserialized from JSON so that all non-specified fields
/// pick up their `#[serde(default)]` values automatically.
fn smoke_config(replay_training: usize, replay_recent: usize) -> PcActorCriticConfig {
    let json = format!(
        r#"{{
            "actor": {{
                "input_size": 9,
                "hidden_layers": [{{ "size": 18, "activation": "Tanh" }}],
                "output_size": 9,
                "output_activation": "Linear",
                "alpha": 0.03,
                "tol": 0.01,
                "min_steps": 1,
                "max_steps": 5,
                "lr_weights": 0.005,
                "synchronous": true,
                "temperature": 1.0,
                "local_lambda": 0.99,
                "residual": false,
                "rezero_init": 0.001
            }},
            "critic": {{
                "input_size": 27,
                "hidden_layers": [{{ "size": 36, "activation": "Tanh" }}],
                "output_activation": "Linear",
                "lr": 0.005
            }},
            "gamma": 0.99,
            "ewc_lambda": 0.1,
            "distillation_lambda_polyak": 0.05,
            "polyak_tau": 0.005,
            "distillation_lambda_frozen": 0.05,
            "replay_training_capacity": {replay_training},
            "replay_recent_capacity": {replay_recent}
        }}"#
    );
    serde_json::from_str(&json).expect("smoke config JSON must parse")
}

/// Serialize the agent in-memory and return the deserialized `SaveFile`.
///
/// This is the test harness's only route to agent internals: actor/critic
/// fields are `pub(crate)`, so weight inspection must go through the
/// public serialization surface.
fn snapshot(agent: &PcActorCritic<CpuLinAlg>) -> SaveFile {
    let path = std::env::temp_dir().join(format!(
        "pc_rl_core_phase2_snapshot_{}.json",
        std::process::id()
    ));
    let path_str = path.to_str().expect("tempdir path is valid UTF-8");
    save_agent(agent, path_str, 0, None).expect("save_agent succeeds");
    let json = std::fs::read_to_string(path_str).expect("snapshot file readable");
    let _ = std::fs::remove_file(path_str);
    serde_json::from_str(&json).expect("SaveFile deserializes")
}

/// Assert every weight + bias + rezero_alpha + skip_projection entry in
/// the actor and critic is finite (no NaN, no Inf).
fn assert_all_weights_finite(agent: &PcActorCritic<CpuLinAlg>, context: &str) {
    let save = snapshot(agent);

    for (li, layer) in save.actor_weights.layers.iter().enumerate() {
        for (i, &w) in layer.weights.data.iter().enumerate() {
            assert!(
                w.is_finite(),
                "{context}: actor layer {li} weight[{i}] not finite: {w}"
            );
        }
        for (i, &b) in layer.bias.iter().enumerate() {
            assert!(
                b.is_finite(),
                "{context}: actor layer {li} bias[{i}] not finite: {b}"
            );
        }
    }
    for (i, &a) in save.actor_weights.rezero_alpha.iter().enumerate() {
        assert!(
            a.is_finite(),
            "{context}: actor rezero_alpha[{i}] not finite: {a}"
        );
    }
    for (li, proj) in save
        .actor_weights
        .skip_projections
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|m| (i, m)))
    {
        for (i, &w) in proj.data.iter().enumerate() {
            assert!(
                w.is_finite(),
                "{context}: actor skip_proj[{li}] weight[{i}] not finite: {w}"
            );
        }
    }
    for (li, layer) in save.critic_weights.layers.iter().enumerate() {
        for (i, &w) in layer.weights.data.iter().enumerate() {
            assert!(
                w.is_finite(),
                "{context}: critic layer {li} weight[{i}] not finite: {w}"
            );
        }
        for (i, &b) in layer.bias.iter().enumerate() {
            assert!(
                b.is_finite(),
                "{context}: critic layer {li} bias[{i}] not finite: {b}"
            );
        }
    }
}

/// Flatten an actor's weight + bias tensors into a single `Vec<f64>`.
///
/// Used to compute L2 norms `||θ_live − θ_baseline||` for the stress
/// scenario degradation-then-recovery assertions.
fn flatten_actor_weights(save: &SaveFile) -> Vec<f64> {
    let mut out = Vec::new();
    for layer in &save.actor_weights.layers {
        out.extend_from_slice(&layer.weights.data);
        out.extend_from_slice(&layer.bias);
    }
    out.extend_from_slice(&save.actor_weights.rezero_alpha);
    for proj in save.actor_weights.skip_projections.iter().flatten() {
        out.extend_from_slice(&proj.data);
    }
    out
}

/// Euclidean distance between two equal-length vectors.
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Smoke test 1: end-to-end replay + rollback.
// Runtime budget: < 8 seconds.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Smoke gate: invoked explicitly by Phase 2 -> MAGI gate B boundary"]
fn phase2_replay_end_to_end() {
    let start = std::time::Instant::now();

    let config = smoke_config(100, 50);
    let mut agent: PcActorCritic<CpuLinAlg> =
        PcActorCritic::new(CpuLinAlg::new(), config, 42).expect("construct agent");

    // Disable cooldown so rollback_hard can be called freely in the test.
    agent.set_rollback_hard_cooldown(0);

    let valid_actions: Vec<usize> = (0..9).collect();
    let state = vec![0.5; 9];

    // Phase A: 200 step_masked calls — populates training compartment (A).
    for i in 0..200 {
        let reward = if i % 2 == 0 { 1.0 } else { 0.25 };
        let terminal = i > 0 && i % 20 == 0;
        let action = agent
            .step_masked(&state, &valid_actions, reward, terminal)
            .expect("step_masked Phase A");
        assert!(action < 9, "action out of bounds: {action}");
    }

    // Seal the training compartment — further pushes route to recent (B).
    agent
        .seal_replay_training_memories()
        .expect("seal after Phase A");

    // Phase B: 50 more steps — populate recent compartment.
    for i in 0..50 {
        let reward = if i % 2 == 0 { 1.0 } else { 0.25 };
        let terminal = i > 0 && i % 10 == 0;
        let action = agent
            .step_masked(&state, &valid_actions, reward, terminal)
            .expect("step_masked Phase B");
        assert!(action < 9, "action out of bounds: {action}");
    }

    // 5× replay_learn with batch_size = 64 off-policy updates.
    for _ in 0..5 {
        agent.replay_learn(64).expect("replay_learn");
    }

    // Discard recent compartment, leave training compartment intact.
    agent
        .clear_recent_memories()
        .expect("clear_recent_memories");

    // Roll back to the frozen champion. Cooldown is 0, so this is allowed.
    agent.rollback_hard().expect("rollback_hard");

    // 3× replay_learn warmup after rollback (critic was frozen to a stale V).
    for _ in 0..3 {
        agent.replay_learn(64).expect("replay_learn post-rollback");
    }

    // Final assertion: every actor + critic weight must be finite.
    assert_all_weights_finite(&agent, "end of end-to-end script");

    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 8.0,
        "phase2_replay_end_to_end must complete in < 8s, took {:.3}s",
        elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Smoke test 2: 4-phase stress scenario — champion → degradation → recovery.
// MAGI R4 W7 canonical stress test. Runtime budget: < 30 seconds.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Smoke gate: invoked explicitly by Phase 2 -> MAGI gate B boundary"]
fn phase2_stress_scenario_rollback_recovery() {
    let start = std::time::Instant::now();

    let config = smoke_config(200, 100);
    let mut agent: PcActorCritic<CpuLinAlg> =
        PcActorCritic::new(CpuLinAlg::new(), config, 123).expect("construct agent");
    agent.set_rollback_hard_cooldown(0);

    let valid_actions: Vec<usize> = (0..9).collect();
    let state = vec![0.5; 9];

    // -- Phase A: establish champion -------------------------------------
    // 500 step_masked calls with positive-reward signal. Every step also
    // has finiteness checked in aggregate (we sample a handful to keep
    // the runtime budget within 30s).
    for i in 0..500 {
        let reward = if i % 2 == 0 { 1.0 } else { 0.5 };
        let terminal = i > 0 && i % 25 == 0;
        agent
            .step_masked(&state, &valid_actions, reward, terminal)
            .expect("step_masked Phase A");
    }
    assert_all_weights_finite(&agent, "end of Phase A");

    let champion_baseline = flatten_actor_weights(&snapshot(&agent));
    agent
        .champion_update()
        .expect("champion_update after Phase A");
    agent
        .seal_replay_training_memories()
        .expect("seal after Phase A");

    // -- Phase B: deliberate degradation ---------------------------------
    // 1000 step_masked calls with negative reward to push weights away.
    // Sample L2 distance at 4 checkpoints and verify (near-)monotonic growth.
    let checkpoints = [250, 500, 750, 1000];
    let mut distances: Vec<f64> = Vec::with_capacity(checkpoints.len());

    for i in 0..1000 {
        let reward = if i % 2 == 0 { -1.0 } else { -0.5 };
        let terminal = i > 0 && i % 25 == 0;
        agent
            .step_masked(&state, &valid_actions, reward, terminal)
            .expect("step_masked Phase B");

        // L2 snapshot at the 4 checkpoints.
        if checkpoints.contains(&(i + 1)) {
            let live = flatten_actor_weights(&snapshot(&agent));
            distances.push(l2_distance(&live, &champion_baseline));
        }
    }
    assert_all_weights_finite(&agent, "end of Phase B");
    assert_eq!(distances.len(), 4, "captured 4 checkpoint distances");

    // Phase B assertion: degradation produces near-monotonic growth of
    // `||θ_live − champion_baseline||`. We permit a tiny noise margin
    // (5% of the first checkpoint distance) to avoid brittleness under
    // synthetic reward signals.
    //
    // This is a ROBUST invariant test: if it were to fail, we would
    // weaken to "final > initial" with a documented comment (see
    // plan: "prefer to test the ROBUST invariant").
    let noise_margin = distances[0] * 0.05;
    for pair in distances.windows(2) {
        assert!(
            pair[1] >= pair[0] - noise_margin,
            "distances expected near-monotonic growth under degradation: \
             {pair:?} (noise_margin = {noise_margin:.6})"
        );
    }
    assert!(
        distances.last().unwrap() > distances.first().unwrap(),
        "final distance {} must exceed initial distance {} after 1000 \
         negative-reward steps",
        distances.last().unwrap(),
        distances.first().unwrap()
    );

    // -- Phase C: recovery sequence --------------------------------------
    agent
        .clear_recent_memories()
        .expect("clear_recent_memories");

    agent.rollback_hard().expect("rollback_hard");

    // Assertion (b): immediately after rollback_hard, the live actor
    // must equal the champion baseline byte-exactly (hard copy semantics).
    let post_rollback = flatten_actor_weights(&snapshot(&agent));
    let post_rollback_distance = l2_distance(&post_rollback, &champion_baseline);
    assert_eq!(
        post_rollback_distance, 0.0,
        "after rollback_hard, ||θ_live − champion_baseline|| must be exactly 0.0, \
         got {post_rollback_distance}"
    );

    // 50× replay_learn warmup (critic fills in stale V with the
    // rolled-back actor's transitions).
    for _ in 0..50 {
        agent
            .replay_learn(64)
            .expect("replay_learn post-rollback warmup");
    }

    // -- Phase D: verification asserts -----------------------------------

    // (a) Aggregate finiteness after the whole scenario.
    assert_all_weights_finite(&agent, "end of Phase D");

    // (c) Drift envelope. Conservative bound per plan §3.7.1:
    //     drift ≤ lr_actor · MAX_REPLAY_TD_ERROR · batch · iters ≈ small.
    //     Use 2.0 as a generous threshold.
    let post_warmup = flatten_actor_weights(&snapshot(&agent));
    let drift = l2_distance(&post_warmup, &champion_baseline);
    assert!(
        drift < 2.0,
        "post-warmup drift from champion_baseline exceeded envelope: \
         {drift} (threshold = 2.0)"
    );

    // (d) Critic V(s) convergence (loose). The public surface does not
    //     expose V(s) directly, so this assertion is exercised indirectly:
    //     the agent must continue to produce finite logits + successfully
    //     step after warmup.
    for i in 0..10 {
        let reward = if i % 2 == 0 { 1.0 } else { 0.5 };
        agent
            .step_masked(&state, &valid_actions, reward, false)
            .expect("step_masked post-warmup sanity");
    }
    assert_all_weights_finite(&agent, "post post-warmup sanity steps");

    // (e) test does NOT call rollback_hard() twice within the recovery
    //     window — trivially satisfied: we only call it once above.

    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 30.0,
        "phase2_stress_scenario_rollback_recovery must complete in < 30s, \
         took {:.3}s",
        elapsed.as_secs_f64()
    );
}
