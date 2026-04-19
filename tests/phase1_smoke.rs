// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-13

//! Phase 1 smoke gate: end-to-end self-recovery integration test.
//!
//! This test is `#[ignore]`-gated and only invoked explicitly at the
//! Phase 1 / Phase 2 boundary. It does NOT run during normal CI.

use pc_rl_core::{CpuLinAlg, PcActorCritic, PcActorCriticConfig};

/// Build a config with both distillation anchors enabled plus EWC.
///
/// Uses serde JSON deserialization so that all non-specified fields
/// pick up their `#[serde(default)]` values automatically.
fn smoke_config() -> PcActorCriticConfig {
    let json = r#"{
        "actor": {
            "input_size": 9,
            "hidden_layers": [{ "size": 18, "activation": "Tanh" }],
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
        },
        "critic": {
            "input_size": 27,
            "hidden_layers": [{ "size": 36, "activation": "Tanh" }],
            "output_activation": "Linear",
            "lr": 0.005
        },
        "gamma": 0.99,
        "ewc_lambda": 0.1,
        "distillation_lambda_polyak": 0.05,
        "polyak_tau": 0.005,
        "distillation_lambda_frozen": 0.05
    }"#;
    serde_json::from_str(json).expect("smoke config JSON must parse")
}

#[test]
#[ignore = "Smoke gate: invoked explicitly by Phase 1 -> Phase 2 boundary"]
fn phase1_self_recovery_end_to_end() {
    let config = smoke_config();
    let mut agent: PcActorCritic<CpuLinAlg> =
        PcActorCritic::new(CpuLinAlg::new(), config, 42).unwrap();

    // Disable cooldown so rollback_hard can be called freely in the test.
    agent.set_rollback_hard_cooldown(0);

    let valid_actions: Vec<usize> = (0..9).collect();
    let state = vec![0.5; 9];

    // Run 200 step_masked calls with alternating +1/-1 reward.
    for i in 0..200 {
        let reward = if i % 2 == 0 { 1.0 } else { -1.0 };
        let terminal = i > 0 && i % 20 == 0;
        let action = agent
            .step_masked(&state, &valid_actions, reward, terminal)
            .expect("step_masked should not fail");
        assert!(action < 9, "action out of bounds: {action}");
    }

    // Exercise champion_update (frozen anchor must be allocated).
    agent
        .champion_update()
        .expect("champion_update should succeed with distillation_lambda_frozen > 0");

    // Exercise rollback_soft (Polyak anchor must be allocated).
    agent
        .rollback_soft()
        .expect("rollback_soft should succeed with distillation_lambda_polyak > 0");

    // Exercise rollback_hard (frozen anchor must be allocated).
    agent
        .rollback_hard()
        .expect("rollback_hard should succeed with distillation_lambda_frozen > 0");

    // Final sanity: infer on a normal input and verify all logits are finite.
    let result = agent.infer(&state);
    for (i, &v) in result.y_conv.iter().enumerate() {
        assert!(
            v.is_finite(),
            "logit[{i}] is not finite after 200 steps + rollback cycle: {v}"
        );
    }
    for (i, &v) in result.latent_concat.iter().enumerate() {
        assert!(
            v.is_finite(),
            "latent[{i}] is not finite after 200 steps + rollback cycle: {v}"
        );
    }
}
