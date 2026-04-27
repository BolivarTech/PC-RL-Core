# PC-Pendulum — Experiment Specification

**Project:** Standalone Rust binary that consumes `pc-rl-core v4.0.0` to train and evaluate a Predictive Coding agent on the Pendulum-v1 swing-up task.

**Status:** Spec — implementation pending. Standalone repository (separate from `pc-rl-core`), following the PC-TicTacToe pattern.

**Target repository name:** `PC-Pendulum` (suggested) on `github.com/BolivarTech/`.

**Purpose:** Empirical validation that v4.0.0's continuous action mode actually trains a working policy on a canonical RL benchmark. Without this validation, v4.0.0 is "code that passes synthetic tests" but unproven against real dynamics.

---

## 1. Goal

Train a Predictive Coding actor-critic agent to **swing up and balance** an inverted pendulum from a random initial state, using only continuous torque actions. Demonstrate that:

1. The Gaussian-policy gradient `(μ − a) / σ²` produces a stable training trajectory on real dynamics.
2. PC inference + continuous output deliver competitive performance vs. a pure feedforward continuous baseline.
3. Surprise-driven learning rate modulation (M1) behaves sensibly on a continuous-reward task.

This is the canonical "hello world" of continuous control. If `pc-rl-core` v4.0.0 cannot solve Pendulum, the implementation has bugs not caught by synthetic tests.

---

## 2. Problem definition

### 2.1 Physical setup

A single rigid rod of length `L` and mass `m` rotates around a fixed pivot in a gravitational field. The agent applies a torque `u` to the pivot. The goal is to swing the rod from any initial position to the upright vertical position and balance there.

```
              ●  (mass m at end)
              │
              │  rod, length L
              │
        ──────┴──────  ← pivot (fixed in space)
                ↻  ← torque u (agent action)
```

### 2.2 State, action, reward (Pendulum-v1 spec)

| Component | Form | Range | Notes |
|---|---|---|---|
| State `s` | `[cos(θ), sin(θ), θ_dot]` | All ∈ ℝ | Angle encoded as cos/sin to avoid wraparound |
| Action `a` | scalar torque `u` | `[-2.0, 2.0]` N·m | Clipped post-sample |
| Reward `r` | `−(θ² + 0.1·θ_dot² + 0.001·u²)` | ≤ 0 | Quadratic cost on angle, velocity, action |

**Conventions:**
- `θ = 0` → upright vertical (goal pose)
- `θ = ±π` → hanging straight down (starting pose typically)
- Reward is **negative cost** — agent maximizes by minimizing all three terms simultaneously
- Per-step max reward: `0.0` (upright, stationary, zero torque)
- Per-step min reward: `≈ −16.27` (worst case: π angle, max velocity, max torque)

### 2.3 Physics model

Standard pendulum dynamics (ODE):

```
θ̈ = (3·g) / (2·L) · sin(θ) + (3·u) / (m·L²)
```

Default constants (Gymnasium Pendulum-v1 spec):
- `g = 10.0` (gravity, slightly higher than Earth for faster dynamics)
- `m = 1.0` (mass)
- `L = 1.0` (length)
- `dt = 0.05` (integration timestep, 20 Hz)
- `θ_dot_max = 8.0` (clamped to prevent runaway)
- `u_max = 2.0` (action clamp)

### 2.4 Episode structure

- **Reset:** `θ ~ Uniform[-π, π]`, `θ_dot ~ Uniform[-1, 1]`
- **Length:** 200 steps (fixed; never terminates)
- **Termination:** None — episode runs to step 200 regardless of state

This is intentional. Pendulum-v1 has no failure state; the agent simply accumulates cost over a fixed window. Balanced episodes converge to ~0 cost; failed episodes accumulate negative cost.

### 2.5 Performance benchmarks

| Agent type | Mean episode reward | Episodes to converge |
|---|---|---|
| Random policy | ~−1500 | N/A |
| REINFORCE vanilla | ~−500 | 500-1000 |
| DDPG / TD3 / SAC (replay) | ~−150 | 50-100 |
| **PC-RL-Core v4.0.0 target** | **~−250 to −350** | **300-500** |

Without replay (rejected in v4.0.0 continuous), expect performance below DDPG but well above random. The point is **convergence**, not SOTA.

---

## 3. Architecture

### 3.1 pc-rl-core configuration

```rust
use pc_rl_core::{
    PcActorCritic, PcActorCriticConfig, ActionSpace,
    pc_actor::PcActorConfig,
    mlp_critic::MlpCriticConfig,
    activation::Activation,
    layer::LayerDef,
    linalg::cpu::CpuLinAlg,
};

fn build_config() -> PcActorCriticConfig {
    PcActorCriticConfig {
        actor: PcActorConfig {
            input_size: 3,                          // [cos(θ), sin(θ), θ_dot]
            hidden_layers: vec![
                LayerDef { size: 32, activation: Activation::Tanh },
            ],
            output_size: 1,                         // single torque value
            output_activation: Activation::Tanh,    // → [-1, 1]; scaled to [-2, 2] post-sample
            alpha: 0.03,
            tol: 0.01,
            min_steps: 1,
            max_steps: 5,
            lr_weights: 0.005,
            synchronous: true,
            temperature: 1.0,
            local_lambda: 0.99,                     // v3.x sweet spot
            residual: false,
            rezero_init: 0.001,
        },
        critic: MlpCriticConfig {
            input_size: 3 + 32,                     // state + latent_concat
            hidden_layers: vec![
                LayerDef { size: 64, activation: Activation::Tanh },
            ],
            output_activation: Activation::Linear,
            lr: 0.005,
        },

        // === v4.0.0 continuous mode ===
        action_space: ActionSpace::Continuous,
        policy_sigma: 0.3,                          // exploration std-dev

        // === Required to be off in continuous (v4.0.0 validation rejects otherwise) ===
        distillation_lambda_polyak: 0.0,
        distillation_lambda_frozen: 0.0,
        replay_training_capacity: 0,
        replay_recent_capacity: 0,
        gae_lambda: None,
        td_steps: 0,

        // === Continuous Learning machinery (M1 active) ===
        gamma: 0.99,
        surprise_low: 0.02,
        surprise_high: 0.15,
        adaptive_surprise: true,
        surprise_buffer_size: 400,
        entropy_coeff: 0.0,
        scale_floor: 0.1,
        scale_ceil: 2.0,

        // === Hysteresis off for first run; revisit later if needed ===
        actor_hysteresis: false,
        critic_hysteresis: false,
        // ... defaults for other hysteresis fields ...

        // === EWC off (single task, no catastrophic forgetting concern) ===
        ewc_lambda: 0.0,
        // ... defaults for other EWC fields ...

        // === Polyak / replay defaults ===
        polyak_tau: 0.005,
        replay_positive_only: true,
        replay_batch_size: 64,
        scale_floor_replay: -1.0,
        critic_floor_replay: -1.0,

        // ... rest of defaults
    }
}
```

### 3.2 Action mapping

The actor output is `Tanh`-bounded to `[-1, 1]`. Scale to physical torque range:

```rust
let (raw_action, _) = agent.act_continuous(&state, mode)?;
let torque = raw_action[0] * 2.0;        // [-1, 1] → [-2, 2]
let torque = torque.clamp(-2.0, 2.0);    // defensive
```

### 3.3 Training loop sketch

```rust
for episode in 0..NUM_EPISODES {
    let mut env = Pendulum::reset(rng);
    let mut episode_reward = 0.0;

    for step in 0..200 {
        let state = env.observation();
        let raw = agent
            .step_continuous(&state, last_reward, false)?;
        let torque = (raw[0] * 2.0).clamp(-2.0, 2.0);

        let (next_state, reward, _done) = env.step(torque);
        episode_reward += reward;
        last_reward = reward;
    }

    // Metrics
    println!("episode {episode}: reward = {episode_reward:.2}");
    log_metrics(episode, episode_reward, agent.actor_hysteresis.as_ref(), ...);

    // Periodic checkpoint
    if episode % 50 == 0 {
        save_agent(&agent, &format!("checkpoints/ep_{episode:04}.json"), episode, None)?;
    }
}
```

**Note:** `step_continuous` is called with `state` of the **current** step but `reward` of the **previous** step's outcome — the canonical RL convention. The first step's previous reward is 0.0.

---

## 4. Implementation phases (TDD)

### Phase 1 — Pendulum environment (no agent yet)

**Goal:** Standalone `Pendulum` struct with deterministic physics.

**Files:**
- `src/env.rs` — Pendulum struct + step + reset
- `tests/env_tests.rs` — physics invariants

**Tests:**
- `test_initial_state_in_distribution` — reset 1000 times, verify θ ∈ [-π, π], θ_dot ∈ [-1, 1]
- `test_zero_torque_falls_naturally` — start at θ=π/4, no torque, observe θ moves toward π over time
- `test_max_torque_pushes_pendulum` — apply +2.0 torque, observe θ_dot increases
- `test_θ_dot_clamps_at_8` — apply continuous max torque for 100 steps, verify θ_dot never exceeds 8
- `test_reward_is_zero_at_upright_stationary` — state (1, 0, 0) with action 0 → reward = 0
- `test_reward_decreases_with_action_magnitude` — same state, |action|=1 vs |action|=2 → second is more negative

### Phase 2 — Agent setup + smoke test

**Goal:** Build the v4.0.0 continuous agent and run 1 episode (no learning expected, just verify integration).

**Files:**
- `src/agent.rs` — `build_config()` + `build_agent()`
- `src/main.rs` — minimal smoke main: build agent, run 1 episode, print reward

**Tests:**
- `test_agent_constructs` — `build_agent()` returns Ok
- `test_agent_step_continuous_returns_finite_action` — single step, action vector contains finite floats
- `test_agent_action_in_range` — over 100 random states, sampled action ∈ [-1, 1] (Tanh bound)

### Phase 3 — Training loop

**Goal:** End-to-end training run. Reward should improve over episodes.

**Files:**
- `src/training.rs` — training loop with metrics logging
- `src/metrics.rs` — episode reward + surprise + hysteresis state CSV output

**Tests:**
- `test_5_episode_smoke` — run 5 episodes, verify (a) no panics, (b) all rewards finite, (c) at least one weight changed since init
- `test_50_episode_improvement` — run 50 episodes, verify mean reward of episodes 40-49 is greater (less negative) than mean of episodes 0-9. **Soft assertion:** strict expectation is reward improves, but if the assertion is too strict, document the observed behavior.

### Phase 4 — Evaluation + checkpointing

**Goal:** Save trained agents, evaluate deterministic Play mode.

**Files:**
- `src/evaluation.rs` — Play-mode eval (no exploration, no learning)
- `bin/evaluate.rs` — CLI tool: load checkpoint, run N episodes, report mean reward

**Tests:**
- `test_save_load_roundtrip` — train 10 eps, save, load, verify next inference matches
- `test_play_mode_deterministic` — Play mode same state twice → same action

### Phase 5 — Multi-seed validation

**Goal:** Statistical confidence the agent works, not just lucky seed.

**Files:**
- `bin/multi_seed.rs` — run N agents with seeds [42, 43, ..., 42+N-1], save all metrics
- `scripts/analyze.py` (optional, Python for plotting) — generate convergence curves

**Procedure:**
- Run 10 seeds × 500 episodes each
- Compute mean ± std of final-100-episodes reward
- Target: mean < −400, std < 200 (rough)
- If one seed wildly diverges, investigate — possible RNG-dependent bug

---

## 5. File structure

```
PC-Pendulum/
├── Cargo.toml
├── README.md
├── src/
│   ├── env.rs              # Pendulum physics
│   ├── agent.rs            # pc-rl-core wrapper + config
│   ├── training.rs         # training loop
│   ├── evaluation.rs       # Play-mode eval
│   ├── metrics.rs          # CSV logging
│   └── main.rs             # train + evaluate
├── bin/
│   ├── evaluate.rs         # CLI: load checkpoint + eval
│   └── multi_seed.rs       # CLI: multi-seed sweep
├── tests/
│   ├── env_tests.rs        # Phase 1 tests
│   ├── integration_tests.rs # Phase 2-3 tests
│   └── eval_tests.rs       # Phase 4 tests
├── checkpoints/             # gitignored, output directory
├── results/                 # gitignored, CSV logs
├── scripts/
│   └── plot_convergence.py # optional: matplotlib convergence curves
└── docs/
    └── results.md          # post-experiment writeup
```

### `Cargo.toml`

```toml
[package]
name = "pc-pendulum"
version = "0.1.0"
edition = "2021"
authors = ["Julian Bolivar"]

[dependencies]
pc-rl-core = "4"           # v4.0.0+
rand = "0.8"
chrono = "0.4"             # for metrics timestamps
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[[bin]]
name = "evaluate"
path = "bin/evaluate.rs"

[[bin]]
name = "multi_seed"
path = "bin/multi_seed.rs"
```

---

## 6. Pendulum implementation reference

Concrete physics implementation (~50 LOC):

```rust
// src/env.rs
use rand::Rng;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Pendulum {
    theta: f64,
    theta_dot: f64,
}

impl Pendulum {
    const G: f64 = 10.0;
    const M: f64 = 1.0;
    const L: f64 = 1.0;
    const DT: f64 = 0.05;
    const THETA_DOT_MAX: f64 = 8.0;
    const TORQUE_MAX: f64 = 2.0;

    pub fn reset(rng: &mut impl Rng) -> Self {
        Self {
            theta: rng.gen_range(-PI..PI),
            theta_dot: rng.gen_range(-1.0..1.0),
        }
    }

    pub fn observation(&self) -> Vec<f64> {
        vec![self.theta.cos(), self.theta.sin(), self.theta_dot]
    }

    pub fn step(&mut self, raw_torque: f64) -> (Vec<f64>, f64, bool) {
        let u = raw_torque.clamp(-Self::TORQUE_MAX, Self::TORQUE_MAX);
        // Cost (negative reward)
        let theta_norm = ((self.theta + PI).rem_euclid(2.0 * PI)) - PI;
        let cost = theta_norm.powi(2)
                 + 0.1 * self.theta_dot.powi(2)
                 + 0.001 * u.powi(2);
        // Dynamics
        let theta_ddot = 3.0 * Self::G / (2.0 * Self::L) * self.theta.sin()
                       + 3.0 / (Self::M * Self::L * Self::L) * u;
        self.theta_dot += theta_ddot * Self::DT;
        self.theta_dot = self.theta_dot.clamp(-Self::THETA_DOT_MAX, Self::THETA_DOT_MAX);
        self.theta += self.theta_dot * Self::DT;
        (self.observation(), -cost, false)
    }
}
```

That's the full physics. ~30 LOC. Verify against Gymnasium Pendulum-v1 source if exact bit-equivalence is needed (for paper-comparison purposes).

---

## 7. Expected results

If everything works correctly:

| Episode range | Mean reward | Interpretation |
|---|---|---|
| 0-50 | −1500 to −1000 | Random behavior, no learning yet |
| 50-150 | −800 to −500 | Policy starts producing torque toward upright |
| 150-300 | −500 to −300 | Swing-up learned; balance unstable |
| 300-500 | −350 to −200 | Stable swing-up + balance; converged |

If you see:
- Stuck at ~−1500 → gradient direction problem (reverse v4.0.0 sign) or numerical issue (NaN propagation)
- Oscillates around −800 → policy_sigma too high (over-exploration) or lr_weights too high
- Slowly improves but plateaus at −800 → policy_sigma too low (under-exploration) or network capacity insufficient
- Improves then crashes back to −1500 → catastrophic forgetting (try `ewc_lambda > 0`) or numerical instability (lower lr)

---

## 8. Reporting & writeup

After multi-seed validation, write `docs/results.md` covering:

1. **Summary**: did v4.0.0 continuous mode solve Pendulum-v1? Convergence reward + episodes-to-converge.
2. **Convergence curves**: per-seed reward over episodes (matplotlib plot).
3. **PC vs MLP baseline**: optional second experiment with `local_lambda = 1.0` and no PC iterations (`max_steps = 1`) to isolate PC's contribution.
4. **policy_sigma sensitivity**: brief sweep over σ ∈ {0.1, 0.3, 0.5, 1.0}.
5. **Surprise dynamics**: plot mean per-step surprise over episodes — should decrease as agent learns.
6. **Hysteresis behavior** (if enabled): plot FROZEN/PLASTIC duty cycle.
7. **Lessons learned**: any deviations from expected, hyperparameter caveats.

This writeup is the empirical-validation deliverable that converts v4.0.0 from "shipped" to "proven".

---

## 9. Out of scope

- **Comparison with DDPG/SAC** — outside this experiment's scope. This is "does PC-RL-Core v4.0.0 work?", not "is it better than DDPG?".
- **Optimization for SOTA performance** — v4.0.0 lacks replay; matching DDPG is unrealistic without v4.x continuous-replay support.
- **Deployed binary** — this is a validation experiment, not a production trading agent.
- **Hyperparameter sweep beyond a few key knobs** — exhaustive HP search out of scope.

---

## 10. Estimated effort

| Phase | Time | Output |
|---|---|---|
| Phase 1 (env) | 2-3 hours | Working Pendulum struct + tests |
| Phase 2 (agent integration) | 1-2 hours | Smoke test passes |
| Phase 3 (training loop) | 2-3 hours | First training runs, basic metrics |
| Phase 4 (eval + checkpointing) | 1-2 hours | CLI eval tool |
| Phase 5 (multi-seed) | 2-3 hours | Statistical validation |
| Hyperparameter tuning | 4-8 hours | If first run doesn't converge cleanly |
| Writeup | 2-3 hours | results.md with plots |
| **Total** | **15-25 hours** | Validation complete |

---

## 11. Success criteria

The experiment is **successful** if:

1. ✓ Multi-seed mean reward (final 100 episodes) is at least 3× better than random baseline (−1500)
2. ✓ At least 5/10 seeds achieve final reward > −400
3. ✓ No seed produces NaN/Inf weights or panics
4. ✓ Training is reproducible under fixed seed (same seed → same trajectory)
5. ✓ Surprise score decreases over training (validates the metric)

The experiment is a **failure** (and reveals real bugs in v4.0.0) if:

- ✗ All seeds stuck at random-policy reward (~−1500)
- ✗ Most seeds NaN/Inf during training
- ✗ Reward improves then catastrophically degrades

If failure occurs, the failure mode is the diagnostic — file a bug report with seed, commit SHA, and metrics CSV. Investigation moves back to `pc-rl-core` repo.

---

## 12. Why this is the right first benchmark

Other candidates were considered and rejected for first-experiment status:

| Benchmark | Why not first |
|---|---|
| MountainCarContinuous | Sparse reward — too noisy without replay |
| BipedalWalker | High-dim action (4-D), requires more tuning |
| HalfCheetah | MuJoCo dependency, more complex |
| Reach (planar arm) | Requires custom env with no standard baseline |
| Pure CartPole-discrete | Wrong test (already covered by TicTacToe) |

Pendulum-v1 is the **lowest-friction** path to "v4.0.0 continuous works on real dynamics". If it fails, the bugs are surfaced early. If it succeeds, there's confidence to invest in more ambitious continuous benchmarks.
