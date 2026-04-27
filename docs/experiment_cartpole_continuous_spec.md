# PC-CartPole-Continuous — Experiment Specification

**Project:** Standalone Rust binary that consumes `pc-rl-core v4.0.0` to train and evaluate a Predictive Coding agent on a continuous-control variant of the cart-pole balance task.

**Status:** Spec — implementation pending. Standalone repository (separate from `pc-rl-core`), following the PC-TicTacToe pattern.

**Target repository name:** `PC-CartPole-Continuous` (suggested) on `github.com/BolivarTech/`.

**Purpose:** Companion to `PC-Pendulum`. Validates v4.0.0 continuous mode on a **different** dynamics regime — pole-on-cart with linear force input — vs. Pendulum's pure rotational torque input. Two complementary benchmarks → broader confidence in the implementation.

---

## 1. Goal

Train a Predictive Coding actor-critic agent to **balance a pole on a moving cart** by applying continuous horizontal force to the cart base. Demonstrate that:

1. v4.0.0 Gaussian-policy gradient handles the cart-pole dynamics regime correctly (different from pure pendulum).
2. The Tanh-bounded continuous output maps cleanly to a bounded force/acceleration command.
3. Sparse-ish termination-on-fail rewards train successfully in continuous mode (or shaped-reward variant if needed).

Where Pendulum-v1 tests "swing-up + continuous-cost reward", this tests "balance + termination signal" — distinct dynamics, distinct reward shape.

---

## 2. Problem definition

### 2.1 Physical setup

A pole of length `2L` is hinged at its lower end on the top of a frictionless cart of mass `M`. The cart slides horizontally on a track of length `2x_threshold`. The agent applies a horizontal force `F` to the cart. Gravity pulls the pole; if the pole tilts past `±θ_threshold` or the cart leaves the track, the episode terminates.

```
              ●  (pole tip, mass m)
              │
              │  pole, length 2L
              │
        ┌─────┴─────┐
        │   CART     │  (mass M)
        └───────────┘
        ──→  Force F (agent action)
        ←───────  x-axis (track) ───────→
                ↑                       ↑
            -x_threshold          +x_threshold
```

### 2.2 State, action, reward

| Component | Form | Range | Notes |
|---|---|---|---|
| State `s` | `[x, ẋ, θ, θ̇]` | All ∈ ℝ | Cart pos, cart vel, pole angle (radians from vertical), pole angular vel |
| Action `a` | scalar force `F` | `[-F_max, F_max]` (default `[-3.0, 3.0]` N) | Tanh-bounded then scaled |
| Reward `r` | `+1` per step alive (Variant A) or shaped (Variant B) | scalar | Sparse — survive longer = more reward |

**Conventions:**
- `θ = 0` → pole upright (goal pose, starting pose)
- `θ > 0` → pole tilts to the right
- `x = 0` → cart centered on track
- Episode terminates when `|x| > x_threshold` (cart fell off) OR `|θ| > θ_threshold` (pole fell)
- No goal state per se — episode lasts as long as agent keeps the pole up

### 2.3 Physics model

Standard cart-pole equations (Russell-Norvig / Barto-Sutton 1983):

```
θ̈ = (g·sin(θ) − cos(θ)·temp) / (L·(4/3 − m·cos²(θ)/(M+m)))
ẍ = temp − m·L·θ̈·cos(θ) / (M+m)

where:
    temp = (F + m·L·θ̇²·sin(θ)) / (M+m)
```

Default constants:
- `g = 9.8` (gravity)
- `M = 1.0` (cart mass)
- `m = 0.1` (pole mass)
- `L = 0.5` (half-pole length, so pole length is 1.0)
- `dt = 0.02` (50 Hz)
- `F_max = 3.0` (force limit)
- `θ_threshold = 12° = 0.2095 rad` (pole-fall termination)
- `x_threshold = 2.4` (track edge)

**Integration:** Forward Euler is sufficient for `dt = 0.02`; semi-implicit Euler is more stable. Either is fine for this task.

### 2.4 Episode structure

- **Reset:** `x, ẋ, θ, θ̇ ~ Uniform[−0.05, 0.05]` (small random perturbation around upright)
- **Length:** Maximum 500 steps (10 seconds at 50 Hz)
- **Termination:** Episode ends when `|x| > x_threshold` OR `|θ| > θ_threshold` OR step count reaches 500

### 2.5 Reward shaping considerations

Default is `+1 per step alive` — sparse signal. For continuous mode without replay buffer, sparse reward may be slow to learn. Two variants are recommended:

#### Variant A — Pure sparse (canonical CartPole)

```
r_t = +1.0  if not terminated this step
r_t =  0.0  if terminated
```

#### Variant B — Shaped continuous (recommended for first attempt)

```
r_t = 1.0 − (θ/θ_threshold)² − 0.1·(x/x_threshold)² − 0.01·F²
```

Penalizes pole tilt, cart drift, and large forces. Provides per-step gradient signal even when not terminated.

**Recommendation:** Implement BOTH variants. Run experiments with B first to validate the agent can learn at all; then run A to test robustness to sparse rewards.

### 2.6 Performance benchmarks

| Agent | Reward (variant B, shaped) | Reward (variant A, sparse) | Episodes to converge |
|---|---|---|---|
| Random policy | ~50 | ~20 | N/A |
| REINFORCE vanilla | ~250 | ~150 | 200-500 |
| DDPG / SAC (replay) | ~480 (near max 500) | ~480 | 50-100 |
| **PC-RL-Core v4.0.0 target (variant B)** | **~350-450** | **~200-350** | **200-400** |

---

## 3. Architecture

### 3.1 pc-rl-core configuration

```rust
fn build_config() -> PcActorCriticConfig {
    PcActorCriticConfig {
        actor: PcActorConfig {
            input_size: 4,                          // [x, ẋ, θ, θ̇]
            hidden_layers: vec![
                LayerDef { size: 32, activation: Activation::Tanh },
            ],
            output_size: 1,                         // single force value
            output_activation: Activation::Tanh,    // → [-1, 1]; scaled to [-3, 3]
            alpha: 0.03,
            tol: 0.01,
            min_steps: 1,
            max_steps: 5,
            lr_weights: 0.005,
            synchronous: true,
            temperature: 1.0,
            local_lambda: 0.99,
            residual: false,
            rezero_init: 0.001,
        },
        critic: MlpCriticConfig {
            input_size: 4 + 32,                     // state + latent_concat
            hidden_layers: vec![
                LayerDef { size: 64, activation: Activation::Tanh },
            ],
            output_activation: Activation::Linear,
            lr: 0.005,
        },

        // === v4.0.0 continuous mode ===
        action_space: ActionSpace::Continuous,
        policy_sigma: 0.5,                          // higher than Pendulum

        // === Required to be off in continuous ===
        distillation_lambda_polyak: 0.0,
        distillation_lambda_frozen: 0.0,
        replay_training_capacity: 0,
        replay_recent_capacity: 0,
        gae_lambda: None,
        td_steps: 0,

        // === CL machinery ===
        gamma: 0.99,
        surprise_low: 0.02,
        surprise_high: 0.15,
        adaptive_surprise: true,
        surprise_buffer_size: 400,
        entropy_coeff: 0.0,
        scale_floor: 0.1,
        scale_ceil: 2.0,

        actor_hysteresis: false,
        critic_hysteresis: false,
        ewc_lambda: 0.0,

        polyak_tau: 0.005,
        replay_positive_only: true,
        replay_batch_size: 64,
        scale_floor_replay: -1.0,
        critic_floor_replay: -1.0,

        // ... rest of defaults ...
    }
}
```

### 3.2 Action mapping

```rust
let (raw_action, _) = agent.act_continuous(&state, mode)?;
let force = raw_action[0] * 3.0;        // [-1, 1] → [-3, 3]
let force = force.clamp(-3.0, 3.0);     // defensive
```

### 3.3 Training loop sketch

```rust
for episode in 0..NUM_EPISODES {
    let mut env = CartPole::reset(rng);
    let mut episode_reward = 0.0;
    let mut step_count = 0;
    let mut last_reward = 0.0;

    for step in 0..500 {
        let state = env.observation();
        let raw = agent.step_continuous(&state, last_reward, false)?;
        let force = (raw[0] * 3.0).clamp(-3.0, 3.0);

        let (next_state, reward, done) = env.step(force);
        episode_reward += reward;
        last_reward = reward;
        step_count += 1;

        if done {
            // Final terminal step — propagate done flag
            let _ = agent.step_continuous(&next_state, last_reward, true);
            break;
        }
    }

    println!("episode {episode}: steps = {step_count}, reward = {episode_reward:.2}");
}
```

**Important:** when episode terminates early, call `step_continuous` one more time with `done = true` so the critic learns the terminal V(s) = 0 boundary correctly.

---

## 4. Implementation phases (TDD)

### Phase 1 — CartPole environment

**Files:**
- `src/env.rs` — CartPole struct + step + reset
- `src/reward.rs` — Variant A and Variant B reward functions
- `tests/env_tests.rs` — physics + termination invariants

**Tests:**
- `test_initial_state_in_distribution` — reset 1000 times, all components ∈ [−0.05, 0.05]
- `test_pole_falls_under_gravity` — start at θ=0.05, no force, observe θ grows
- `test_force_moves_cart` — apply +3.0 force, x increases
- `test_terminates_on_pole_fall` — manually set θ = 0.5 (> threshold), step, verify done=true
- `test_terminates_on_cart_off_track` — apply max force for many steps, verify x exceeds threshold
- `test_episode_length_capped_at_500` — verify upper bound on episode length
- `test_reward_variants` — verify Variant A returns +1, Variant B returns shaped value

### Phase 2 — Agent setup + smoke

- `test_agent_constructs`
- `test_agent_step_continuous_returns_finite_action`
- `test_agent_action_in_range`

### Phase 3 — Training (variant B first)

- `test_5_episode_smoke` — 5 episodes, no panics, finite metrics
- `test_50_episode_improvement` — episodes 40-49 mean > episodes 0-9 mean

### Phase 4 — Variant A (sparse reward)

- `test_50_episode_sparse_smoke` — runs without panics
- `test_episode_length_distribution` — distribution shifts toward longer episodes over training

### Phase 5 — Evaluation + checkpointing + multi-seed

Identical pattern to Pendulum spec.

---

## 5. File structure

```
PC-CartPole-Continuous/
├── Cargo.toml
├── README.md
├── src/
│   ├── env.rs              # CartPole physics
│   ├── reward.rs           # Both reward variants
│   ├── agent.rs            # pc-rl-core wrapper
│   ├── training.rs         # Training loop with done-flag handling
│   ├── evaluation.rs       # Play-mode eval
│   ├── metrics.rs          # CSV logging (includes episode_length)
│   └── main.rs             # Entry point
├── bin/
│   ├── evaluate.rs
│   ├── multi_seed.rs
│   └── compare_variants.rs # Variant A vs B comparison runner
├── tests/
│   ├── env_tests.rs
│   ├── integration_tests.rs
│   └── eval_tests.rs
├── checkpoints/
├── results/
├── scripts/
│   └── plot_convergence.py
└── docs/
    └── results.md
```

### `Cargo.toml`

```toml
[package]
name = "pc-cartpole-continuous"
version = "0.1.0"
edition = "2021"
authors = ["Julian Bolivar"]

[dependencies]
pc-rl-core = "4"
rand = "0.8"
chrono = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[[bin]]
name = "evaluate"
path = "bin/evaluate.rs"

[[bin]]
name = "multi_seed"
path = "bin/multi_seed.rs"

[[bin]]
name = "compare_variants"
path = "bin/compare_variants.rs"
```

---

## 6. CartPole implementation reference

```rust
// src/env.rs
use rand::Rng;

#[derive(Debug, Clone)]
pub struct CartPole {
    x: f64,
    x_dot: f64,
    theta: f64,
    theta_dot: f64,
    steps: usize,
}

impl CartPole {
    const G: f64 = 9.8;
    const CART_MASS: f64 = 1.0;
    const POLE_MASS: f64 = 0.1;
    const TOTAL_MASS: f64 = Self::CART_MASS + Self::POLE_MASS;
    const HALF_LENGTH: f64 = 0.5;
    const DT: f64 = 0.02;
    const FORCE_MAX: f64 = 3.0;
    const THETA_THRESHOLD: f64 = 0.2095;
    const X_THRESHOLD: f64 = 2.4;
    const MAX_STEPS: usize = 500;

    pub fn reset(rng: &mut impl Rng) -> Self {
        Self {
            x: rng.gen_range(-0.05..0.05),
            x_dot: rng.gen_range(-0.05..0.05),
            theta: rng.gen_range(-0.05..0.05),
            theta_dot: rng.gen_range(-0.05..0.05),
            steps: 0,
        }
    }

    pub fn observation(&self) -> Vec<f64> {
        vec![self.x, self.x_dot, self.theta, self.theta_dot]
    }

    pub fn step(&mut self, raw_force: f64) -> (Vec<f64>, f64, bool) {
        let f = raw_force.clamp(-Self::FORCE_MAX, Self::FORCE_MAX);
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        let temp = (f + Self::POLE_MASS * Self::HALF_LENGTH 
                  * self.theta_dot.powi(2) * sin_t) / Self::TOTAL_MASS;
        let theta_acc = (Self::G * sin_t - cos_t * temp)
            / (Self::HALF_LENGTH * (4.0 / 3.0
                - Self::POLE_MASS * cos_t.powi(2) / Self::TOTAL_MASS));
        let x_acc = temp - Self::POLE_MASS * Self::HALF_LENGTH * theta_acc * cos_t
                  / Self::TOTAL_MASS;
        // Semi-implicit Euler
        self.x_dot += x_acc * Self::DT;
        self.x += self.x_dot * Self::DT;
        self.theta_dot += theta_acc * Self::DT;
        self.theta += self.theta_dot * Self::DT;
        self.steps += 1;
        let done = self.x.abs() > Self::X_THRESHOLD
                || self.theta.abs() > Self::THETA_THRESHOLD
                || self.steps >= Self::MAX_STEPS;
        (self.observation(), self.reward_shaped(f), done)
    }

    fn reward_shaped(&self, force: f64) -> f64 {
        // Variant B (shaped). For Variant A, return 1.0 if not done else 0.0.
        let theta_term = (self.theta / Self::THETA_THRESHOLD).powi(2);
        let x_term = (self.x / Self::X_THRESHOLD).powi(2);
        let f_term = force.powi(2);
        1.0 - theta_term - 0.1 * x_term - 0.01 * f_term
    }
}
```

~70 LOC. Verify against canonical CartPole-v1 source if exact bit-equivalence is needed.

---

## 7. Expected results

### Variant B (shaped reward)

| Episode range | Mean reward | Mean episode length | Interpretation |
|---|---|---|---|
| 0-50 | ~50 | ~50 steps | Random fall |
| 50-150 | ~150 | ~150 steps | Agent learns to push opposite to fall |
| 150-300 | ~300 | ~300 steps | Mostly balanced, occasional fall |
| 300-500 | ~400-450 | ~400-500 steps | Stable balance |

### Variant A (sparse +1 reward)

| Episode range | Mean reward | Mean episode length |
|---|---|---|
| 0-50 | ~20 | ~20 steps |
| 50-150 | ~80 | ~80 steps |
| 150-300 | ~200 | ~200 steps |
| 300-500 | ~250-350 | ~250-350 steps |

---

## 8. Reporting

`docs/results.md` should cover:

1. **Summary**: did v4.0.0 train CartPole successfully under both reward variants?
2. **Convergence curves** per-seed for both variants.
3. **Episode-length distribution** over training — visual proof of balance learning.
4. **Variant A vs B comparison** — sample efficiency difference.
5. **policy_sigma sensitivity** — sweep over σ ∈ {0.2, 0.3, 0.5, 0.7, 1.0}.
6. **Comparison with Pendulum-v1 results** — does v4.0.0 generalize across continuous-control regimes?

---

## 9. Out of scope

- DDPG/SAC comparison
- High-dimensional cart-pole (multiple links)
- Vision input (pixel-based)
- Real hardware deployment

---

## 10. Estimated effort

| Phase | Time | Output |
|---|---|---|
| Phase 1 (env) | 3-4 hours | CartPole struct + tests + both reward variants |
| Phase 2 (agent) | 1-2 hours | Smoke test |
| Phase 3 (training, variant B) | 2-3 hours | First runs, basic metrics |
| Phase 4 (variant A) | 1-2 hours | Sparse-reward training validated |
| Phase 5 (eval + multi-seed) | 3-4 hours | Statistical validation |
| Hyperparameter tuning | 4-6 hours | If first runs underperform |
| Comparison with Pendulum | 1-2 hours | Cross-reference write-up |
| Writeup | 2-3 hours | results.md with plots |
| **Total** | **17-26 hours** | Validation complete |

---

## 11. Success criteria

The experiment is **successful** if:

1. ✓ Variant B mean reward (final 100 episodes) > 350 across seeds
2. ✓ Variant B mean episode length (final 100) > 350 steps
3. ✓ Variant A trains to mean reward > 200
4. ✓ At least 7/10 seeds achieve "balanced" behavior
5. ✓ No NaN/Inf weights, no panics
6. ✓ Surprise score decreases over training

The experiment **reveals bugs** if:

- ✗ Variant B fails to improve under any seed
- ✗ Most seeds get NaN during training
- ✗ Mean episode length stuck at ~50 steps

---

## 12. Relationship to PC-Pendulum

| Aspect | Pendulum-v1 | CartPole-Continuous |
|---|---|---|
| Action interpretation | Torque (rotational) | Force (linear) |
| Termination | Never (fixed-length) | Yes (pole-fall or cart-off) |
| Reward shape | Continuous quadratic cost | +1/step (sparse) or shaped |
| State complexity | 3-D (cos, sin, dot) | 4-D (x, ẋ, θ, θ̇) |
| Task difficulty | Swing-up + balance (harder) | Balance only (easier) |
| Convergence target | ~−300 (negative cost) | ~400 (positive accumulated) |

**Recommended order:** Pendulum first (simpler dynamics, no termination handling). CartPole second (validates termination logic + sparse reward).

If both succeed, v4.0.0 continuous mode is empirically validated across two distinct regimes — strong evidence the implementation is correct.

---

## 13. Future extensions (post-validation)

- **Add hysteresis (M2)** — does FROZEN/PLASTIC modulation help under perturbations?
- **EWC across reward variants** — train on Variant B then switch to Variant A; does EWC prevent catastrophic forgetting?
- **CCA crossover** — train two seeds, crossover, evaluate offspring (validates v1.1.0 GA tooling on continuous).
- **Ablation: PC vs MLP** — `local_lambda = 1.0` + `max_steps = 1` to compare.

These are **post-validation** extensions, not part of the core experiment success criteria.
