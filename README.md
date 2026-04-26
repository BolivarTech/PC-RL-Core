# PC-RL-Core

[![CI](https://github.com/BolivarTech/PC-RL-Core/actions/workflows/ci.yml/badge.svg)](https://github.com/BolivarTech/PC-RL-Core/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/pc-rl-core.svg)](https://crates.io/crates/pc-rl-core)
[![docs.rs](https://docs.rs/pc-rl-core/badge.svg)](https://docs.rs/pc-rl-core)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A **Deliberative Predictive Coding (DPC)** reinforcement learning framework implemented entirely in Rust with zero ML framework dependencies.

The actor **deliberates before acting** by running an iterative free energy minimization loop (predictive coding inference), and a **residual echo of that deliberation** feeds back into weight updates as a structured micro-regularizer. These two mechanisms form a coupled system: deliberation generates the signal, the signal improves learning, and better learning improves future deliberation.

The library is **backend-agnostic**: all linear algebra operations are abstracted behind a `LinAlg` trait, enabling future GPU backends (wgpu, CUDA) without changing the RL logic.

## Installation

```toml
[dependencies]
pc-rl-core = "2.0"
```

## Quick Start

```rust
use pc_rl_core::{
    CpuLinAlg, PcActorCritic, PcActorCriticConfig, PcActorConfig, MlpCriticConfig,
    Activation, LayerDef, SelectionMode,
};

// Configure the agent
let actor_config = PcActorConfig {
    input_size: 9,
    output_size: 9,
    hidden_layers: vec![LayerDef { size: 27, activation: Activation::Softsign }],
    output_activation: Activation::Linear,
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
};

let critic_config = MlpCriticConfig {
    input_size: 36,  // state_dim + latent_dim
    hidden_layers: vec![LayerDef { size: 36, activation: Activation::Softsign }],
    output_activation: Activation::Linear,
    lr: 0.005,
};

let config = PcActorCriticConfig {
    actor: actor_config,
    critic: critic_config,
    gamma: 0.99,
    surprise_low: 0.02,
    surprise_high: 0.15,
    adaptive_surprise: true,
    surprise_buffer_size: 400,
    entropy_coeff: 0.0,
    td_steps: 0,       // TD(n) n-step returns (0 = default, >=2 for multi-step)
    gae_lambda: Some(0.95), // GAE(λ) eligibility traces (None = disabled, recommended 0.95)
    ..Default::default()  // CL features default to disabled
};

let backend = CpuLinAlg::new();
let mut agent = PcActorCritic::new(backend, config, 42)?;

// Continuous learning loop (step API — TD(0) or TD(n))
loop {
    let action = agent.step(&state, reward, terminal);
    // ... or with action masking:
    let action = agent.step_masked(&state, &valid_actions, reward, terminal)?;

    if terminal { break; }
    // ... execute action in environment, get next state + reward ...
}

// Episodic learning (REINFORCE — alternative to step API)
let (action, infer_result) = agent.act(&state, &valid_actions, SelectionMode::Training);
// ... collect TrajectoryStep per timestep ...
let avg_loss = agent.learn(&trajectory);

// Evaluation (deterministic)
let (action, _) = agent.act(&state, &valid_actions, SelectionMode::Play);
```

## Self-Recovery Workflow

Self-recovery is the library's answer to catastrophic forgetting and
policy cascades during continuous learning. It exposes three
complementary mechanisms:

- **Polyak-tracked target** (noise smoothing, `~1/polyak_tau` step lag).
- **Frozen champion anchor** (cascade recovery, immutable between
  explicit promotions).
- **Dual-compartment replay buffer** (off-policy TD updates from stored
  positive-reward trajectories).

All three are **opt-in** (`lambda = 0` and `capacity = 0` defaults make
them no-ops). The consumer drives the recovery pipeline — the library
never calls `rollback_*` or `champion_update` on its own.

### Enabling

```rust
use pc_rl_core::{CpuLinAlg, PcActorCritic, PcActorCriticConfig};

let config = PcActorCriticConfig {
    // ... existing actor/critic/gamma/... fields ...

    // Phase 1: dual anchors
    distillation_lambda_polyak: 0.05,   // KL weight toward Polyak target
    polyak_tau:                 0.005,  // EMA rate (~200-step lag)
    distillation_lambda_frozen: 0.05,   // KL weight toward frozen champion

    // Phase 2: replay buffer
    replay_training_capacity:   200,    // compartment A size (0 disables)
    replay_recent_capacity:     100,    // compartment B size
    replay_positive_only:       true,   // drop reward <= 0.0 transitions
    replay_batch_size:          64,     // sample size per replay_learn call

    // EWC regularization composes naturally with the anchors
    ewc_lambda: 0.1,

    ..Default::default()
};

let mut agent = PcActorCritic::new(CpuLinAlg::new(), config, 42)?;
```

### Typical consumer pipeline

```rust
// 1) Train normally — transitions auto-record into compartment A.
for step in 0..n_warmup {
    let action = agent.step_masked(&state, &valid, reward, terminal)?;
    // ... environment step ...
}

// 2) Lock in a champion once fitness is acceptable. `champion_update`
//    promotes the live actor into the frozen slot; `seal_replay...`
//    freezes compartment A and routes further pushes to compartment B.
agent.champion_update()?;
agent.seal_replay_training_memories()?;

// 3) Continue learning. Compartment B now collects recent successes.
for step in 0..n_stress {
    let action = agent.step_masked(&state, &valid, reward, terminal)?;
}

// 4) If a fitness regression is detected (consumer-side check):
if consumer_detects_cascade(&agent) {
    agent.clear_recent_memories()?;   // drop contaminated compartment B
    agent.rollback_hard()?;           // live actor <- frozen champion
    for _ in 0..50 {                  // critic warmup from stored A transitions
        agent.replay_learn(64)?;
    }
}

// 5) For short-horizon noise smoothing (NOT cascade recovery), use:
agent.rollback_soft()?;               // live actor <- Polyak target
```

### When to use each method

| Symptom observed by consumer | Recommended response |
|---|---|
| Policy oscillating around a local minimum | `rollback_soft` (undoes ~`1/polyak_tau` steps of noise) |
| Fitness regression sustained over hundreds of steps | `clear_recent_memories` + `rollback_hard` + warmup |
| Starting fresh recovery cycle with a new champion | `champion_update` + `seal_replay_training_memories` |
| Monitoring off-policy stability | `agent.replay_clamp_count()` (monotonic counter) |

`rollback_hard` enforces a cooldown window (default 100 steps) — calling
it in a tight loop returns `Err(PcError::ConfigValidation)` without
mutating state. Override via `set_rollback_hard_cooldown(n)`.

### Observability

- `replay_clamp_count() -> u64` — monotonic counter of replay updates
  where the TD-error clamp bound (±5.0). Sustained incrementing is the
  leading indicator that off-policy drift is close to its envelope; the
  consumer can size warmup windows from the growth rate.
- Save/load persists all self-recovery state: anchor weights, replay
  buffer contents, clamp counter, and cooldown timers. Legacy
  pre-Phase-1 save files load cleanly with anchors auto-initialized
  from the live actor.

See the `# When to use` sections on `rollback_soft` / `rollback_hard`
and the "Stale V(s) batch semantics" rustdoc on `replay_learn` for the
full design rationale and parameter-tuning guidance.

### Replay under actor and critic hysteresis

By default, when actor or critic hysteresis is enabled and the
corresponding network is in FROZEN state, `replay_learn` updates
the gated network at the `scale_floor` clamp (default 0.0 → no
update). The protected network is shielded from off-policy
gradients during stress; the un-gated network continues learning.

To let replay reinforce a FROZEN network anyway, set the
corresponding opt-in floor to a strict-positive value:

- **Actor:** `scale_floor_replay = 0.3` (or higher; up to `10 ×
  scale_ceil`). Actor opt-in also enables Polyak and Frozen KL
  anchor gradients in the replay update.
- **Critic (v3.0.0+):** `critic_floor_replay = 0.3` — parallel
  knob for the critic. Same tri-state sentinel semantics, same
  validation rule.

Recommended pairs (set both fields together to keep
actor-critic dynamics symmetric):

- `(-1.0, -1.0)` (default) — both networks protected during
  FROZEN-replay.
- `(0.3, 0.3)` — mild symmetric recovery.
- `(1.0, 1.0)` — aggressive symmetric recovery.

Asymmetric pairs (one strict-positive, the other at sentinel)
are allowed but produce desynchronization — the moving network
learns from storage while the gated network stays frozen.

### Migration from v3.x to v4.0.0 — generic action space

In v4.0.0, `pc-rl-core` adds continuous-action support alongside
the existing discrete pipeline. v3.x consumers:

**Discrete (default) — minimal migration:**

- Add `?` to `act()` calls (return type became `Result`).
- Replace deprecated `step(state, reward, done)` with
  `step_masked(state, &(0..output_size).collect::<Vec<_>>(), reward, done)?`.
- All other API surface unchanged.

**Adopting continuous:**

```rust,ignore
use pc_rl_core::{ActionSpace, PcActorCriticConfig};
use pc_rl_core::activation::Activation;

// Start from your existing v3.x config literal (PcActorCriticConfig
// has no Default impl — replicate field-by-field or reload via serde).
let mut config: PcActorCriticConfig = existing_v3_config;

config.action_space = ActionSpace::Continuous;
config.policy_sigma = 0.1;             // Gaussian std-dev
config.distillation_lambda_polyak = 0.0;  // required: continuous mode
config.distillation_lambda_frozen = 0.0;  // required: continuous mode
config.actor.output_activation = Activation::Tanh;  // for bounded actions

let mut agent = PcActorCritic::new(backend, config, seed)?;

loop {
    let action = agent.step_continuous(&state, reward, done)?;
    let next_state = env.apply(&action);
    state = next_state;
}
```

**Self-recovery toolkit availability:**

| Mode | rollback_soft | rollback_hard | champion_update |
|---|---|---|---|
| Discrete | ✓ | ✓ | ✓ |
| Continuous | ✗ (Polyak distillation rejected) | ✗ (Frozen distillation rejected) | ✗ |

L2-anchored continuous distillation is experimental future work.
For continuous training requiring self-recovery, evaluate via
discrete intermediate or wait for the experimental branch.

### Migration from v2.2.x to v3.0.0 — critic hysteresis enforcement

In v3.0.0 the critic's `critic_hysteresis.state` is enforced on
weight updates for the first time. Consumers running with
`critic_hysteresis = true` who relied on the v2.2.x implicit
"critic always updates" behavior will see changed dynamics. See
`CHANGELOG.md` `[3.0.0] - Breaking changes` for the full
migration table.

**Recommended migration — paired opt-in.** Start from your existing
v2.2.x `PcActorCriticConfig` literal (or your serde-loaded config)
and override the four self-recovery fields below:

```rust,ignore
use pc_rl_core::PcActorCriticConfig;

// `existing_v2_2_x_config` is your current full PcActorCriticConfig
// literal (PcActorCriticConfig does NOT implement Default — replicate
// your existing field-by-field literal here, or reload via serde).
let mut config: PcActorCriticConfig = existing_v2_2_x_config;

// Paired opt-in: both actor and critic learn during FROZEN-replay,
// preserving the v2.2.x effective behaviour of "critic always
// learning during stress" while also activating the actor side.
// The two fields should be set TOGETHER to avoid actor-critic
// desynchronization.
config.actor_hysteresis = true;
config.critic_hysteresis = true;

// 0.3 is "mild recovery". For behavioural equivalence to the
// v2.2.x dynamic surprise→scale band, use `config.scale_ceil`
// (typically 2.0); see CHANGELOG [3.0.0] migration table.
config.scale_floor_replay = 0.3;
config.critic_floor_replay = 0.3;
```

Leaving both at their default `-1.0` sentinel is also valid and
corresponds to "full stress protection" — neither network updates
during FROZEN windows, and cross-wake coupling eventually
re-activates learning. Partial opt-in (one field positive, the
other `-1.0`) is allowed but produces asymmetric dynamics and is
not recommended for most workloads.

## Architecture

### Core Components

- **`PcActor<L: LinAlg>`** -- Policy network with predictive coding inference loop, residual skip connections, surprise scoring, and CCA crossover
- **`MlpCritic<L: LinAlg>`** -- Standard MLP value function with MSE loss backpropagation and CCA crossover
- **`PcActorCritic<L: LinAlg>`** -- Integrated agent combining actor and critic with surprise-based learning rate scheduling, continuous learning (CL), and TD(n) n-step returns
- **`Layer<L: LinAlg>`** -- Dense layer with forward, transpose (PC top-down), and backward passes
- **`LinAlg` trait** -- Backend-agnostic linear algebra interface (31 instance methods). Default implementation: `CpuLinAlg`
- **`GolubKahanSvd`** -- O(n^3) SVD via bidiagonalization, used for CCA neuron alignment

### Key Mechanisms

**Predictive Coding Inference**: Instead of a single feedforward pass, the actor runs an iterative inference loop where higher layers generate top-down predictions of lower layer states. The prediction error (surprise) between layers drives hidden state updates until convergence.

**Residual Echo (local_lambda)**: A small fraction of prediction errors from deliberation is blended into backpropagation gradients: `delta = lambda * backprop_grad + (1-lambda) * pc_error`. This couples inference and learning into a synergistic system.

**Adaptive Surprise Scheduling**: A circular buffer of recent surprise scores dynamically calibrates learning rate thresholds. Low surprise reduces LR (familiar states), high surprise boosts LR (novel states). Buffer-mediated damping protects learned representations during environment transitions.

**CCA Crossover**: GA-ready crossover operator using Canonical Correlation Analysis to align neurons functionally before blending weights, solving the permutation problem. Supports dimension mismatches, layer count differences, and residual components.

**Continuous Learning (v2.1.0)**: Surprise-driven plasticity modulation for non-stationary environments:
- *M1 Scale Range*: Configurable surprise-to-learning-rate mapping (`scale_floor`/`scale_ceil`)
- *M2 Dual-EWMA Hysteresis*: Automatic FROZEN/PLASTIC transitions via fast/slow surprise EWMAs
- *M3 Consolidation Decay*: Per-layer exponential decay (fixed M3a) or adaptive sigmoid (M3b)
- *M4 EWC Regularization*: Fisher diagonal with 3-step lifecycle (decay/accumulate/merge)
- *Bidirectional Coupling*: `actor_wakes_critic` + `critic_wakes_actor` (both default true) with EWMA k reset to prevent re-freeze
- *NaN Safety*: Guards in EwmaTracker, learn_continuous, push_surprise, push_td_error

**TD(n) N-Step Returns (v2.1.0)**: Configurable n-step temporal difference learning via `td_steps`. Buffers n transitions before bootstrapping with V(s_{t+n}). Terminal flush uses pre-computed V(s) to avoid stale-estimate bias. `td_steps=0` (default) preserves exact TD(0) behavior with zero overhead. See [docs/td_n_spec.md](docs/td_n_spec.md).

**GAE(λ) Eligibility Traces (v2.1.0)**: Output-level eligibility traces via `gae_lambda: Option<f64>`. Accumulates policy gradient direction across steps: `trace = γλ*trace + ∇log π`, then `delta = td_error * trace`. Smoothly interpolates between TD(0) (λ=0) and Monte Carlo (λ=1). Trace clipped at GRAD_CLIP=5.0. Mutually exclusive with `td_steps > 0`. Default `None` (disabled); recommended `Some(0.95)` for short episodes.

### Type Aliases

```rust
type PcActorCpu = PcActor<CpuLinAlg>;
type MlpCriticCpu = MlpCritic<CpuLinAlg>;
type PcActorCriticCpu = PcActorCritic<CpuLinAlg>;
type LayerCpu = Layer<CpuLinAlg>;
```

## Project Structure

```
PC-RL-Core/
├── src/
│   ├── linalg/
│   │   ├── mod.rs                  # LinAlg trait (31 methods, backend-agnostic)
│   │   ├── cpu.rs                  # CpuLinAlg (Vec<f64> + Matrix)
│   │   └── golub_kahan.rs          # Golub-Kahan SVD (O(n^3))
│   ├── activation.rs               # Tanh, ReLU, Sigmoid, ELU, Softsign, Linear
│   ├── error.rs                    # PcError crate-wide error type
│   ├── matrix.rs                   # Dense matrix, softmax, CCA alignment, Hungarian assignment
│   ├── layer.rs                    # Layer<L: LinAlg> with PC top-down support
│   ├── pc_actor.rs                 # PcActor<L> with inference loop, residual, crossover
│   ├── mlp_critic.rs               # MlpCritic<L> value function, crossover
│   ├── pc_actor_critic/            # PcActorCritic<L> directory submodule
│   │   ├── mod.rs                  # Agent impl: act, step, learn, crossover, CL pipeline
│   │   ├── config.rs               # PcActorCriticConfig + 31 serde defaults
│   │   ├── ewma.rs                 # EwmaTracker + PlasticityState (with NaN guard)
│   │   ├── hysteresis.rs           # HysteresisState dual-EWMA state machine
│   │   ├── fisher.rs               # FisherState<L> for EWC regularization
│   │   └── trajectory.rs           # TrajectoryStep<L> + ActivationCache<L>
│   └── serializer.rs               # JSON persistence, ClState with backward compat
├── docs/
│   ├── experiment_analysis.md      # 20 experimental phases, ~3,800 runs
│   ├── pc_actor_critic_paper.md    # DPC architecture paper
│   ├── continuous_learning_spec.md # CL v2.1.0 specification
│   └── td_n_spec.md               # TD(n) technical specification
└── Cargo.toml
```

## Research Findings

Validated through 20 experimental phases (~3,800 training runs) on Tic-Tac-Toe ([PC-TicTacToe](https://github.com/BolivarTech/PC-TicTacToe)):

- **Deliberation is the primary advantage** -- PC inference loop adds +2-3 depth levels over equivalent MLP
- **Residual echo breaks performance ceilings** -- 1% PC error blend (lambda=0.99) is statistically significant (p<0.034)
- **Depth-Lambda Scaling Law: `lambda = 1 - 10^(-(L+1))`** -- PC error must decrease exponentially with network depth
- **Lambda and training budget interact** -- ultra-low PC error needs more episodes to accumulate its regularization effect
- **Adaptive surprise eliminates catastrophic forgetting** -- buffer-mediated transition damping protects learned representations during curriculum transitions
- **Optimal buffer ratio: 0.3-0.4 x environment transition window** -- too small resonates, too large over-damps
- **Bounded activations required for PC** -- ReLU dies, ELU explodes; tanh and softsign work
- **Softsign + residual + projection cooperate** -- three mechanisms enable gradient flow in deep networks
- **Parameter efficiency** -- ~550 actor parameters matching networks 4-330x larger through iterative inference

## Documentation

| Document | Audience | Content |
|---|---|---|
| [docs/pc_actor_critic_paper.md](docs/pc_actor_critic_paper.md) | Researchers, contributors | Formal architecture spec, mathematical justification (including §1.6 on equilibrium-snapshot backprop strategy), full empirical results, lessons learned |
| [docs/pc_inference_intuitive_guide.md](docs/pc_inference_intuitive_guide.md) | New users, mental-model builders | Conversational walkthrough of PC inference + backprop with metaphors, numerical examples, FAQ — companion to the formal paper |
| [docs/experiment_analysis.md](docs/experiment_analysis.md) | Anyone reproducing results | Complete experimental methodology, statistical validation across 35 seeds × 8 configurations |
| [docs/generic_action_space_spec.md](docs/generic_action_space_spec.md) | v4.0.0 adopters | v4.0.0 generic action space spec — discrete vs continuous design, brainstorm decisions Q1-Q8 |
| [docs/continuous_learning_spec.md](docs/continuous_learning_spec.md) | CL adopters | Continuous Learning (M1-M4) detailed spec |
| [docs/td_n_spec.md](docs/td_n_spec.md) | TD(n) users | n-step temporal difference learning spec |
| [docs/gae_spec.md](docs/gae_spec.md) | GAE users | Generalized Advantage Estimation spec |
| [docs/crossover_technical_spec.md](docs/crossover_technical_spec.md) | GA evolution users | CCA-based neural network crossover for genetic algorithms |
| [docs/apply_config_spec.md](docs/apply_config_spec.md) | Runtime config mutation | Hot-reload of config fields without rebuilding the agent |
| [CHANGELOG.md](CHANGELOG.md) | Migration | Per-release breaking changes, migration tables, mitigation matrices |

## Dependencies

- `serde` / `serde_json` -- Serialization
- `rand` -- Random number generation
- `chrono` -- Timestamps

No PyTorch, TensorFlow, or any ML framework. Pure Rust from scratch.

## Testing

538 unit tests + 21 doctests:

```bash
cargo nextest run
cargo test --doc
cargo clippy --tests -- -D warnings
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
