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
    td_steps: 4,   // TD(4) n-step returns (0 = TD(0) default)
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

For the complete experimental methodology and statistical analysis, see [docs/experiment_analysis.md](docs/experiment_analysis.md). For the full architecture description and lessons learned, see [docs/pc_actor_critic_paper.md](docs/pc_actor_critic_paper.md).

## Dependencies

- `serde` / `serde_json` -- Serialization
- `rand` -- Random number generation
- `chrono` -- Timestamps

No PyTorch, TensorFlow, or any ML framework. Pure Rust from scratch.

## Testing

524 unit tests + 21 doctests:

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
