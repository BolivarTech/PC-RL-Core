# PC-RL-Core

[![CI](https://github.com/BolivarTech/PC-RL-Core/actions/workflows/ci.yml/badge.svg)](https://github.com/BolivarTech/PC-RL-Core/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A **Deliberative Predictive Coding (DPC)** reinforcement learning agent that learns to play Tic-Tac-Toe from scratch, implemented entirely in Rust with zero ML framework dependencies.

The actor **deliberates before acting** by running an iterative free energy minimization loop (predictive coding inference), and a **residual echo of that deliberation** feeds back into weight updates as a structured micro-regularizer. These two mechanisms form a coupled system: deliberation generates the signal, the signal improves learning, and better learning improves future deliberation. The agent trains via REINFORCE with baseline against minimax opponents with curriculum learning.

The core library (`pc-rl-core`) is **backend-agnostic**: all linear algebra operations are abstracted behind a `LinAlg` trait, enabling future GPU backends (wgpu, CUDA) without changing the RL logic.

## Results

The agent reaches **minimax depth 9** (near-perfect play) in **23% of seeds** (functional models) with a 3-layer [27,27,18] architecture, ultra-low PC error (`local_lambda=0.9999`), and adaptive surprise scheduling:

At depth 9, the agent achieves **~99% draws** against a near-perfect minimax opponent -- essentially optimal play for Tic-Tac-Toe.

### Statistical Validation (N=35 seeds, 20 phases, ~3,800 runs)

| Topology | Lambda | Activation | Residual | Surprise | Episodes | Mean | D=9 (functional) |
|----------|--------|------------|----------|----------|----------|------|-------------------|
| **[27,27,18]** | **0.9999** | **softsign** | **yes (proj)** | **adaptive buf=400** | **200k** | **7.63** | **23%** |
| [27,27,18] | 0.9999 | softsign | yes (proj) | fixed | 200k | 7.69 | 14% |
| 1×27 | 0.99 | tanh | no | fixed | 50k | 7.94 | 37% |
| 1×27 | 0.99 | softsign | no | fixed | 50k | 7.89 | 31% |

**Phase 20 discovery**: The baseline 40% D=9 rate was misleading -- 64% of those models had 100% loss rate (collapsed immediately after advancing). Adaptive surprise with buffer=400 eliminates most collapses and increases functional D=9 from 14% to 23%, producing 3 perfect-play models (including one with theoretically optimal 50W/0L/50D).

See the [full experiment analysis](docs/experiment_analysis.md) for details across all 20 experimental phases.

### Parameter Efficiency

The PC actor achieves near-optimal play with only **~550-1,000 parameters** -- 3-330x smaller than typical published architectures for the same task (which range from ~2,700 to ~183,000 parameters). The PC inference loop trades compute for parameters: 5 iterative inference steps extract more representational capacity per parameter than a single feedforward pass through a larger network.

## Architecture

```
Input (9) ──> [H1 27, Softsign] ──> [H2 27, Softsign] ──> [H3 18, Softsign] ──> [Output 9, Linear] ──> Softmax ──> Action
                  ^    |     ↕ skip        ↕ skip+proj
                  |    v
              PC Inference Loop (top-down / bottom-up)
                  |
                  v
            Latent Concat (27+27+18 = 72)
                  |
         [Board State (9)] ++ [Latent (72)] = Critic Input (81)
                  |
                  v
         [Critic Hidden 36, Softsign] ──> V(s)
```

All core structs are generic over `L: LinAlg` (default `CpuLinAlg`), enabling future GPU backends. The library is **GA-ready** with CCA-based crossover operators for evolving network populations.

**Predictive Coding Loop**: Instead of a single feedforward pass, the actor runs an iterative inference loop where higher layers generate top-down predictions of lower layer states. The prediction error (surprise) between layers drives hidden state updates. This process converges to a stable internal representation before action selection.

**Curriculum Learning**: The agent starts against a weak opponent (minimax depth 1) and advances when it achieves >95% non-loss rate over a 1000-game window. Metrics reset on each advancement to prevent cascading.

## Project Structure

```
PC-RL-Core/
├── pc-rl-core/                    # Reusable RL library (v1.2.1)
│   └── src/
│       ├── linalg/
│       │   ├── mod.rs          # LinAlg trait (32 methods, backend-agnostic)
│       │   └── cpu.rs          # CpuLinAlg (Vec<f64> + Matrix, Golub-Kahan SVD)
│       ├── activation.rs       # Tanh, ReLU, Sigmoid, ELU, Softsign, Linear
│       ├── error.rs            # PcError crate-wide error type
│       ├── matrix.rs           # Dense matrix, softmax, CCA alignment, Hungarian assignment
│       ├── layer.rs            # Layer<L: LinAlg> with PC top-down support
│       ├── pc_actor.rs         # PcActor<L> with inference loop, residual, crossover
│       ├── mlp_critic.rs       # MlpCritic<L> value function, crossover
│       ├── pc_actor_critic.rs  # PcActorCritic<L> agent, ActivationCache, crossover
│       └── serializer.rs       # JSON persistence (CPU concrete bridge)
├── pc_tictactoe/               # Game binary
│   ├── config.toml             # Training configuration
│   └── src/
│       ├── env/                # TicTacToe + Minimax opponent
│       ├── training/           # Episodic + continuous + experiment runners
│       ├── ui/                 # CLI: train, play, evaluate, experiment, seed-test, init
│       └── utils/              # Config, logger, metrics
```

## Quick Start

```bash
# Build
cargo build --release

# Train (uses pc_tictactoe/config.toml)
cargo run --release -- train -c pc_tictactoe/config.toml

# Play against the trained agent
cargo run --release -- play --model model.json

# Play as first player
cargo run --release -- play --model model.json --first

# Evaluate against minimax
cargo run --release -- evaluate --model model.json --games 100 --depth 9
```

## Configuration

All hyperparameters are configured via TOML. See [`pc_tictactoe/config.toml`](pc_tictactoe/config.toml) for the full configuration with the optimal parameters.

Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `output_activation` | `linear` | Unbounded logits for softmax (tanh prevents learning) |
| `alpha` | `0.03` | PC inference loop update rate |
| `lr_weights` | `0.005` | Actor learning rate |
| `hidden_layers` | `[27,27,18] softsign` | 3-layer with dimensionality reduction |
| `residual` | `true` | Skip connections with ReZero + projection |
| `rezero_init` | `0.1` | ReZero initial scaling factor |
| `gamma` | `0.99` | Discount factor |
| `entropy_coeff` | `0.0` | No entropy regularization |
| `local_lambda` | `0.9999` | Ultra-low PC error for deep networks (0.99 for 1-layer) |
| `adaptive_surprise` | `true` | Dynamic surprise thresholds from recent history |
| `surprise_buffer_size` | `400` | Circular buffer size (~0.4x curriculum window) |

## Key Findings

- **Adaptive surprise eliminates D=9 collapse** -- 64% of fixed-threshold D=9 models were collapsed (100% loss); adaptive buffer=400 raises functional D=9 from 14% to 23%
- **Buffer-mediated transition damping** -- circular buffer of recent surprise scores creates a decaying LR envelope during curriculum transitions, protecting learned representations
- **Optimal buffer ratio: 0.3-0.4 x curriculum window** -- buffer=400 with window=1000 is the sweet spot; too small (100) resonates, too large (500) over-damps
- **3-layer [27,27,18] with lambda=0.9999** -- best deep configuration with adaptive surprise
- **Depth-Lambda Scaling Law: `lambda = 1 - 10^(-(L+1))`** -- PC error must decrease exponentially with network depth
- **Lambda and training budget interact** -- lambda=0.9999 needs 200k episodes (6% D=9 at 50k, 40% nominal at 200k)
- **Deliberation is the primary advantage** -- PC inference loop adds +2-3 depth levels over MLP
- **Softsign + residual + projection cooperate** -- three mechanisms enable gradient flow in deep networks
- **Output activation must be Linear** -- Tanh bounds logits to [-1,1], preventing policy learning
- **Bounded activations required for PC** -- ReLU dies, ELU explodes; tanh and softsign work
- **Golub-Kahan SVD** -- O(n^3) bidiagonalization replacing Jacobi O(n^4); practical for layers up to n~1024
- **Backend-agnostic architecture** -- `LinAlg` trait enables CPU/GPU swap with zero logic changes
- **GA-ready crossover** -- CCA neuron alignment solves the permutation problem; Hungarian optimal assignment; supports topology mutation (dimension/layer count changes)

Validated through 20 experimental phases, ~3,800 training runs across multiple architectural configurations.

For the complete experimental methodology and statistical analysis, see [docs/experiment_analysis.md](docs/experiment_analysis.md). For the full architecture description, lessons learned, and applicability to other PC projects, see [docs/pc_actor_critic_paper.md](docs/pc_actor_critic_paper.md).

## Dependencies

The `pc-rl-core` library uses only:
- `serde` / `serde_json` -- Serialization
- `rand` -- Random number generation
- `chrono` -- Timestamps

The `pc_tictactoe` binary adds:
- `toml` -- Configuration parsing
- `clap` -- CLI argument parsing
- `ctrlc` -- Graceful shutdown

No PyTorch, TensorFlow, or any ML framework. Pure Rust from scratch.

## Testing

472 unit tests + 20 doctests covering all modules:

```bash
# Run all tests
cargo nextest run --workspace

# Run specific crate
cargo nextest run -p pc-rl-core
cargo nextest run -p pc_tictactoe

# Lint
cargo clippy --workspace --tests -- -D warnings
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
