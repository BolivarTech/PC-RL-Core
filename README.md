# PC-TicTacToe

[![CI](https://github.com/BolivarTech/PC-TicTacToe/actions/workflows/ci.yml/badge.svg)](https://github.com/BolivarTech/PC-TicTacToe/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A Predictive Coding Actor-Critic reinforcement learning agent that learns to play Tic-Tac-Toe from scratch, implemented entirely in Rust with zero ML framework dependencies.

The actor uses iterative top-down/bottom-up **predictive coding inference loops** instead of standard feedforward passes, combined with a standard MLP critic for value estimation. The agent trains via REINFORCE with baseline against a minimax opponent with curriculum learning.

## Results

With only **27 hidden neurons** (~500 parameters), the agent reaches **minimax depth 9** (near-perfect play) with a hybrid PC-backprop learning rule (`local_lambda=0.99`):

```
  >> Curriculum advanced: depth 7 -> 8
[ep  14000/50000] win=0.0% loss=12.5% draw=87.5%  | depth=8
  >> Curriculum advanced: depth 8 -> 9
[ep  15000/50000] win=0.0% loss=0.0%  draw=100.0% | depth=9
[ep  50000/50000] win=0.0% loss=0.6%  draw=99.4%  | depth=9
```

At depth 9, the agent achieves **~99% draws** against a near-perfect minimax opponent -- essentially optimal play for Tic-Tac-Toe.

### Statistical Validation (N=35 seeds)

| Lambda | Mean Depth | Depth>=8 | Depth=9 | p-value vs baseline |
|--------|-----------|----------|---------|-------------------|
| **0.99 (hybrid)** | **7.57** | **37%** | **20%** | **0.034\*** |
| 1.00 (backprop) | 7.14 | 26% | 9% | baseline |

lambda=0.99 is the **only statistically significant improvement** (p < 0.05) over pure backprop across all tested values. See the [full experiment analysis](docs/experiment_analysis.md) for details.

### Architecture Comparison

| Configuration | Depth Reached | Performance |
|---------------|:---:|-------------|
| Pure MLP (no PC), 18 neurons | 6 | Draws as P1 |
| PC inference, 18 neurons | 7 | Draws as P1 |
| PC inference, 27 neurons, lr=0.01 | 7 | Wins as P1 |
| PC inference, 27 neurons, lr=0.005 | 8 | Draws as P1 vs near-perfect |
| **PC + hybrid lambda=0.99** | **9** | **~99% draws vs near-perfect** |

Predictive coding inference consistently adds **+1 depth level** over the equivalent MLP architecture. The hybrid learning rule adds another level on top.

### Parameter Efficiency

The PC actor achieves near-optimal play with only **~550 parameters** -- 4-330x smaller than typical published architectures for the same task (which range from ~2,700 to ~183,000 parameters). The PC inference loop trades compute for parameters: 5 iterative inference steps extract more representational capacity per parameter than a single feedforward pass through a larger network.

## Architecture

```
Input (9) ──> [Hidden 27, Tanh] ──> [Output 9, Linear] ──> Softmax ──> Action
                  ^    |
                  |    v
              PC Inference Loop (top-down / bottom-up)
                  |
                  v
            Latent Concat (27)
                  |
         [Board State (9)] ++ [Latent (27)] = Critic Input (36)
                  |
                  v
         [Critic Hidden 36, Tanh] ──> V(s)
```

**Predictive Coding Loop**: Instead of a single feedforward pass, the actor runs an iterative inference loop where higher layers generate top-down predictions of lower layer states. The prediction error (surprise) between layers drives hidden state updates. This process converges to a stable internal representation before action selection.

**Curriculum Learning**: The agent starts against a weak opponent (minimax depth 1) and advances when it achieves >95% non-loss rate over a 1000-game window. Metrics reset on each advancement to prevent cascading.

## Project Structure

```
PC-TicTacToe/
├── pc_core/                    # Reusable RL library (publishable)
│   └── src/
│       ├── activation.rs       # Tanh, ReLU, Sigmoid, ELU, Linear
│       ├── error.rs            # PcError crate-wide error type
│       ├── matrix.rs           # Dense matrix ops, softmax, sampling
│       ├── layer.rs            # Dense layer with PC top-down support
│       ├── pc_actor.rs         # PC actor with inference loop
│       ├── mlp_critic.rs       # MLP value function
│       ├── pc_actor_critic.rs  # Integrated agent
│       └── serializer.rs       # JSON model persistence
├── pc_tictactoe/               # Game binary
│   ├── config.toml             # Training configuration
│   └── src/
│       ├── env/                # TicTacToe + Minimax opponent
│       ├── training/           # Episodic + continuous trainers
│       ├── ui/                 # CLI interface
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
| `hidden_layers` | `[27, tanh]` | Single hidden layer, 27 neurons |
| `gamma` | `0.99` | Discount factor |
| `entropy_coeff` | `0.0` | No entropy regularization |
| `local_lambda` | `0.99` | Hybrid PC-backprop blend (1.0=backprop, 0.0=local PC) |

## Key Findings

- **Hybrid lambda=0.99 breaks the depth ceiling** -- 1% PC error as regularizer enables depth 9 (p=0.034, N=35 seeds)
- **Output activation must be Linear** -- Tanh bounds logits to [-1,1], making softmax nearly uniform and preventing any policy learning
- **PC inference adds measurable value** -- Consistently +1 minimax depth level vs equivalent MLP
- **Bounded activations required for PC** -- ReLU dies, ELU explodes; tanh's self-regulation is essential
- **Single hidden layer outperforms deeper networks** -- 2-layer architectures suffer vanishing gradients through double Tanh
- **27 neurons is the sweet spot** -- 18 too small, 32 no improvement
- **Entropy regularization hurts** -- Destabilizes learned defensive play in this architecture

**Next frontier**: `local_lambda` is a hyperparameter with an ultra-narrow sweet spot (only 0.99 works out of 6 values tested) that likely interacts with alpha, lr, and topology. A genetic algorithm co-evolving all hyperparameters -- chromosome `[hidden_size, alpha, lr, lambda, ...]` with fitness = max depth -- could discover optimal configurations that grid search misses.

For the complete experimental methodology and statistical analysis, see [docs/experiment_analysis.md](docs/experiment_analysis.md). For the full architecture description, lessons learned, and applicability to other PC projects, see [docs/pc_actor_critic_paper.md](docs/pc_actor_critic_paper.md).

## Dependencies

The `pc_core` library uses only:
- `serde` / `serde_json` -- Serialization
- `rand` -- Random number generation
- `chrono` -- Timestamps

The `pc_tictactoe` binary adds:
- `toml` -- Configuration parsing
- `clap` -- CLI argument parsing
- `ctrlc` -- Graceful shutdown

No PyTorch, TensorFlow, or any ML framework. Pure Rust from scratch.

## Testing

268 tests covering all modules:

```bash
# Run all tests
cargo nextest run --workspace

# Run specific crate
cargo nextest run -p pc_core
cargo nextest run -p pc_tictactoe

# Lint
cargo clippy --workspace --tests -- -D warnings
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
