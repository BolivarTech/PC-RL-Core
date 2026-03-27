# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive Coding Actor-Critic (PC-Actor-Critic) reinforcement learning agent applied to Tic-Tac-Toe. Implements a novel architecture where the actor uses iterative top-down/bottom-up predictive coding inference loops instead of standard feedforward passes, combined with a standard MLP critic for value estimation.

## Build & Test Commands

```bash
# Build
cargo build

# Run tests (standard)
cargo nextest run

# Run tests with TDD Guard (used by hooks)
bash ./run-tests.sh

# Run a single test
cargo nextest run test_name

# Run tests for a specific module
cargo nextest run -p pc_core
cargo nextest run -p pc_tictactoe

# Check without building
cargo check

# Lint
cargo clippy -- -D warnings
```

## Architecture

The project is planned as a **Cargo workspace** with two crates:

- **`pc_core`** — Library crate (publishable). Contains the PC-Actor-Critic framework:
  - `activation.rs` — Activation functions enum (Tanh, Relu, Sigmoid, Elu, Linear) with apply/derivative
  - `error.rs` — Crate-wide `PcError` enum (ConfigValidation, DimensionMismatch, Serialization, Io)
  - `matrix.rs` — Dense matrix ops, softmax, argmax, RMS error, weight/gradient clipping (WEIGHT_CLIP=5.0, GRAD_CLIP=5.0)
  - `layer.rs` — Single dense layer with forward, transpose_forward (PC top-down), and backward
  - `pc_actor.rs` — PC network with variable hidden topology, iterative inference loop, surprise scoring
  - `mlp_critic.rs` — Standard MLP value function, MSE loss backprop
  - `pc_actor_critic.rs` — Integrated agent: act, learn (episode/continuous), surprise-based scheduling, save/load
  - `serializer.rs` — JSON weight persistence

- **`pc_tictactoe`** — Binary crate. Game environment and training:
  - `env/tictactoe.rs` — Game rules, board state, valid actions
  - `env/minimax.rs` — Minimax baseline opponent
  - `training/trainer.rs` — Episode-based training loop
  - `training/continuous.rs` — Continuous learning with surprise scheduling
  - `ui/cli.rs` — CLI interface (clap)
  - `utils/` — Config (TOML), logger, metrics

### Key Concepts

- **PC Inference**: Bottom-up feedforward → top-down prediction loop until convergence (alpha, tol, max_steps)
- **Surprise Score**: RMS prediction error across layers; drives learning rate scaling (0.1× when low, 2.0× when high)
- **Latent Concat**: All hidden layer activations concatenated → fed to critic alongside raw state
- **SelectionMode**: Training (stochastic sampling) vs Play (deterministic argmax)
- **local_lambda**: Blend factor for hidden layer weight updates. `delta = λ * backprop_grad + (1-λ) * pc_prediction_error`. Default 1.0 (pure backprop). Output layer always uses backprop.

## TDD Setup

This project uses TDD Guard hooks (configured in `.claude/settings.json`):
- **PreToolUse**: `tdd-guard` validates Red-Green-Refactor cycle before file writes
- **PostToolUse**: `bash ./run-tests.sh` runs tests after every file write
- **Quick commands**: `tdd-guard on` / `tdd-guard off` to toggle enforcement

## Implementation Order

Follow the plan in `plans/pc_tictactoe_plan_rust_tdd_v2_1.md`. Modules must be implemented in dependency order:
1. activation → 2. matrix → 3. layer → 4. pc_actor → 5. mlp_critic → 6. pc_actor_critic → 7. serializer → 8-15. pc_tictactoe modules

## Dependencies

Only `serde`, `serde_json`, `rand`, `chrono` in pc_core. The binary adds `toml`, `clap`, `ctrlc`.

## Training Results

Best configuration found through systematic experimentation:

### Best Config (depth=8 baseline, depth=9 with local_lambda=0.99)

```toml
[agent.actor]
output_activation = "linear"   # Critical: tanh crushes policy gradients
alpha = 0.03                   # PC loop active but gentle
max_steps = 5
lr_weights = 0.005             # Matched with critic lr for stability
hidden_layers = [{ size = 27, activation = "tanh" }]
local_lambda = 1.0             # 1.0=backprop, 0.99=best hybrid (test branch)

[agent.critic]
input_size = 36                # 9 board + 27 latent
lr = 0.005
hidden_layers = [{ size = 36, activation = "tanh" }]

[agent]
gamma = 0.99
entropy_coeff = 0.0            # Entropy hurts this architecture
adaptive_surprise = false
```

### Experiment Comparison

| Config | Depth | Pattern at Final Depth | Notes |
|--------|-------|----------------------|-------|
| MLP α=0, 18h, lr=0.01 | 6 | 0% win / 50% loss / 50% draw | Pure MLP baseline |
| PC α=0.03, 18h, lr=0.01 | 7 | 0% win / 50% loss / 50% draw | PC adds 1 depth level |
| PC α=0.03, 27h, lr=0.01 | 7 | 50% win / 50% loss / 0% draw | Larger net wins as P1 |
| PC α=0.03, 27→18h, lr=0.005 | 7 | 0% win / 50% loss / 50% draw | 2 layers hurts: 8x slower, -1 depth |
| PC α=0.01, 32h, lr=0.005 | 8 | 0% win / 50% loss / 50% draw | Lower alpha, same ceiling |
| **PC α=0.03, 27h, lr=0.005** | **8** | **0% win / 50% loss / 50% draw** | **Baseline — reproducible (seed=42)** |
| **PC λ=0.99, 27h, lr=0.005** | **9** | **0% win / ~1% loss / ~99% draw** | **Best — seed=42 only (test branch)** |

### local_lambda Experiments (test branch)

Hybrid blend: `delta = λ * backprop_grad + (1-λ) * pc_prediction_error`

| Topology | lambda | seed=42 | seed=123 |
|----------|--------|---------|----------|
| 27h | 1.0 | depth 8 | depth 7 |
| 27h | 0.99 | **depth 9** | depth 7 |
| 27h | 0.975 | depth 7 | — |
| 27h | 0.75 | depth 7 | — |
| 27h | 0.50 | depth 7 | — |
| 27h | 0.25 | depth 7 | — |
| 27h | 0.0 | depth 7 | — |
| 27→18h | 0.75 | depth 7 | — |
| 27→18h | 0.50 | depth 7 | — |
| 27→18h | 0.25 | depth 6 | — |
| 27→18h | 0.0 | depth 6 | — |

### Activation Experiments (test branch, seed=42, 27h)

| Activation | Depth | Notes |
|------------|-------|-------|
| tanh | 8 | Baseline. Bounded output stabilizes PC loop |
| relu | 4 | Dying neurons: PC loop pushes activations negative permanently |
| elu | 6 | Unbounded positives: PC loop creates unstable feedback → policy collapse |

### Key Findings

- `output_activation` MUST be `"linear"` — tanh bounds logits to [-1,1], making softmax nearly uniform and preventing policy learning
- `entropy_coeff = 0.0` works best — any entropy destabilizes learned defensive play
- PC inference (α=0.03) outperforms pure MLP (α=0) by one depth level
- 27 hidden neurons is the sweet spot — 18 too small, 32 no improvement
- Lower actor lr (0.005 vs 0.01) trades speed for stability — slower to scale but reaches higher depth
- 2-layer networks always worse — vanishing gradients through double tanh confirmed across all lambdas
- Bounded activations (tanh) required — PC loop destabilizes unbounded activations (relu, elu)
- Curriculum must reset metrics on advancement to prevent cascading depth jumps
- Advancement criterion uses non-loss rate (win+draw > 95%), not just win rate
- **local_lambda=0.99 broke depth 8 ceiling** → depth 9 with ~99% draws (seed=42). The 1% PC error acts as a subtle regularizer that prevents weight stagnation at local minima
- The sweet spot is ultra-narrow: λ=0.975 fails (depth 7), λ=0.99 succeeds (depth 9). A 1.5% difference = 2 depth levels
- **Seed-dependent**: baseline depth=8 only confirmed with seed=42. With seed=123 both λ=1.0 and λ=0.99 reach depth 7. Multi-seed evaluation needed
- λ=0.99 reaches intermediate depths faster even when not reaching higher depths (3.4× faster with seed=123)
- Training reaches depth=8 in ~11,000 episodes (baseline), depth=9 in ~14,500 episodes (λ=0.99)

### Statistical Validation (N=35 random seeds, 210 runs)

| Lambda | Mean Depth | StdDev | Min | Max | D>=8 | D=9 | p-value vs 1.0 |
|--------|-----------|--------|-----|-----|------|-----|----------------|
| 0.95 | 7.00 | 1.19 | 4 | 9 | 23% | 11% | 0.564 |
| 0.96 | 6.66 | 1.33 | 4 | 9 | 14% | 9% | 0.072 |
| 0.97 | 6.94 | 1.16 | 4 | 9 | 14% | 11% | 0.413 |
| 0.98 | 7.00 | 0.87 | 4 | 9 | 11% | 9% | 0.490 |
| **0.99** | **7.57** | **0.81** | **7** | **9** | **37%** | **20%** | **0.034\*** |
| 1.00 | 7.14 | 0.85 | 6 | 9 | 26% | 9% | baseline |

**λ=0.99 is the only statistically significant improvement** (p < 0.05). It has the highest mean depth, doubles the rate of depth=9, and never drops below depth 7. All other lambda values perform equal or worse than baseline.

### Seed Dependency Analysis

Seed-dependence is expected: different seeds place the optimizer in different basins of attraction in the loss landscape. The λ=0.99 perturbation (1% PC error) increases the probability of escaping to deeper basins but cannot guarantee it for all starting points. This explains both the statistical improvement (more seeds reach depth 9) and the remaining variance.

### Evolutionary Optimization Potential

`local_lambda` has an ultra-narrow sweet spot (only 0.99 of 6 values tested) that likely interacts with alpha, lr, and topology. A genetic algorithm co-evolving all hyperparameters — chromosome `[hidden_size, alpha, lr, lambda, ...]` with fitness = max depth — could discover optimal configurations that grid search misses. Each GA individual has its own weight initialization, so lambda evolves to match its particular basin of attraction.

Full analysis: [docs/experiment_analysis.md](docs/experiment_analysis.md)
