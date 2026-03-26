# PC-TicTacToe

A Predictive Coding Actor-Critic reinforcement learning agent that learns to play Tic-Tac-Toe from scratch, implemented entirely in Rust with zero ML framework dependencies.

The actor uses iterative top-down/bottom-up **predictive coding inference loops** instead of standard feedforward passes, combined with a standard MLP critic for value estimation. The agent trains via REINFORCE with baseline against a minimax opponent with curriculum learning.

## Results

With only **27 hidden neurons** (~500 parameters), the agent reaches **minimax depth 8** (near-perfect play) in ~11,000 training episodes:

```
[ep   1000/50000] win=78.3% loss=20.4% draw=1.3%  | depth=1
  >> Curriculum advanced: depth 1 -> 2
[ep   2000/50000] win=0.1% loss=13.2% draw=86.7%  | depth=2
  >> Curriculum advanced: depth 2 -> 3
  >> ...
  >> Curriculum advanced: depth 7 -> 8
[ep  16000/50000] win=0.0% loss=50.0% draw=50.0%  | depth=8
```

At depth 8, the agent **never loses as first player** (50% draw) against a minimax opponent that searches 8 moves ahead. The 50% loss rate corresponds exclusively to games played as second player against near-perfect play.

### Experiment Summary

| Configuration | Depth Reached | Performance |
|---------------|:---:|-------------|
| Pure MLP (no PC), 18 neurons | 6 | Draws as P1 |
| PC inference, 18 neurons | 7 | Draws as P1 |
| PC inference, 27 neurons, lr=0.01 | 7 | **Wins as P1** |
| **PC inference, 27 neurons, lr=0.005** | **8** | **Draws as P1 vs near-perfect** |

Predictive coding inference consistently adds **+1 depth level** over the equivalent MLP architecture.

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
│       ├── activation.rs       # Tanh, ReLU, Sigmoid, Linear
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

## Key Findings

- **Output activation must be Linear** -- Tanh bounds logits to [-1,1], making softmax nearly uniform and preventing any policy learning
- **PC inference adds measurable value** -- Consistently +1 minimax depth level vs equivalent MLP
- **Single hidden layer outperforms deeper networks** -- 2-layer architectures suffer vanishing gradients through double Tanh
- **27 neurons is the sweet spot** -- 18 too small, 32 no improvement
- **Entropy regularization hurts** -- Destabilizes learned defensive play in this architecture
- **Lower learning rate reaches higher depth** -- 0.005 vs 0.01 trades speed for stability

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

237 tests covering all modules:

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

MIT
