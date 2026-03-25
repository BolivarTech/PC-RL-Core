# Ralph Progress Log

Started: 2026-03-24
Project: PC-TicTacToe (Predictive Coding Actor-Critic for Tic-Tac-Toe)

## Codebase Patterns

- Cargo workspace: pc_core (lib) + pc_tictactoe (bin)
- TDD enforced via tdd-guard hooks
- Tests live inside impl files under #[cfg(test)]
- Module dependency order defined in planning/sections/index.md
- Verification: cargo nextest run --workspace

## Key Files

- plans/prd.json — Task tracking (14 features, F-001 to F-014)
- planning/sections/index.md — Dependency graph and execution order
- planning/claude-plan-tdd.md — All test definitions per module
- CLAUDE.md — Project architecture and conventions
- CLAUDE.local.md — Code standards enforcement rules

---

## Iteration 1 — F-001: Workspace Setup (2026-03-25)

**Status:** PASSED

Workspace structure was already in place from prior setup:
- Root Cargo.toml is a workspace manifest with pc_core and pc_tictactoe members
- pc_core: lib crate with stub files (activation, matrix, layer, pc_actor, mlp_critic, pc_actor_critic, serializer)
- pc_tictactoe: bin crate with subdirectory stubs (env, training, ui, utils)
- All dependencies configured (serde, serde_json, rand, chrono + toml, clap, ctrlc for bin)
- `cargo build --workspace`, `cargo nextest run --workspace`, and `cargo clippy --workspace -- -D warnings` all pass clean

---

## Iteration 2 — F-002: Activation Functions (2026-03-25)

**Status:** PASSED

- Implemented `Activation` enum (Tanh, Relu, Sigmoid, Linear) with `apply`, `derivative`, `apply_vec`
- 24 tests all passing: apply tests, derivative tests, apply_vec tests, robustness, serde roundtrip
- Note: sigmoid(100.0) rounds to exactly 1.0 in f64 due to precision limits; test uses x=30.0 instead
- Added `pub mod activation;` to lib.rs

---

## Iteration 3 — F-003: Matrix Operations (2026-03-25)

**Status:** PASSED

- Implemented `Matrix` struct with `zeros`, `xavier`, `get/set`, `transpose`, `mul_vec`, `outer`, `scale_add`
- Implemented free functions: `softmax_masked`, `argmax_masked`, `rms_error`, `sample_from_probs`, `clip_vec`, `vec_sub`, `vec_add`, `vec_scale`
- Constants: `WEIGHT_CLIP=5.0`, `GRAD_CLIP=1.0`
- 45 matrix tests all passing (69 total workspace tests)
- Note: `SmallRng` requires `small_rng` feature; used `StdRng` instead
- Clippy required iterator-based `mul_vec` instead of range loop indexing
- Added `pub mod matrix;` to lib.rs

---

## Iteration 4 — F-004: Dense Layer (2026-03-25)

**Status:** PASSED

- Implemented `Layer` struct with `new`, `forward`, `transpose_forward`, `backward` methods
- Implemented `LayerDef` struct for topology configuration
- `forward`: W*input + bias → activation
- `transpose_forward`: W^T*input → custom activation (no bias, used for PC top-down)
- `backward`: gradient clipping (GRAD_CLIP), weight update via scale_add (WEIGHT_CLIP clamping), bias update with clamping
- 18 layer tests all passing (87 total workspace tests)
- Added `pub mod layer;` to lib.rs

---

## Iteration 5 — F-005: PC Actor (2026-03-25)

**Status:** PASSED

- Implemented `PcActor` with `PcActorConfig`, `InferResult`, `SelectionMode`
- `new`: builds hidden layers + output layer from config
- `latent_size`: sum of hidden layer sizes
- `infer`: PC inference loop with synchronous (snapshot) and in-place modes; top-down predictions via `transpose_forward`, error-driven state updates, convergence check
- `select_action`: temperature-scaled softmax, argmax (Play) or sample (Training)
- `update_weights`: backprop through all layers from output to input
- Key learnings:
  - alpha=0 needs explicit guard to prevent false convergence (RMS error can be below tol with zero updates)
  - Synchronous vs in-place difference shows in lower hidden layers' latent_concat, not in y_conv (output is computed from top hidden layer which gets identical treatment)
  - Zero input produces zero gradients for first layer — tests need non-zero input for weight change assertions
- 22 tests passing (109 total workspace tests)
- Added `pub mod pc_actor;` to lib.rs

## Iteration 6 — F-006: MLP Critic (2026-03-25)

**Status:** PASSED

- Implemented `MlpCritic` with `MlpCriticConfig`, `MlpCriticWeights`
- `new`: builds hidden layers + output layer (1 neuron) from config
- `forward`: sequential forward through all layers, returns scalar `output[0]`
- `update`: forward pass storing intermediates, MSE loss, output gradient `-2*(target-predicted)`, backprop through layers in reverse with `surprise_scale=1.0`
- `to_weights`/`from_weights`: serialization support without RNG
- 10 tests passing (119 total workspace tests)
- Added `pub mod mlp_critic;` to lib.rs

---

## Iteration 7 — F-007: PC Actor-Critic Agent (2026-03-25)

**Status:** PASSED

- Implemented `PcActorCritic` with `PcActorCriticConfig`, `TrajectoryStep`
- `new`: builds actor + critic from config with seeded StdRng
- `act`: delegates to actor.infer + select_action
- `learn`: REINFORCE with baseline — discounted returns, advantage = G[t] - V(s), policy gradient + entropy regularization, re-infers to get hidden_states for backprop
- `learn_continuous`: single-step TD(0) — V(s)/V(s'), target = reward + gamma*V(s') or just reward if terminal
- `surprise_scale`: piecewise linear interpolation between 0.1 and 2.0, with adaptive override using buffer mean/std
- `push_surprise`: circular buffer (max 100) for adaptive thresholds
- `save`/`load`: stubs returning Err until serializer (F-008) is implemented
- Key learnings:
  - Re-inference needed in `learn` to get full `hidden_states` for `update_weights` (TrajectoryStep only stores latent_concat, not per-layer states)
  - Adaptive surprise with zero std makes low == high; test needs varied data for meaningful assertions
  - `rand::Rng` import not needed when only using concrete StdRng type
- 18 tests passing (137 total workspace tests)
- Added `pub mod pc_actor_critic;` to lib.rs

---

## Iteration 8 — F-008: Serializer (2026-03-25)

**Status:** PASSED

- Implemented `PcError` enum with `DimensionMismatch`, `ConfigValidation`, `Serialization`, `Io` variants
- Implemented `Display`, `Error`, `From<std::io::Error>`, `From<serde_json::Error>` for `PcError`
- Implemented `AgentMetadata`, `TrainingMetrics`, `PcActorWeights`, `SerializerCriticWeights`, `SaveFile` structs
- Implemented `save_agent`, `load_agent`, `checkpoint_filename`, `save_checkpoint` functions
- Added `from_parts` constructor to `PcActorCritic` for reconstructing from loaded components
- Key learnings:
  - `serde_json` f64 roundtrip can have last-digit precision differences (~1e-16); tests use approximate equality (1e-15 tolerance)
  - `PcActor` and `MlpCritic` don't implement `Debug`, so `Result::unwrap_err()` won't compile; use `result.err().unwrap()` + `matches!` macro instead
  - `pub(crate)` access on actor/critic layers works since serializer is in the same crate
- 10 serializer tests + 137 prior = 147 total workspace tests passing
- Added `pub mod serializer;` to lib.rs

---

## Iteration 9 — F-009: Library Exports (2026-03-25)

**Status:** PASSED (CHECKPOINT 1)

- Added crate-level `//!` documentation describing the PC-Actor-Critic framework
- Added `pub use` re-exports for all public types, constants, and functions
- Fixed broken intra-doc links in `layer.rs` (`PcActor`/`MlpCritic` → `crate::PcActor`/`crate::MlpCritic`)
- All 3 acceptance criteria met: 147 tests pass, `cargo doc --no-deps` clean, clippy clean

---

## Iteration 10 — F-010: Utilities (Logger, Config, Metrics) (2026-03-25)

**Status:** PASSED (CHECKPOINT 2)

- Implemented `config.rs`: `AppConfig` with TOML deserialization, `validate()` checking critic_input == actor_input + latent_sum, `to_agent_config()` converting to pc_core types, `apply_cli_overrides()`, `load()` with default fallback
- Implemented `logger.rs`: `Logger` with level filtering, file + CSV output, log rotation with max_backups, `SharedLogger` for thread safety, progress bar
- Implemented `metrics.rs`: `Metrics` sliding window with `GameOutcome` enum, win/loss/draw rates, surprise average
- Updated `main.rs` to declare `pub mod env; pub mod training; pub mod ui; pub mod utils;`
- 23 new tests (170 total workspace), clippy clean
- Key: clippy requires `#[derive(Default)]` instead of manual impl when all fields have Default

---
