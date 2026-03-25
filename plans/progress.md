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
