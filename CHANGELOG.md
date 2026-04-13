# Changelog

## [2.1.0] - 2026-04-11

### Added
- **Continuous Learning (CL)** via intrinsic surprise modulation:
  - M1: Configurable surprise-to-learning-rate scale range (`scale_floor`/`scale_ceil`)
  - M2: Dual-EWMA hysteresis with automatic FROZEN/PLASTIC transitions
  - M3a: Fixed exponential layer-wise consolidation decay
  - M3b: Adaptive sigmoid layer-wise decay driven by per-layer prediction error EMA
  - M4: EWC regularization with Fisher diagonal lifecycle (decay/accumulate/merge)
  - Logits reversal for Fisher estimation (optional)
  - NaN safety guards: EwmaTracker, learn_continuous (td_error), push_surprise, push_td_error
  - ClState serialization with `PartialEq` + `Default` build-then-compare
- **TD(n)** generic n-step temporal difference returns:
  - `td_steps: usize` config field (0=TD(0) default, >=2 for multi-step)
  - Buffer + terminal flush architecture with `gamma^n` bootstrap
  - Pre-computed V(s) in flush via `pre_v_s` to avoid stale-estimate bias
  - O(K) suffix-sum for flush returns
  - NaN guard at buffer insertion; `td_steps=1` rejected at construction
- **Bidirectional hysteresis coupling**:
  - `critic_wakes_actor` coupling symmetric to existing `actor_wakes_critic`
  - Both default `true` (active when hysteresis enabled)
  - `actor_frozen_steps` counter symmetric to `critic_frozen_steps`
  - EWMA k reset on coupling-forced wake prevents immediate re-freeze
  - Anti-cascade proven safe (Frozen guard + warmup guard)
- `step()` and `step_masked()` unified step API with internal TD(0)/TD(n) learning
- `step_masked()` returns `Result<usize, PcError>` (was panic on empty valid_actions)
- `compute_decay_factors()` single source of truth (extracted from 3x duplication)
- Monolith `pc_actor_critic.rs` refactored into 6-file directory submodule:
  `config.rs`, `ewma.rs`, `hysteresis.rs`, `fisher.rs`, `trajectory.rs`, `mod.rs`
- 524 tests (was 392), 21 doctests

### Changed
- `PcActorCriticConfig` has ~15 new serde-default fields (all backward compatible)
- `Matrix` now derives `Default` + `PartialEq`
- `ClState`, `FisherStateSerialized`, `EwmaTrackerSerialized` derive `Default` + `PartialEq`
- `EwmaTracker.k` can be reset to 0 on coupling-forced wake (doc updated)
- `actor_wakes_critic` default changed from `false` to `true`

### Fixed
- M3b `layer_error_ema` was never updated during learning (now updates in `learn_continuous`)
- NaN reward propagates silently through weight updates (td_error guard added)
- NaN/Inf silently corrupts EwmaTracker (finite guard added)

## [2.0.0] - 2026-04-05

### Breaking Changes
- `LinAlg` trait methods now require `&self` (instance methods instead of static)
- `vec_as_slice` removed from `LinAlg` trait (31 methods remain)
- All struct constructors (`PcActorCritic::new`, `PcActor::new`, `MlpCritic::new`) now take `backend: L` as first parameter
- `load_agent` and `load_agent_generic` now take `backend` as second parameter
- `PcActorCritic::from_parts` now takes `backend: L` as last parameter
- Generic functions in `matrix.rs` (`cca_neuron_alignment`, `standardize_columns`) now take `backend: &L` as first parameter
- `LinAlg` backends used with serde-derived structs must implement `Default`

### Migration Guide
1. Create a backend instance: `let backend = CpuLinAlg::new();`
2. Pass it to constructors: `PcActorCritic::new(backend, config, seed)`
3. Pass it to load functions: `load_agent("path.json", CpuLinAlg::new())`
4. Replace `L::method(args)` with `backend.method(args)` in generic code
5. Update calls to `cca_neuron_alignment(&backend, ...)`, `standardize_columns(&backend, ...)`, etc.
6. Custom `LinAlg` backends must implement `Default` for serde compatibility
7. Serialization format is unchanged — v1.x JSON files load in v2.0

### Added
- `CpuLinAlg::new()` constructor
- `impl Default for CpuLinAlg`
- `backend: L` field on `Layer`, `PcActor`, `MlpCritic`, `PcActorCritic`

## [1.2.3] - 2026-04-04

- Resolve skip projection validation, SVD doc, NaN sort, and buffer size guard

## [1.2.2] - 2026-04-03

- Standalone library crate restructuring
- CD release pipeline with tag-version validation

## [1.2.1] - 2026-04-02

- Adaptive surprise with configurable buffer
- Serde defaults for all config fields

## [1.2.0] - 2026-04-01

- Golub-Kahan SVD O(n^3) replacing Jacobi O(n^4)

## [1.1.0] - 2026-03-30

- CCA crossover with Hungarian matching for GA evolution

## [1.0.0] - 2026-03-25

- Initial release: PC Actor-Critic with predictive coding inference
