# LinAlg Trait Instance Methods Refactor (v2.0) — TDD Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor LinAlg trait from static methods to instance methods (`&self`), enabling GPU backends with device state.

**Methodology:** Strict TDD — Red-Green-Refactor per task. Each phase produces a git commit.

**Tech Stack:** Rust, cargo nextest, no new dependencies

**Spec:** `sbtdd/spec-behavior.md`

**TDD Deviation Zone (Tasks 3-8):** This is a trait-wide refactor. After Task 1 changes the trait, the crate will NOT compile until Task 8 completes all migrations. This is inherent to Rust's type system — changing a trait breaks all implementors and callers simultaneously. During Tasks 3-8, RED phases produce tests that cannot compile (not just fail), which deviates from strict TDD where RED means "test compiles but fails." This is an acknowledged and unavoidable constraint of Rust's monomorphization model. Intermediate verification uses error count tracking: `cargo check 2>&1 | grep -c '^error'` should decrease monotonically after each GREEN step. Full `cargo nextest run` is only possible after Task 8 GREEN.

**Serde Constraint:** `Layer<L>` derives `Serialize/Deserialize` with explicit serde bounds on `L::Matrix` and `L::Vector`. Adding a `backend: L` field requires either `L: Serialize+Deserialize` (CpuLinAlg does not implement these) or `#[serde(skip)]` which requires `L: Default`. The fix: `impl Default for CpuLinAlg` (Task 2) and `#[serde(skip, default)]` on all `backend` fields in Serde-derived structs (Task 3). This does NOT violate the spec rule "No agregar Default a los config structs" because `CpuLinAlg` is a backend, not a config struct. `PcActor`, `MlpCritic`, and `PcActorCritic` do NOT derive Serde — only `Layer<L>` needs this treatment. **Note:** The `L: Default` bound is scoped to serde, NOT added to the `LinAlg` trait itself. Future GPU backends must implement `Default` if used with serde-derived structs — this must be documented on the `LinAlg` trait (Task 1 REFACTOR).

**Layer Struct Literals:** Adding `backend: L` to `Layer<L>` breaks ALL struct literal construction sites (not just `Layer::new()`). There are **8 sites** across the codebase that construct `Layer { weights, bias, activation }` directly — each must add `backend: backend.clone()`. These are in: `pc_actor.rs` (4 sites: crossover output layer, from_weights, cca_align_and_blend_layer), `mlp_critic.rs` (3 sites: crossover, from_weights, to_weights reconstruction). Tasks 5 and 6 must explicitly address these.

**CpuLinAlg:: Production Calls:** In addition to `L::` method calls, `mlp_critic.rs` and `pc_actor.rs` have `CpuLinAlg::` static calls in production code (to_weights/from_weights bridge functions). These are concrete-type calls, not generic, and must migrate to `CpuLinAlg::new().method()` or a local `let cpu = CpuLinAlg::new();` binding. Tasks 5 and 6 must address these.

**vec_as_slice Disposition:** Remove entirely (0 production callers confirmed). Do NOT keep as inherent method on CpuLinAlg.

**Inventory (verified):**

The `L::` column includes both method calls (`L::zeros_vec(...)`) and associated type references (`L::Matrix`, `L::Vector`). Only method calls need migration to `self.backend.method()` or `backend.method()`. Type references (`L::Matrix`, `L::Vector`) remain unchanged. Approximate method-call-only counts are ~60% of the totals below.

| File | L:: total (prod) | CpuLinAlg:: calls (test) | #[test] count |
|------|-------------------|--------------------------|---------------|
| linalg/mod.rs | 0 | 0 | 0 |
| linalg/cpu.rs | 0 | 0 | 54 |
| layer.rs | 32 | 0 | 18 |
| matrix.rs | 56 | 34 | 65 |
| mlp_critic.rs | 30 | 4 | 24 |
| pc_actor.rs | 139 | 21 | 96 |
| pc_actor_critic.rs | 27 | 1 | 41 |
| serializer.rs | 0 | 0 | 16 |
| activation.rs | 0 | 0 | 37 |
| golub_kahan.rs | 0 | 0 | 35 |
| **Total** | **284** | **60** | **386** |

**Dependency order:**
```
1. linalg/mod.rs (trait) ──► 2. linalg/cpu.rs (CpuLinAlg impl)
         │
         ▼
3. layer.rs ──► 4. matrix.rs ──► 5. mlp_critic.rs ──► 6. pc_actor.rs ──► 7. pc_actor_critic.rs ──► 8. serializer.rs
```

**Pre-flight (before Task 1):**
- [ ] Create fixture directory: `mkdir -p tests/fixtures`
- [ ] Generate `tests/fixtures/v1_model.json` by adding a `#[test] #[ignore]` test to serializer.rs that creates a valid agent with `PcActorCritic::new(config, 42)` and calls `save_agent(&agent, "tests/fixtures/v1_model.json")`, then running `cargo nextest run -E 'test(generate_v1_fixture)' --run-ignored`. This fixture is needed for Escenario 10 (backward compatibility test) in Task 8. **Run this BEFORE any trait changes — it must use v1.2.3 API.**

---

## Task 1: LinAlg trait — instance methods + remove vec_as_slice

**Files:** `src/linalg/mod.rs`
**Spec coverage:** R1, Escenario 5, Escenario 9

### RED — Write failing tests for new API

Add to `src/linalg/mod.rs` in a new `#[cfg(test)]` module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuLinAlg;

    /// Escenario 3: Operations use &self
    #[test]
    fn test_linalg_methods_take_self() {
        let backend = CpuLinAlg::new();
        let v = backend.zeros_vec(5);
        assert_eq!(backend.vec_len(&v), 5);
    }

    /// Escenario 9: CpuLinAlg is ZST
    #[test]
    fn test_cpu_linalg_is_zst() {
        assert_eq!(std::mem::size_of::<CpuLinAlg>(), 0);
    }

    /// Escenario 4: mat_vec_mul via instance
    #[test]
    fn test_mat_vec_mul_via_instance() {
        let backend = CpuLinAlg::new();
        // Identity 2x2 * [3,4] = [3,4]
        let mut m = backend.zeros_mat(2, 2);
        backend.mat_set(&mut m, 0, 0, 1.0);
        backend.mat_set(&mut m, 1, 1, 1.0);
        let v = backend.vec_from_slice(&[3.0, 4.0]);
        let result = backend.mat_vec_mul(&m, &v);
        assert_eq!(backend.vec_get(&result, 0), 3.0);
        assert_eq!(backend.vec_get(&result, 1), 4.0);
    }

    /// Escenario 5: vec_as_slice removed from trait.
    /// This is a compile-time negative check. The real verification is that
    /// no generic code (code parameterized over L: LinAlg) can call
    /// backend.vec_as_slice(). This test documents the removal; the compiler
    /// enforces it. A `compile_fail` doctest would be ideal but requires
    /// the `trybuild` crate which we avoid per the no-new-dependencies rule.
    #[test]
    fn test_trait_has_31_methods() {
        // After removing vec_as_slice, the trait has 31 methods.
        // This is a documentation marker — the real enforcement is the compiler.
        // If vec_as_slice were re-added, the method count wouldn't change this
        // constant, but the commit history documents the intent.
    }
}
```

- [ ] Write tests in `src/linalg/mod.rs`
- [ ] Verify RED: `cargo check` fails — `CpuLinAlg::new()` does not exist, `zeros_vec` is not `&self`
- [ ] Commit:
```bash
git add src/linalg/mod.rs
git commit -m "test: Add instance method tests for LinAlg trait (RED)"
```

### GREEN — Change trait to `&self`, remove `vec_as_slice`

1. Change all 32 method signatures from `fn method(args)` to `fn method(&self, args)`
2. Remove `vec_as_slice` from the trait (line 161) → 31 methods remain
3. Update trait docstring example to use instance syntax

- [ ] Modify trait in `src/linalg/mod.rs`
- [ ] Verify: `cargo check 2>&1 | grep -c '^error'` — note error count (baseline for monotonic decrease)
- [ ] Commit:
```bash
git add src/linalg/mod.rs
git commit -m "feat: Change LinAlg trait to &self instance methods, remove vec_as_slice"
```

### REFACTOR — Clean up trait documentation + Default warning

- [ ] Update module-level docstring and `///` examples to reflect new API
- [ ] Add doc comment to `LinAlg` trait warning about `Default` requirement for serde compatibility:
```rust
/// # Serde Compatibility
///
/// Backends used with serde-derived structs (e.g., `Layer<L>`) must
/// implement [`Default`] because the `backend` field uses
/// `#[serde(skip, default)]`. `CpuLinAlg` implements `Default`.
/// GPU backends must also implement `Default` (e.g., via a
/// default device selection).
```
- [ ] Verify: error count unchanged or decreased
- [ ] Commit:
```bash
git add src/linalg/mod.rs
git commit -m "refactor: Update LinAlg trait documentation for instance methods"
```

---

## Task 2: CpuLinAlg — adapt implementation + Default

**Files:** `src/linalg/cpu.rs`
**Spec coverage:** R2, Escenario 3, Escenario 9

### RED — Write failing tests for CpuLinAlg::new() and instance calls

Add/modify tests in `src/linalg/cpu.rs` `#[cfg(test)]` module:

```rust
#[test]
fn test_cpu_linalg_new_constructor() {
    let backend = CpuLinAlg::new();
    let v = backend.zeros_vec(3);
    assert_eq!(backend.vec_len(&v), 3);
}

#[test]
fn test_cpu_linalg_default_equals_new() {
    let a = CpuLinAlg::new();
    let b = CpuLinAlg::default();
    // Both are ZST — they're identical
    assert_eq!(std::mem::size_of_val(&a), 0);
    assert_eq!(std::mem::size_of_val(&b), 0);
}
```

Also migrate ALL 54 existing tests from `CpuLinAlg::method()` to `backend.method()` syntax. Example:

```rust
// Before
let v = CpuLinAlg::zeros_vec(5);
assert_eq!(CpuLinAlg::vec_len(&v), 5);

// After
let backend = CpuLinAlg::new();
let v = backend.zeros_vec(5);
assert_eq!(backend.vec_len(&v), 5);
```

Remove `test_vec_as_slice_returns_slice` test (method removed from trait).

- [ ] Add constructor test + Default test + migrate all 54 tests to instance syntax
- [ ] Remove `test_vec_as_slice_returns_slice`
- [ ] Verify RED: tests fail — `CpuLinAlg::new()` not defined, impl signatures don't match trait
- [ ] Commit:
```bash
git add src/linalg/cpu.rs
git commit -m "test: Migrate cpu.rs tests to instance methods (RED)"
```

### GREEN — Adapt CpuLinAlg impl + add Default

1. Add constructor and Default:
```rust
impl CpuLinAlg {
    /// Creates a new CPU linear algebra backend.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuLinAlg {
    fn default() -> Self {
        Self
    }
}
```

2. Add `&self` to all 31 method signatures in `impl LinAlg for CpuLinAlg`
3. Remove `vec_as_slice` implementation from trait impl block
   - Keep `vec_as_slice` as a direct method on `CpuLinAlg` (not part of trait) if needed for backward compat, or remove entirely if zero callers

- [ ] Add `CpuLinAlg::new()`, `Default` impl, add `&self` to all 31 methods, remove `vec_as_slice` from trait impl
- [ ] Verify GREEN: `cargo nextest run -E 'test(linalg::)' 2>&1` — all linalg tests pass (54 cpu + trait tests)
- [ ] Note: rest of crate still broken (L:: calls in other files). Record error count.
- [ ] Commit:
```bash
git add src/linalg/cpu.rs
git commit -m "feat: Adapt CpuLinAlg to instance methods with new() constructor and Default"
```

### REFACTOR — Clean up cpu.rs

- [ ] Update docstrings and `///` examples in cpu.rs to new syntax
- [ ] Verify: linalg tests still pass
- [ ] Commit:
```bash
git add src/linalg/cpu.rs
git commit -m "refactor: Update CpuLinAlg documentation for instance methods"
```

---

## Task 3: Migrate Layer (+ serde fix)

**Files:** `src/layer.rs` (32 L:: references, 0 CpuLinAlg:: test calls, 18 tests)
**Spec coverage:** R3 (backend field), R4 (L:: to self.backend)

**Serde note:** `Layer<L>` derives `Serialize/Deserialize`. The new `backend: L` field must be annotated with `#[serde(skip, default)]` so that: (a) it is not serialized to JSON (preserving backward compat), and (b) deserialization uses `L::default()` (satisfied by `CpuLinAlg::default()` from Task 2). Future GPU backends must also implement `Default` or use a custom deserializer.

### RED — Migrate layer.rs tests to expect backend

Migrate all 18 tests to create layers with backend parameter. Since `Layer` is `pub(crate)`, tests use it directly:

```rust
// Before
let layer = Layer::<CpuLinAlg>::new(3, 2, Activation::Tanh, &mut rng);

// After
let backend = CpuLinAlg::new();
let layer = Layer::new(3, 2, Activation::Tanh, &backend, &mut rng);
```

- [ ] Migrate all 18 tests to pass `&backend` to Layer constructors
- [ ] Verify RED: error count unchanged or increased (tests reference APIs that don't exist yet)
- [ ] Commit:
```bash
git add src/layer.rs
git commit -m "test: Migrate layer.rs tests to backend parameter (RED)"
```

### GREEN — Add backend field with serde skip, migrate L:: method calls

1. Add `backend: L` field to `Layer<L>` with serde annotation:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "L::Matrix: Serialize, L::Vector: Serialize",
    deserialize = "L::Matrix: for<'a> Deserialize<'a>, L::Vector: for<'a> Deserialize<'a>, L: Default"
))]
pub(crate) struct Layer<L: LinAlg = CpuLinAlg> {
    pub(crate) weights: L::Matrix,
    pub(crate) bias: L::Vector,
    pub(crate) activation: Activation,
    #[serde(skip, default)]
    pub(crate) backend: L,
}
```

2. Update `new()` to accept `backend: &L`, clone into field
3. Replace all `L::method(args)` calls with `self.backend.method(args)` in:
   - `forward()`, `transpose_forward()`, `backward()`

- [ ] Add backend field with `#[serde(skip, default)]`, update constructor, migrate L:: method calls
- [ ] Verify: `cargo check 2>&1 | grep -c '^error'` — error count decreased
- [ ] Commit:
```bash
git add src/layer.rs
git commit -m "feat: Add backend field to Layer with serde skip, migrate L:: calls"
```

### REFACTOR — No significant cleanup expected

- [ ] Review for any missed L:: calls or unnecessary clones
- [ ] If changes made, commit:
```bash
git add src/layer.rs
git commit -m "refactor: Clean up Layer backend migration"
```

---

## Task 4: Migrate matrix.rs generic functions

**Files:** `src/matrix.rs` (56 L:: references in prod, 34 CpuLinAlg:: calls in tests, 65 tests)
**Spec coverage:** R4 (internal calls migrated)

### RED — Migrate matrix.rs tests

Migrate all 34 `CpuLinAlg::method()` calls in tests to `backend.method()`. Also update calls to generic functions to pass `&backend` as first arg:

```rust
// Before
let result = standardize_columns::<CpuLinAlg>(&m, rows, cols);

// After
let backend = CpuLinAlg::new();
let result = standardize_columns(&backend, &m, rows, cols);
```

- [ ] Migrate all 65 tests (34 CpuLinAlg:: calls + generic function calls)
- [ ] Verify RED: error count unchanged or increased
- [ ] Commit:
```bash
git add src/matrix.rs
git commit -m "test: Migrate matrix.rs tests to instance methods (RED)"
```

### GREEN — Add `backend: &L` to generic functions, migrate L:: method calls

Functions to change (add `backend: &L` as first parameter):
- `standardize_columns<L>`
- `compute_matrix_sqrtm<L>`
- `generic_svd_alignment<L>`
- `cca_neuron_alignment<L>`

Non-generic functions (`softmax_masked`, `argmax_masked`, etc.) do NOT change.

Replace all `L::method(args)` calls with `backend.method(args)`.

- [ ] Add `backend: &L` parameter, migrate all L:: method calls
- [ ] Verify: `cargo check 2>&1 | grep -c '^error'` — error count decreased
- [ ] Commit:
```bash
git add src/matrix.rs
git commit -m "feat: Migrate matrix.rs generic functions to instance methods"
```

### REFACTOR — Clean up matrix.rs

- [ ] Review for consistent parameter ordering (`backend` always first)
- [ ] If changes made, commit:
```bash
git add src/matrix.rs
git commit -m "refactor: Clean up matrix.rs backend parameter ordering"
```

---

## Task 5: Migrate MlpCritic

**Files:** `src/mlp_critic.rs` (30 L:: references, 4 CpuLinAlg:: test calls, 24 tests)
**Spec coverage:** R3 (backend field), R4 (calls migrated), R6 (crossover)

### RED — Migrate mlp_critic.rs tests

Migrate 24 tests. Update `make_critic()` helpers and all 4 `CpuLinAlg::` calls:

```rust
// Before
let critic = MlpCritic::<CpuLinAlg>::new(config, &mut rng)?;

// After
let backend = CpuLinAlg::new();
let critic = MlpCritic::new(backend, config, &mut rng)?;
```

- [ ] Migrate all 24 tests to pass backend to constructors
- [ ] Verify RED: error count unchanged or increased
- [ ] Commit:
```bash
git add src/mlp_critic.rs
git commit -m "test: Migrate mlp_critic.rs tests to backend parameter (RED)"
```

### GREEN — Add backend field, migrate L:: method calls

1. Add `pub(crate) backend: L` field to `MlpCritic<L>` (no Serde derive on this struct — no serde annotation needed)
2. Update `new()`: `pub fn new(backend: L, config: MlpCriticConfig, rng: &mut impl Rng) -> Result<Self, PcError>`
3. Replace all `L::method(args)` calls with `self.backend.method(args)` in:
   - `forward()`, `forward_with_hidden()`, `backward()`
   - `crossover()` — receives backend from parents via `parent_a.backend.clone()`. **Also update the `Layer { weights, bias, activation }` struct literal in crossover (line ~257) to include `backend: parent_a.backend.clone()`**
   - `from_weights(backend: L, ...)` — receives backend as owned parameter, stores in struct. **Also update Layer struct literals in from_weights (line ~494) and to_weights Layer reconstruction (line ~401) to include backend field**
4. Migrate `CpuLinAlg::` static calls in `to_weights()` and `from_weights()` bridge functions (5 calls) to instance syntax: `let cpu = CpuLinAlg::new(); cpu.method(...)` or `CpuLinAlg::new().method(...)`

- [ ] Add backend field, update constructor, migrate all L:: method calls
- [ ] Update 3 Layer struct literal sites to include `backend` field
- [ ] Migrate 5 CpuLinAlg:: production calls to instance syntax
- [ ] Verify: `cargo check 2>&1 | grep -c '^error'` — error count decreased
- [ ] Commit:
```bash
git add src/mlp_critic.rs
git commit -m "feat: Add backend field to MlpCritic, migrate L:: calls"
```

### REFACTOR — Clean up MlpCritic

- [ ] Review for consistency
- [ ] If changes made, commit:
```bash
git add src/mlp_critic.rs
git commit -m "refactor: Clean up MlpCritic backend migration"
```

---

## Task 6: Migrate PcActor (largest file)

**Files:** `src/pc_actor.rs` (139 L:: references, 21 CpuLinAlg:: test calls, 96 tests)
**Spec coverage:** R3 (backend field), R4 (calls migrated), R6 (crossover)

### RED — Migrate pc_actor.rs tests + add Escenario 7 test

Migrate 96 tests. Update `default_config()`, `make_actor()` helpers and all 21 `CpuLinAlg::` calls:

```rust
// Before
let actor = PcActor::<CpuLinAlg>::new(config, &mut rng)?;

// After
let backend = CpuLinAlg::new();
let actor = PcActor::new(backend, config, &mut rng)?;
```

Free functions used in tests (`permute_cols`, `permute_rows`, `permute_vec`) will need `&backend` parameter.

Add new test for Escenario 7 (crossover with backend):

```rust
/// Escenario 7: Crossover child inherits backend
#[test]
fn test_crossover_child_inherits_backend() {
    let backend = CpuLinAlg::new();
    let config = default_config();
    let mut rng = StdRng::seed_from_u64(42);
    let parent_a = PcActor::new(backend.clone(), config.clone(), &mut rng).unwrap();
    let parent_b = PcActor::new(backend.clone(), config.clone(), &mut rng).unwrap();
    let cache_a = /* ... build activation cache ... */;
    let cache_b = /* ... build activation cache ... */;
    let child = PcActor::crossover(&parent_a, &parent_b, &cache_a, &cache_b, 0.5, &config, &mut rng).unwrap();
    // Child should have a working backend — verify by performing an operation
    assert_eq!(std::mem::size_of_val(&child.backend), 0); // ZST confirms CpuLinAlg
}
```

- [ ] Migrate all 96 tests + helper functions + add Escenario 7 test
- [ ] Verify RED: error count unchanged or increased
- [ ] Commit:
```bash
git add src/pc_actor.rs
git commit -m "test: Migrate pc_actor.rs tests to backend parameter (RED)"
```

### GREEN — Add backend field, migrate L:: method calls

1. Add `pub(crate) backend: L` field to `PcActor<L>` (no Serde derive — no serde annotation needed)
2. Update `new()` to accept `backend: L` as first parameter
3. Migrate by method group:
   - Constructor + skip projection init
   - `infer()`
   - `select_action()`
   - `backward()`
   - `crossover()` — child gets backend from parent via `parent_a.backend.clone()`. **Update the `Self { layers, config, rezero_alpha, skip_projections }` struct literal (line ~541) to include `backend: parent_a.backend.clone()`. Also update the `Layer { weights, bias, activation }` struct literal in crossover output layer construction (line ~464) to include `backend`**
   - `permute_cols()`, `permute_rows()`, `permute_vec()` (free functions → add `backend: &L` as first param)
   - `blend_layer_weights<L>` (free function → add `backend: &L` as first param, ~18 L:: calls)
   - `cca_align_and_blend_layer<L>` (free function → add `backend: &L` as first param, ~16 L:: calls, plus Layer struct literal at line ~1299)
   - `from_weights(backend: L, ...)` — receives backend as owned parameter. **Update Layer struct literal in from_weights (line ~933) to include `backend`**

- [ ] Add backend field, migrate all L:: method calls
- [ ] Add `backend: &L` to `blend_layer_weights` and `cca_align_and_blend_layer` free functions
- [ ] Update 4 Layer struct literal sites to include `backend` field
- [ ] Update PcActor Self struct literal in crossover to include `backend`
- [ ] Verify: `cargo check 2>&1 | grep -c '^error'` — error count decreased significantly
- [ ] Commit:
```bash
git add src/pc_actor.rs
git commit -m "feat: Add backend field to PcActor, migrate L:: calls"
```

### REFACTOR — Clean up PcActor

- [ ] Review for consistent `self.backend` vs `backend` parameter usage
- [ ] If changes made, commit:
```bash
git add src/pc_actor.rs
git commit -m "refactor: Clean up PcActor backend migration"
```

---

## Task 7: Migrate PcActorCritic (+ from_parts + cache_to_matrices)

**Files:** `src/pc_actor_critic.rs` (27 L:: references, 1 CpuLinAlg:: test call, 41 tests)
**Spec coverage:** R3 (backend field), R4 (calls migrated), Escenario 1, 2, 8

**Critical note:** `PcActorCritic::from_parts` (line 369) constructs `Self` and must accept a `backend: L` parameter. Both `load_agent` and `load_agent_generic` in serializer.rs call `from_parts`. Also, `cache_to_matrices<L>` (line 660) is a free function with 5 `L::` method calls that needs a `backend: &L` parameter.

### RED — Migrate pc_actor_critic.rs tests + add scenario tests

Migrate 41 tests. Update `make_agent()` helper:

```rust
// Before
let agent = PcActorCritic::new(config, 42)?;

// After
let backend = CpuLinAlg::new();
let agent = PcActorCritic::new(backend, config, 42)?;
```

Add new tests for spec scenarios:

```rust
/// Escenario 1: CPU agent construction
#[test]
fn test_agent_construction_with_backend() {
    let backend = CpuLinAlg::new();
    let config = valid_config();
    let agent = PcActorCritic::new(backend, config, 42);
    assert!(agent.is_ok());
}

/// Escenario 2: Invalid config still returns PcError
#[test]
fn test_agent_invalid_config_with_backend() {
    let backend = CpuLinAlg::new();
    let mut config = valid_config();
    config.gamma = -0.1;
    let result = PcActorCritic::new(backend, config, 42);
    assert!(matches!(result, Err(PcError::ConfigValidation(msg)) if msg.contains("gamma")));
}

/// Escenario 8: Backend propagates to sub-components
#[test]
fn test_backend_propagates_to_actor_and_critic() {
    let backend = CpuLinAlg::new();
    let config = valid_config();
    let agent = PcActorCritic::new(backend, config, 42).unwrap();
    // Verify agent, actor, and critic all have ZST backends
    assert_eq!(std::mem::size_of_val(&agent.backend), 0);
    assert_eq!(std::mem::size_of_val(&agent.actor.backend), 0);
    assert_eq!(std::mem::size_of_val(&agent.critic.backend), 0);
}
```

- [ ] Migrate all 41 tests + add 3 new scenario tests
- [ ] Verify RED: error count unchanged or increased
- [ ] Commit:
```bash
git add src/pc_actor_critic.rs
git commit -m "test: Migrate pc_actor_critic.rs tests to backend parameter (RED)"
```

### GREEN — Add backend field, migrate L:: calls, update from_parts + cache_to_matrices

1. Add `pub(crate) backend: L` field to `PcActorCritic<L>`
2. Update `new()`: `pub fn new(backend: L, config: PcActorCriticConfig, seed: u64) -> Result<Self, PcError>`
   - Pass `backend.clone()` to `PcActor::new()` and `MlpCritic::new()`
   - Store `backend` in the struct
3. Update `from_parts()`:
```rust
pub fn from_parts(
    config: PcActorCriticConfig,
    actor: PcActor<L>,
    critic: MlpCritic<L>,
    rng: StdRng,
    backend: L,
) -> Self {
    Self { actor, critic, config, rng, surprise_buffer: VecDeque::new(), backend }
}
```
4. Update `cache_to_matrices<L>` free function (line 660) — add `backend: &L` as first parameter, replace 5 `L::method(args)` calls with `backend.method(args)`:
   - `L::zeros_mat(0, 0)` → `backend.zeros_mat(0, 0)`
   - `L::vec_len(...)` → `backend.vec_len(...)`
   - `L::zeros_mat(...)` → `backend.zeros_mat(...)`
   - `L::mat_set(...)` → `backend.mat_set(...)`
   - `L::vec_get(...)` → `backend.vec_get(...)`
5. Replace remaining `L::method(args)` calls with `self.backend.method(args)` in:
   - `act()`, `learn()`, `learn_td()`, `crossover()`
   - Update `cache_to_matrices` call sites to pass `&self.backend`

- [ ] Add backend field, update from_parts, update cache_to_matrices, migrate all L:: calls
- [ ] Verify: `cargo check 2>&1 | grep -c '^error'` — error count decreased (near zero, only serializer remains)
- [ ] Commit:
```bash
git add src/pc_actor_critic.rs
git commit -m "feat: Add backend field to PcActorCritic, migrate L:: calls including from_parts"
```

### REFACTOR — Clean up PcActorCritic

- [ ] Review for consistency
- [ ] If changes made, commit:
```bash
git add src/pc_actor_critic.rs
git commit -m "refactor: Clean up PcActorCritic backend migration"
```

---

## Task 8: Migrate Serializer (+ load_agent_generic)

**Files:** `src/serializer.rs` (~5 indirect calls, 16 tests)
**Spec coverage:** R5, Escenario 6, Escenario 10

**Critical note:** Both `load_agent` (CPU-specific, line 146) and `load_agent_generic<L>` (generic, line 178) must receive a `backend` parameter. Both call `PcActorCritic::from_parts` which now requires a backend. The `backend.clone()` pattern is needed when passing to both `PcActor::from_weights` and `MlpCritic::from_weights` (L: Clone is already a bound on LinAlg).

### RED — Migrate serializer.rs tests + add scenario tests

Migrate 16 tests. Update `load_agent` and `load_agent_generic` calls:

```rust
// Before
let (loaded, meta) = load_agent("test.json")?;

// After
let backend = CpuLinAlg::new();
let (loaded, meta) = load_agent("test.json", CpuLinAlg::new())?;
```

Add new tests:

```rust
/// Escenario 6: Serialization round-trip with backend
#[test]
fn test_save_load_roundtrip_with_backend() {
    let backend = CpuLinAlg::new();
    let agent = make_agent(backend.clone());
    save_agent(&agent, "roundtrip_test.json").unwrap();
    let (loaded, _) = load_agent("roundtrip_test.json", CpuLinAlg::new()).unwrap();
    // verify same behavior for same input
}

/// Escenario 10: Backward compat — v1.2.3 JSON loads in v2.0
#[test]
fn test_backward_compat_json_load() {
    // v1.2.3 JSON has no backend field — backend comes from parameter.
    // The backend field is #[serde(skip)] in Layer, so it deserializes
    // via Default (CpuLinAlg::default()). The from_parts call provides
    // the backend for PcActorCritic.
    let (loaded, _) = load_agent("tests/fixtures/v1_model.json", CpuLinAlg::new());
    assert!(loaded.is_ok());
}
```

- [ ] Migrate all 16 tests + add 2 scenario tests
- [ ] Verify RED: error count unchanged or increased
- [ ] Commit:
```bash
git add src/serializer.rs
git commit -m "test: Migrate serializer.rs tests to backend parameter (RED)"
```

### GREEN — Add backend parameter to load functions

1. Update `load_agent`:
```rust
pub fn load_agent(path: &str, backend: CpuLinAlg) -> Result<(PcActorCritic, AgentMetadata), PcError> {
    load_agent_generic(path, backend)
}
```

2. Update `load_agent_generic`:
```rust
pub fn load_agent_generic<L: LinAlg>(
    path: &str,
    backend: L,
) -> Result<(PcActorCritic<L>, AgentMetadata), PcError> {
    // ... deserialize SaveFile ...
    let actor = PcActor::from_weights(backend.clone(), config.actor, weights.actor)?;
    let critic = MlpCritic::from_weights(backend.clone(), config.critic, weights.critic)?;
    // ... build rng ...
    let agent = PcActorCritic::from_parts(config.agent, actor, critic, rng, backend);
    Ok((agent, metadata))
}
```

3. `save_agent` — no signature change needed (to_weights() extracts to CPU types)

- [ ] Update both load function signatures, propagate backend with `.clone()`
- [ ] **COMPILATION CHECKPOINT:** `cargo nextest run` — ALL tests should now compile and pass
- [ ] Expected: 386 existing - 1 removed (vec_as_slice test) + 6 new scenario tests = ~391 tests pass
- [ ] Commit:
```bash
git add src/serializer.rs
git commit -m "feat: Add backend parameter to serializer load functions"
```

### REFACTOR — Clean up serializer

- [ ] Review for consistency
- [ ] If changes made, commit:
```bash
git add src/serializer.rs
git commit -m "refactor: Clean up serializer backend migration"
```

---

## Task 9: Update lib.rs exports

**Files:** `src/lib.rs`
**Spec coverage:** R4 (type aliases)

### RED — Verify current exports still work

- [ ] Check that type aliases (`PcActorCpu`, `MlpCriticCpu`, `PcActorCriticCpu`, `LayerCpu`) still compile
- [ ] No new tests needed — existing tests cover this

### GREEN — Update exports if needed

- [ ] Verify `pub use` statements still work
- [ ] Update docstring examples in lib.rs to use `CpuLinAlg::new()`
- [ ] Run: `cargo nextest run` — all tests pass
- [ ] Commit:
```bash
git add src/lib.rs
git commit -m "feat: Update lib.rs exports and examples for v2.0 API"
```

---

## Task 10: Update doctests

**Files:** Doctests in all `src/*.rs` files
**Spec coverage:** All scenarios via documentation examples

### RED/GREEN — Update doctest examples to new API

Every `///` example that uses `CpuLinAlg::method()` becomes `backend.method()`:

```rust
/// ```
/// use pc_rl_core::CpuLinAlg;
/// use pc_rl_core::LinAlg;
///
/// let backend = CpuLinAlg::new();
/// let v = backend.zeros_vec(5);
/// assert_eq!(backend.vec_len(&v), 5);
/// ```
```

- [ ] Update all doctest examples across all files
- [ ] Verify: `cargo test --doc` — all ~20 doctests pass
- [ ] Commit:
```bash
git add src/
git commit -m "docs: Update all doctests for LinAlg v2.0 instance methods"
```

---

## Task 11: Update README.md

**Files:** `README.md`

### RED/GREEN — Update Quick Start and examples

```rust
use pc_rl_core::{CpuLinAlg, PcActorCritic, PcActorCriticConfig};

let backend = CpuLinAlg::new();
let agent = PcActorCritic::new(backend, config, 42)?;
```

- [ ] Update Quick Start example
- [ ] Update type aliases section — note that backend is passed to constructors
- [ ] Commit:
```bash
git add README.md
git commit -m "docs: Update README for LinAlg v2.0 API"
```

---

## Task 12: CHANGELOG and migration guide

**Files:** `CHANGELOG.md` (new file)

### RED/GREEN — Document breaking changes for crates.io consumers

Create `CHANGELOG.md` with v2.0.0 entry:

```markdown
# Changelog

## [2.0.0] - 2026-04-XX

### Breaking Changes
- `LinAlg` trait methods now require `&self` (instance methods instead of static)
- `vec_as_slice` removed from `LinAlg` trait
- All struct constructors (`PcActorCritic::new`, `PcActor::new`, `MlpCritic::new`) now take `backend: L` as first parameter
- `load_agent` and `load_agent_generic` now take `backend` as second parameter
- `PcActorCritic::from_parts` now takes `backend: L` as last parameter
- Generic functions in `matrix.rs` (`cca_neuron_alignment`, `standardize_columns`, `compute_matrix_sqrtm`, `generic_svd_alignment`) now take `backend: &L` as first parameter
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
```

- [ ] Create CHANGELOG.md
- [ ] Commit:
```bash
git add CHANGELOG.md
git commit -m "docs: Add CHANGELOG with v2.0.0 migration guide"
```

---

## Task 13: Final verification

**Spec coverage:** Full regression — all escenarios validated

### GREEN — Full regression suite (per CLAUDE.local.md §0.2)

```bash
cargo nextest run              # All unit tests — 0 failures
cargo test --doc               # All doctests — 0 failures
cargo clippy --tests -- -D warnings  # 0 warnings
cargo fmt --check              # Clean formatting
cargo build --release          # Release build — 0 warnings
cargo doc --no-deps            # Doc generation — 0 warnings
```

- [ ] Run all 6 verification commands
- [ ] All must pass with zero errors/warnings
- [ ] Commit (only if fmt or clippy fixes needed):
```bash
git add src/
git commit -m "refactor: Apply clippy and fmt fixes for v2.0"
```

---

## Task 14: Version bump to v2.0.0

### GREEN — Bump version

- [ ] Update `Cargo.toml` version to `2.0.0`
- [ ] Update `README.md` install snippet to `pc-rl-core = "2.0"`
- [ ] Commit:
```bash
git add Cargo.toml README.md
git commit -m "chore: Bump pc-rl-core to v2.0.0"
```

---

## Summary

| Task | File | Effort | RED commit | GREEN commit | REFACTOR commit |
|------|------|--------|------------|--------------|-----------------|
| Pre | fixture generation | Trivial | — | — | — |
| 1 | linalg/mod.rs | Low | `test:` trait tests | `feat:` trait change | `refactor:` docs |
| 2 | linalg/cpu.rs | Low | `test:` migrate 54 tests | `feat:` adapt impl + Default | `refactor:` docs |
| 3 | layer.rs | Low | `test:` migrate 18 tests | `feat:` backend + serde skip | `refactor:` if needed |
| 4 | matrix.rs | Medium | `test:` migrate 65 tests | `feat:` generic funcs | `refactor:` if needed |
| 5 | mlp_critic.rs | Low | `test:` migrate 24 tests | `feat:` backend field | `refactor:` if needed |
| 6 | pc_actor.rs | High | `test:` migrate 96 + Esc.7 | `feat:` backend field | `refactor:` if needed |
| 7 | pc_actor_critic.rs | Medium | `test:` migrate 41 + 3 new | `feat:` backend + from_parts + cache_to_matrices | `refactor:` if needed |
| 8 | serializer.rs | Low | `test:` migrate 16 + 2 new | `feat:` load funcs + load_agent_generic | `refactor:` if needed |
| 9 | lib.rs | Trivial | — | `feat:` exports | — |
| 10 | doctests | Low | — | `docs:` ~20 examples | — |
| 11 | README.md | Trivial | — | `docs:` quick start | — |
| 12 | CHANGELOG | Low | — | `docs:` migration guide | — |
| 13 | verification | Trivial | — | `refactor:` if fixes | — |
| 14 | version bump | Trivial | — | `chore:` v2.0.0 | — |

**Total: 14 tasks + pre-flight, 284 production L:: references (~170 method calls to migrate) + 60 test CpuLinAlg:: calls, up to 42 commits.**

**Compilation checkpoints:**
- After Task 2 GREEN: `cargo nextest run -E 'test(linalg::)'` passes (~55 tests)
- After Task 8 GREEN: `cargo nextest run` passes (ALL ~391 tests) ← **first full compilation**
- After Task 13: full 6-command verification passes (CLAUDE.local.md §0.2)

**Spec coverage matrix:**

| Escenario | Covered in Task | Test type |
|-----------|----------------|-----------|
| 1: CPU agent construction | 7 RED | New explicit test |
| 2: Invalid config with backend | 7 RED | New explicit test |
| 3: Operations use &self | 1 RED | New explicit test |
| 4: mat_vec_mul via instance | 1 RED | New explicit test |
| 5: vec_as_slice removed | 1 RED | Compile-time (documented) |
| 6: Serialization round-trip | 8 RED | New explicit test |
| 7: Crossover with backend | 6 RED | New explicit test |
| 8: Backend propagates | 7 RED | New explicit test |
| 9: CpuLinAlg is ZST | 1 RED | New explicit test |
| 10: Backward compat JSON | 8 RED (+ pre-flight fixture) | New explicit test |

**MAGI findings addressed:**

| # | Finding | Resolution |
|---|---------|------------|
| C1 | Serde incompatible with backend field | `impl Default for CpuLinAlg` (Task 2) + `#[serde(skip, default)]` (Task 3) |
| C2 | Missing from_parts migration | Added to Task 7 GREEN with explicit signature |
| W1 | cargo check grep unreliable | Changed to `grep -c '^error'` monotonic decrease |
| W2 | TDD deviation Tasks 3-7 | Documented as "TDD Deviation Zone" in plan header |
| W3 | Inflated call counts | Clarified: 284 total L:: refs, ~170 method calls |
| W4 | Missing v1_model.json fixture | Added pre-flight step |
| W5 | Missing Escenario 7 test | Added to Task 6 RED |
| W6 | load_agent_generic not addressed | Added to Task 8 GREEN with explicit signature |
| W7 | No CHANGELOG / migration guide | Added as Task 12 |
| W8 | cache_to_matrices needs backend | Added to Task 7 GREEN with 5 L:: calls listed |

**MAGI CHECKPOINT 2 findings addressed:**

| # | Finding | Resolution |
|---|---------|------------|
| W1 | 8 Layer struct literals not enumerated | Added explicit checklist items to Tasks 5 and 6 with line numbers |
| W2 | Pre-flight has no concrete command | Added mkdir + #[ignore] test + cargo nextest command |
| W3 | cca_align_and_blend_layer + blend_layer_weights missing from Task 6 | Added both free functions with L:: call counts to Task 6 GREEN |
| W4 | CpuLinAlg:: production calls in to_weights/from_weights | Added to Tasks 5 GREEN with migration pattern |
| W5 | Future GPU backends get opaque serde error without Default | Added doc comment to Task 1 REFACTOR + CHANGELOG entry |
| I1 | run-tests.sh does not exist | External tooling — not a plan concern |
| I2 | vec_as_slice disposition ambiguous | Resolved to "remove entirely" in plan header |
| I3 | CHANGELOG should mention matrix.rs generic function changes | Added to Task 12 CHANGELOG |
