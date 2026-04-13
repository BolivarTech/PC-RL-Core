# TD(n) — N-Step Temporal Difference Returns

**Version:** 1.0
**Date:** 2026-04-11
**Author:** Julian Bolivar
**Crate:** `pc-rl-core`
**Status:** Implemented on branch `TD_n`, pending merge to `main`

---

## 1. Overview

TD(n) extends the existing TD(0) learning in `step_masked()` with configurable
n-step returns. Instead of bootstrapping from the immediate next state, the agent
accumulates `n` steps of real rewards before bootstrapping with V(s_{t+n}):

```
TD(0):  target = r_t + gamma * V(s_{t+1})
TD(n):  target = G_n + gamma^n * V(s_{t+n})
TD(oo): target = G_T                          (Monte Carlo / REINFORCE)
```

Where the n-step discounted return is:

```
G_n = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^(n-1) * r_{t+n-1}
```

### 1.1 Motivation

TD(0) produces noisy gradient signal for short-episode games (TicTacToe:
5-9 steps). Episodic REINFORCE provides better signal but higher variance
for long episodes. TD(n) is the middle ground:

| Method | Signal Quality | Variance | CL Compatible |
|---|---|---|---|
| TD(0) | Low (1-step bootstrap) | Low | Yes |
| TD(n) | Medium (n-step bootstrap) | Medium | Yes |
| REINFORCE | High (full return) | High | No (episodic) |

Experimental evidence from PC-TicTacToe (N=35 seeds):

| Method | Mean Depth |
|---|---|
| Episodic REINFORCE | 7.57 |
| TD(0) with CL | 6.71 |
| TD(0) without CL | 6.43 |

TD(n) with n=4-5 is expected to approach REINFORCE quality while retaining
the `step_masked()` infrastructure and all CL features.

---

## 2. Configuration

### 2.1 Config Field

```rust
/// Number of steps for TD(n) return computation.
/// 0 = standard TD(0) (default, zero overhead). No buffer allocated.
/// n > 0 = accumulate n real reward steps before bootstrapping with V(s_{t+n}).
///
/// Recommended values: 0 (TD(0)), 4-5 (TicTacToe), 8-10 (longer games).
/// td_steps=1 is rejected at construction — use 0 for TD(0) or >= 2 for multi-step.
///
/// Memory: O(td_steps x network_size) per agent. ~3KB/step for [27,27,18].
/// Large td_steps introduces stale-inference approximation (analogous to
/// experience replay); keep td_steps <= episode_length / 2.
///
/// NaN safety: non-finite rewards are rejected at buffer insertion time.
#[serde(default = "default_td_steps")]
pub td_steps: usize,
```

### 2.2 Validation Rules

| Condition | Action |
|---|---|
| `td_steps = 0` | Standard TD(0). Zero overhead. No buffer allocated. |
| `td_steps = 1` | **Rejected** — `PcError::ConfigValidation`. Use 0 or >= 2. |
| `td_steps >= 2` | N-step TD enabled. Buffer allocated. |
| No upper bound | Graceful degradation to Monte Carlo for short episodes. |

### 2.3 TOML Usage

```toml
[agent]
td_steps = 0      # TD(0) — default, single-step bootstrap
td_steps = 4      # TD(4) — good for TicTacToe (5-9 step episodes)
td_steps = 8      # TD(8) — good for Qubic (10-30 step episodes)
td_steps = 100    # TD(100) — effectively REINFORCE for short episodes
```

### 2.4 Backward Compatibility

- `td_steps` has `#[serde(default)]` defaulting to 0
- Legacy JSON configs without `td_steps` load as TD(0)
- Existing save files load without changes
- `td_steps=0` produces bit-identical behavior to pre-TD(n) code

---

## 3. Architecture

### 3.1 Data Structures

#### TdTransition

```rust
/// A single buffered transition for TD(n) computation.
/// Transient — not serialized. Cleared on reset_step(), terminal, and crossover.
struct TdTransition<L: LinAlg> {
    state: L::Vector,
    infer: InferResult<L>,
    action: usize,
    valid_actions: Vec<usize>,
    reward: f64,
}
```

#### Buffer

```rust
// Field in PcActorCritic<L>
td_buffer: VecDeque<TdTransition<L>>,
```

The buffer is initialized as `VecDeque::new()` in all constructors
(`new`, `crossover`, `from_parts`) and is never serialized.

### 3.2 Core Methods

#### compute_n_step_reward

```rust
/// Pure function — computes G = Sigma(gamma^i * r_i).
/// Takes gamma as parameter (no &self) to avoid borrow conflicts during flush.
fn compute_n_step_reward(gamma: f64, rewards: &[f64]) -> f64
```

Used in the non-terminal buffer-full path. For terminal flush, a more
efficient O(K) suffix-sum is used instead.

#### learn_continuous_inner

```rust
/// Internal learning method with explicit gamma_power and optional pre-computed V(s).
fn learn_continuous_inner(
    &mut self,
    input, infer, action, valid_actions, reward,
    next_input, next_infer,
    terminal: bool,
    gamma_power: f64,      // gamma^n for bootstrap (replaces gamma^1)
    pre_v_s: Option<f64>,  // pre-computed V(s) to bypass critic.forward()
) -> f64
```

- `learn_continuous` delegates with `gamma_power=gamma`, `pre_v_s=None`
- Non-terminal buffer-full calls with `gamma_power=gamma^n`, `pre_v_s=None`
- Terminal flush calls with `gamma_power=gamma^k`, `pre_v_s=Some(v_s_values[k])`

The `pre_v_s` parameter enables the flush to inject pre-computed V(s) values
that are not affected by weight updates from prior flush iterations, preventing
stale-estimate bias.

#### flush_td_buffer

```rust
/// Flushes all buffered transitions at episode end.
/// Pre-computes V(s) before weight updates to avoid stale-estimate bias.
/// Calls process_hysteresis after each learn step.
fn flush_td_buffer(&mut self, terminal_state: &[f64], terminal_infer: &InferResult<L>)
```

Uses O(K) suffix-sum: `g[k] = r[k] + gamma * g[k+1]`, computed right-to-left.

### 3.3 Control Flow

#### TD(0) Path (td_steps = 0)

```
step_masked(state, valid, reward, terminal):
  1. If state_prev exists:
     learn_continuous(prev -> current, reward, terminal)
     process_hysteresis()
  2. Infer, select action, store as prev
  3. If terminal: clear prev state
  4. Return action
```

Zero overhead — no buffer operations, no TdTransition allocation.

#### TD(n) Non-Terminal Path (td_steps > 0, terminal = false)

```
step_masked(state, valid, reward, terminal=false):
  1. If state_prev exists AND reward is finite:
     Push TdTransition(prev_state, prev_infer, prev_action, prev_valid, reward)
  2. If buffer.len() >= td_steps:
     Collect rewards from buffer
     G_n = compute_n_step_reward(gamma, rewards)
     Pop oldest transition from buffer
     learn_continuous_inner(oldest -> current, G_n, gamma^n, pre_v_s=None)
     process_hysteresis()
  3. Infer, select action, store as prev
  4. Return action
```

#### TD(n) Terminal Path (td_steps > 0, terminal = true)

```
step_masked(state, valid, reward, terminal=true):
  1. If state_prev exists AND reward is finite:
     Push TdTransition(prev_state, prev_infer, prev_action, prev_valid, reward)
  2. flush_td_buffer(current_state, current_infer):
     a. Drain buffer into local Vec
     b. Pre-compute V(s) for all transitions (before any weight update)
     c. Compute suffix-sum returns: g[k] = r[k] + gamma * g[k+1]
     d. For each transition k:
        learn_continuous_inner(state_k, G_k, terminal=true, pre_v_s=Some(v_s_k))
        process_hysteresis()
  3. Clear prev state
  4. Return action
```

---

## 4. Interaction with CL Features

All continuous learning features operate per `learn_continuous_inner` call
and are agnostic to the target computation method:

| CL Feature | TD(n) Interaction | Impact |
|---|---|---|
| **M1 Scale Range** | `surprise_scale` applied per learn call | None |
| **M2 Hysteresis** | Updated per learn call; fewer updates (1 per n steps) but higher quality signal | Improved signal quality |
| **M3 Consolidation** | Per-layer decay applied per learn call | None |
| **M4 EWC** | Fisher accumulation per learn call | None |
| **NaN Guard** | `td_error.is_finite()` check in learn_continuous_inner | Protects all paths |

### 4.1 Hysteresis During Terminal Flush

During terminal flush, `process_hysteresis()` is called after each
`learn_continuous_inner()` invocation, ensuring the EWMA signals see
every td_error from the flush. This maintains hysteresis signal
consistency regardless of buffer depth.

---

## 5. Edge Cases

### 5.1 Episode Shorter Than n Steps

If an episode has k < n steps, the buffer never fills during play. All k
transitions are flushed at terminal with progressively shorter returns:

```
Episode: 3 steps, td_steps=5
  Transition 0: G = r0 + gamma*r1 + gamma^2*r2  (3-step Monte Carlo)
  Transition 1: G = r1 + gamma*r2               (2-step Monte Carlo)
  Transition 2: G = r2                          (1-step Monte Carlo)
```

Short episodes gracefully degrade to full Monte Carlo returns.

### 5.2 NaN Reward

Non-finite rewards (`NaN`, `Inf`, `-Inf`) are rejected at buffer insertion
via `reward.is_finite()`. The transition is not buffered, creating a gap
in the TD(n) chain. The next valid transition pairs with the post-NaN state.
This is the correct behavior — dropping is safer than corrupting.

### 5.3 reset_step() Mid-Episode

Clears the buffer, discarding unbacked transitions. The next `step()` call
behaves as the first call of a new episode (no learning). Also clears
`state_prev`, `action_prev`, `infer_prev`, `valid_actions_prev`.

### 5.4 Crossover

`crossover()` creates a child with `td_buffer: VecDeque::new()`. Parent
buffers are not inherited. Crossover is an episode boundary — the child
starts with a clean buffer.

---

## 6. Memory and Performance

### 6.1 Memory Scaling

Each `TdTransition<L>` stores a full `InferResult<L>` (y_conv, latent_concat,
hidden_states, prediction_errors, tanh_components). Memory per transition:

| Network | Memory/Transition | td_steps=4 | td_steps=100 |
|---|---:|---:|---:|
| 1x18 | ~1 KB | ~4 KB | ~100 KB |
| [27,27,18] | ~3 KB | ~12 KB | ~300 KB |
| 1x1024 (GPU) | ~80 KB | ~320 KB | ~8 MB |

For typical RL networks (n <= 128 neurons), td_steps up to 100 uses < 1 MB.

### 6.2 Stale Inference

The plan buffers `InferResult` at time t and uses it for learning at time
t+n. By then, actor weights have been updated up to n times. The gradient
is computed from stale parameters — analogous to experience replay (DQN
uses far staler data).

**Impact:** For small n (4-5), staleness is minimal. For large n (50+),
gradient direction errors increase.

**Recommendation:** Keep `td_steps <= episode_length / 2`.

### 6.3 Computational Complexity

| Operation | TD(0) | TD(n) Buffer-Full | TD(n) Terminal Flush |
|---|---|---|---|
| Buffer push | None | O(1) | O(1) |
| N-step return | None | O(n) forward | O(K) suffix-sum |
| V(s) pre-compute | None | None | O(K) critic forwards |
| Learn calls per step | 1 | 1 per n steps | K at terminal |
| Total per episode (L steps) | L | L/n + K | L/n + K |

### 6.4 Zero Overhead Guarantee

When `td_steps=0`:
- `VecDeque::new()` allocates nothing
- The `if self.config.td_steps == 0` branch enters the exact original TD(0) code
- No `TdTransition` construction, no `compute_n_step_reward`, no buffer operations
- Bit-identical weight trajectories to pre-TD(n) code (verified by regression test)

---

## 7. Serialization

| Component | Serialized | Notes |
|---|---|---|
| `td_steps` config field | Yes | `#[serde(default)]` — legacy configs load as 0 |
| `td_buffer` | No | Transient mid-episode state |
| `TdTransition` | No | Private struct, not in save format |
| `ClState` | No changes | CL state unaffected by TD(n) |

---

## 8. Testing

### 8.1 Unit Tests (11 tests in `pc-rl-core`)

| Test | Purpose |
|---|---|
| `test_td0_unchanged_with_td_steps_zero` | Regression: td_steps=0 = bit-identical TD(0) |
| `test_td_n_return_computation` | G = r0 + gamma*r1 + gamma^2*r2 math verification |
| `test_td_n_return_single_step` | G = r0 for single reward |
| `test_td_n_return_empty` | G = 0 for empty rewards |
| `test_td_n_buffer_fills_at_n` | No learning until buffer reaches td_steps |
| `test_td_n_terminal_flush` | Short episode flushes all transitions at terminal |
| `test_td_n_reset_clears_buffer` | reset_step() empties td_buffer |
| `test_td_n_short_episode_monte_carlo` | Episode < n degrades to Monte Carlo |
| `test_td_n_nan_reward_rejected_at_buffer` | NaN reward doesn't enter buffer |
| `test_td_n_serialization_config` | td_steps survives JSON save/load round-trip |
| `test_td_n_gamma_power_bootstrap` | TD(2) produces different weights than TD(0) |

### 8.2 Planned Integration Tests (PC-TicTacToe)

| Test | Purpose |
|---|---|
| `test_td_n_completes_game` | Agent with td_steps=4 plays a full game without panic |
| `test_td_n_trains_above_random` | Agent learns to beat random after 500 episodes |

### 8.3 Follow-up Tests (tracked)

| Test | Purpose |
|---|---|
| `test_td_n_hysteresis_interaction` | Hysteresis transitions work with buffered updates |
| `test_td_n_hysteresis_frozen_buffer` | Buffer fills while frozen; stale transitions learned on wake |

---

## 9. Files Modified

| File | Change |
|---|---|
| `src/pc_actor_critic/config.rs` | `td_steps: usize` field + `default_td_steps()` |
| `src/pc_actor_critic/mod.rs` | `TdTransition`, `td_buffer`, `compute_n_step_reward`, `learn_continuous_inner`, `flush_td_buffer`, step_inner branching, reset_step buffer clear |
| `src/serializer.rs` | `td_steps: 0` in test helper `default_config()` |

---

## 10. Experimental Plan

| Experiment | Configuration | Metric |
|---|---|---|
| Baseline verification | td_steps=0, seed-test | Must match current TD(0) results |
| TD(4) on TicTacToe | 35 seeds, 200k episodes, 3-layer | Target: mean depth ~7.5 |
| TD(4) + CL | Above + M1+M2+M3 | Measure CL benefit with TD(4) |
| td_steps sweep | [0, 2, 3, 4, 5, 8] on TicTacToe | Find optimal n for 5-9 step episodes |
| Qubic application | TD(5-10) with CL | Longer episodes (10-30 steps) |

