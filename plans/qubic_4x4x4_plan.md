# Plan: Qubic 4×4×4 3D Tic-Tac-Toe Implementation

## Context

Extend the PC-TicTacToe project to support Qubic (4×4×4 3D Tic-Tac-Toe). This is the next domain for validating the DPC architecture on a significantly more complex problem: 64 board positions, ~10²⁰ states, 76 winning lines. The current 3×3 TTT has 9 positions and 8 winning lines.

The `pc_core` library is domain-agnostic — no changes needed. All changes are in `pc_tictactoe` (environment, training, config).

## Architecture

### Board Representation

```
4 levels × 4 rows × 4 cols = 64 positions
Index: level * 16 + row * 4 + col (0..63)

Level 0 (bottom):     Level 1:          Level 2:          Level 3 (top):
 0  1  2  3          16 17 18 19       32 33 34 35       48 49 50 51
 4  5  6  7          20 21 22 23       36 37 38 39       52 53 54 55
 8  9 10 11          24 25 26 27       40 41 42 43       56 57 58 59
12 13 14 15          28 29 30 31       44 45 46 47       60 61 62 63
```

### Winning Lines (76 total)

- **Rows** (horizontal within level): 4 per level × 4 levels = 16
- **Columns** (vertical within level): 4 per level × 4 levels = 16
- **Level diagonals** (within level): 2 per level × 4 levels = 8
- **Pillars** (vertical through levels): 4 × 4 = 16
- **Level-spanning diagonals** (through levels, along rows): 4 × 2 = 8
- **Level-spanning diagonals** (through levels, along cols): 4 × 2 = 8
- **Space diagonals** (corner to corner through 3D): 4

Total: 16 + 16 + 8 + 16 + 8 + 8 + 4 = 76

### Network Topology

```
Input (64) → [Hidden layers, configurable] → Output (64) → Softmax → Action
```

- Input: 64 board positions encoded as f64 (+1 own, -1 opponent, 0 empty)
- Output: 64 action logits (one per position)
- Hidden: configurable via TOML, suggested starting point: [128, softsign] or [64, 64, softsign] with residual
- Critic input: 64 + sum(hidden_sizes)

### Minimax Opponent

Full minimax to depth 64 is infeasible (~10²⁰ states). Use alpha-beta pruning with configurable depth limit. At low depths (1-4), alpha-beta is fast. At higher depths, evaluation heuristic needed for non-terminal states.

**Evaluation heuristic** for non-terminal states:
- Count lines where player has pieces and opponent doesn't (potential wins)
- Weight by number of pieces already in line (2-in-a-row > 1-in-a-row)
- Center positions more valuable (participate in more lines)

## Critical Files

### New Files

| File | Purpose |
|------|---------|
| `pc_tictactoe/src/env/qubic.rs` | Game rules, board state, valid actions, winning lines |
| `pc_tictactoe/src/env/qubic_minimax.rs` | Alpha-beta minimax with depth limit and evaluation heuristic |

### Modified Files

| File | Changes |
|------|---------|
| `pc_tictactoe/src/env/mod.rs` | Add `pub mod qubic; pub mod qubic_minimax;` |
| `pc_tictactoe/src/ui/cli.rs` | Add `train-qubic` subcommand or `--game qubic` flag |
| `pc_tictactoe/config.toml` | Qubic-specific config example (or separate `qubic_config.toml`) |

### Unchanged

| File | Reason |
|------|--------|
| `pc_core/src/*` | Domain-agnostic — works with any input/output size |
| `pc_tictactoe/src/training/trainer.rs` | Already generic over environment interface |

## TDD Implementation Phases

### Phase 1: Qubic Board (env/qubic.rs)

**File header**: `// Author: Julian Bolivar`, `// Version: 1.0.0`, `// Date: YYYY-MM-DD`

#### Cycle 1.1: Board struct and basic operations
**RED**: Tests for:
- `Qubic::new()` creates empty 64-position board
- `board_size()` returns 64
- `current_player()` returns Player::One initially
- `is_terminal()` returns false on empty board
- `valid_actions()` returns all 64 positions on empty board

**GREEN**: Implement `Qubic` struct with:
```rust
pub struct Qubic {
    board: [Option<Player>; 64],
    current_player: Player,
    move_count: usize,
}
```

#### Cycle 1.2: Move execution
**RED**: Tests for:
- `step(pos)` places piece at position
- `step(pos)` switches current player
- `step(occupied)` returns error
- `step(pos)` removes position from valid_actions
- `valid_actions()` shrinks after each move

**GREEN**: Implement `step()` with validation.

#### Cycle 1.3: Win detection
**RED**: Tests for:
- Horizontal 4-in-a-row detected (within level)
- Vertical 4-in-a-row detected (within level)
- Diagonal 4-in-a-row detected (within level)
- Pillar 4-in-a-row detected (through levels)
- Space diagonal 4-in-a-row detected
- Level-spanning diagonal detected
- Draw detection (all 64 filled, no winner)
- No false win on partial board

**GREEN**: Implement winning line lookup table (76 lines × 4 positions each). Check all lines on each move.

```rust
const WINNING_LINES: [[usize; 4]; 76] = [
    // Rows (16): level 0
    [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
    // ... level 1-3
    // Columns (16)
    // Level diagonals (8)
    // Pillars (16)
    // Level-spanning diagonals (16)
    // Space diagonals (4)
];
```

#### Cycle 1.4: Board encoding for neural network
**RED**: Tests for:
- `board_as_f64(player)` returns [f64; 64]
- Own pieces encoded as +1.0, opponent as -1.0, empty as 0.0
- Encoding flips perspective based on player argument

**GREEN**: Implement `board_as_f64()`.

#### Cycle 1.5: Game result
**RED**: Tests for:
- `game_result()` returns None during play
- `game_result()` returns Win(player) after 4-in-a-row
- `game_result()` returns Draw after 64 moves with no winner
- `reset()` clears board

**GREEN**: Implement game result tracking.

### Phase 2: Qubic Minimax (env/qubic_minimax.rs)

#### Cycle 2.1: Evaluation heuristic
**RED**: Tests for:
- Empty board evaluates to 0
- Board with player advantage evaluates positive
- Board with opponent advantage evaluates negative
- Terminal win evaluates to large positive/negative

**GREEN**: Implement heuristic:
```rust
fn evaluate(board: &Qubic, player: Player) -> f64 {
    let mut score = 0.0;
    for line in &WINNING_LINES {
        let (own, opp, empty) = count_pieces(board, line, player);
        if opp == 0 && own > 0 {
            score += 10.0_f64.powi(own as i32);  // 10, 100, 1000
        }
        if own == 0 && opp > 0 {
            score -= 10.0_f64.powi(opp as i32);
        }
    }
    score
}
```

#### Cycle 2.2: Alpha-beta minimax
**RED**: Tests for:
- Depth 1: chooses winning move if available
- Depth 1: blocks opponent winning move
- Depth 2: chooses forced win in 2
- Returns valid action always

**GREEN**: Implement alpha-beta with depth limit:
```rust
pub struct QubicMinimax {
    depth: usize,
}

impl QubicMinimax {
    pub fn choose_action(&self, board: &Qubic) -> usize {
        // Alpha-beta search to self.depth
    }
}
```

#### Cycle 2.3: Performance validation
**RED**: Tests for:
- Depth 1-3 completes in < 1 second
- Depth 4 completes in reasonable time
- Never returns invalid action

**GREEN**: Optimize with move ordering (center first, then edges, then corners).

### Phase 3: Training Integration

#### Cycle 3.1: Qubic trainer
**RED**: Tests for:
- Trainer creates with Qubic environment
- Single episode runs without panic
- Agent learns (loss decreases over episodes)

**GREEN**: Either:
- **Option A**: Generalize existing `Trainer` to accept any environment trait
- **Option B**: Create `QubicTrainer` parallel to `Trainer`

Option A is preferred if the interface is compatible. The key methods are:
- `env.reset()`
- `env.current_player()`
- `env.board_as_f64(player)`
- `env.valid_actions()`
- `env.step(action)`
- `env.is_terminal()`
- `env.game_result()` or `env.episode_outcome()`

If `Trainer` can be generic over an `Environment` trait, this is the cleanest approach.

#### Cycle 3.2: CLI integration
**RED**: Tests for:
- `cargo run -- train-qubic -c qubic_config.toml` parses
- Config loads with 64 input/output size

**GREEN**: Add `TrainQubic` subcommand or `--game` flag.

#### Cycle 3.3: Qubic config
**RED**: Tests for:
- Default qubic config validates
- Input/output size = 64
- Critic input size matches

**GREEN**: Create `qubic_config.toml`:
```toml
[agent.actor]
input_size = 64
output_size = 64
output_activation = "linear"
alpha = 0.03
max_steps = 5
lr_weights = 0.005
local_lambda = 0.999
residual = true
rezero_init = 0.1

[[agent.actor.hidden_layers]]
size = 128
activation = "softsign"

[agent.critic]
input_size = 192  # 64 + 128
lr = 0.005

[[agent.critic.hidden_layers]]
size = 128
activation = "softsign"

[training]
episodes = 200000
seed = 42

[curriculum]
advance_threshold = 0.95
window_size = 2000
```

### Phase 4: Environment Trait (Optional but Recommended)

If Phase 3 reveals significant duplication between TTT and Qubic trainers:

#### Cycle 4.1: Extract Environment trait
```rust
pub trait Environment {
    fn reset(&mut self);
    fn current_player(&self) -> Player;
    fn board_as_f64(&self, player: Player) -> Vec<f64>;
    fn valid_actions(&self) -> Vec<usize>;
    fn step(&mut self, action: usize) -> Result<(), GameError>;
    fn is_terminal(&self) -> bool;
    fn game_result(&self) -> Option<GameResult>;
}
```

#### Cycle 4.2: Implement trait for TicTacToe and Qubic
#### Cycle 4.3: Generic Trainer<E: Environment>

## Verification

### Per-phase verification (CLAUDE.local.md)
```bash
cargo test                              # All pass, 0 fail
cargo clippy --tests -- -D warnings     # 0 warnings
cargo fmt --check                       # Clean
cargo build --release                   # Compiles
cargo doc --no-deps                     # No doc warnings
```

### Functional tests
```bash
# Qubic plays a full game without panic
cargo run --release -- train-qubic -c qubic_config.toml

# Agent improves over episodes (depth 1 → 2 advancement)
# Minimax at depth 1-3 is fast enough for training
```

### Expected training timeline
- Depth 1-2: fast (similar to TTT)
- Depth 3-4: slow but feasible with alpha-beta
- Depth 5+: may need heuristic-only opponent or self-play

## Dependencies

No new dependencies. Reuses `pc_core` as-is. Only new code is in `pc_tictactoe`.

## Risks

1. **Minimax too slow at depth 4+**: Alpha-beta pruning with move ordering should keep depth 3-4 feasible. Beyond that, curriculum may stall.
2. **Network capacity**: 64 inputs may need larger hidden layers than 27. Start with 128, experiment from there.
3. **Training time**: 50,000 episodes may not be enough for 64-position game. Plan for 200,000+.
4. **Winning line enumeration**: Must enumerate all 76 lines correctly. Off-by-one errors are likely. Unit test each category separately.

## Estimated Effort

| Phase | Cycles | Complexity |
|-------|--------|------------|
| Phase 1: Board | 5 | Medium — careful line enumeration |
| Phase 2: Minimax | 3 | Medium — alpha-beta + heuristic |
| Phase 3: Integration | 3 | Low-Medium — reuse existing training |
| Phase 4: Trait (optional) | 3 | Medium — refactoring |
| **Total** | **11-14** | |
