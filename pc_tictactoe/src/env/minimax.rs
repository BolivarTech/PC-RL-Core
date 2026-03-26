// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Minimax opponent with alpha-beta pruning for Tic-Tac-Toe.
//!
//! Provides [`MinimaxPlayer`] with configurable search depth (1–9),
//! move ordering, depth-aware scoring, and a transposition table
//! that persists across calls.
//!
//! # Examples
//!
//! ```
//! use pc_tictactoe::env::tictactoe::TicTacToe;
//! use pc_tictactoe::env::minimax::MinimaxPlayer;
//!
//! let mut game = TicTacToe::new();
//! let mut player = MinimaxPlayer::new(9);
//! let action = player.choose_action(&game);
//! assert!(game.valid_actions().contains(&action));
//! ```

use std::collections::HashMap;

use super::tictactoe::{GameResult, Player, TicTacToe};

/// Move ordering priority: center > corners > edges.
const MOVE_ORDER: [usize; 9] = [4, 0, 2, 6, 8, 1, 3, 5, 7];

/// Alpha-beta minimax player with transposition table.
///
/// Depth controls search strength: 1 (weak) to 9 (perfect play).
/// The transposition table persists across calls to [`choose_action`](MinimaxPlayer::choose_action),
/// accelerating repeated evaluations.
pub struct MinimaxPlayer {
    /// Maximum search depth.
    depth: usize,
    /// Cached board evaluations keyed by board hash.
    transposition_table: HashMap<u32, i8>,
}

impl MinimaxPlayer {
    /// Creates a new minimax player with the given search depth.
    ///
    /// # Arguments
    ///
    /// * `depth` - Search depth from 1 (weak) to 9 (perfect).
    ///
    /// # Panics
    ///
    /// Panics if `depth` is not in `[1, 9]`.
    pub fn new(depth: usize) -> Self {
        assert!(
            (1..=9).contains(&depth),
            "depth must be in [1, 9], got {depth}"
        );
        Self {
            depth,
            transposition_table: HashMap::new(),
        }
    }

    /// Returns the best action for the current player on `board`.
    ///
    /// Evaluates all valid moves using alpha-beta search up to the
    /// configured depth, with move ordering and transposition caching.
    ///
    /// # Panics
    ///
    /// Panics if the board has no valid actions.
    pub fn choose_action(&mut self, board: &TicTacToe) -> usize {
        let valid = board.valid_actions();
        assert!(!valid.is_empty(), "No valid actions on terminal board");

        let ordered = Self::order_moves(&valid);
        let mut best_action = ordered[0];
        let mut best_score: i8 = -120;

        for action in ordered {
            let mut clone = board.clone();
            clone.step(action).unwrap();
            // After our move, opponent is maximizing from their perspective,
            // but we evaluate from our perspective: negate opponent's score.
            let score = -self.alpha_beta(&clone, self.depth - 1, -120, -best_score, 1);
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        best_action
    }

    /// Returns the number of entries in the transposition table.
    pub fn transposition_table_len(&self) -> usize {
        self.transposition_table.len()
    }

    /// Alpha-beta search returning a score from the perspective of the
    /// player whose turn it is on `board`.
    ///
    /// Positive scores favor the current player; negative scores favor
    /// the opponent. Depth-aware scoring ensures faster wins rank higher.
    fn alpha_beta(
        &mut self,
        board: &TicTacToe,
        depth: usize,
        mut alpha: i8,
        beta: i8,
        depth_from_root: usize,
    ) -> i8 {
        // Terminal check
        if board.is_terminal() {
            return self.terminal_score(board, depth_from_root);
        }
        if depth == 0 {
            return 0; // heuristic: draw at depth limit
        }

        // Transposition table lookup
        let key = Self::board_key(board);
        if let Some(&cached) = self.transposition_table.get(&key) {
            return cached;
        }

        let valid = board.valid_actions();
        let ordered = Self::order_moves(&valid);
        let mut best: i8 = -120;

        for action in ordered {
            let mut clone = board.clone();
            clone.step(action).unwrap();
            let score = -self.alpha_beta(&clone, depth - 1, -beta, -alpha, depth_from_root + 1);
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }

        self.transposition_table.insert(key, best);
        best
    }

    /// Evaluates a terminal board in negamax convention.
    ///
    /// After a winning `step()`, `current_player()` remains the winner
    /// (the turn does not switch on terminal states). In negamax, the
    /// caller negates the returned score, so we return from the
    /// perspective of the logical next-to-move player:
    /// - If the winner is `current_player()` (the one who just moved),
    ///   the next-to-move player lost → negative score.
    /// - Draw → 0.
    fn terminal_score(&self, board: &TicTacToe, depth_from_root: usize) -> i8 {
        match board.result() {
            GameResult::Draw => 0,
            GameResult::Win(winner) => {
                let d = depth_from_root as i8;
                if winner == board.current_player() {
                    // Winner is still marked as current (step doesn't switch
                    // on win). The logical next-to-move player lost.
                    d - 10 // negative: bad for the evaluating side
                } else {
                    // Shouldn't happen in normal play, but handle for safety.
                    10 - d // positive: good for the evaluating side
                }
            }
            GameResult::InProgress => 0,
        }
    }

    /// Orders moves by priority: center > corners > edges.
    fn order_moves(valid: &[usize]) -> Vec<usize> {
        let mut ordered: Vec<usize> = Vec::with_capacity(valid.len());
        for &m in &MOVE_ORDER {
            if valid.contains(&m) {
                ordered.push(m);
            }
        }
        ordered
    }

    /// Computes a unique hash key for the board state including current player.
    ///
    /// Layout: bit 31 = current player flag, bits 16–24 = board_x, bits 0–8 = board_o.
    fn board_key(board: &TicTacToe) -> u32 {
        let (bx, bo) = board.bitboards();
        let player_bit: u32 = match board.current_player() {
            Player::One => 0,
            Player::Two => 1 << 31,
        };
        player_bit | ((bx as u32) << 16) | (bo as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: play a sequence of moves on a fresh board.
    fn play_moves(moves: &[usize]) -> TicTacToe {
        let mut game = TicTacToe::new();
        for &m in moves {
            game.step(m).unwrap();
        }
        game
    }

    #[test]
    fn test_minimax_takes_immediate_win() {
        // P1 has cells 0,1. Winning move is 2 (row 0).
        // P2 has cells 3,4. It's P1's turn.
        let game = play_moves(&[0, 3, 1, 4]);
        // P1's turn, should take cell 2
        let mut player = MinimaxPlayer::new(9);
        let action = player.choose_action(&game);
        assert_eq!(action, 2, "Minimax should take the winning move");
    }

    #[test]
    fn test_minimax_blocks_opponent_win() {
        // P1 has cells 0,1 (about to win with 2). P2 must block.
        // P2 has cell 4. It's P2's turn after P1 plays.
        // Moves: P1=0, P2=4, P1=1 → P2's turn, must block at 2.
        let game = play_moves(&[0, 4, 1]);
        // P2's turn; P1 threatens row 0 (cells 0,1 → needs 2)
        let mut player = MinimaxPlayer::new(9);
        let action = player.choose_action(&game);
        assert_eq!(action, 2, "Minimax should block opponent's winning move");
    }

    #[test]
    fn test_minimax_perfect_never_loses_as_first_player() {
        // Perfect minimax (depth 9) as P1 vs perfect minimax as P2 → draw
        let mut p1 = MinimaxPlayer::new(9);
        let mut p2 = MinimaxPlayer::new(9);
        let mut game = TicTacToe::new();

        while !game.is_terminal() {
            let action = match game.current_player() {
                Player::One => p1.choose_action(&game),
                Player::Two => p2.choose_action(&game),
            };
            game.step(action).unwrap();
        }

        assert_ne!(
            game.result(),
            GameResult::Win(Player::Two),
            "Perfect minimax as P1 should never lose"
        );
    }

    #[test]
    fn test_minimax_perfect_never_loses_as_second_player() {
        // Same test but checking P2 never loses
        let mut p1 = MinimaxPlayer::new(9);
        let mut p2 = MinimaxPlayer::new(9);
        let mut game = TicTacToe::new();

        while !game.is_terminal() {
            let action = match game.current_player() {
                Player::One => p1.choose_action(&game),
                Player::Two => p2.choose_action(&game),
            };
            game.step(action).unwrap();
        }

        assert_ne!(
            game.result(),
            GameResult::Win(Player::One),
            "Perfect minimax as P2 should never lose"
        );
    }

    #[test]
    fn test_minimax_vs_minimax_always_draws() {
        let mut p1 = MinimaxPlayer::new(9);
        let mut p2 = MinimaxPlayer::new(9);
        let mut game = TicTacToe::new();

        while !game.is_terminal() {
            let action = match game.current_player() {
                Player::One => p1.choose_action(&game),
                Player::Two => p2.choose_action(&game),
            };
            game.step(action).unwrap();
        }

        assert_eq!(
            game.result(),
            GameResult::Draw,
            "Two perfect minimax players should always draw"
        );
    }

    #[test]
    fn test_choose_action_always_returns_valid_cell() {
        // Test on multiple board states
        let mut player = MinimaxPlayer::new(9);

        // Empty board
        let game = TicTacToe::new();
        let action = player.choose_action(&game);
        assert!(game.valid_actions().contains(&action));

        // After some moves
        let states = vec![
            vec![4],
            vec![0, 4],
            vec![0, 4, 8],
            vec![0, 1, 2, 3],
            vec![4, 0, 8, 2, 6],
        ];

        for moves in &states {
            let board = play_moves(moves);
            if !board.is_terminal() {
                let a = player.choose_action(&board);
                assert!(
                    board.valid_actions().contains(&a),
                    "Action {} not valid for state {:?}",
                    a,
                    moves
                );
            }
        }
    }

    #[test]
    fn test_transposition_table_populated_after_call() {
        let mut player = MinimaxPlayer::new(9);
        assert_eq!(player.transposition_table_len(), 0);

        let game = TicTacToe::new();
        player.choose_action(&game);

        assert!(
            player.transposition_table_len() > 0,
            "Transposition table should be populated after search"
        );
    }

    #[test]
    fn test_move_ordering_prefers_center() {
        let valid = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let ordered = MinimaxPlayer::order_moves(&valid);
        assert_eq!(ordered[0], 4, "Center should be evaluated first");
    }

    #[test]
    fn test_depth_1_returns_valid_action() {
        let mut player = MinimaxPlayer::new(1);
        let game = TicTacToe::new();
        let action = player.choose_action(&game);
        assert!(
            game.valid_actions().contains(&action),
            "Depth-1 minimax should return a valid action"
        );
    }

    #[test]
    #[should_panic(expected = "depth must be")]
    fn test_minimax_zero_depth_panics() {
        MinimaxPlayer::new(0);
    }

    #[test]
    #[should_panic(expected = "depth must be")]
    fn test_minimax_depth_over_9_panics() {
        MinimaxPlayer::new(10);
    }
}
