// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Tic-Tac-Toe game environment using bitboard representation.
//!
//! Provides [`TicTacToe`] as the game state manager, [`Player`] for turn
//! tracking, and [`GameResult`] for outcome detection. Uses two `u16`
//! bitboards (one per player) with precomputed win masks for efficient
//! win detection.
//!
//! # Board Layout
//!
//! ```text
//! 0 | 1 | 2
//! ---------
//! 3 | 4 | 5
//! ---------
//! 6 | 7 | 8
//! ```
//!
//! # Examples
//!
//! ```
//! use pc_tictactoe::env::tictactoe::{TicTacToe, Player, GameResult};
//!
//! let mut game = TicTacToe::new();
//! assert_eq!(game.valid_actions().len(), 9);
//! let result = game.step(4).unwrap();
//! assert_eq!(result, GameResult::InProgress);
//! ```

/// All eight winning line bitmasks for a 3x3 board.
///
/// Each mask represents one winning pattern:
/// rows (3), columns (3), and diagonals (2).
const WIN_MASKS: [u16; 8] = [
    0b_000_000_111, // Row 0: cells 0,1,2
    0b_000_111_000, // Row 1: cells 3,4,5
    0b_111_000_000, // Row 2: cells 6,7,8
    0b_001_001_001, // Col 0: cells 0,3,6
    0b_010_010_010, // Col 1: cells 1,4,7
    0b_100_100_100, // Col 2: cells 2,5,8
    0b_100_010_001, // Diagonal: cells 0,4,8
    0b_001_010_100, // Anti-diagonal: cells 2,4,6
];

/// Represents which player is active or has won.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    /// First player (X).
    One,
    /// Second player (O).
    Two,
}

/// Outcome of the game at any point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    /// A player has won.
    Win(Player),
    /// All cells filled with no winner.
    Draw,
    /// Game is still ongoing.
    InProgress,
}

/// Tic-Tac-Toe game state using bitboard representation.
///
/// Two `u16` bitboards track pieces for each player. Bits 0–8 map
/// to board cells. Win detection uses precomputed bitmasks.
///
/// # Examples
///
/// ```
/// use pc_tictactoe::env::tictactoe::{TicTacToe, GameResult};
///
/// let mut game = TicTacToe::new();
/// game.step(0).unwrap(); // Player One at cell 0
/// game.step(3).unwrap(); // Player Two at cell 3
/// assert!(!game.is_terminal());
/// ```
#[derive(Debug, Clone)]
pub struct TicTacToe {
    /// Bitboard for Player One (X).
    board_x: u16,
    /// Bitboard for Player Two (O).
    board_o: u16,
    /// Player whose turn it is.
    current: Player,
    /// Cached game result.
    result: GameResult,
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl TicTacToe {
    /// Creates a new game with an empty board. Player One moves first.
    pub fn new() -> Self {
        Self {
            board_x: 0,
            board_o: 0,
            current: Player::One,
            result: GameResult::InProgress,
        }
    }

    /// Resets the game to initial state.
    pub fn reset(&mut self) {
        self.board_x = 0;
        self.board_o = 0;
        self.current = Player::One;
        self.result = GameResult::InProgress;
    }

    /// Returns the current player.
    pub fn current_player(&self) -> Player {
        self.current
    }

    /// Returns indices of empty cells (valid moves).
    ///
    /// Returns an empty vector if the game is terminal.
    pub fn valid_actions(&self) -> Vec<usize> {
        if self.is_terminal() {
            return Vec::new();
        }
        let occupied = self.board_x | self.board_o;
        let empty = !occupied & 0x1FF;
        (0..9).filter(|&i| empty & (1 << i) != 0).collect()
    }

    /// Places a piece at `action` for the current player.
    ///
    /// Returns the resulting [`GameResult`] or an error string if the
    /// action is invalid (out of range or cell already occupied).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `action >= 9` or the cell is already occupied.
    pub fn step(&mut self, action: usize) -> Result<GameResult, String> {
        if action >= 9 {
            return Err(format!("Action {} out of range [0, 9)", action));
        }
        let mask = 1u16 << action;
        if (self.board_x | self.board_o) & mask != 0 {
            return Err(format!("Cell {} is already occupied", action));
        }
        if self.is_terminal() {
            return Err("Game is already over".to_string());
        }

        match self.current {
            Player::One => self.board_x |= mask,
            Player::Two => self.board_o |= mask,
        }

        // Check for win
        let board = match self.current {
            Player::One => self.board_x,
            Player::Two => self.board_o,
        };
        if Self::has_winning_line(board) {
            self.result = GameResult::Win(self.current);
            return Ok(self.result);
        }

        // Check for draw (all 9 cells filled)
        if (self.board_x | self.board_o) == 0x1FF {
            self.result = GameResult::Draw;
            return Ok(self.result);
        }

        // Switch player
        self.current = match self.current {
            Player::One => Player::Two,
            Player::Two => Player::One,
        };

        Ok(GameResult::InProgress)
    }

    /// Returns `true` if the game has ended (win or draw).
    pub fn is_terminal(&self) -> bool {
        self.result != GameResult::InProgress
    }

    /// Returns the current game result.
    pub fn result(&self) -> GameResult {
        self.result
    }

    /// Encodes the board as 9 `f64` values from `perspective` player's view.
    ///
    /// - `+1.0` for cells occupied by the perspective player.
    /// - `-1.0` for cells occupied by the opponent.
    /// - `0.0` for empty cells.
    ///
    /// # Arguments
    ///
    /// * `perspective` - The player whose viewpoint to encode from.
    pub fn board_as_f64(&self, perspective: Player) -> [f64; 9] {
        let (mine, theirs) = match perspective {
            Player::One => (self.board_x, self.board_o),
            Player::Two => (self.board_o, self.board_x),
        };
        let mut out = [0.0f64; 9];
        for (i, cell) in out.iter_mut().enumerate() {
            let bit = 1u16 << i;
            if mine & bit != 0 {
                *cell = 1.0;
            } else if theirs & bit != 0 {
                *cell = -1.0;
            }
        }
        out
    }

    /// Returns the reward for the given player based on the current result.
    ///
    /// - `+1.0` if the player won.
    /// - `-1.0` if the player lost.
    /// - `0.0` for draw or in-progress.
    pub fn reward(&self, player: Player) -> f64 {
        match self.result {
            GameResult::Win(winner) => {
                if winner == player {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => 0.0,
        }
    }

    /// Checks whether a bitboard contains any winning line.
    fn has_winning_line(board: u16) -> bool {
        for &mask in &WIN_MASKS {
            if board & mask == mask {
                return true;
            }
        }
        false
    }

    /// Returns the win masks used for win detection.
    pub fn win_masks() -> &'static [u16; 8] {
        &WIN_MASKS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game_has_9_valid_actions() {
        let game = TicTacToe::new();
        assert_eq!(game.valid_actions().len(), 9);
    }

    #[test]
    fn test_after_one_move_has_8_valid_actions() {
        let mut game = TicTacToe::new();
        game.step(4).unwrap();
        assert_eq!(game.valid_actions().len(), 8);
        assert!(!game.valid_actions().contains(&4));
    }

    #[test]
    fn test_step_on_occupied_returns_error() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap();
        assert!(game.step(0).is_err());
    }

    #[test]
    fn test_step_out_of_range_returns_error() {
        let mut game = TicTacToe::new();
        assert!(game.step(9).is_err());
        assert!(game.step(100).is_err());
    }

    #[test]
    fn test_win_detection_all_8_patterns() {
        // Test all 8 win patterns for Player One
        let patterns: [(usize, usize, usize); 8] = [
            (0, 1, 2), // Row 0
            (3, 4, 5), // Row 1
            (6, 7, 8), // Row 2
            (0, 3, 6), // Col 0
            (1, 4, 7), // Col 1
            (2, 5, 8), // Col 2
            (0, 4, 8), // Diagonal
            (2, 4, 6), // Anti-diagonal
        ];

        for (i, &(a, b, c)) in patterns.iter().enumerate() {
            let mut game = TicTacToe::new();
            // Player One takes cells a, b, c with Player Two taking
            // filler cells in between (avoiding the winning line)
            let filler: Vec<usize> = (0..9)
                .filter(|&x| x != a && x != b && x != c)
                .collect();

            game.step(a).unwrap(); // P1
            game.step(filler[0]).unwrap(); // P2
            game.step(b).unwrap(); // P1
            game.step(filler[1]).unwrap(); // P2
            let result = game.step(c).unwrap(); // P1
            assert_eq!(
                result,
                GameResult::Win(Player::One),
                "Pattern {} ({},{},{}) should be a win",
                i,
                a,
                b,
                c
            );
        }
    }

    #[test]
    fn test_draw_detection() {
        // Sequence that leads to a draw:
        // X O X
        // X O O
        // O X X
        // Moves: 0(X),1(O),2(X),4(O),3(X),5(O),8(X),6(O),7(X)
        let mut game = TicTacToe::new();
        let moves = [0, 1, 2, 4, 3, 5, 8, 6, 7];
        for (i, &m) in moves.iter().enumerate() {
            let result = game.step(m).unwrap();
            if i < 8 {
                assert_eq!(result, GameResult::InProgress, "Move {} should be in progress", i);
            } else {
                assert_eq!(result, GameResult::Draw);
            }
        }
    }

    #[test]
    fn test_board_as_f64_player_one_perspective() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap(); // P1 at 0
        game.step(1).unwrap(); // P2 at 1
        let board = game.board_as_f64(Player::One);
        assert_eq!(board[0], 1.0);
        assert_eq!(board[1], -1.0);
        for &cell in &board[2..] {
            assert_eq!(cell, 0.0);
        }
    }

    #[test]
    fn test_board_as_f64_player_two_perspective_flipped() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap(); // P1 at 0
        game.step(1).unwrap(); // P2 at 1
        let board = game.board_as_f64(Player::Two);
        assert_eq!(board[0], -1.0); // P1's piece is opponent
        assert_eq!(board[1], 1.0); // P2's piece is self
        for &cell in &board[2..] {
            assert_eq!(cell, 0.0);
        }
    }

    #[test]
    fn test_player_alternates_after_each_step() {
        let mut game = TicTacToe::new();
        assert_eq!(game.current_player(), Player::One);
        game.step(0).unwrap();
        assert_eq!(game.current_player(), Player::Two);
        game.step(1).unwrap();
        assert_eq!(game.current_player(), Player::One);
    }

    #[test]
    fn test_is_terminal_true_after_win() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap(); // P1
        game.step(3).unwrap(); // P2
        game.step(1).unwrap(); // P1
        game.step(4).unwrap(); // P2
        game.step(2).unwrap(); // P1 wins row 0
        assert!(game.is_terminal());
    }

    #[test]
    fn test_valid_actions_empty_when_terminal() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap(); // P1
        game.step(3).unwrap(); // P2
        game.step(1).unwrap(); // P1
        game.step(4).unwrap(); // P2
        game.step(2).unwrap(); // P1 wins row 0
        assert!(game.valid_actions().is_empty());
    }

    #[test]
    fn test_reset_clears_board_and_state() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap();
        game.step(3).unwrap();
        game.step(1).unwrap();
        game.step(4).unwrap();
        game.step(2).unwrap(); // P1 wins
        assert!(game.is_terminal());

        game.reset();
        assert!(!game.is_terminal());
        assert_eq!(game.current_player(), Player::One);
        assert_eq!(game.valid_actions().len(), 9);
    }

    #[test]
    fn test_win_reward_is_plus_one() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap(); // P1
        game.step(3).unwrap(); // P2
        game.step(1).unwrap(); // P1
        game.step(4).unwrap(); // P2
        game.step(2).unwrap(); // P1 wins
        assert_eq!(game.reward(Player::One), 1.0);
    }

    #[test]
    fn test_loss_reward_is_minus_one() {
        let mut game = TicTacToe::new();
        game.step(0).unwrap(); // P1
        game.step(3).unwrap(); // P2
        game.step(1).unwrap(); // P1
        game.step(4).unwrap(); // P2
        game.step(2).unwrap(); // P1 wins
        assert_eq!(game.reward(Player::Two), -1.0);
    }

    #[test]
    fn test_bitboard_win_masks_cover_all_8_patterns() {
        let masks = TicTacToe::win_masks();
        assert_eq!(masks.len(), 8);
        // Verify specific mask values
        assert_eq!(masks[0], 7);   // Row 0
        assert_eq!(masks[1], 56);  // Row 1
        assert_eq!(masks[2], 448); // Row 2
        assert_eq!(masks[3], 73);  // Col 0
        assert_eq!(masks[4], 146); // Col 1
        assert_eq!(masks[5], 292); // Col 2
        assert_eq!(masks[6], 273); // Diagonal
        assert_eq!(masks[7], 84);  // Anti-diagonal
    }
}
