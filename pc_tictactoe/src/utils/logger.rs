// Author: Jose Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Logging and CSV metrics export.
//!
//! Provides console + file logging with level filtering, CSV metrics
//! writing for training analysis, and log file rotation with configurable
//! maximum backups.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Log level severity ordering.
///
/// Messages are written only if their level is >= the configured minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Verbose debugging information.
    Debug = 0,
    /// General informational messages.
    Info = 1,
    /// Training progress updates.
    Training = 2,
    /// Potential issues.
    Warning = 3,
    /// Errors requiring attention.
    Error = 4,
}

impl LogLevel {
    /// Parses a log level from a string (case-insensitive).
    ///
    /// Defaults to [`LogLevel::Info`] for unrecognized strings.
    ///
    /// # Parameters
    ///
    /// * `s` - Level name (e.g. "debug", "info", "training", "warning", "error").
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "debug" => Self::Debug,
            "info" => Self::Info,
            "training" => Self::Training,
            "warning" | "warn" => Self::Warning,
            "error" => Self::Error,
            _ => Self::Info,
        }
    }

    /// Returns the label string for this level.
    fn label(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Training => "TRAIN",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

/// Thread-safe logger with optional file and CSV output.
///
/// # Examples
///
/// ```no_run
/// use pc_tictactoe::utils::logger::{Logger, LogLevel};
///
/// let logger = Logger::new(LogLevel::Info, None, None, 3, 10_000_000);
/// logger.log(LogLevel::Info, "training started");
/// ```
pub struct Logger {
    /// Minimum level for messages to be written.
    min_level: LogLevel,
    /// Optional file handle for log output.
    file: Option<Arc<Mutex<BufWriter<File>>>>,
    /// Optional CSV handle for metrics output.
    csv: Option<Arc<Mutex<BufWriter<File>>>>,
    /// Path to the log file (for rotation).
    file_path: Option<String>,
    /// Maximum number of backup files to keep.
    max_backups: usize,
    /// Maximum log file size in bytes before rotation.
    max_size: u64,
}

impl Logger {
    /// Creates a new logger.
    ///
    /// Opens the log file and CSV file if paths are provided. Writes the CSV
    /// header row on creation.
    ///
    /// # Parameters
    ///
    /// * `min_level` - Minimum log level to output.
    /// * `file_path` - Optional path for log file.
    /// * `csv_path` - Optional path for CSV metrics file.
    /// * `max_backups` - Maximum rotated backup count.
    /// * `max_size` - File size threshold for rotation (bytes).
    pub fn new(
        min_level: LogLevel,
        file_path: Option<&str>,
        csv_path: Option<&str>,
        max_backups: usize,
        max_size: u64,
    ) -> Self {
        let file = file_path.map(|p| {
            if let Some(parent) = Path::new(p).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("failed to open log file");
            Arc::new(Mutex::new(BufWriter::new(f)))
        });

        let csv = csv_path.map(|p| {
            if let Some(parent) = Path::new(p).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("failed to open CSV file");
            // Write header if file is empty.
            let meta = f.metadata().expect("failed to read CSV metadata");
            if meta.len() == 0 {
                writeln!(f, "episode,win_rate,avg_reward,avg_surprise,curriculum_depth,timestamp")
                    .expect("failed to write CSV header");
            }
            Arc::new(Mutex::new(BufWriter::new(f)))
        });

        Self {
            min_level,
            file: file.clone(),
            csv,
            file_path: file_path.map(String::from),
            max_backups,
            max_size,
        }
    }

    /// Logs a message at the given level.
    ///
    /// Writes to stderr and (if configured) to the log file. Messages below
    /// the minimum level are silently dropped.
    ///
    /// # Parameters
    ///
    /// * `level` - Severity of this message.
    /// * `message` - The message text.
    pub fn log(&self, level: LogLevel, message: &str) {
        if level < self.min_level {
            return;
        }
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let line = format!("[{timestamp}] [{label}] {message}", label = level.label());
        eprintln!("{line}");

        if let Some(ref file) = self.file {
            self.maybe_rotate();
            if let Ok(mut f) = file.lock() {
                let _ = writeln!(f, "{line}");
                let _ = f.flush();
            }
        }
    }

    /// Writes a metrics row to the CSV file.
    ///
    /// # Parameters
    ///
    /// * `episode` - Current episode number.
    /// * `win_rate` - Win rate over recent window.
    /// * `avg_reward` - Average reward over recent window.
    /// * `avg_surprise` - Average surprise score.
    /// * `depth` - Current curriculum depth.
    pub fn log_metrics(
        &self,
        episode: usize,
        win_rate: f64,
        avg_reward: f64,
        avg_surprise: f64,
        depth: usize,
    ) {
        if let Some(ref csv) = self.csv {
            let timestamp = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S");
            if let Ok(mut f) = csv.lock() {
                let _ = writeln!(
                    f,
                    "{episode},{win_rate:.4},{avg_reward:.4},{avg_surprise:.4},{depth},{timestamp}"
                );
                let _ = f.flush();
            }
        }
    }

    /// Prints a simple progress bar to stderr.
    ///
    /// # Parameters
    ///
    /// * `current` - Current step.
    /// * `total` - Total steps.
    pub fn progress(&self, current: usize, total: usize) {
        if total == 0 {
            return;
        }
        let pct = (current as f64 / total as f64 * 100.0).min(100.0);
        let filled = (pct / 2.0) as usize;
        let bar: String = "=".repeat(filled) + &" ".repeat(50 - filled);
        eprint!("\r[{bar}] {pct:5.1}% ({current}/{total})");
        if current == total {
            eprintln!();
        }
        let _ = io::stderr().flush();
    }

    /// Rotates the log file if it exceeds `max_size`.
    fn maybe_rotate(&self) {
        let path = match &self.file_path {
            Some(p) => p,
            None => return,
        };
        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return,
        };
        if meta.len() < self.max_size {
            return;
        }
        // Rotate: shift existing backups.
        for i in (1..self.max_backups).rev() {
            let from = format!("{path}.{i}");
            let to = format!("{path}.{}", i + 1);
            let _ = fs::rename(&from, &to);
        }
        let backup = format!("{path}.1");
        let _ = fs::rename(path, &backup);

        // Remove excess backups.
        let excess = format!("{path}.{}", self.max_backups + 1);
        let _ = fs::remove_file(&excess);

        // Re-open the log file.
        if let Some(ref file) = self.file {
            if let Ok(mut guard) = file.lock() {
                if let Ok(f) = OpenOptions::new().create(true).append(true).open(path) {
                    *guard = BufWriter::new(f);
                }
            }
        }
    }
}

/// Thread-safe reference to a logger.
///
/// Wraps a [`Logger`] in `Arc<Mutex<>>` for concurrent access from
/// multiple threads.
pub type SharedLogger = Arc<Mutex<Logger>>;

/// Creates a [`SharedLogger`] from a logger instance.
///
/// # Parameters
///
/// * `logger` - The logger to wrap.
pub fn shared_logger(logger: Logger) -> SharedLogger {
    Arc::new(Mutex::new(logger))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("pc_logger_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_creates_log_file() {
        let dir = temp_dir();
        let log_path = dir.join("test.log");
        let _ = fs::remove_file(&log_path);

        let logger = Logger::new(LogLevel::Debug, Some(log_path.to_str().unwrap()), None, 3, 10_000_000);
        logger.log(LogLevel::Info, "hello");

        assert!(log_path.exists());
        let content = fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("hello"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_all_levels_written_at_debug() {
        let dir = temp_dir();
        let log_path = dir.join("levels.log");
        let _ = fs::remove_file(&log_path);

        let logger = Logger::new(LogLevel::Debug, Some(log_path.to_str().unwrap()), None, 3, 10_000_000);
        logger.log(LogLevel::Debug, "d");
        logger.log(LogLevel::Info, "i");
        logger.log(LogLevel::Training, "t");
        logger.log(LogLevel::Warning, "w");
        logger.log(LogLevel::Error, "e");

        let content = fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("[DEBUG]"));
        assert!(content.contains("[INFO]"));
        assert!(content.contains("[TRAIN]"));
        assert!(content.contains("[WARN]"));
        assert!(content.contains("[ERROR]"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_min_level_info_filters_debug_and_training() {
        let dir = temp_dir();
        let log_path = dir.join("filtered.log");
        let _ = fs::remove_file(&log_path);

        let logger = Logger::new(LogLevel::Info, Some(log_path.to_str().unwrap()), None, 3, 10_000_000);
        logger.log(LogLevel::Debug, "should_not_appear");
        logger.log(LogLevel::Info, "should_appear");

        let content = fs::read_to_string(&log_path).unwrap();
        assert!(!content.contains("should_not_appear"));
        assert!(content.contains("should_appear"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rotation_creates_backup() {
        let dir = temp_dir();
        let log_path = dir.join("rotate.log");
        let _ = fs::remove_file(&log_path);
        let _ = fs::remove_file(dir.join("rotate.log.1"));

        // Create logger with very small max_size to trigger rotation.
        let logger = Logger::new(
            LogLevel::Debug,
            Some(log_path.to_str().unwrap()),
            None,
            3,
            50, // 50 bytes
        );

        // Write enough to exceed 50 bytes.
        for i in 0..20 {
            logger.log(LogLevel::Info, &format!("message number {i} padding"));
        }

        let backup = dir.join("rotate.log.1");
        assert!(backup.exists(), "backup file should exist after rotation");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rotation_respects_max_backups() {
        let dir = temp_dir();
        let log_path = dir.join("rot_max.log");
        for i in 0..5 {
            let _ = fs::remove_file(dir.join(format!("rot_max.log.{i}")));
        }
        let _ = fs::remove_file(&log_path);

        let logger = Logger::new(
            LogLevel::Debug,
            Some(log_path.to_str().unwrap()),
            None,
            2,  // max 2 backups
            30, // very small
        );

        for i in 0..100 {
            logger.log(LogLevel::Info, &format!("line {i} with padding data here"));
        }

        // .1 and .2 should exist; .3 should not.
        assert!(dir.join("rot_max.log.1").exists());
        assert!(dir.join("rot_max.log.2").exists());
        assert!(!dir.join("rot_max.log.3").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_csv_export_creates_file_with_header() {
        let dir = temp_dir();
        let csv_path = dir.join("metrics.csv");
        let _ = fs::remove_file(&csv_path);

        let _logger = Logger::new(LogLevel::Info, None, Some(csv_path.to_str().unwrap()), 3, 10_000_000);

        assert!(csv_path.exists());
        let content = fs::read_to_string(&csv_path).unwrap();
        assert!(content.starts_with("episode,win_rate,avg_reward,avg_surprise,curriculum_depth,timestamp"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_csv_appends_one_row_per_call() {
        let dir = temp_dir();
        let csv_path = dir.join("append.csv");
        let _ = fs::remove_file(&csv_path);

        let logger = Logger::new(LogLevel::Info, None, Some(csv_path.to_str().unwrap()), 3, 10_000_000);
        logger.log_metrics(1, 0.5, 0.1, 0.05, 1);
        logger.log_metrics(2, 0.6, 0.2, 0.04, 1);

        let content = fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // 1 header + 2 data rows = 3 lines
        assert_eq!(lines.len(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_thread_safe_40_concurrent_writes() {
        use std::thread;

        let dir = temp_dir();
        let log_path = dir.join("concurrent.log");
        let _ = fs::remove_file(&log_path);

        let logger = shared_logger(Logger::new(
            LogLevel::Debug,
            Some(log_path.to_str().unwrap()),
            None,
            3,
            10_000_000,
        ));

        let mut handles = vec![];
        for i in 0..40 {
            let lg = Arc::clone(&logger);
            handles.push(thread::spawn(move || {
                if let Ok(l) = lg.lock() {
                    l.log(LogLevel::Info, &format!("thread {i}"));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let content = fs::read_to_string(&log_path).unwrap();
        let line_count = content.lines().count();
        assert_eq!(line_count, 40, "expected 40 log lines, got {line_count}");

        let _ = fs::remove_dir_all(&dir);
    }
}
