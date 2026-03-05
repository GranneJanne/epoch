use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Mutex;

use color_eyre::{Result, eyre::ContextCompat};
use notify::Watcher;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::config::Config;
use crate::parsers::LogParser;
use crate::parsers::csv::CsvParser;
use crate::parsers::detect_parser;
use crate::parsers::hf_trainer::parse_trainer_state;
use crate::parsers::jsonl::JsonlParser;
use crate::parsers::regex_parser::RegexParser;
use crate::types::TrainingMetrics;

struct CsvBootstrapParser {
    parser: Mutex<Option<CsvParser>>,
}

impl CsvBootstrapParser {
    fn new() -> Self {
        Self {
            parser: Mutex::new(None),
        }
    }
}

impl LogParser for CsvBootstrapParser {
    fn parse_line(&self, line: &str) -> Result<Option<TrainingMetrics>> {
        let mut guard = self
            .parser
            .lock()
            .map_err(|_| color_eyre::eyre::eyre!("csv parser mutex poisoned"))?;

        if let Some(parser) = guard.as_ref() {
            return parser.parse_line(line);
        }

        if let Ok(parser) = CsvParser::new(line) {
            *guard = Some(parser);
        }

        Ok(None)
    }
}

enum AutoState {
    Undetected,
    Jsonl,
    Csv(CsvParser),
}

struct AutoDetectingParser {
    state: Mutex<AutoState>,
}

impl AutoDetectingParser {
    fn new() -> Self {
        Self {
            state: Mutex::new(AutoState::Undetected),
        }
    }
}

impl LogParser for AutoDetectingParser {
    fn parse_line(&self, line: &str) -> Result<Option<TrainingMetrics>> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| color_eyre::eyre::eyre!("auto parser mutex poisoned"))?;

        match &mut *guard {
            AutoState::Undetected => {
                let jsonl = JsonlParser;
                if let Some(metrics) = jsonl.parse_line(line)? {
                    *guard = AutoState::Jsonl;
                    return Ok(Some(metrics));
                }

                if let Ok(csv) = CsvParser::new(line) {
                    *guard = AutoState::Csv(csv);
                    return Ok(None);
                }

                Ok(None)
            }
            AutoState::Jsonl => JsonlParser.parse_line(line),
            AutoState::Csv(csv) => csv.parse_line(line),
        }
    }
}

fn read_sample_lines(path: &PathBuf, max_lines: usize) -> Vec<String> {
    let Ok(file) = std::fs::File::open(path) else {
        return vec![];
    };

    BufReader::new(file)
        .lines()
        .map_while(|line_result| line_result.ok())
        .take(max_lines)
        .collect()
}

fn read_first_non_empty_line(path: &PathBuf) -> Result<String> {
    let file = std::fs::File::open(path)?;
    for line in BufReader::new(file)
        .lines()
        .map_while(|line_result| line_result.ok())
    {
        if !line.trim().is_empty() {
            return Ok(line);
        }
    }

    color_eyre::eyre::bail!("csv parser requires a non-empty header line")
}

pub fn create_parser(config: &Config) -> Result<Box<dyn LogParser + Send>> {
    match config.parser.as_str() {
        "jsonl" => Ok(Box::new(JsonlParser)),
        "csv" => {
            if let Some(path) = config.log_file.as_ref()
                && let Ok(header) = read_first_non_empty_line(path)
            {
                return Ok(Box::new(CsvParser::new(&header)?));
            }

            Ok(Box::new(CsvBootstrapParser::new()))
        }
        "regex" => {
            let pattern = config
                .regex_pattern
                .as_deref()
                .context("regex_pattern required when parser is 'regex'")?;
            Ok(Box::new(RegexParser::new(pattern)?))
        }
        "auto" => {
            if let Some(path) = config.log_file.as_ref() {
                let sample_lines = read_sample_lines(path, 20);
                if !sample_lines.is_empty() {
                    let sample_refs: Vec<&str> = sample_lines.iter().map(String::as_str).collect();
                    return Ok(detect_parser(&sample_refs));
                }
            }
            Ok(Box::new(AutoDetectingParser::new()))
        }
        _ => Ok(Box::new(JsonlParser)),
    }
}

pub fn spawn_stdin_reader(
    parser: Box<dyn LogParser + Send>,
    tx: mpsc::Sender<TrainingMetrics>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let stdin = tokio::io::stdin();
        let reader = BufReader::new(stdin);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            match parser.parse_line(&line) {
                Ok(Some(metrics)) => {
                    if tx.send(metrics).await.is_err() {
                        break;
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    tracing::debug!("parse error: {err}");
                }
            }
        }
    })
}

pub fn spawn_file_watcher(
    path: PathBuf,
    parser: Box<dyn LogParser + Send>,
    tx: mpsc::Sender<TrainingMetrics>,
) -> Result<JoinHandle<()>> {
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        color_eyre::eyre::bail!("parent directory does not exist: {}", parent.display());
    }

    Ok(tokio::spawn(async move {
        use notify::{Config as NotifyConfig, RecommendedWatcher, RecursiveMode};

        // NOTE: notify crate requires std::sync::mpsc::Sender (unbounded).
        // Back-pressure is applied via the bounded tokio channel bridge below.
        let (notify_tx, notify_rx) = std::sync::mpsc::channel();
        let _watcher_guard = {
            let mut watcher = match RecommendedWatcher::new(notify_tx, NotifyConfig::default()) {
                Ok(watcher) => watcher,
                Err(err) => {
                    tracing::error!("failed to create file watcher: {err}");
                    return;
                }
            };

            let watch_path = if path.exists() {
                path.as_path()
            } else {
                path.parent().unwrap_or(path.as_path())
            };

            if let Err(err) = watcher.watch(watch_path, RecursiveMode::NonRecursive) {
                tracing::error!("failed to watch path {}: {err}", watch_path.display());
                return;
            }

            watcher
        };

        let mut position: u64 = 0;

        if path.exists() {
            if let Ok(file) = std::fs::File::open(&path) {
                let reader = BufReader::new(file);
                for line in reader.lines().map_while(|line_result| line_result.ok()) {
                    if let Ok(Some(metrics)) = parser.parse_line(&line)
                        && tx.send(metrics).await.is_err()
                    {
                        return;
                    }
                }
                position = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            }
        } else {
            tracing::info!("waiting for {} to be created...", path.display());
        }

        let (async_tx, mut async_rx) = mpsc::channel::<()>(16);
        tokio::task::spawn_blocking(move || {
            for event in notify_rx.into_iter().flatten() {
                use notify::event::{EventKind, ModifyKind};

                if matches!(
                    event.kind,
                    EventKind::Modify(ModifyKind::Data(_))
                        | EventKind::Modify(ModifyKind::Any)
                        | EventKind::Create(_)
                ) && async_tx.blocking_send(()).is_err()
                {
                    break;
                }
            }
        });

        while async_rx.recv().await.is_some() {
            if !path.exists() {
                continue;
            }

            let current_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

            if current_size < position {
                position = 0;
            }

            if current_size > position
                && let Ok(mut file) = std::fs::File::open(&path)
            {
                if file.seek(SeekFrom::Start(position)).is_ok() {
                    let reader = BufReader::new(&file);
                    for line in reader.lines().map_while(|line_result| line_result.ok()) {
                        if let Ok(Some(metrics)) = parser.parse_line(&line)
                            && tx.send(metrics).await.is_err()
                        {
                            return;
                        }
                    }
                }

                position = current_size;
            }
        }
    }))
}

pub fn spawn_trainer_state_poller(
    path: PathBuf,
    tx: mpsc::Sender<TrainingMetrics>,
    interval: std::time::Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        let mut last_modified = std::time::UNIX_EPOCH;
        let mut last_emitted_step: Option<u64> = None;

        loop {
            ticker.tick().await;

            let Ok(metadata) = std::fs::metadata(&path) else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };

            if modified <= last_modified {
                continue;
            }
            last_modified = modified;

            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(metrics_list) = parse_trainer_state(&content) else {
                continue;
            };

            let newest_step = metrics_list.iter().filter_map(|m| m.step).max();
            if let (Some(previous), Some(newest)) = (last_emitted_step, newest_step)
                && newest < previous
            {
                last_emitted_step = None;
            }

            for metrics in metrics_list {
                let should_send = metrics.step.is_some_and(|step| {
                    last_emitted_step
                        .map(|previous| step > previous)
                        .unwrap_or(true)
                });
                if should_send {
                    last_emitted_step = metrics.step;
                    if tx.send(metrics).await.is_err() {
                        return;
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use tokio::sync::mpsc;
    use tokio::time::timeout;

    use crate::config::Config;

    #[tokio::test]
    async fn test_create_parser_jsonl() {
        let config = Config {
            parser: "jsonl".to_string(),
            ..Config::default()
        };

        let parser = create_parser(&config).expect("jsonl parser should be created");
        let parsed = parser
            .parse_line(r#"{"loss": 0.25, "step": 10}"#)
            .expect("jsonl parse should succeed");

        assert!(parsed.is_some());
        assert_eq!(parsed.expect("metrics should exist").loss, Some(0.25));
    }

    #[tokio::test]
    async fn test_create_parser_auto() {
        let config = Config {
            parser: "auto".to_string(),
            ..Config::default()
        };

        let parser = create_parser(&config).expect("auto parser should be created");
        let parsed = parser
            .parse_line(r#"{"step": 99}"#)
            .expect("default parser parse should succeed");

        assert!(parsed.is_some());
        assert_eq!(parsed.expect("metrics should exist").step, Some(99));
    }

    #[tokio::test]
    async fn test_create_parser_auto_detects_csv_from_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("epoch-training-csv-test-{unique}"));
        fs::create_dir_all(&root).expect("test directory should be created");
        let file_path = root.join("train.csv");
        fs::write(&file_path, "loss,step,lr\n0.5,100,0.001\n")
            .expect("test csv file should be written");

        let config = Config {
            parser: "auto".to_string(),
            log_file: Some(file_path.clone()),
            ..Config::default()
        };

        let parser = create_parser(&config).expect("auto parser should be created");
        let parsed = parser
            .parse_line("0.5,100,0.001")
            .expect("auto-detected csv parser should parse csv data");

        assert!(parsed.is_some());
        let metrics = parsed.expect("metrics should exist");
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.learning_rate, Some(0.001));

        fs::remove_file(&file_path).expect("test file should be removed");
        fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[tokio::test]
    async fn test_create_parser_auto_detects_csv_after_header_line() {
        let config = Config {
            parser: "auto".to_string(),
            log_file: Some(std::env::temp_dir().join("epoch-does-not-exist.csv")),
            ..Config::default()
        };

        let parser = create_parser(&config).expect("auto parser should be created");
        assert!(
            parser
                .parse_line("loss,step,lr")
                .expect("header parse should succeed")
                .is_none()
        );

        let parsed = parser
            .parse_line("0.5,100,0.001")
            .expect("csv parse should succeed");
        assert!(parsed.is_some());
        let metrics = parsed.expect("metrics should exist");
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.step, Some(100));
    }

    #[tokio::test]
    async fn test_create_parser_regex_with_pattern() {
        let config = Config {
            parser: "regex".to_string(),
            regex_pattern: Some(r"step=(?P<step>\d+) loss=(?P<loss>[\d.]+)".to_string()),
            ..Config::default()
        };

        let parser = create_parser(&config).expect("regex parser should be created");
        let parsed = parser
            .parse_line("step=42 loss=0.123")
            .expect("regex parse should succeed");

        assert!(parsed.is_some());
        let metrics = parsed.expect("metrics should exist");
        assert_eq!(metrics.step, Some(42));
        assert_eq!(metrics.loss, Some(0.123));
    }

    #[tokio::test]
    async fn test_create_parser_regex_without_pattern() {
        let config = Config {
            parser: "regex".to_string(),
            regex_pattern: None,
            ..Config::default()
        };

        let result = create_parser(&config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_parser_unknown_defaults_to_jsonl() {
        let config = Config {
            parser: "something-new".to_string(),
            ..Config::default()
        };

        let parser = create_parser(&config).expect("unknown parser should default to jsonl");
        let parsed = parser
            .parse_line(r#"{"lr": 0.001}"#)
            .expect("jsonl parse should succeed");

        assert!(parsed.is_some());
        assert_eq!(
            parsed.expect("metrics should exist").learning_rate,
            Some(0.001)
        );
    }

    #[tokio::test]
    async fn test_file_watcher_reads_existing_content() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("epoch-training-test-{unique}"));
        fs::create_dir_all(&root).expect("test directory should be created");
        let file_path = root.join("train.log");

        let contents = [
            r#"{"loss": 0.9, "step": 1}"#,
            "not-json",
            r#"{"loss": 0.8, "step": 2}"#,
        ]
        .join("\n");
        fs::write(&file_path, contents).expect("test log file should be written");

        let config = Config {
            parser: "jsonl".to_string(),
            ..Config::default()
        };
        let parser = create_parser(&config).expect("jsonl parser should be created");
        let (tx, mut rx) = mpsc::channel(8);

        let handle = spawn_file_watcher(file_path.clone(), parser, tx)
            .expect("watcher should spawn for existing file");

        let first = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("first recv should complete")
            .expect("first metric should exist");
        let second = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("second recv should complete")
            .expect("second metric should exist");

        assert_eq!(first.step, Some(1));
        assert_eq!(second.step, Some(2));

        handle.abort();
        fs::remove_file(&file_path).expect("test file should be removed");
        fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[tokio::test]
    async fn test_file_watcher_missing_parent_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let missing_parent =
            std::env::temp_dir().join(format!("epoch-nonexistent-parent-{unique}/child"));
        let path = missing_parent.join("train.log");

        let config = Config {
            parser: "jsonl".to_string(),
            ..Config::default()
        };
        let parser = create_parser(&config).expect("jsonl parser should be created");
        let (tx, _rx) = mpsc::channel(8);

        let result = spawn_file_watcher(path, parser, tx);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_trainer_state_poller_emits_new_entries_on_rewrite() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("epoch-trainer-state-test-{unique}"));
        fs::create_dir_all(&root).expect("test directory should be created");
        let file_path = root.join("trainer_state.json");

        fs::write(
            &file_path,
            r#"{"log_history":[{"loss":1.0,"learning_rate":0.001,"step":10}]}"#,
        )
        .expect("initial trainer state should be written");

        let (tx, mut rx) = mpsc::channel(8);
        let handle = spawn_trainer_state_poller(file_path.clone(), tx, Duration::from_millis(100));

        let first = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("first recv should complete")
            .expect("first metric should exist");
        assert_eq!(first.step, Some(10));

        fs::write(
            &file_path,
            r#"{"log_history":[{"loss":1.0,"learning_rate":0.001,"step":10},{"loss":0.9,"learning_rate":0.001,"step":20}]}"#,
        )
        .expect("rewritten trainer state should be written");

        let second = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("second recv should complete")
            .expect("second metric should exist");
        assert_eq!(second.step, Some(20));

        handle.abort();
        fs::remove_file(&file_path).expect("test file should be removed");
        fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[tokio::test]
    async fn test_trainer_state_poller_emits_step_zero() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("epoch-trainer-state-zero-{unique}"));
        fs::create_dir_all(&root).expect("test directory should be created");
        let file_path = root.join("trainer_state.json");

        fs::write(
            &file_path,
            r#"{"log_history":[{"loss":1.0,"learning_rate":0.001,"step":0}]}"#,
        )
        .expect("initial trainer state should be written");

        let (tx, mut rx) = mpsc::channel(8);
        let handle = spawn_trainer_state_poller(file_path.clone(), tx, Duration::from_millis(100));

        let first = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("recv should complete")
            .expect("metric should exist");
        assert_eq!(first.step, Some(0));

        handle.abort();
        fs::remove_file(&file_path).expect("test file should be removed");
        fs::remove_dir_all(&root).expect("test directory should be removed");
    }
}
