use color_eyre::{Result, eyre::Context};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub tick_rate_ms: u64,
    pub history_size: usize,
    pub stale_after_secs: u64,
    pub parser: String,
    pub regex_pattern: Option<String>,
    pub log_file: Option<PathBuf>,
    pub stdin_mode: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tick_rate_ms: 250,
            history_size: 300,
            stale_after_secs: 10,
            parser: "auto".to_string(),
            regex_pattern: None,
            log_file: None,
            stdin_mode: false,
        }
    }
}

impl Config {
    /// Load configuration from XDG config directory (~/.config/epoch/config.toml)
    /// Returns default config if file doesn't exist.
    pub fn load() -> Result<Self> {
        let config_path = directories::ProjectDirs::from("", "", "epoch")
            .map(|dirs| dirs.config_dir().join("config.toml"));

        if let Some(path) = config_path {
            if path.exists() {
                let contents = std::fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read config file: {}", path.display()))?;
                let config: Config = toml::from_str(&contents)
                    .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?;
                return Ok(config);
            }
        }

        Ok(Config::default())
    }

    /// Merge CLI arguments into config (CLI takes precedence)
    pub fn merge_cli_args(
        &mut self,
        log_file: Option<PathBuf>,
        stdin: bool,
        parser: Option<String>,
    ) {
        if let Some(lf) = log_file {
            self.log_file = Some(lf);
        }
        if stdin {
            self.stdin_mode = true;
        }
        if let Some(p) = parser {
            self.parser = p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.tick_rate_ms, 250);
        assert_eq!(config.history_size, 300);
        assert_eq!(config.stale_after_secs, 10);
        assert_eq!(config.parser, "auto");
    }

    #[test]
    fn test_config_defaults_expanded() {
        let config = Config::default();
        assert_eq!(config.tick_rate_ms, 250);
        assert_eq!(config.history_size, 300);
        assert_eq!(config.stale_after_secs, 10);
        assert_eq!(config.parser, "auto");
        assert!(config.regex_pattern.is_none());
        assert!(config.log_file.is_none());
        assert!(!config.stdin_mode);
    }

    #[test]
    fn test_config_parse_toml() {
        let toml_str = r#"
            tick_rate_ms = 100
            history_size = 500
            stale_after_secs = 20
            parser = "jsonl"
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.tick_rate_ms, 100);
        assert_eq!(config.history_size, 500);
        assert_eq!(config.stale_after_secs, 20);
        assert_eq!(config.parser, "jsonl");
    }

    #[test]
    fn test_config_parse_partial_toml() {
        let toml_str = r#"tick_rate_ms = 100"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.tick_rate_ms, 100);
        assert_eq!(config.history_size, 300); // default
        assert_eq!(config.stale_after_secs, 10); // default
        assert_eq!(config.parser, "auto"); // default
    }

    #[test]
    fn test_config_stale_after_secs_default() {
        let config = Config::default();
        assert_eq!(config.stale_after_secs, 10);
    }

    #[test]
    fn test_config_invalid_toml_errors() {
        let result: Result<Config, _> = toml::from_str("this is not valid toml {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_unknown_fields_accepted() {
        let toml_str = r#"
            tick_rate_ms = 100
            unknown_field = "oops"
        "#;
        let result: Result<Config, _> = toml::from_str(toml_str);
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_merge_cli_overrides() {
        let mut config = Config::default();
        // Simulate CLI with parser set
        config.merge_cli_args(
            Some(PathBuf::from("/tmp/train.log")),
            false,
            Some("jsonl".to_string()),
        );
        assert_eq!(config.parser, "jsonl");
        assert_eq!(config.log_file, Some(PathBuf::from("/tmp/train.log")));
    }

    #[test]
    fn test_config_merge_cli_stdin() {
        let mut config = Config::default();
        config.merge_cli_args(None, true, None);
        assert!(config.stdin_mode);
        assert_eq!(config.parser, "auto"); // not overridden
    }

    #[test]
    fn test_config_merge_cli_no_override_when_none() {
        let mut config = Config {
            parser: "jsonl".to_string(),
            ..Config::default()
        };
        config.merge_cli_args(None, false, None);
        assert_eq!(config.parser, "jsonl"); // kept, not overridden to "auto"
    }

    #[test]
    fn test_config_load_missing_file_returns_defaults() {
        // Config::load should not error if config file doesn't exist
        let config = Config::load().unwrap();
        assert_eq!(config.tick_rate_ms, 250);
    }
}
