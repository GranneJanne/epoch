use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub tick_rate_ms: u64,
    pub history_size: usize,
    pub parser: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tick_rate_ms: 250,
            history_size: 300,
            parser: "auto".to_string(),
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
        assert_eq!(config.parser, "auto");
    }
}
