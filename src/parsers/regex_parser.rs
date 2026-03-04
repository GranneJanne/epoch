use color_eyre::Result;
use regex::Regex;
use std::time::Instant;

use super::LogParser;
use crate::types::TrainingMetrics;

pub const DEFAULT_PATTERN: &str = r"[Ss]tep\s*[:=]?\s*(?P<step>\d+).*[Ll]oss\s*[:=]?\s*(?P<loss>[\d.]+)(?:.*[Ll](?:earning_)?[Rr](?:ate)?\s*[:=]?\s*(?P<lr>[\d.eE-]+))?";

pub struct RegexParser {
    pattern: Regex,
}

impl RegexParser {
    pub fn new(pattern: &str) -> Result<Self> {
        let compiled = Regex::new(pattern)?;
        Ok(Self { pattern: compiled })
    }
}

impl LogParser for RegexParser {
    fn parse_line(&self, line: &str) -> Result<Option<TrainingMetrics>> {
        let captures = match self.pattern.captures(line) {
            Some(caps) => caps,
            None => return Ok(None),
        };

        let mut metrics = TrainingMetrics {
            timestamp: Instant::now(),
            ..Default::default()
        };

        let mut has_any_field = false;

        if let Some(loss_match) = captures.name("loss") {
            if let Ok(val) = loss_match.as_str().parse::<f64>() {
                metrics.loss = Some(val);
                has_any_field = true;
            }
        }

        if let Some(lr_match) = captures.name("lr") {
            if let Ok(val) = lr_match.as_str().parse::<f64>() {
                metrics.learning_rate = Some(val);
                has_any_field = true;
            }
        }

        if let Some(step_match) = captures.name("step") {
            if let Ok(val) = step_match.as_str().parse::<u64>() {
                metrics.step = Some(val);
                has_any_field = true;
            }
        }

        if let Some(throughput_match) = captures.name("throughput") {
            if let Ok(val) = throughput_match.as_str().parse::<f64>() {
                metrics.throughput = Some(val);
                has_any_field = true;
            }
        }

        if let Some(tokens_match) = captures.name("tokens") {
            if let Ok(val) = tokens_match.as_str().parse::<u64>() {
                metrics.tokens = Some(val);
                has_any_field = true;
            }
        }

        if has_any_field {
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_parser_with_all_fields() {
        let pattern = r"step=(?P<step>\d+) loss=(?P<loss>[\d.]+) lr=(?P<lr>[\d.eE-]+) throughput=(?P<throughput>[\d.]+) tokens=(?P<tokens>\d+)";
        let parser = RegexParser::new(pattern).expect("valid pattern");

        let line = "step=100 loss=0.5 lr=1e-4 throughput=1000.0 tokens=50000";
        let result = parser.parse_line(line).expect("parse should succeed");

        assert!(result.is_some());
        let metrics = result.unwrap();
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.learning_rate, Some(1e-4));
        assert_eq!(metrics.throughput, Some(1000.0));
        assert_eq!(metrics.tokens, Some(50000));
    }

    #[test]
    fn test_regex_parser_partial_fields() {
        let pattern = r"step=(?P<step>\d+) loss=(?P<loss>[\d.]+)";
        let parser = RegexParser::new(pattern).expect("valid pattern");

        let line = "step=100 loss=0.5";
        let result = parser.parse_line(line).expect("parse should succeed");

        assert!(result.is_some());
        let metrics = result.unwrap();
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.learning_rate, None);
        assert_eq!(metrics.throughput, None);
        assert_eq!(metrics.tokens, None);
    }

    #[test]
    fn test_regex_parser_no_match() {
        let pattern = r"step=(?P<step>\d+) loss=(?P<loss>[\d.]+)";
        let parser = RegexParser::new(pattern).expect("valid pattern");

        let line = "this line does not match the pattern";
        let result = parser.parse_line(line).expect("parse should succeed");

        assert!(result.is_none());
    }

    #[test]
    fn test_regex_parser_invalid_pattern() {
        let pattern = "[invalid";
        let result = RegexParser::new(pattern);

        assert!(result.is_err());
    }

    #[test]
    fn test_regex_parser_unparseable_capture() {
        let pattern = r"step=(?P<step>\d+) loss=(?P<loss>\w+)";
        let parser = RegexParser::new(pattern).expect("valid pattern");

        // "abc" can't be parsed as f64
        let line = "step=100 loss=abc";
        let result = parser.parse_line(line).expect("parse should succeed");

        assert!(result.is_some());
        let metrics = result.unwrap();
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.loss, None); // Should be None, not panic
    }

    #[test]
    fn test_default_pattern_matches() {
        let parser = RegexParser::new(DEFAULT_PATTERN).expect("default pattern should be valid");

        let line = "Step 100 | Loss: 0.5 | LR: 1e-4";
        let result = parser.parse_line(line).expect("parse should succeed");

        assert!(result.is_some());
        let metrics = result.unwrap();
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.learning_rate, Some(1e-4));
    }

    #[test]
    fn test_default_pattern_no_match() {
        let parser = RegexParser::new(DEFAULT_PATTERN).expect("default pattern should be valid");

        let line = "unrelated log message";
        let result = parser.parse_line(line).expect("parse should succeed");

        assert!(result.is_none());
    }
}
