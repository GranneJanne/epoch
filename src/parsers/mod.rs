pub mod aliases;
pub mod csv;
pub mod hf_trainer;
pub mod jsonl;
pub mod regex_parser;
pub mod tensorboard;

use color_eyre::Result;

use crate::types::TrainingMetrics;

pub trait LogParser: Send {
    fn parse_line(&self, line: &str) -> Result<Option<TrainingMetrics>>;
}

pub fn detect_parser(sample_lines: &[&str]) -> Box<dyn LogParser + Send> {
    let jsonl_parser = jsonl::JsonlParser;

    if sample_lines
        .iter()
        .any(|line| matches!(jsonl_parser.parse_line(line), Ok(Some(_))))
    {
        return Box::new(jsonl::JsonlParser);
    }

    if let Some(first_non_empty) = sample_lines.iter().find(|line| !line.trim().is_empty())
        && (first_non_empty.contains(',') || first_non_empty.contains('\t'))
        && let Ok(csv_parser) = csv::CsvParser::new(first_non_empty)
    {
        return Box::new(csv_parser);
    }

    Box::new(jsonl::JsonlParser)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::aliases::{
        EVAL_LOSS_KEYS, GRAD_NORM_KEYS, LEARNING_RATE_KEYS, LOSS_KEYS, SAMPLES_PER_SECOND_KEYS,
        STEP_KEYS, STEPS_PER_SECOND_KEYS, THROUGHPUT_KEYS, TOKEN_KEYS, TOKENS_PER_SECOND_KEYS,
    };

    #[test]
    fn test_detect_jsonl_from_sample() {
        let sample_lines = vec![
            r#"{"loss": 0.5, "step": 100}"#,
            r#"{"lr": 0.001, "step": 200}"#,
            "some text",
        ];

        let parser = detect_parser(&sample_lines);

        let result = parser
            .parse_line(r#"{"loss": 0.3}"#)
            .expect("parse should succeed");
        assert!(result.is_some());

        let metrics = result.unwrap();
        assert_eq!(metrics.loss, Some(0.3));
    }

    #[test]
    fn test_detect_empty_sample_returns_jsonl() {
        let sample_lines: Vec<&str> = vec![];
        let parser = detect_parser(&sample_lines);

        let result = parser
            .parse_line(r#"{"loss": 1.0}"#)
            .expect("parse should succeed");
        assert!(result.is_some());
    }

    #[test]
    fn test_detect_csv_from_header() {
        let sample_lines = vec!["loss,step,lr", "0.5,100,0.001"];
        let parser = detect_parser(&sample_lines);

        let result = parser
            .parse_line("0.5,100,0.001")
            .expect("parse should succeed");
        assert!(result.is_some());

        let metrics = result.unwrap();
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.learning_rate, Some(0.001));
    }

    #[test]
    fn test_detect_csv_tab_from_header() {
        let sample_lines = vec!["loss\tstep\tlr", "0.5\t100\t0.001"];
        let parser = detect_parser(&sample_lines);

        let result = parser
            .parse_line("0.5\t100\t0.001")
            .expect("parse should succeed");
        assert!(result.is_some());

        let metrics = result.unwrap();
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.learning_rate, Some(0.001));
    }

    #[test]
    fn test_detect_fallback_to_jsonl() {
        let sample_lines = vec!["garbage text", "still not data"];
        let parser = detect_parser(&sample_lines);

        let result = parser
            .parse_line(r#"{"loss": 1.0}"#)
            .expect("fallback parser should parse jsonl");
        assert!(result.is_some());
    }

    #[test]
    fn test_detect_mixed_jsonl_and_noise() {
        let sample_lines = vec!["INFO start", r#"{"loss": 0.9, "step": 1}"#, "garbage"];
        let parser = detect_parser(&sample_lines);

        let result = parser
            .parse_line(r#"{"step": 42}"#)
            .expect("parse should succeed");
        assert!(result.is_some());
    }

    #[test]
    fn test_detect_parser_returns_send() {
        fn assert_send<T: Send>(_: T) {}

        let parser = detect_parser(&[]);
        assert_send(parser);
    }

    #[test]
    fn test_parser_alias_contract_consistency_jsonl_csv_hf() {
        assert!(LOSS_KEYS.contains(&"loss"));
        assert!(LEARNING_RATE_KEYS.contains(&"learning_rate"));
        assert!(STEP_KEYS.contains(&"step"));
        assert!(THROUGHPUT_KEYS.contains(&"throughput"));
        assert!(TOKEN_KEYS.contains(&"tokens"));
        assert!(EVAL_LOSS_KEYS.contains(&"eval_loss"));
        assert!(GRAD_NORM_KEYS.contains(&"grad_norm"));
        assert!(SAMPLES_PER_SECOND_KEYS.contains(&"samples_per_second"));
        assert!(STEPS_PER_SECOND_KEYS.contains(&"steps_per_second"));
        assert!(TOKENS_PER_SECOND_KEYS.contains(&"tokens_per_second"));
    }
}
