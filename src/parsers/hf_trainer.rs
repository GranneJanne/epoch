use std::time::Instant;

use color_eyre::Result;

use crate::parsers::aliases::{
    EVAL_LOSS_KEYS, GRAD_NORM_KEYS, LEARNING_RATE_KEYS, LOSS_KEYS, SAMPLES_PER_SECOND_KEYS,
    STEP_KEYS, STEPS_PER_SECOND_KEYS, THROUGHPUT_KEYS, TOKEN_KEYS, TOKENS_PER_SECOND_KEYS,
};
use crate::types::TrainingMetrics;

fn try_extract_f64(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(|value| value.as_f64()))
}

fn try_extract_u64(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(|value| value.as_u64()))
}

pub fn parse_trainer_state(content: &str) -> Result<Vec<TrainingMetrics>> {
    let value: serde_json::Value = serde_json::from_str(content)?;

    let Some(log_history) = value.get("log_history").and_then(|v| v.as_array()) else {
        return Ok(vec![]);
    };

    let mut metrics_vec = Vec::with_capacity(log_history.len());

    for entry in log_history {
        let Some(obj) = entry.as_object() else {
            continue;
        };

        let loss = try_extract_f64(obj, LOSS_KEYS);
        let learning_rate = try_extract_f64(obj, LEARNING_RATE_KEYS);
        let step = try_extract_u64(obj, STEP_KEYS);
        let throughput = try_extract_f64(obj, THROUGHPUT_KEYS);
        let tokens = try_extract_u64(obj, TOKEN_KEYS);
        let eval_loss = try_extract_f64(obj, EVAL_LOSS_KEYS);
        let grad_norm = try_extract_f64(obj, GRAD_NORM_KEYS);
        let samples_per_second = try_extract_f64(obj, SAMPLES_PER_SECOND_KEYS);
        let steps_per_second = try_extract_f64(obj, STEPS_PER_SECOND_KEYS);
        let tokens_per_second = try_extract_f64(obj, TOKENS_PER_SECOND_KEYS);

        if loss.is_none()
            && learning_rate.is_none()
            && step.is_none()
            && throughput.is_none()
            && tokens.is_none()
            && eval_loss.is_none()
            && grad_norm.is_none()
            && samples_per_second.is_none()
            && steps_per_second.is_none()
            && tokens_per_second.is_none()
        {
            continue;
        }

        metrics_vec.push(TrainingMetrics {
            loss,
            learning_rate,
            step,
            throughput,
            tokens,
            eval_loss,
            grad_norm,
            samples_per_second,
            steps_per_second,
            tokens_per_second,
            timestamp: Instant::now(),
        });
    }

    Ok(metrics_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_trainer_state_valid() {
        let content = r#"{
            "log_history": [
                {"loss": 4.5, "learning_rate": 0.0001, "step": 10, "epoch": 0.1},
                {"loss": 3.8, "learning_rate": 0.0002, "step": 20, "epoch": 0.2},
                {"loss": 3.1, "learning_rate": 0.0003, "step": 30, "epoch": 0.3}
            ]
        }"#;

        let parsed = parse_trainer_state(content).expect("parse should succeed");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].loss, Some(4.5));
        assert_eq!(parsed[0].learning_rate, Some(0.0001));
        assert_eq!(parsed[0].step, Some(10));
    }

    #[test]
    fn test_parse_trainer_state_missing_log_history() {
        let content = r#"{"best_metric": 0.9}"#;
        let parsed = parse_trainer_state(content).expect("parse should succeed");
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_trainer_state_empty_log_history() {
        let content = r#"{"log_history": []}"#;
        let parsed = parse_trainer_state(content).expect("parse should succeed");
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_trainer_state_partial_fields() {
        let content = r#"{"log_history": [{"loss": 1.25}]}"#;

        let parsed = parse_trainer_state(content).expect("parse should succeed");
        assert_eq!(parsed.len(), 1);
        let metrics = &parsed[0];
        assert_eq!(metrics.loss, Some(1.25));
        assert_eq!(metrics.learning_rate, None);
        assert_eq!(metrics.step, None);
        assert_eq!(metrics.throughput, None);
        assert_eq!(metrics.tokens, None);
    }

    #[test]
    fn test_parse_trainer_state_invalid_json() {
        let result = parse_trainer_state("{invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_trainer_state_ignores_epoch_field() {
        let content = r#"{"log_history": [{"loss": 0.5, "epoch": 2.0, "step": 100}]}"#;

        let parsed = parse_trainer_state(content).expect("parse should succeed");
        assert_eq!(parsed.len(), 1);
        let metrics = &parsed[0];
        assert_eq!(metrics.loss, Some(0.5));
        assert_eq!(metrics.step, Some(100));
        assert_eq!(metrics.learning_rate, None);
        assert_eq!(metrics.throughput, None);
        assert_eq!(metrics.tokens, None);
    }

    #[test]
    fn test_parse_trainer_state_new_core_fields() {
        let content = r#"{"log_history": [{"eval_loss": 0.7, "gradient_norm": 1.3, "train_samples_per_second": 12.0, "train_steps_per_second": 0.8, "train_tokens_per_second": 2048.0}]}"#;

        let parsed = parse_trainer_state(content).expect("parse should succeed");
        assert_eq!(parsed.len(), 1);
        let metrics = &parsed[0];
        assert_eq!(metrics.eval_loss, Some(0.7));
        assert_eq!(metrics.grad_norm, Some(1.3));
        assert_eq!(metrics.samples_per_second, Some(12.0));
        assert_eq!(metrics.steps_per_second, Some(0.8));
        assert_eq!(metrics.tokens_per_second, Some(2048.0));
    }
}
