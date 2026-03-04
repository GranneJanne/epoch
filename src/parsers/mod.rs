pub mod csv;
pub mod jsonl;
pub mod regex_parser;
pub mod tensorboard;

use color_eyre::Result;

use crate::types::TrainingMetrics;

pub trait LogParser {
    fn parse_line(&self, line: &str) -> Result<Option<TrainingMetrics>>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        assert!(true);
    }
}
