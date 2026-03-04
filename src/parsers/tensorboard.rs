use color_eyre::Result;

use super::LogParser;
use crate::types::TrainingMetrics;

pub struct TensorboardParser;

impl LogParser for TensorboardParser {
    fn parse_line(&self, _line: &str) -> Result<Option<TrainingMetrics>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensorboard_parser_instantiation() {
        let _parser = TensorboardParser;
    }
}
