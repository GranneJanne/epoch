use color_eyre::Result;

use super::LogParser;
use crate::types::TrainingMetrics;

pub struct RegexParser;

impl LogParser for RegexParser {
    fn parse_line(&self, _line: &str) -> Result<Option<TrainingMetrics>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_parser_instantiation() {
        let _parser = RegexParser;
    }
}
