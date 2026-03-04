use color_eyre::Result;

use super::LogParser;
use crate::types::TrainingMetrics;

pub struct CsvParser;

impl LogParser for CsvParser {
    fn parse_line(&self, _line: &str) -> Result<Option<TrainingMetrics>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_parser_instantiation() {
        let _parser = CsvParser;
    }
}
