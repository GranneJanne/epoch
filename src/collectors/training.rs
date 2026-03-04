use color_eyre::Result;

use super::Collector;

pub struct TrainingCollector;

impl Collector for TrainingCollector {
    async fn collect(&mut self) -> Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_collector_instantiation() {
        let _collector = TrainingCollector;
    }
}
