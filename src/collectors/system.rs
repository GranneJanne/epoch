use color_eyre::Result;

use super::Collector;

pub struct SystemCollector;

impl Collector for SystemCollector {
    async fn collect(&mut self) -> Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_collector_instantiation() {
        let _collector = SystemCollector;
    }
}
