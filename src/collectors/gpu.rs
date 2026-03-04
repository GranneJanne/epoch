use color_eyre::Result;

use super::Collector;

pub struct GpuCollector;

#[cfg(feature = "nvidia")]
impl Collector for GpuCollector {
    async fn collect(&mut self) -> Result<()> {
        todo!()
    }
}

#[cfg(not(feature = "nvidia"))]
impl Collector for GpuCollector {
    async fn collect(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_collector_instantiation() {
        let _collector = GpuCollector;
    }
}
