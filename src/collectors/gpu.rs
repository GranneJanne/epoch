use crate::types::GpuMetrics;
use color_eyre::Result;

#[cfg(feature = "nvidia")]
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
#[cfg(feature = "nvidia")]
use nvml_wrapper::{Nvml, error::NvmlError};
#[cfg(feature = "nvidia")]
use std::sync::OnceLock;

#[cfg(feature = "nvidia")]
static NVML: OnceLock<Result<Nvml, NvmlError>> = OnceLock::new();

#[cfg(feature = "nvidia")]
fn init_nvml() -> Result<Nvml, NvmlError> {
    Nvml::init()
}

pub struct GpuCollector {
    tx: tokio::sync::mpsc::Sender<Vec<GpuMetrics>>,
}

impl GpuCollector {
    pub fn new(tx: tokio::sync::mpsc::Sender<Vec<GpuMetrics>>) -> Self {
        Self { tx }
    }

    #[cfg(feature = "nvidia")]
    pub async fn collect(&mut self) -> Result<()> {
        let nvml = NVML.get_or_init(init_nvml);

        let metrics = match nvml.as_ref() {
            Err(e) => {
                tracing::debug!("NVML initialization failed: {:?}", e);
                vec![]
            }
            Ok(nvml) => {
                let device_count = match nvml.device_count() {
                    Ok(count) => count,
                    Err(e) => {
                        tracing::debug!("Failed to get device count: {:?}", e);
                        0
                    }
                };

                let mut gpu_metrics = Vec::new();

                for i in 0..device_count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        let name = match device.name() {
                            Ok(n) => n,
                            Err(_) => continue,
                        };

                        let utilization = match device.utilization_rates() {
                            Ok(rates) => rates.gpu as f64,
                            Err(_) => continue,
                        };

                        let (memory_used, memory_total) = match device.memory_info() {
                            Ok(mem) => (mem.used, mem.total),
                            Err(_) => continue,
                        };

                        let temperature = match device.temperature(TemperatureSensor::Gpu) {
                            Ok(temp) => temp as f64,
                            Err(_) => continue,
                        };

                        gpu_metrics.push(GpuMetrics {
                            name,
                            utilization,
                            memory_used,
                            memory_total,
                            temperature,
                        });
                    }
                }

                gpu_metrics
            }
        };

        self.tx.send(metrics).await.ok();

        Ok(())
    }

    #[cfg(not(feature = "nvidia"))]
    pub async fn collect(&mut self) -> Result<()> {
        self.tx.send(vec![]).await.ok();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_collector_creation() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let _collector = GpuCollector::new(tx);
    }

    #[tokio::test]
    async fn test_gpu_collect_no_panic() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let mut collector = GpuCollector::new(tx);

        let result = collector.collect().await;
        assert!(result.is_ok(), "collect should not panic or return error");
    }

    #[tokio::test]
    async fn test_gpu_collect_sends_metrics() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let mut collector = GpuCollector::new(tx);

        collector.collect().await.expect("collect should succeed");

        let _metrics = rx.recv().await.expect("should receive metrics");
    }

    #[cfg(not(feature = "nvidia"))]
    #[tokio::test]
    async fn test_gpu_collect_sends_empty_vec_without_nvidia_feature() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let mut collector = GpuCollector::new(tx);

        collector.collect().await.expect("collect should succeed");

        let metrics = rx.recv().await.expect("should receive metrics");
        assert!(
            metrics.is_empty(),
            "should send empty vec when nvidia feature is disabled"
        );
    }

    #[tokio::test]
    async fn test_gpu_collector_multiple_collects() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let mut collector = GpuCollector::new(tx);

        for _ in 0..3 {
            collector.collect().await.expect("collect should succeed");
            let _metrics = rx.recv().await.expect("should receive metrics");
        }
    }
}
