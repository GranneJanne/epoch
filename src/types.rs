use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: Option<f64>,
    pub learning_rate: Option<f64>,
    pub step: Option<u64>,
    pub throughput: Option<f64>,
    pub tokens: Option<u64>,
    pub timestamp: Instant,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            loss: None,
            learning_rate: None,
            step: None,
            throughput: None,
            tokens: None,
            timestamp: Instant::now(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_used: u64,
    pub memory_total: u64,
    pub gpus: Vec<GpuMetrics>,
}

#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    pub name: String,
    pub utilization: f32,
    pub memory_used: u64,
    pub memory_total: u64,
    pub temperature: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_default() {
        let metrics = TrainingMetrics::default();
        assert!(metrics.loss.is_none());
        assert!(metrics.timestamp.elapsed().as_millis() < 100);
    }

    #[test]
    fn test_system_metrics_default() {
        let metrics = SystemMetrics::default();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.gpus.len(), 0);
    }

    #[test]
    fn test_gpu_metrics_default() {
        let metrics = GpuMetrics::default();
        assert_eq!(metrics.name, "");
        assert_eq!(metrics.utilization, 0.0);
    }
}
