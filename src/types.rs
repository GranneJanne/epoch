use std::collections::VecDeque;
use std::time::Instant;

pub type MetricHistory = VecDeque<TrainingMetrics>;
pub const DEFAULT_HISTORY_SIZE: usize = 300;

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

impl TrainingMetrics {
    pub fn is_empty(&self) -> bool {
        self.loss.is_none()
            && self.learning_rate.is_none()
            && self.step.is_none()
            && self.throughput.is_none()
            && self.tokens.is_none()
    }

    pub fn merge(&mut self, other: &TrainingMetrics) {
        if self.loss.is_none() {
            self.loss = other.loss;
        }
        if self.learning_rate.is_none() {
            self.learning_rate = other.learning_rate;
        }
        if self.step.is_none() {
            self.step = other.step;
        }
        if self.throughput.is_none() {
            self.throughput = other.throughput;
        }
        if self.tokens.is_none() {
            self.tokens = other.tokens;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_used: u64,
    pub memory_total: u64,
    pub gpus: Vec<GpuMetrics>,
}

impl SystemMetrics {
    pub fn cpu_usage_percent(&self) -> f64 {
        self.cpu_usage
    }

    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_total == 0 {
            return 0.0;
        }
        (self.memory_used as f64 / self.memory_total as f64) * 100.0
    }

    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    pub name: String,
    pub utilization: f64,
    pub memory_used: u64,
    pub memory_total: u64,
    pub temperature: f64,
}

impl GpuMetrics {
    pub fn vram_usage_percent(&self) -> f64 {
        if self.memory_total == 0 {
            return 0.0;
        }
        (self.memory_used as f64 / self.memory_total as f64) * 100.0
    }
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

    // TrainingMetrics::is_empty tests
    #[test]
    fn test_training_metrics_is_empty_all_none() {
        let m = TrainingMetrics::default();
        assert!(m.is_empty());
    }

    #[test]
    fn test_training_metrics_is_empty_with_loss() {
        let mut m = TrainingMetrics::default();
        m.loss = Some(1.5);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_training_metrics_is_empty_with_step() {
        let mut m = TrainingMetrics::default();
        m.step = Some(100);
        assert!(!m.is_empty());
    }

    // TrainingMetrics::merge tests
    #[test]
    fn test_training_metrics_merge_fills_none() {
        let mut m1 = TrainingMetrics::default();
        m1.loss = Some(1.0);
        let m2 = TrainingMetrics {
            step: Some(100),
            ..TrainingMetrics::default()
        };
        m1.merge(&m2);
        assert_eq!(m1.loss, Some(1.0)); // kept
        assert_eq!(m1.step, Some(100)); // filled from other
    }

    #[test]
    fn test_training_metrics_merge_does_not_overwrite() {
        let mut m1 = TrainingMetrics {
            loss: Some(1.0),
            ..TrainingMetrics::default()
        };
        let m2 = TrainingMetrics {
            loss: Some(2.0),
            ..TrainingMetrics::default()
        };
        m1.merge(&m2);
        assert_eq!(m1.loss, Some(1.0)); // NOT overwritten
    }

    #[test]
    fn test_training_metrics_merge_multiple_fields() {
        let mut m1 = TrainingMetrics {
            loss: Some(1.0),
            ..TrainingMetrics::default()
        };
        let m2 = TrainingMetrics {
            loss: Some(2.0),
            learning_rate: Some(0.001),
            step: Some(100),
            throughput: Some(1000.0),
            tokens: Some(50000),
            ..TrainingMetrics::default()
        };
        m1.merge(&m2);
        assert_eq!(m1.loss, Some(1.0)); // NOT overwritten
        assert_eq!(m1.learning_rate, Some(0.001)); // filled
        assert_eq!(m1.step, Some(100)); // filled
        assert_eq!(m1.throughput, Some(1000.0)); // filled
        assert_eq!(m1.tokens, Some(50000)); // filled
    }

    // SystemMetrics::memory_usage_percent tests
    #[test]
    fn test_system_metrics_memory_percent() {
        let m = SystemMetrics {
            memory_used: 4_000_000_000,
            memory_total: 16_000_000_000,
            ..Default::default()
        };
        assert!((m.memory_usage_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_system_metrics_memory_percent_zero_total() {
        let m = SystemMetrics {
            memory_used: 0,
            memory_total: 0,
            ..Default::default()
        };
        assert_eq!(m.memory_usage_percent(), 0.0); // no div-by-zero
    }

    #[test]
    fn test_system_metrics_memory_percent_full() {
        let m = SystemMetrics {
            memory_used: 16_000_000_000,
            memory_total: 16_000_000_000,
            ..Default::default()
        };
        assert!((m.memory_usage_percent() - 100.0).abs() < 0.01);
    }

    // SystemMetrics::cpu_usage_percent tests
    #[test]
    fn test_system_metrics_cpu_usage_percent() {
        let m = SystemMetrics {
            cpu_usage: 75.5,
            ..Default::default()
        };
        assert_eq!(m.cpu_usage_percent(), 75.5);
    }

    // SystemMetrics::has_gpu tests
    #[test]
    fn test_system_metrics_has_gpu() {
        let m = SystemMetrics::default();
        assert!(!m.has_gpu());
    }

    #[test]
    fn test_system_metrics_has_gpu_with_one() {
        let m = SystemMetrics {
            gpus: vec![GpuMetrics::default()],
            ..Default::default()
        };
        assert!(m.has_gpu());
    }

    #[test]
    fn test_system_metrics_has_gpu_with_multiple() {
        let m = SystemMetrics {
            gpus: vec![GpuMetrics::default(), GpuMetrics::default()],
            ..Default::default()
        };
        assert!(m.has_gpu());
    }

    // GpuMetrics::vram_usage_percent tests
    #[test]
    fn test_gpu_vram_percent() {
        let g = GpuMetrics {
            memory_used: 4_000_000_000,
            memory_total: 8_000_000_000,
            ..Default::default()
        };
        assert!((g.vram_usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_vram_percent_zero_total() {
        let g = GpuMetrics {
            memory_used: 0,
            memory_total: 0,
            ..Default::default()
        };
        assert_eq!(g.vram_usage_percent(), 0.0);
    }

    #[test]
    fn test_gpu_vram_percent_full() {
        let g = GpuMetrics {
            memory_used: 8_000_000_000,
            memory_total: 8_000_000_000,
            ..Default::default()
        };
        assert!((g.vram_usage_percent() - 100.0).abs() < 0.01);
    }

    // Type alias and constant tests
    #[test]
    fn test_metric_history_type() {
        let _history: MetricHistory = VecDeque::new();
        assert_eq!(DEFAULT_HISTORY_SIZE, 300);
    }
}
