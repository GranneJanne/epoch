#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricUnit {
    Loss,
    Rate,
    Count,
    Percent,
    Bytes,
    TempC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayTier {
    Overview,
    Advanced,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricId {
    Loss,
    LearningRate,
    Step,
    Throughput,
    Tokens,
    EvalLoss,
    GradNorm,
    SamplesPerSecond,
    StepsPerSecond,
    TokensPerSecond,
    Perplexity,
    LossSpikeCount,
    NanInfCount,
    CpuUsage,
    MemoryUsed,
    MemoryTotal,
    GpuUtilization,
    GpuMemoryUsed,
    GpuMemoryTotal,
    GpuTemperature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub id: MetricId,
    pub canonical_name: &'static str,
    pub unit: MetricUnit,
    pub tier: DisplayTier,
    pub aliases: &'static [&'static str],
}

pub const V02_CORE_STABILITY_METRICS: &[MetricDescriptor] = &[
    MetricDescriptor {
        id: MetricId::Loss,
        canonical_name: "loss",
        unit: MetricUnit::Loss,
        tier: DisplayTier::Overview,
        aliases: &[
            "loss",
            "train_loss",
            "training_loss",
            "train/loss",
            "lm_loss",
            "nll_loss",
        ],
    },
    MetricDescriptor {
        id: MetricId::LearningRate,
        canonical_name: "learning_rate",
        unit: MetricUnit::Rate,
        tier: DisplayTier::Overview,
        aliases: &["learning_rate", "lr", "train/learning_rate", "train/lr"],
    },
    MetricDescriptor {
        id: MetricId::Step,
        canonical_name: "step",
        unit: MetricUnit::Count,
        tier: DisplayTier::Overview,
        aliases: &[
            "step",
            "global_step",
            "iteration",
            "train/global_step",
            "_step",
        ],
    },
    MetricDescriptor {
        id: MetricId::Throughput,
        canonical_name: "throughput",
        unit: MetricUnit::Rate,
        tier: DisplayTier::Overview,
        aliases: &["throughput", "tps", "items_per_second"],
    },
    MetricDescriptor {
        id: MetricId::Tokens,
        canonical_name: "tokens",
        unit: MetricUnit::Count,
        tier: DisplayTier::Overview,
        aliases: &[
            "tokens",
            "total_tokens",
            "num_tokens",
            "num_input_tokens_seen",
        ],
    },
    MetricDescriptor {
        id: MetricId::EvalLoss,
        canonical_name: "eval_loss",
        unit: MetricUnit::Loss,
        tier: DisplayTier::Advanced,
        aliases: &["eval_loss", "validation_loss", "val/loss"],
    },
    MetricDescriptor {
        id: MetricId::GradNorm,
        canonical_name: "grad_norm",
        unit: MetricUnit::Rate,
        tier: DisplayTier::Advanced,
        aliases: &["grad_norm", "gradient_norm"],
    },
    MetricDescriptor {
        id: MetricId::SamplesPerSecond,
        canonical_name: "samples_per_second",
        unit: MetricUnit::Rate,
        tier: DisplayTier::Advanced,
        aliases: &["samples_per_second", "train_samples_per_second"],
    },
    MetricDescriptor {
        id: MetricId::StepsPerSecond,
        canonical_name: "steps_per_second",
        unit: MetricUnit::Rate,
        tier: DisplayTier::Advanced,
        aliases: &["steps_per_second", "train_steps_per_second"],
    },
    MetricDescriptor {
        id: MetricId::TokensPerSecond,
        canonical_name: "tokens_per_second",
        unit: MetricUnit::Rate,
        tier: DisplayTier::Advanced,
        aliases: &["tokens_per_second", "train_tokens_per_second"],
    },
    MetricDescriptor {
        id: MetricId::Perplexity,
        canonical_name: "perplexity",
        unit: MetricUnit::Rate,
        tier: DisplayTier::Advanced,
        aliases: &[],
    },
    MetricDescriptor {
        id: MetricId::LossSpikeCount,
        canonical_name: "loss_spike_count",
        unit: MetricUnit::Count,
        tier: DisplayTier::Advanced,
        aliases: &[],
    },
    MetricDescriptor {
        id: MetricId::NanInfCount,
        canonical_name: "nan_inf_count",
        unit: MetricUnit::Count,
        tier: DisplayTier::Advanced,
        aliases: &[],
    },
    MetricDescriptor {
        id: MetricId::CpuUsage,
        canonical_name: "cpu_usage",
        unit: MetricUnit::Percent,
        tier: DisplayTier::Overview,
        aliases: &["cpu_usage"],
    },
    MetricDescriptor {
        id: MetricId::MemoryUsed,
        canonical_name: "memory_used",
        unit: MetricUnit::Bytes,
        tier: DisplayTier::Overview,
        aliases: &["memory_used", "ram_used"],
    },
    MetricDescriptor {
        id: MetricId::MemoryTotal,
        canonical_name: "memory_total",
        unit: MetricUnit::Bytes,
        tier: DisplayTier::Hidden,
        aliases: &["memory_total", "ram_total"],
    },
    MetricDescriptor {
        id: MetricId::GpuUtilization,
        canonical_name: "gpu_utilization",
        unit: MetricUnit::Percent,
        tier: DisplayTier::Overview,
        aliases: &["utilization", "gpu_utilization"],
    },
    MetricDescriptor {
        id: MetricId::GpuMemoryUsed,
        canonical_name: "gpu_memory_used",
        unit: MetricUnit::Bytes,
        tier: DisplayTier::Advanced,
        aliases: &["gpu_memory_used", "vram_used", "memory_used"],
    },
    MetricDescriptor {
        id: MetricId::GpuMemoryTotal,
        canonical_name: "gpu_memory_total",
        unit: MetricUnit::Bytes,
        tier: DisplayTier::Hidden,
        aliases: &["gpu_memory_total", "vram_total", "memory_total"],
    },
    MetricDescriptor {
        id: MetricId::GpuTemperature,
        canonical_name: "gpu_temperature",
        unit: MetricUnit::TempC,
        tier: DisplayTier::Advanced,
        aliases: &["temperature", "gpu_temperature"],
    },
];

pub fn descriptor_for(id: MetricId) -> Option<&'static MetricDescriptor> {
    V02_CORE_STABILITY_METRICS
        .iter()
        .find(|descriptor| descriptor.id == id)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_metric_ids_unique() {
        let mut seen = HashSet::new();
        for descriptor in V02_CORE_STABILITY_METRICS {
            assert!(
                seen.insert(descriptor.id),
                "duplicate metric id: {descriptor:?}"
            );
        }
    }

    #[test]
    fn test_each_core_metric_has_unit_and_tier() {
        assert!(!V02_CORE_STABILITY_METRICS.is_empty());

        for descriptor in V02_CORE_STABILITY_METRICS {
            assert!(!descriptor.canonical_name.is_empty());

            match descriptor.unit {
                MetricUnit::Loss
                | MetricUnit::Rate
                | MetricUnit::Count
                | MetricUnit::Percent
                | MetricUnit::Bytes
                | MetricUnit::TempC => {}
            }

            match descriptor.tier {
                DisplayTier::Overview | DisplayTier::Advanced | DisplayTier::Hidden => {}
            }
        }
    }

    #[test]
    fn test_v02_scope_excludes_distributed_and_cost() {
        let excluded_tokens = [
            "all_reduce",
            "nccl",
            "bandwidth",
            "network",
            "communication",
            "cost",
            "carbon",
            "mfu",
            "co2",
            "dollar",
        ];

        for descriptor in V02_CORE_STABILITY_METRICS {
            let name = descriptor.canonical_name.to_ascii_lowercase();
            assert!(
                excluded_tokens.iter().all(|token| !name.contains(token)),
                "v0.2 scope should exclude distributed/cost metrics: {}",
                descriptor.canonical_name
            );
        }
    }
}
