# Epoch

Terminal-based real-time monitoring dashboard for AI/ML training runs.

<!-- TODO: Add CI badge (GitHub Actions) -->
<!-- TODO: Add crates.io badge -->
<!-- TODO: Add license badge -->

## Features

- **Training Metrics**: Real-time loss, learning rate, training steps, throughput, and token tracking
- **System Metrics**: GPU utilization, VRAM, CPU usage, and RAM monitoring
- **Multi-View TUI**: Dynamic tab-based interface for switching between dashboards
- **Multiple Log Formats**: Support for JSONL (primary) and custom regex patterns
- **Stdin Support**: Pipe training metrics directly from your training script

## Installation

### From Source

```bash
git clone https://github.com/epoch-ml/epoch.git
cd epoch
cargo build --release
./target/release/epoch
```

### Using Cargo

```bash
cargo install --path .
```

(Publishing to crates.io coming soon)

## Usage

### Monitor a Training Log File

```bash
epoch --log-file train.log
```

### Read Metrics from Stdin

```bash
python train.py 2>&1 | epoch --stdin
```

### Monitor Multiple Metrics

Epoch automatically detects and parses training metrics from your logs, displaying them alongside live system metrics (GPU, CPU, RAM).

## Supported Log Formats

### v0.1.0 (Current)

- **JSONL** (JSON Lines): `{"loss": 0.5, "step": 100, "lr": 1e-4}`
- **Custom Regex**: User-defined patterns for framework-specific logs

### Planned for v0.2.0+

- CSV format
- TensorBoard event files
- HuggingFace Trainer native format

## Configuration

Configuration can be provided via TOML file at `~/.config/epoch/config.toml`:

```toml
[ui]
refresh_rate_ms = 100
theme = "dark"

[metrics]
auto_detect = true
```

See CONTRIBUTING.md for configuration examples and defaults.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
