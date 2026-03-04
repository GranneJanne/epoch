# Contributing to Epoch

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/epoch-ml/epoch.git
   cd epoch
   ```

2. **Build the project**:
   ```bash
   cargo build
   ```

3. **Run tests**:
   ```bash
   cargo test
   ```

## Code Style

We follow Rust conventions enforced by `cargo fmt` and `cargo clippy`:

```bash
# Format code
cargo fmt

# Lint (must pass with no warnings)
cargo clippy -- -D warnings
```

All PRs must pass these checks before merging.

## Submitting a Pull Request

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test them
   ```bash
   cargo test
   cargo fmt
   cargo clippy -- -D warnings
   ```

3. **Push to your fork** and create a pull request
   - Provide a clear title and description
   - Link any related issues
   - Reference AGENTS.md for architectural guidelines

## Questions?

Open an issue or discussion in the repository.
