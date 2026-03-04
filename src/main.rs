use clap::Parser;
use color_eyre::Result;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Path to training log file to monitor
    log_file: Option<PathBuf>,

    /// Read training metrics from stdin
    #[arg(long, conflicts_with = "log_file")]
    stdin: bool,

    /// Override log parser (auto, jsonl, csv, regex)
    #[arg(long)]
    parser: Option<String>,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let _cli = Cli::parse();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse() {
        let cli = Cli::parse_from(["epoch"]);
        assert!(cli.log_file.is_none());
        assert!(!cli.stdin);
    }
}
