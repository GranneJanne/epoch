use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use color_eyre::Result;

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "target",
    ".tox",
    ".mypy_cache",
    ".sisyphus",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Jsonl,
    Csv,
    HfTrainerState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub format: FileFormat,
    pub modified: SystemTime,
}

pub fn discover_training_files(root: &Path) -> Result<Vec<DiscoveredFile>> {
    let mut walker = ignore::WalkBuilder::new(root);
    walker
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .filter_entry(|entry| {
            if entry.depth() == 0 {
                return true;
            }

            let name = entry.file_name().to_string_lossy();
            !SKIP_DIRS.contains(&name.as_ref())
        });

    let mut discovered = Vec::new();

    for entry in walker.build() {
        let Ok(entry) = entry else {
            continue;
        };

        let Some(file_type) = entry.file_type() else {
            continue;
        };

        if !file_type.is_file() {
            continue;
        }

        let path = entry.path().to_path_buf();
        let Some(format) = detect_format(&path) else {
            continue;
        };

        let modified = entry
            .metadata()
            .ok()
            .and_then(|metadata| metadata.modified().ok())
            .unwrap_or(UNIX_EPOCH);

        discovered.push(DiscoveredFile {
            path,
            format,
            modified,
        });
    }

    discovered.sort_by(|a, b| b.modified.cmp(&a.modified));
    Ok(discovered)
}

fn detect_format(path: &Path) -> Option<FileFormat> {
    let file_name = path.file_name().and_then(|name| name.to_str())?;

    if file_name == "trainer_state.json" {
        return Some(FileFormat::HfTrainerState);
    }

    if file_name == "wandb-events.jsonl" {
        return Some(FileFormat::Jsonl);
    }

    match path.extension().and_then(|extension| extension.to_str()) {
        Some("jsonl") => Some(FileFormat::Jsonl),
        Some("csv") => Some(FileFormat::Csv),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_temp_root(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("epoch-{prefix}-{unique}"));
        std::fs::create_dir_all(&root).expect("test directory should be created");
        root
    }

    #[test]
    fn test_discover_finds_jsonl_files() {
        let root = create_temp_root("discover-jsonl");
        let file_path = root.join("train.jsonl");
        std::fs::write(&file_path, "{}\n").expect("file should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].format, FileFormat::Jsonl);

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_finds_csv_files() {
        let root = create_temp_root("discover-csv");
        let file_path = root.join("metrics.csv");
        std::fs::write(&file_path, "step,loss\n1,0.5\n").expect("file should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].format, FileFormat::Csv);

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_finds_trainer_state() {
        let root = create_temp_root("discover-hf");
        let file_path = root.join("trainer_state.json");
        std::fs::write(&file_path, "{}\n").expect("file should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].format, FileFormat::HfTrainerState);

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_finds_wandb_events() {
        let root = create_temp_root("discover-wandb");
        let file_path = root.join("wandb-events.jsonl");
        std::fs::write(&file_path, "{}\n").expect("file should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].format, FileFormat::Jsonl);

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_newest_first() {
        let root = create_temp_root("discover-order");
        let older = root.join("older.jsonl");
        std::fs::write(&older, "{}\n").expect("older file should be written");

        std::thread::sleep(std::time::Duration::from_millis(20));

        let newer = root.join("newer.jsonl");
        std::fs::write(&newer, "{}\n").expect("newer file should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert!(discovered.len() >= 2);
        assert_eq!(discovered[0].path, newer);

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_empty_dir() {
        let root = create_temp_root("discover-empty");
        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert!(discovered.is_empty());

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_skips_gitignored() {
        let root = create_temp_root("discover-gitignore");
        std::fs::write(root.join(".gitignore"), "*.jsonl\n").expect("gitignore should be written");
        std::fs::write(root.join("train.jsonl"), "{}\n").expect("jsonl should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert!(discovered.is_empty());

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_skips_node_modules() {
        let root = create_temp_root("discover-node-modules");
        let node_modules = root.join("node_modules");
        std::fs::create_dir_all(&node_modules).expect("node_modules should be created");
        std::fs::write(node_modules.join("artifact.jsonl"), "{}\n")
            .expect("node_modules file should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert!(discovered.is_empty());

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }

    #[test]
    fn test_discover_ignores_non_training_files() {
        let root = create_temp_root("discover-ignore-non-training");
        std::fs::write(root.join("README.md"), "docs").expect("readme should be written");
        std::fs::write(root.join("main.py"), "print('hello')")
            .expect("python file should be written");

        let discovered = discover_training_files(&root).expect("discovery should succeed");
        assert!(discovered.is_empty());

        std::fs::remove_dir_all(&root).expect("test directory should be removed");
    }
}
