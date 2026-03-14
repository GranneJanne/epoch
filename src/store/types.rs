#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunStatus {
    Active,
    Completed,
    Failed,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "active" => Some(Self::Active),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunSourceKind {
    LogFile,
    Stdin,
    Process,
}

impl RunSourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LogFile => "log_file",
            Self::Stdin => "stdin",
            Self::Process => "process",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "log_file" => Some(Self::LogFile),
            "stdin" => Some(Self::Stdin),
            "process" => Some(Self::Process),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RunMetadata {
    pub display_name: Option<String>,
    pub project_root: Option<String>,
    pub command: Option<String>,
    pub cwd: Option<String>,
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
    pub source_locator: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RunRecord {
    pub run_id: String,
    pub source_fingerprint: String,
    pub source_kind: RunSourceKind,
    pub source_locator: Option<String>,
    pub project_root: Option<String>,
    pub display_name: Option<String>,
    pub tags: Vec<String>,
    pub status: RunStatus,
    pub command: Option<String>,
    pub cwd: Option<String>,
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
    pub started_at_epoch_secs: i64,
    pub ended_at_epoch_secs: Option<i64>,
    pub last_step: Option<u64>,
    pub last_updated_epoch_secs: i64,
}

pub fn normalize_run_tags(tags: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    for tag in tags {
        let clean = tag.trim().to_ascii_lowercase();
        if clean.is_empty() || normalized.iter().any(|existing| existing == &clean) {
            continue;
        }
        normalized.push(clean);
    }
    normalized
}

pub fn parse_run_tags(raw: &str) -> Vec<String> {
    let parts = raw
        .split(',')
        .map(|tag| tag.trim().to_string())
        .collect::<Vec<_>>();
    normalize_run_tags(&parts)
}

pub fn encode_run_tags(tags: &[String]) -> String {
    normalize_run_tags(tags).join(",")
}

#[derive(Debug, Clone)]
pub struct RunAttachResult {
    pub run_id: String,
    pub reused_existing_active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunEventRecord {
    pub id: i64,
    pub run_id: String,
    pub kind: String,
    pub note: Option<String>,
    pub pinned: bool,
    pub event_epoch_secs: i64,
    pub step: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventJumpTarget {
    pub run_id: String,
    pub event_epoch_secs: i64,
    pub step: Option<u64>,
}

pub fn now_epoch_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now();
    now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

pub fn run_explorer_columns() -> Vec<&'static str> {
    vec![
        "Name",
        "Project",
        "Status",
        "Duration",
        "Best Metric",
        "Current/Final Step",
        "Start Date",
        "Git State",
        "Device Info",
    ]
}

pub fn filter_runs_by_project_status_date(
    rows: &[(String, String, String)],
    project: &str,
    status: &str,
    date: &str,
) -> Vec<(String, String, String)> {
    rows.iter()
        .filter(|(p, s, d)| p == project && s == status && d == date)
        .cloned()
        .collect()
}

pub fn fuzzy_search_runs(rows: &[String], query: &str) -> Vec<String> {
    if query.is_empty() {
        return rows.to_vec();
    }
    let query_lower = query.to_ascii_lowercase();
    rows.iter()
        .filter(|row| row.to_ascii_lowercase().contains(&query_lower))
        .cloned()
        .collect()
}

pub fn system_processes_columns() -> [&'static str; 5] {
    ["PID", "Command", "CWD", "CPU", "Memory"]
}
