use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config as MatcherConfig, Matcher};

use crate::config::Config;
use crate::discovery::DiscoveredFile;
use crate::event::Event;
use crate::types::{SystemMetrics, TrainingMetrics};
use crate::ui::Tab;

#[derive(Debug)]
pub struct TrainingState {
    pub latest: Option<TrainingMetrics>,
    pub loss_history: VecDeque<u64>,
    pub lr_history: VecDeque<u64>,
    pub step_history: VecDeque<u64>,
    pub throughput_history: VecDeque<u64>,
    pub tokens_history: VecDeque<u64>,
    pub eval_loss_history: VecDeque<u64>,
    pub grad_norm_history: VecDeque<u64>,
    pub samples_per_second_history: VecDeque<u64>,
    pub steps_per_second_history: VecDeque<u64>,
    pub tokens_per_second_history: VecDeque<u64>,
    pub perplexity_latest: Option<f64>,
    pub loss_spike_count: u64,
    pub nan_inf_count: u64,
    pub last_loss_spike_at: Option<Instant>,
    pub last_nan_inf_at: Option<Instant>,
    pub total_steps: u64,
    pub start_time: Option<Instant>,
    pub input_active: bool,
    pub last_data_at: Option<Instant>,
}

#[derive(Debug)]
pub struct SystemState {
    pub latest: Option<SystemMetrics>,
    pub cpu_history: VecDeque<u64>,
    pub ram_history: VecDeque<u64>,
    pub gpu_history: VecDeque<u64>,
}

#[derive(Debug)]
pub struct UiState {
    pub selected_tab: Tab,
    pub mode: AppMode,
    pub selected_file: Option<PathBuf>,
    pub scanning_frame: usize,
    pub training_viewport: ViewportState,
    pub system_viewport: ViewportState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportState {
    pub follow_latest: bool,
    pub offset_samples: usize,
    pub zoom_level: u8,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            follow_latest: true,
            offset_samples: 0,
            zoom_level: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataHealthState {
    Live,
    Stale,
    NoData,
}

impl DataHealthState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Live => "Live",
            Self::Stale => "Stale",
            Self::NoData => "No data",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppMode {
    Scanning,
    FilePicker(FilePickerState),
    Monitoring,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FilePickerState {
    pub files: Vec<DiscoveredFile>,
    pub query: String,
    pub filtered_indices: Vec<usize>,
    pub selected_index: usize,
}

#[derive(Clone)]
struct FuzzyCandidate {
    index: usize,
    text: String,
}

impl AsRef<str> for FuzzyCandidate {
    fn as_ref(&self) -> &str {
        &self.text
    }
}

impl FilePickerState {
    pub fn new(files: Vec<DiscoveredFile>) -> Self {
        Self {
            filtered_indices: (0..files.len()).collect(),
            files,
            query: String::new(),
            selected_index: 0,
        }
    }

    pub fn refresh_filter(&mut self) {
        if self.query.is_empty() {
            self.filtered_indices = (0..self.files.len()).collect();
        } else {
            let candidates = self
                .files
                .iter()
                .enumerate()
                .map(|(index, file)| FuzzyCandidate {
                    index,
                    text: file.path.to_string_lossy().to_string(),
                })
                .collect::<Vec<_>>();

            let pattern = Pattern::parse(&self.query, CaseMatching::Smart, Normalization::Smart);
            let mut matcher = Matcher::new(MatcherConfig::DEFAULT.match_paths());
            self.filtered_indices = pattern
                .match_list(candidates, &mut matcher)
                .into_iter()
                .map(|(candidate, _)| candidate.index)
                .collect();
        }

        if self.filtered_indices.is_empty() {
            self.selected_index = 0;
        } else if self.selected_index >= self.filtered_indices.len() {
            self.selected_index = self.filtered_indices.len() - 1;
        }
    }

    fn move_down(&mut self) {
        if self.filtered_indices.is_empty() {
            return;
        }

        self.selected_index = (self.selected_index + 1) % self.filtered_indices.len();
    }

    fn move_up(&mut self) {
        if self.filtered_indices.is_empty() {
            return;
        }

        if self.selected_index == 0 {
            self.selected_index = self.filtered_indices.len() - 1;
        } else {
            self.selected_index -= 1;
        }
    }
}

#[derive(Debug)]
pub struct App {
    pub running: bool,
    pub training: TrainingState,
    pub system: SystemState,
    pub ui_state: UiState,
    pub config: Config,
}

impl App {
    const VIEWPORT_PAN_STEP: usize = 10;
    const VIEWPORT_MAX_ZOOM_LEVEL: u8 = 6;

    pub fn new(config: Config) -> Self {
        let capacity = config.history_size;
        Self {
            running: true,
            training: TrainingState {
                latest: None,
                loss_history: VecDeque::with_capacity(capacity),
                lr_history: VecDeque::with_capacity(capacity),
                step_history: VecDeque::with_capacity(capacity),
                throughput_history: VecDeque::with_capacity(capacity),
                tokens_history: VecDeque::with_capacity(capacity),
                eval_loss_history: VecDeque::with_capacity(capacity),
                grad_norm_history: VecDeque::with_capacity(capacity),
                samples_per_second_history: VecDeque::with_capacity(capacity),
                steps_per_second_history: VecDeque::with_capacity(capacity),
                tokens_per_second_history: VecDeque::with_capacity(capacity),
                perplexity_latest: None,
                loss_spike_count: 0,
                nan_inf_count: 0,
                last_loss_spike_at: None,
                last_nan_inf_at: None,
                total_steps: 0,
                start_time: None,
                input_active: false,
                last_data_at: None,
            },
            system: SystemState {
                latest: None,
                cpu_history: VecDeque::with_capacity(capacity),
                ram_history: VecDeque::with_capacity(capacity),
                gpu_history: VecDeque::with_capacity(capacity),
            },
            ui_state: UiState {
                selected_tab: Tab::Dashboard,
                mode: AppMode::Monitoring,
                selected_file: None,
                scanning_frame: 0,
                training_viewport: ViewportState::default(),
                system_viewport: ViewportState::default(),
            },
            config,
        }
    }

    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) => self.handle_key(key),
            Event::Tick => self.on_tick(),
            Event::Metrics(m) => self.push_metrics(m),
            Event::System(s) => self.push_system(s),
            Event::Resize(..) | Event::Mouse(..) => {}
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), KeyModifiers::NONE) => {
                self.running = false;
                return;
            }
            (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.running = false;
                return;
            }
            _ => {}
        }

        if let AppMode::FilePicker(ref mut picker) = self.ui_state.mode {
            match (key.code, key.modifiers) {
                (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                    picker.move_up();
                }
                (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                    picker.move_down();
                }
                (KeyCode::Backspace, _) => {
                    picker.query.pop();
                    picker.refresh_filter();
                }
                (KeyCode::Enter, _) => {
                    if let Some(index) = picker.filtered_indices.get(picker.selected_index).copied()
                    {
                        self.ui_state.selected_file = Some(picker.files[index].path.clone());
                        self.ui_state.mode = AppMode::Monitoring;
                    } else if !picker.query.trim().is_empty() {
                        self.ui_state.selected_file = Some(PathBuf::from(picker.query.clone()));
                        self.ui_state.mode = AppMode::Monitoring;
                    }
                }
                (KeyCode::Esc, _) => {
                    self.running = false;
                }
                (KeyCode::Char(c), KeyModifiers::NONE) => {
                    picker.query.push(c);
                    picker.refresh_filter();
                }
                _ => {}
            }
            return;
        }

        match (key.code, key.modifiers) {
            (KeyCode::Tab, _) => {
                let current = self.ui_state.selected_tab as usize;
                self.ui_state.selected_tab =
                    Tab::from_repr((current + 1) % 4).unwrap_or(Tab::Dashboard);
            }
            (KeyCode::BackTab, _) => {
                let current = self.ui_state.selected_tab as usize;
                self.ui_state.selected_tab =
                    Tab::from_repr((current + 3) % 4).unwrap_or(Tab::Dashboard);
            }
            (KeyCode::Char('1'), KeyModifiers::NONE) => {
                self.ui_state.selected_tab = Tab::Dashboard;
            }
            (KeyCode::Char('2'), KeyModifiers::NONE) => {
                self.ui_state.selected_tab = Tab::Metrics;
            }
            (KeyCode::Char('3'), KeyModifiers::NONE) => {
                self.ui_state.selected_tab = Tab::System;
            }
            (KeyCode::Char('4'), KeyModifiers::NONE) => {
                self.ui_state.selected_tab = Tab::Advanced;
            }
            (KeyCode::Char(' '), KeyModifiers::NONE) => {
                let follow_latest = !self.ui_state.training_viewport.follow_latest;
                self.ui_state.training_viewport.follow_latest = follow_latest;
                self.ui_state.system_viewport.follow_latest = follow_latest;
                if follow_latest {
                    self.ui_state.training_viewport.offset_samples = 0;
                    self.ui_state.system_viewport.offset_samples = 0;
                }
            }
            (KeyCode::Left, KeyModifiers::NONE) => {
                if !self.ui_state.training_viewport.follow_latest {
                    self.ui_state.training_viewport.offset_samples = self
                        .ui_state
                        .training_viewport
                        .offset_samples
                        .saturating_add(Self::VIEWPORT_PAN_STEP);
                }
                if !self.ui_state.system_viewport.follow_latest {
                    self.ui_state.system_viewport.offset_samples = self
                        .ui_state
                        .system_viewport
                        .offset_samples
                        .saturating_add(Self::VIEWPORT_PAN_STEP);
                }
            }
            (KeyCode::Right, KeyModifiers::NONE) => {
                self.ui_state.training_viewport.offset_samples = self
                    .ui_state
                    .training_viewport
                    .offset_samples
                    .saturating_sub(Self::VIEWPORT_PAN_STEP);
                self.ui_state.system_viewport.offset_samples = self
                    .ui_state
                    .system_viewport
                    .offset_samples
                    .saturating_sub(Self::VIEWPORT_PAN_STEP);
            }
            (KeyCode::Char('g'), KeyModifiers::NONE) => {
                self.ui_state.training_viewport.follow_latest = true;
                self.ui_state.system_viewport.follow_latest = true;
                self.ui_state.training_viewport.offset_samples = 0;
                self.ui_state.system_viewport.offset_samples = 0;
            }
            (KeyCode::Char('-'), KeyModifiers::NONE) => {
                self.ui_state.training_viewport.zoom_level = self
                    .ui_state
                    .training_viewport
                    .zoom_level
                    .saturating_add(1)
                    .min(Self::VIEWPORT_MAX_ZOOM_LEVEL);
                self.ui_state.system_viewport.zoom_level = self
                    .ui_state
                    .system_viewport
                    .zoom_level
                    .saturating_add(1)
                    .min(Self::VIEWPORT_MAX_ZOOM_LEVEL);
            }
            (KeyCode::Char('='), KeyModifiers::NONE) => {
                self.ui_state.training_viewport.zoom_level =
                    self.ui_state.training_viewport.zoom_level.saturating_sub(1);
                self.ui_state.system_viewport.zoom_level =
                    self.ui_state.system_viewport.zoom_level.saturating_sub(1);
            }
            _ => {}
        }
    }

    pub fn on_tick(&mut self) {
        if matches!(self.ui_state.mode, AppMode::Scanning) {
            self.ui_state.scanning_frame = (self.ui_state.scanning_frame + 1) % 4;
        }

        if let Some(last_data) = self.training.last_data_at {
            if last_data.elapsed() > Duration::from_secs(self.config.stale_after_secs) {
                self.training.input_active = false;
            }
        }
    }

    pub fn training_data_health_state(&self) -> DataHealthState {
        if self.training.input_active {
            DataHealthState::Live
        } else if self.training.last_data_at.is_some() {
            DataHealthState::Stale
        } else {
            DataHealthState::NoData
        }
    }

    pub fn training_viewport_series(&self, history: &VecDeque<u64>, width: usize) -> Vec<u64> {
        Self::viewport_series(history, self.ui_state.training_viewport, width)
    }

    pub fn system_viewport_series(&self, history: &VecDeque<u64>, width: usize) -> Vec<u64> {
        Self::viewport_series(history, self.ui_state.system_viewport, width)
    }

    pub fn push_metrics(&mut self, m: TrainingMetrics) {
        let capacity = self.config.history_size;

        let invalid_count = Self::count_non_finite_metrics(&m);
        self.training.nan_inf_count += invalid_count;
        if invalid_count > 0 {
            self.training.last_nan_inf_at = Some(Instant::now());
        }

        if let Some(loss) = m.loss
            && loss.is_finite()
        {
            self.training.perplexity_latest = Some(Self::safe_perplexity(loss));

            if Self::is_loss_spike(&self.training.loss_history, loss, 1000.0, 20, 1.2) {
                self.training.loss_spike_count += 1;
                self.training.last_loss_spike_at = Some(Instant::now());
            }
        }

        self.training.latest = Some(m.clone());

        if let Some(loss) = m.loss {
            let scaled = Self::scale_to_u64(loss, 1000.0);
            Self::push_bounded(&mut self.training.loss_history, scaled, capacity);
        }

        if let Some(lr) = m.learning_rate {
            let scaled = Self::scale_to_u64(lr, 1_000_000.0);
            Self::push_bounded(&mut self.training.lr_history, scaled, capacity);
        }

        if let Some(step) = m.step {
            Self::push_bounded(&mut self.training.step_history, step, capacity);
            self.training.total_steps = self.training.total_steps.max(step);
        }

        let throughput_value = m
            .tokens_per_second
            .or(m.samples_per_second)
            .or(m.throughput);
        if let Some(throughput) = throughput_value {
            let scaled = Self::scale_to_u64(throughput, 1.0);
            Self::push_bounded(&mut self.training.throughput_history, scaled, capacity);
        }

        if let Some(tokens) = m.tokens {
            Self::push_bounded(&mut self.training.tokens_history, tokens, capacity);
        }

        if let Some(eval_loss) = m.eval_loss {
            let scaled = Self::scale_to_u64(eval_loss, 1000.0);
            Self::push_bounded(&mut self.training.eval_loss_history, scaled, capacity);
        }

        if let Some(grad_norm) = m.grad_norm {
            let scaled = Self::scale_to_u64(grad_norm, 1000.0);
            Self::push_bounded(&mut self.training.grad_norm_history, scaled, capacity);
        }

        if let Some(samples_per_second) = m.samples_per_second {
            let scaled = Self::scale_to_u64(samples_per_second, 1.0);
            Self::push_bounded(
                &mut self.training.samples_per_second_history,
                scaled,
                capacity,
            );
        }

        if let Some(steps_per_second) = m.steps_per_second {
            let scaled = Self::scale_to_u64(steps_per_second, 1000.0);
            Self::push_bounded(
                &mut self.training.steps_per_second_history,
                scaled,
                capacity,
            );
        }

        if let Some(tokens_per_second) = m.tokens_per_second {
            let scaled = Self::scale_to_u64(tokens_per_second, 1.0);
            Self::push_bounded(
                &mut self.training.tokens_per_second_history,
                scaled,
                capacity,
            );
        }

        self.training.input_active = true;
        self.training.last_data_at = Some(Instant::now());

        if self.training.start_time.is_none() {
            self.training.start_time = Some(Instant::now());
        }
    }

    pub fn push_system(&mut self, s: SystemMetrics) {
        let capacity = self.config.history_size;

        self.system.latest = Some(s.clone());

        let cpu_scaled = Self::scale_to_u64(s.cpu_usage_percent(), 100.0);
        Self::push_bounded(&mut self.system.cpu_history, cpu_scaled, capacity);

        let ram_scaled = Self::scale_to_u64(s.memory_usage_percent(), 100.0);
        Self::push_bounded(&mut self.system.ram_history, ram_scaled, capacity);

        if s.has_gpu() && !s.gpus.is_empty() {
            let gpu_scaled = Self::scale_to_u64(s.gpus[0].utilization, 100.0);
            Self::push_bounded(&mut self.system.gpu_history, gpu_scaled, capacity);
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.training
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    fn push_bounded(buf: &mut VecDeque<u64>, value: u64, capacity: usize) {
        buf.push_back(value);
        if buf.len() > capacity {
            buf.pop_front();
        }
    }

    fn scale_to_u64(value: f64, factor: f64) -> u64 {
        if !value.is_finite() || value <= 0.0 || !factor.is_finite() || factor <= 0.0 {
            return 0;
        }

        let clamped = value.clamp(0.0, f64::MAX / factor);
        (clamped * factor) as u64
    }

    fn safe_perplexity(loss: f64) -> f64 {
        loss.clamp(0.0, 50.0).exp()
    }

    fn count_non_finite_metrics(m: &TrainingMetrics) -> u64 {
        [
            m.loss,
            m.learning_rate,
            m.throughput,
            m.eval_loss,
            m.grad_norm,
            m.samples_per_second,
            m.steps_per_second,
            m.tokens_per_second,
        ]
        .iter()
        .filter(|v| v.is_some_and(|n| !n.is_finite()))
        .count() as u64
    }

    fn is_loss_spike(
        history: &VecDeque<u64>,
        current_loss: f64,
        scale: f64,
        window: usize,
        threshold_multiplier: f64,
    ) -> bool {
        let baseline_values: Vec<f64> = history
            .iter()
            .rev()
            .take(window)
            .copied()
            .map(|v| v as f64 / scale)
            .collect();

        if baseline_values.len() < 5 {
            return false;
        }

        let baseline_mean = baseline_values.iter().sum::<f64>() / baseline_values.len() as f64;
        current_loss > baseline_mean * threshold_multiplier
    }

    fn viewport_series(history: &VecDeque<u64>, viewport: ViewportState, width: usize) -> Vec<u64> {
        if history.is_empty() {
            return Vec::new();
        }

        let width = width.max(1);
        let zoom_factor = 1usize << viewport.zoom_level.min(Self::VIEWPORT_MAX_ZOOM_LEVEL);
        let window = width.saturating_mul(zoom_factor).max(1);
        let history_len = history.len();
        let max_start = history_len.saturating_sub(window);
        let offset = if viewport.follow_latest {
            0
        } else {
            viewport.offset_samples.min(max_start)
        };
        let start = max_start.saturating_sub(offset);
        let end = (start + window).min(history_len);

        let sampled: Vec<u64> = history
            .iter()
            .skip(start)
            .take(end - start)
            .copied()
            .collect();
        if sampled.len() <= width {
            return sampled;
        }

        let step = sampled.len().div_ceil(width);
        sampled.into_iter().step_by(step).take(width).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;
    use strum::IntoEnumIterator;

    use crate::discovery::FileFormat;
    use crate::types::GpuMetrics;

    fn sample_discovered_files() -> Vec<DiscoveredFile> {
        vec![
            DiscoveredFile {
                path: PathBuf::from("/tmp/a.jsonl"),
                format: FileFormat::Jsonl,
                modified: UNIX_EPOCH,
            },
            DiscoveredFile {
                path: PathBuf::from("/tmp/b.csv"),
                format: FileFormat::Csv,
                modified: UNIX_EPOCH,
            },
        ]
    }

    #[test]
    fn test_app_new_defaults() {
        let app = App::new(Config::default());
        assert!(app.running);
        assert!(app.training.loss_history.is_empty());
        assert!(app.training.lr_history.is_empty());
        assert!(app.training.step_history.is_empty());
        assert!(app.training.throughput_history.is_empty());
        assert!(app.training.tokens_history.is_empty());
        assert!(app.training.eval_loss_history.is_empty());
        assert!(app.training.grad_norm_history.is_empty());
        assert!(app.training.samples_per_second_history.is_empty());
        assert!(app.training.steps_per_second_history.is_empty());
        assert!(app.training.tokens_per_second_history.is_empty());
        assert!(app.training.perplexity_latest.is_none());
        assert_eq!(app.training.loss_spike_count, 0);
        assert_eq!(app.training.nan_inf_count, 0);
        assert!(app.training.last_loss_spike_at.is_none());
        assert!(app.training.last_nan_inf_at.is_none());
        assert_eq!(app.ui_state.selected_tab, Tab::Dashboard);
        assert_eq!(app.ui_state.mode, AppMode::Monitoring);
        assert!(app.ui_state.selected_file.is_none());
        assert_eq!(app.ui_state.scanning_frame, 0);
        assert!(app.ui_state.training_viewport.follow_latest);
        assert_eq!(app.ui_state.training_viewport.offset_samples, 0);
        assert_eq!(app.ui_state.training_viewport.zoom_level, 0);
        assert!(app.ui_state.system_viewport.follow_latest);
        assert_eq!(app.ui_state.system_viewport.offset_samples, 0);
        assert_eq!(app.ui_state.system_viewport.zoom_level, 0);
        assert!(app.training.latest.is_none());
        assert!(app.system.latest.is_none());
    }

    #[test]
    fn test_scanning_mode_advances_spinner_on_tick() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::Scanning;
        assert_eq!(app.ui_state.scanning_frame, 0);

        app.on_tick();
        assert_eq!(app.ui_state.scanning_frame, 1);

        app.on_tick();
        app.on_tick();
        app.on_tick();
        assert_eq!(app.ui_state.scanning_frame, 0);
    }

    #[test]
    fn test_app_default_mode_is_monitoring() {
        let app = App::new(Config::default());
        assert_eq!(app.ui_state.mode, AppMode::Monitoring);
    }

    #[test]
    fn test_file_picker_state_creation() {
        let files = sample_discovered_files();
        let state = FilePickerState::new(files.clone());

        assert_eq!(state.files, files);
        assert_eq!(state.query, "");
        assert_eq!(state.filtered_indices, vec![0, 1]);
        assert_eq!(state.selected_index, 0);
    }

    #[test]
    fn test_file_picker_navigation_down() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::FilePicker(FilePickerState::new(sample_discovered_files()));

        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));

        assert!(
            matches!(app.ui_state.mode, AppMode::FilePicker(ref state) if state.selected_index == 1)
        );
    }

    #[test]
    fn test_file_picker_navigation_up() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::FilePicker(FilePickerState::new(sample_discovered_files()));

        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));

        assert!(
            matches!(app.ui_state.mode, AppMode::FilePicker(ref state) if state.selected_index == 1)
        );
    }

    #[test]
    fn test_file_picker_query_input() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::FilePicker(FilePickerState::new(sample_discovered_files()));

        app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));

        assert!(matches!(app.ui_state.mode, AppMode::FilePicker(ref state) if state.query == "a"));
    }

    #[test]
    fn test_file_picker_query_fuzzy_match() {
        let mut state = FilePickerState::new(sample_discovered_files());
        state.query = "ajsn".to_string();
        state.refresh_filter();

        assert!(!state.filtered_indices.is_empty());
        let first = state.filtered_indices[0];
        assert_eq!(state.files[first].path, PathBuf::from("/tmp/a.jsonl"));
    }

    #[test]
    fn test_file_picker_backspace() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::FilePicker(FilePickerState {
            query: "ab".to_string(),
            ..FilePickerState::new(sample_discovered_files())
        });

        app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));

        assert!(matches!(app.ui_state.mode, AppMode::FilePicker(ref state) if state.query == "a"));
    }

    #[test]
    fn test_file_picker_enter_selects() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::FilePicker(FilePickerState::new(sample_discovered_files()));

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.ui_state.mode, AppMode::Monitoring);
        assert_eq!(
            app.ui_state.selected_file,
            Some(PathBuf::from("/tmp/a.jsonl"))
        );
    }

    #[test]
    fn test_file_picker_enter_uses_query_path_when_no_matches() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::FilePicker(FilePickerState {
            files: vec![],
            query: "/tmp/manual.jsonl".to_string(),
            filtered_indices: vec![],
            selected_index: 0,
        });

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.ui_state.mode, AppMode::Monitoring);
        assert_eq!(
            app.ui_state.selected_file,
            Some(PathBuf::from("/tmp/manual.jsonl"))
        );
    }

    #[test]
    fn test_file_picker_escape_quits() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::FilePicker(FilePickerState::new(sample_discovered_files()));

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        assert!(!app.running);
    }

    #[test]
    fn test_tab_key_in_monitoring_mode_still_works() {
        let mut app = App::new(Config::default());
        app.ui_state.mode = AppMode::Monitoring;

        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.ui_state.selected_tab, Tab::Metrics);
    }

    #[test]
    fn test_push_metrics_stores_latest() {
        let mut app = App::new(Config::default());
        let metrics = TrainingMetrics {
            loss: Some(0.5),
            learning_rate: Some(0.001),
            step: Some(100),
            throughput: Some(1000.0),
            tokens: Some(50000),
            eval_loss: None,
            grad_norm: None,
            samples_per_second: None,
            steps_per_second: None,
            tokens_per_second: None,
            timestamp: Instant::now(),
        };
        app.push_metrics(metrics);
        assert!(app.training.latest.is_some());
        assert_eq!(app.training.latest.as_ref().unwrap().loss, Some(0.5));
    }

    #[test]
    fn test_push_metrics_appends_to_history() {
        let mut app = App::new(Config::default());
        let metrics = TrainingMetrics {
            loss: Some(0.5),
            ..TrainingMetrics::default()
        };
        app.push_metrics(metrics);
        assert_eq!(app.training.loss_history.len(), 1);
        assert_eq!(app.training.loss_history[0], 500); // 0.5 * 1000
    }

    #[test]
    fn test_history_respects_capacity() {
        let config = Config {
            history_size: 300,
            ..Config::default()
        };
        let mut app = App::new(config);
        // Push 400 items
        for i in 0..400 {
            let metrics = TrainingMetrics {
                loss: Some(i as f64),
                ..TrainingMetrics::default()
            };
            app.push_metrics(metrics);
        }
        assert_eq!(app.training.loss_history.len(), 300);
    }

    #[test]
    fn test_handle_key_q_quits() {
        let mut app = App::new(Config::default());
        let key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        app.handle_key(key);
        assert!(!app.running);
    }

    #[test]
    fn test_handle_key_ctrl_c_quits() {
        let mut app = App::new(Config::default());
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        app.handle_key(key);
        assert!(!app.running);
    }

    #[test]
    fn test_tab_cycle_forward() {
        let mut app = App::new(Config::default());
        assert_eq!(app.ui_state.selected_tab, Tab::Dashboard);

        let tab_key = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::Metrics);

        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::System);

        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::Advanced);

        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::Dashboard); // wrap
    }

    #[test]
    fn test_tab_cycle_backward() {
        let mut app = App::new(Config::default());
        assert_eq!(app.ui_state.selected_tab, Tab::Dashboard);

        let backtab_key = KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT);
        app.handle_key(backtab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::Advanced); // wrap around
    }

    #[test]
    fn test_tab_direct_number() {
        let mut app = App::new(Config::default());

        let key1 = KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE);
        app.handle_key(key1);
        assert_eq!(app.ui_state.selected_tab, Tab::Dashboard);

        let key2 = KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE);
        app.handle_key(key2);
        assert_eq!(app.ui_state.selected_tab, Tab::Metrics);

        let key3 = KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE);
        app.handle_key(key3);
        assert_eq!(app.ui_state.selected_tab, Tab::System);

        let key4 = KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE);
        app.handle_key(key4);
        assert_eq!(app.ui_state.selected_tab, Tab::Advanced);
    }

    #[test]
    fn test_tab_iteration_count_is_4() {
        let tabs: Vec<Tab> = Tab::iter().collect();
        assert_eq!(tabs.len(), 4);
    }

    #[test]
    fn test_tab_cycle_forward_wraps_with_advanced() {
        let mut app = App::new(Config::default());
        let tab_key = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);

        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::Metrics);
        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::System);
        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::Advanced);
        app.handle_key(tab_key);
        assert_eq!(app.ui_state.selected_tab, Tab::Dashboard);
    }

    #[test]
    fn test_direct_tab_jump_4_opens_advanced() {
        let mut app = App::new(Config::default());
        app.handle_key(KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE));
        assert_eq!(app.ui_state.selected_tab, Tab::Advanced);
    }

    #[test]
    fn test_on_tick_staleness() {
        let mut app = App::new(Config::default());
        // Simulate old data
        app.training.last_data_at = Some(Instant::now() - Duration::from_secs(11));
        app.training.input_active = true;

        app.on_tick();
        assert!(!app.training.input_active);
    }

    #[test]
    fn test_staleness_threshold_uses_config_value() {
        let mut app = App::new(Config {
            stale_after_secs: 30,
            ..Config::default()
        });
        app.training.last_data_at = Some(Instant::now() - Duration::from_secs(11));
        app.training.input_active = true;

        app.on_tick();
        assert!(app.training.input_active);

        app.training.last_data_at = Some(Instant::now() - Duration::from_secs(31));
        app.on_tick();
        assert!(!app.training.input_active);
    }

    #[test]
    fn test_viewport_live_follow_shows_latest() {
        let mut app = App::new(Config::default());

        for i in 0..100 {
            app.push_metrics(TrainingMetrics {
                step: Some(i),
                ..TrainingMetrics::default()
            });
        }

        let series = app.training_viewport_series(&app.training.step_history, 12);
        assert_eq!(series.len(), 12);
        assert_eq!(series.last().copied(), Some(99));

        app.push_metrics(TrainingMetrics {
            step: Some(100),
            ..TrainingMetrics::default()
        });

        let updated = app.training_viewport_series(&app.training.step_history, 12);
        assert_eq!(updated.last().copied(), Some(100));
    }

    #[test]
    fn test_viewport_pan_clamps_bounds() {
        let mut app = App::new(Config::default());

        for i in 0..50 {
            app.push_metrics(TrainingMetrics {
                step: Some(i),
                ..TrainingMetrics::default()
            });
        }

        app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
        app.ui_state.training_viewport.offset_samples = usize::MAX;

        let series = app.training_viewport_series(&app.training.step_history, 10);
        assert_eq!(series.len(), 10);
        assert_eq!(series.first().copied(), Some(0));
        assert_eq!(series.last().copied(), Some(9));
    }

    #[test]
    fn test_viewport_zoom_clamps_and_reslices() {
        let mut app = App::new(Config::default());

        for i in 0..256 {
            app.push_metrics(TrainingMetrics {
                step: Some(i),
                ..TrainingMetrics::default()
            });
        }

        let baseline = app.training_viewport_series(&app.training.step_history, 16);
        assert_eq!(baseline.len(), 16);

        for _ in 0..20 {
            app.handle_key(KeyEvent::new(KeyCode::Char('-'), KeyModifiers::NONE));
        }
        assert_eq!(
            app.ui_state.training_viewport.zoom_level,
            App::VIEWPORT_MAX_ZOOM_LEVEL
        );

        let zoomed_out = app.training_viewport_series(&app.training.step_history, 16);
        assert_eq!(zoomed_out.len(), 16);
        assert_ne!(zoomed_out.first(), baseline.first());

        for _ in 0..20 {
            app.handle_key(KeyEvent::new(KeyCode::Char('='), KeyModifiers::NONE));
        }
        assert_eq!(app.ui_state.training_viewport.zoom_level, 0);

        let zoomed_in = app.training_viewport_series(&app.training.step_history, 16);
        assert_eq!(zoomed_in.len(), 16);
        assert_eq!(zoomed_in.last().copied(), Some(255));
    }

    #[test]
    fn test_push_metrics_sets_active() {
        let mut app = App::new(Config::default());
        let metrics = TrainingMetrics {
            loss: Some(0.5),
            ..TrainingMetrics::default()
        };
        app.push_metrics(metrics);
        assert!(app.training.input_active);
    }

    #[test]
    fn test_push_system_updates() {
        let mut app = App::new(Config::default());
        let system = SystemMetrics {
            cpu_usage: 50.0,
            memory_used: 4_000_000_000,
            memory_total: 16_000_000_000,
            gpus: vec![],
        };
        app.push_system(system);
        assert_eq!(app.system.cpu_history.len(), 1);
        assert_eq!(app.system.cpu_history[0], 5000); // 50.0 * 100
    }

    #[test]
    fn test_elapsed_zero_before_data() {
        let app = App::new(Config::default());
        assert_eq!(app.elapsed(), Duration::ZERO);
    }

    #[test]
    fn test_handle_event_dispatches() {
        let mut app = App::new(Config::default());

        // Test Event::Tick dispatch
        app.training.last_data_at = Some(Instant::now() - Duration::from_secs(11));
        app.training.input_active = true;
        app.handle_event(Event::Tick);
        assert!(!app.training.input_active);

        // Test Event::Metrics dispatch
        let metrics = TrainingMetrics {
            loss: Some(0.5),
            ..TrainingMetrics::default()
        };
        app.handle_event(Event::Metrics(metrics));
        assert!(app.training.latest.is_some());
    }

    #[test]
    fn test_push_metrics_all_fields() {
        let mut app = App::new(Config::default());
        let metrics = TrainingMetrics {
            loss: Some(0.5),
            learning_rate: Some(0.001),
            step: Some(100),
            throughput: Some(1000.0),
            tokens: Some(50000),
            eval_loss: Some(0.45),
            grad_norm: Some(1.75),
            samples_per_second: Some(12.0),
            steps_per_second: Some(0.5),
            tokens_per_second: Some(1500.0),
            timestamp: Instant::now(),
        };
        app.push_metrics(metrics);

        assert_eq!(app.training.loss_history.len(), 1);
        assert_eq!(app.training.loss_history[0], 500); // 0.5 * 1000

        assert_eq!(app.training.lr_history.len(), 1);
        assert_eq!(app.training.lr_history[0], 1000); // 0.001 * 1_000_000

        assert_eq!(app.training.step_history.len(), 1);
        assert_eq!(app.training.step_history[0], 100);

        assert_eq!(app.training.throughput_history.len(), 1);
        assert_eq!(app.training.throughput_history[0], 1500);

        assert_eq!(app.training.tokens_history.len(), 1);
        assert_eq!(app.training.tokens_history[0], 50000);

        assert_eq!(app.training.eval_loss_history.len(), 1);
        assert_eq!(app.training.eval_loss_history[0], 450);

        assert_eq!(app.training.grad_norm_history.len(), 1);
        assert_eq!(app.training.grad_norm_history[0], 1750);

        assert_eq!(app.training.samples_per_second_history.len(), 1);
        assert_eq!(app.training.samples_per_second_history[0], 12);

        assert_eq!(app.training.steps_per_second_history.len(), 1);
        assert_eq!(app.training.steps_per_second_history[0], 500);

        assert_eq!(app.training.tokens_per_second_history.len(), 1);
        assert_eq!(app.training.tokens_per_second_history[0], 1500);

        assert_eq!(app.training.total_steps, 100);
    }

    #[test]
    fn test_push_metrics_appends_new_core_histories() {
        let mut app = App::new(Config::default());
        app.push_metrics(TrainingMetrics {
            tokens: Some(1200),
            eval_loss: Some(0.75),
            grad_norm: Some(2.0),
            samples_per_second: Some(21.0),
            steps_per_second: Some(0.75),
            tokens_per_second: Some(3000.0),
            ..TrainingMetrics::default()
        });

        assert_eq!(app.training.tokens_history.len(), 1);
        assert_eq!(app.training.eval_loss_history.len(), 1);
        assert_eq!(app.training.grad_norm_history.len(), 1);
        assert_eq!(app.training.samples_per_second_history.len(), 1);
        assert_eq!(app.training.steps_per_second_history.len(), 1);
        assert_eq!(app.training.tokens_per_second_history.len(), 1);
    }

    #[test]
    fn test_new_histories_respect_capacity() {
        let config = Config {
            history_size: 3,
            ..Config::default()
        };
        let mut app = App::new(config);

        for i in 0..10 {
            app.push_metrics(TrainingMetrics {
                tokens: Some(i),
                eval_loss: Some(i as f64),
                grad_norm: Some(i as f64),
                samples_per_second: Some(i as f64),
                steps_per_second: Some(i as f64),
                tokens_per_second: Some(i as f64),
                ..TrainingMetrics::default()
            });
        }

        assert_eq!(app.training.tokens_history.len(), 3);
        assert_eq!(app.training.eval_loss_history.len(), 3);
        assert_eq!(app.training.grad_norm_history.len(), 3);
        assert_eq!(app.training.samples_per_second_history.len(), 3);
        assert_eq!(app.training.steps_per_second_history.len(), 3);
        assert_eq!(app.training.tokens_per_second_history.len(), 3);
    }

    #[test]
    fn test_legacy_throughput_fallback_remains_intact() {
        let mut app = App::new(Config::default());
        app.push_metrics(TrainingMetrics {
            throughput: Some(42.0),
            ..TrainingMetrics::default()
        });

        assert_eq!(app.training.throughput_history.len(), 1);
        assert_eq!(app.training.throughput_history[0], 42);
    }

    #[test]
    fn test_perplexity_derived_from_loss() {
        let mut app = App::new(Config::default());
        app.push_metrics(TrainingMetrics {
            loss: Some(1.0),
            ..TrainingMetrics::default()
        });

        let perplexity = app
            .training
            .perplexity_latest
            .expect("perplexity should be calculated");
        assert!((perplexity - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_loss_spike_counter_increments_on_threshold_cross() {
        let mut app = App::new(Config::default());

        for _ in 0..25 {
            app.push_metrics(TrainingMetrics {
                loss: Some(1.0),
                ..TrainingMetrics::default()
            });
        }

        let before = app.training.loss_spike_count;
        app.push_metrics(TrainingMetrics {
            loss: Some(1.5),
            ..TrainingMetrics::default()
        });
        let after = app.training.loss_spike_count;

        assert_eq!(after, before + 1);
        assert!(app.training.last_loss_spike_at.is_some());
    }

    #[test]
    fn test_nan_inf_counter_tracks_invalid_metrics() {
        let mut app = App::new(Config::default());
        app.push_metrics(TrainingMetrics {
            loss: Some(f64::NAN),
            grad_norm: Some(f64::INFINITY),
            ..TrainingMetrics::default()
        });

        assert_eq!(app.training.nan_inf_count, 2);
        assert!(app.training.last_nan_inf_at.is_some());
    }

    #[test]
    fn test_push_system_with_gpu() {
        let mut app = App::new(Config::default());
        let system = SystemMetrics {
            cpu_usage: 50.0,
            memory_used: 8_000_000_000,
            memory_total: 16_000_000_000,
            gpus: vec![GpuMetrics {
                name: "RTX 4090".to_string(),
                utilization: 75.5,
                memory_used: 12_000_000_000,
                memory_total: 24_000_000_000,
                temperature: 65.0,
            }],
        };
        app.push_system(system);

        assert_eq!(app.system.cpu_history.len(), 1);
        assert_eq!(app.system.cpu_history[0], 5000); // 50.0 * 100

        assert_eq!(app.system.ram_history.len(), 1);
        assert_eq!(app.system.ram_history[0], 5000); // 50.0 * 100

        assert_eq!(app.system.gpu_history.len(), 1);
        assert_eq!(app.system.gpu_history[0], 7550); // 75.5 * 100
    }

    #[test]
    fn test_app_new() {
        let app = App::new(Config::default());
        assert!(app.running);
        assert_eq!(app.ui_state.selected_tab, Tab::Dashboard);
        assert!(app.training.latest.is_none());
    }
}
