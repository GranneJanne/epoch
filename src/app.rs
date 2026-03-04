use std::collections::VecDeque;

use crate::config::Config;
use crate::types::{SystemMetrics, TrainingMetrics};

#[derive(Debug)]
pub struct TrainingState {
    pub latest: Option<TrainingMetrics>,
    pub history: VecDeque<TrainingMetrics>,
}

#[derive(Debug)]
pub struct SystemState {
    pub latest: Option<SystemMetrics>,
}

#[derive(Debug)]
pub struct UiState {
    pub selected_tab: usize,
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
    pub fn new(config: Config) -> Self {
        Self {
            running: true,
            training: TrainingState {
                latest: None,
                history: VecDeque::new(),
            },
            system: SystemState { latest: None },
            ui_state: UiState { selected_tab: 0 },
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_new() {
        let app = App::new(Config::default());
        assert!(app.running);
        assert_eq!(app.ui_state.selected_tab, 0);
        assert!(app.training.latest.is_none());
    }
}
