use crossterm::event::{KeyEvent, MouseEvent};

use crate::types::{SystemMetrics, TrainingMetrics};

#[derive(Debug)]
pub enum Event {
    Tick,
    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),
    Metrics(TrainingMetrics),
    System(SystemMetrics),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = Event::Tick;
        matches!(event, Event::Tick);
    }
}
