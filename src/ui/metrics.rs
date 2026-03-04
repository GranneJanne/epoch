use ratatui::Frame;
use ratatui::layout::Rect;

use crate::app::TrainingState;

pub fn render_metrics(_frame: &mut Frame, _area: Rect, _training: &TrainingState) {
    todo!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        assert!(true);
    }
}
