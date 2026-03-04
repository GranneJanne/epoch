pub mod dashboard;
pub mod header;
pub mod metrics;
pub mod system;

use ratatui::Frame;

use crate::app::App;

pub fn render(_frame: &mut Frame, _app: &App) {
    todo!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        assert!(true);
    }
}
