use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::app::{App, HomeFocusTarget};
use crate::store::types::RunStatus;
use crate::ui::alerts_panel::{AlertPanelData, render_alert_panel};
use crate::ui::components::{centered_text_area, format_duration, format_epoch_date, format_step};
use crate::ui::run_explorer::{run_display_name, run_status_label};
use crate::ui::theme::resolve_palette_from_config;

const DETAILS_PANEL_HEIGHT: u16 = 11;
const SYSTEM_PANEL_HEIGHT: u16 = 5;
const RIGHT_COLUMN_TARGET_WIDTH: u16 = 58;
const RIGHT_COLUMN_MIN_WIDTH: u16 = 34;
const LEFT_COLUMN_MIN_WIDTH: u16 = 44;

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let palette = resolve_palette_from_config(&app.config);

    let right_width = right_column_width(area.width);

    let [left_col, right_col] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(right_width)])
        .areas(area);

    let [runs_area, processes_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(12),
            Constraint::Length(process_panel_height(app)),
        ])
        .areas(left_col);

    let [details_area, alerts_area, system_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(DETAILS_PANEL_HEIGHT),
            Constraint::Min(8),
            Constraint::Length(SYSTEM_PANEL_HEIGHT),
        ])
        .areas(right_col);

    let focus = &app.ui_state.monitoring.home_focus;

    crate::ui::run_explorer::render_runs_panel(
        frame,
        runs_area,
        app,
        *focus == HomeFocusTarget::Runs,
    );
    crate::ui::system_processes::render_processes_table(
        frame,
        processes_area,
        app,
        &palette,
        *focus == HomeFocusTarget::Processes,
    );
    render_run_details_panel(
        frame,
        details_area,
        app,
        &palette,
        *focus == HomeFocusTarget::RunDetails,
    );
    render_alerts(frame, alerts_area, app, &palette);
    crate::ui::system_processes::render_resource_strip(frame, system_area, app, &palette);
}

fn render_run_details_panel(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    palette: &crate::ui::theme::ThemePalette,
    is_focused: bool,
) {
    let mut border_style = Style::default().fg(palette.muted);
    let mut title_style = Style::default().fg(palette.header_fg);
    if is_focused {
        border_style = border_style.fg(palette.accent).add_modifier(Modifier::BOLD);
        title_style = title_style.fg(palette.accent).add_modifier(Modifier::BOLD);
    }

    let block = Block::default()
        .title("[1] Run Details")
        .title_style(title_style)
        .borders(Borders::ALL)
        .border_style(border_style);

    let Some(record) = app.selected_run_record() else {
        let message = "No run selected. Focus Runs to choose a run, then press Enter to open it.";
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let paragraph = Paragraph::new(message)
            .style(Style::default().fg(palette.muted))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, centered_text_area(inner, message));
        return;
    };

    let (status_text, status_color) = match record.status {
        RunStatus::Active => (run_status_label(&record.status), palette.success),
        RunStatus::Completed => (run_status_label(&record.status), palette.accent),
        RunStatus::Failed => (run_status_label(&record.status), palette.error),
    };

    let step = record
        .last_step
        .map(format_step)
        .unwrap_or_else(|| "-".to_string());
    let started = format_epoch_date(record.started_at_epoch_secs);
    let duration = app
        .selected_run_elapsed()
        .map(format_duration)
        .unwrap_or_else(|| "-".to_string());
    let latest_loss = app
        .selected_run_live_loss()
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "-".to_string());
    let source = record.source_locator.as_deref().unwrap_or("-");
    let source_line = if source.len() > 48 {
        format!("{}...", &source[..45])
    } else {
        source.to_string()
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Run ", Style::default().fg(palette.muted)),
            Span::styled(
                run_display_name(record),
                Style::default()
                    .fg(palette.header_fg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("   "),
            Span::styled("Status ", Style::default().fg(palette.muted)),
            Span::styled(
                status_text,
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(format!("Started {started}   Duration {duration}")),
        Line::from(format!("Step {step}   Latest loss {latest_loss}")),
        Line::from(format!("Source {source_line}")),
    ];

    if !record.tags.is_empty() {
        lines.push(Line::from(format!("Tags {}", record.tags.join(", "))));
    }

    if let Some(overlay) = app.compare_run_display_name() {
        lines.push(Line::from(format!("Overlay {overlay}")));
    }

    lines.push(Line::from(Span::styled(
        if is_focused {
            "Enter opens the selected run from Runs"
        } else {
            "Selection follows the Runs panel"
        },
        Style::default().fg(palette.muted),
    )));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .style(Style::default().fg(palette.header_fg))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn render_alerts(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    palette: &crate::ui::theme::ThemePalette,
) {
    let (active, resolved) = app.home_alert_records();
    let data = AlertPanelData::from_records(&active, &resolved);
    render_alert_panel(frame, area, &data, palette, "Alerts", false, 5, 3);
}

fn right_column_width(total_width: u16) -> u16 {
    let max_allowed = total_width.saturating_sub(LEFT_COLUMN_MIN_WIDTH);
    let target = RIGHT_COLUMN_TARGET_WIDTH.min(max_allowed);
    let min_width = RIGHT_COLUMN_MIN_WIDTH.min(total_width.saturating_sub(1));
    target.max(min_width)
}

fn process_panel_height(app: &App) -> u16 {
    if app.discovered_processes.is_empty() {
        return 5;
    }

    let visible_rows = app.discovered_processes.len().min(6) as u16;
    (visible_rows + 3).min(10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn make_terminal() -> Terminal<TestBackend> {
        Terminal::new(TestBackend::new(120, 40)).unwrap()
    }

    #[test]
    fn test_home_renders_without_panic() {
        let app = crate::app::App::new(Default::default());
        let mut terminal = make_terminal();
        terminal
            .draw(|frame| {
                render(frame, frame.area(), &app);
            })
            .unwrap();
    }
}
