use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders, Paragraph, Sparkline};

use crate::app::App;
use crate::ui::{ACCENT, LOSS_COLOR, LR_COLOR, MUTED, WARNING};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let Some(latest) = app.training.latest.as_ref() else {
        let text = Paragraph::new("No diagnostics available yet")
            .alignment(Alignment::Center)
            .style(Style::default().fg(MUTED));
        frame.render_widget(text, area);
        return;
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(35),
            Constraint::Percentage(35),
            Constraint::Percentage(30),
        ])
        .split(area);

    let trends = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[0]);

    let eval_width = usize::from(trends[0].width.saturating_sub(2).max(1));
    let eval_history = app.training_viewport_series(&app.training.eval_loss_history, eval_width);
    let eval_block = Block::default()
        .title("Eval Loss (unit: loss)")
        .borders(Borders::ALL);
    if eval_history.is_empty() {
        frame.render_widget(
            Paragraph::new("No eval loss data")
                .block(eval_block)
                .style(Style::default().fg(MUTED)),
            trends[0],
        );
    } else {
        frame.render_widget(
            Sparkline::default()
                .block(eval_block)
                .data(&eval_history)
                .style(Style::default().fg(LOSS_COLOR)),
            trends[0],
        );
    }

    let grad_width = usize::from(trends[1].width.saturating_sub(2).max(1));
    let grad_history = app.training_viewport_series(&app.training.grad_norm_history, grad_width);
    let grad_block = Block::default()
        .title("Grad Norm (unit: norm)")
        .borders(Borders::ALL);
    if grad_history.is_empty() {
        frame.render_widget(
            Paragraph::new("No grad norm data")
                .block(grad_block)
                .style(Style::default().fg(MUTED)),
            trends[1],
        );
    } else {
        frame.render_widget(
            Sparkline::default()
                .block(grad_block)
                .data(&grad_history)
                .style(Style::default().fg(LR_COLOR)),
            trends[1],
        );
    }

    let throughput = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(rows[1]);

    render_rate_panel(
        frame,
        throughput[0],
        "Tokens/s",
        "tok/s",
        latest.tokens_per_second,
    );
    render_rate_panel(
        frame,
        throughput[1],
        "Samples/s",
        "samples/s",
        latest.samples_per_second,
    );
    render_rate_panel(
        frame,
        throughput[2],
        "Steps/s",
        "steps/s",
        latest.steps_per_second,
    );

    let diagnostics = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[2]);

    let summary = Paragraph::new(format!(
        "Perplexity: {}\nLoss spikes: {}\nNaN/Inf count: {}",
        latest_or_dash(app.training.perplexity_latest, 3),
        app.training.loss_spike_count,
        app.training.nan_inf_count,
    ))
    .block(
        Block::default()
            .title("Stability Summary")
            .borders(Borders::ALL),
    )
    .style(Style::default().fg(ACCENT));
    frame.render_widget(summary, diagnostics[0]);

    let anomaly = Paragraph::new(format!(
        "Last loss spike: {}\nLast NaN/Inf: {}\nHealth: {}",
        anomaly_age(app.training.last_loss_spike_at),
        anomaly_age(app.training.last_nan_inf_at),
        app.training_data_health_state().label(),
    ))
    .block(
        Block::default()
            .title("Anomaly Timing")
            .borders(Borders::ALL),
    )
    .style(Style::default().fg(WARNING));
    frame.render_widget(anomaly, diagnostics[1]);
}

fn render_rate_panel(frame: &mut Frame, area: Rect, label: &str, unit: &str, value: Option<f64>) {
    let value_text = value
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "-".to_string());
    let text = format!("{value_text}\nunit: {unit}");
    let panel = Paragraph::new(text)
        .alignment(Alignment::Center)
        .block(Block::default().title(label).borders(Borders::ALL))
        .style(Style::default().fg(ACCENT));
    frame.render_widget(panel, area);
}

fn latest_or_dash(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(v) => format!("{v:.decimals$}"),
        None => "-".to_string(),
    }
}

fn anomaly_age(ts: Option<std::time::Instant>) -> String {
    match ts {
        Some(t) => format!("{}s ago", t.elapsed().as_secs()),
        None => "never".to_string(),
    }
}
