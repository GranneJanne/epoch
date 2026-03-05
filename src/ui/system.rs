use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders, LineGauge, Paragraph, Sparkline};

use crate::app::App;
use crate::ui::{CPU_COLOR, GPU_COLOR, HEADER_FG, MUTED, RAM_COLOR, WARNING};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let Some(system) = &app.system.latest else {
        let msg = Paragraph::new("Collecting system metrics...")
            .style(Style::default().fg(MUTED))
            .alignment(Alignment::Center);

        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(1),
                Constraint::Fill(1),
            ])
            .split(area);

        frame.render_widget(msg, layout[1]);
        return;
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(6),
            Constraint::Min(0),
        ])
        .split(area);

    let cpu_ratio = (system.cpu_usage / 100.0).clamp(0.0, 1.0);
    let cpu_color = if system.cpu_usage > 80.0 {
        WARNING
    } else {
        CPU_COLOR
    };

    let cpu_gauge = LineGauge::default()
        .block(Block::default().title("CPU").borders(Borders::ALL))
        .filled_style(Style::default().fg(cpu_color))
        .filled_symbol("█")
        .unfilled_symbol(" ")
        .ratio(cpu_ratio)
        .label(format!("{:.1}%", system.cpu_usage));

    frame.render_widget(cpu_gauge, layout[0]);

    let ram_ratio = (system.memory_used as f64 / system.memory_total.max(1) as f64).clamp(0.0, 1.0);
    let ram_gauge = LineGauge::default()
        .block(Block::default().title("RAM").borders(Borders::ALL))
        .filled_style(Style::default().fg(RAM_COLOR))
        .filled_symbol("█")
        .unfilled_symbol(" ")
        .ratio(ram_ratio)
        .label(format!(
            "{} / {}",
            format_bytes(system.memory_used),
            format_bytes(system.memory_total)
        ));

    frame.render_widget(ram_gauge, layout[1]);

    if system.gpus.is_empty() {
        let block = Block::default()
            .title("GPU: Not available")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(MUTED));
        let p = Paragraph::new("No GPU detected")
            .style(Style::default().fg(MUTED))
            .block(block);
        frame.render_widget(p, layout[2]);
    } else {
        let max_visible = usize::from((layout[2].height / 3).max(1));
        let visible_count = system.gpus.len().min(max_visible);
        let hidden_count = system.gpus.len().saturating_sub(visible_count);

        let mut gpu_constraints = vec![Constraint::Length(3); visible_count];
        if hidden_count > 0 {
            gpu_constraints.push(Constraint::Length(1));
        }

        let gpus_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(gpu_constraints)
            .split(layout[2]);

        let avg_util =
            system.gpus.iter().map(|g| g.utilization).sum::<f64>() / system.gpus.len() as f64;

        for (i, gpu) in system.gpus.iter().take(visible_count).enumerate() {
            let gpu_area = gpus_layout[i];
            let gpu_ratio = (gpu.utilization / 100.0).clamp(0.0, 1.0);
            let is_outlier = (gpu.utilization - avg_util).abs() >= 30.0;
            let title = if is_outlier {
                format!("GPU {}: {} [OUTLIER]", i, gpu.name)
            } else {
                format!("GPU {}: {}", i, gpu.name)
            };

            let block = Block::default().title(title).borders(Borders::ALL);
            let inner_area = block.inner(gpu_area);
            frame.render_widget(block, gpu_area);

            let gpu_inner_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Length(1)])
                .split(inner_area);

            let gpu_gauge = LineGauge::default()
                .filled_style(Style::default().fg(if is_outlier { WARNING } else { GPU_COLOR }))
                .filled_symbol("█")
                .unfilled_symbol(" ")
                .ratio(gpu_ratio)
                .label(format!("{:.1}%", gpu.utilization));
            frame.render_widget(gpu_gauge, gpu_inner_layout[0]);

            let detail_text = format!(
                "{} / {}   {}",
                format_bytes(gpu.memory_used),
                format_bytes(gpu.memory_total),
                format_temp(gpu.temperature)
            );

            let detail = Paragraph::new(detail_text).style(Style::default().fg(HEADER_FG));
            frame.render_widget(detail, gpu_inner_layout[1]);
        }

        if hidden_count > 0 {
            let hidden = Paragraph::new(format!("+{} more GPUs", hidden_count))
                .alignment(Alignment::Right)
                .style(Style::default().fg(MUTED));
            frame.render_widget(hidden, gpus_layout[visible_count]);
        }
    }

    let history_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(layout[3]);

    let history_width = usize::from(history_layout[0].width.saturating_sub(2).max(1));
    let cpu_data = app.system_viewport_series(&app.system.cpu_history, history_width);
    let ram_data = app.system_viewport_series(&app.system.ram_history, history_width);

    let cpu_sparkline = Sparkline::default()
        .block(Block::default().title("CPU History").borders(Borders::ALL))
        .data(&cpu_data)
        .style(Style::default().fg(CPU_COLOR))
        .max(10000);
    frame.render_widget(cpu_sparkline, history_layout[0]);

    let ram_sparkline = Sparkline::default()
        .block(Block::default().title("RAM History").borders(Borders::ALL))
        .data(&ram_data)
        .style(Style::default().fg(RAM_COLOR))
        .max(10000);
    frame.render_widget(ram_sparkline, history_layout[1]);
}

fn format_bytes(bytes: u64) -> String {
    let tb = 1_099_511_627_776;
    let gb = 1_073_741_824;
    let mb = 1_048_576;
    let kb = 1024;

    if bytes >= tb {
        format!("{:.1} TB", bytes as f64 / tb as f64)
    } else if bytes >= gb {
        format!("{:.1} GB", bytes as f64 / gb as f64)
    } else if bytes >= mb {
        format!("{:.1} MB", bytes as f64 / mb as f64)
    } else {
        format!("{:.1} KB", bytes as f64 / kb as f64)
    }
}

fn format_temp(celsius: f64) -> String {
    format!("{}°C", celsius as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::{GpuMetrics, SystemMetrics};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn format_percent(v: f64) -> String {
        format!("{:.1}%", v)
    }

    #[test]
    fn test_system_empty_state() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new(Config::default());

        terminal
            .draw(|frame| {
                render(frame, frame.area(), &app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer.cell((x, y)).unwrap().symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<String>>()
            .join("\n");
        assert!(content.contains("Collecting system metrics..."));
    }

    #[test]
    fn test_system_with_full_data() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default());
        app.push_system(SystemMetrics {
            cpu_usage: 45.0,
            memory_used: 8_589_934_592,
            memory_total: 17_179_869_184,
            gpus: vec![GpuMetrics {
                name: "RTX 4090".into(),
                utilization: 95.0,
                memory_used: 20_000_000_000,
                memory_total: 24_000_000_000,
                temperature: 72.0,
            }],
        });

        terminal
            .draw(|frame| {
                render(frame, frame.area(), &app);
            })
            .unwrap();
    }

    #[test]
    fn test_system_no_gpu() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default());
        app.push_system(SystemMetrics {
            cpu_usage: 45.0,
            memory_used: 8_589_934_592,
            memory_total: 17_179_869_184,
            gpus: vec![],
        });

        terminal
            .draw(|frame| {
                render(frame, frame.area(), &app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer.cell((x, y)).unwrap().symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<String>>()
            .join("\n");
        assert!(content.contains("Not available"));
    }

    #[test]
    fn test_system_multiple_gpus() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default());
        app.push_system(SystemMetrics {
            cpu_usage: 45.0,
            memory_used: 8_589_934_592,
            memory_total: 17_179_869_184,
            gpus: vec![
                GpuMetrics {
                    name: "RTX 4090".into(),
                    utilization: 95.0,
                    memory_used: 20_000_000_000,
                    memory_total: 24_000_000_000,
                    temperature: 72.0,
                },
                GpuMetrics {
                    name: "RTX 3080".into(),
                    utilization: 10.0,
                    memory_used: 2_000_000_000,
                    memory_total: 10_000_000_000,
                    temperature: 45.0,
                },
            ],
        });

        terminal
            .draw(|frame| {
                render(frame, frame.area(), &app);
            })
            .unwrap();
    }

    #[test]
    fn test_cpu_warning_color() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default());
        app.push_system(SystemMetrics {
            cpu_usage: 85.0,
            memory_used: 8_589_934_592,
            memory_total: 17_179_869_184,
            gpus: vec![],
        });

        terminal
            .draw(|frame| {
                render(frame, frame.area(), &app);
            })
            .unwrap();
    }

    #[test]
    fn test_format_bytes_gb() {
        assert_eq!(format_bytes(8_589_934_592), "8.0 GB");
    }

    #[test]
    fn test_format_bytes_mb() {
        assert_eq!(format_bytes(536_870_912), "512.0 MB");
    }

    #[test]
    fn test_format_bytes_tb() {
        assert_eq!(format_bytes(1_099_511_627_776), "1.0 TB");
    }

    #[test]
    fn test_format_temp() {
        assert_eq!(format_temp(72.0), "72°C");
    }

    #[test]
    fn test_format_percent() {
        assert_eq!(format_percent(45.2), "45.2%");
    }

    #[test]
    fn test_system_tab_shows_hidden_gpu_count_indicator() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default());
        app.push_system(SystemMetrics {
            cpu_usage: 30.0,
            memory_used: 1,
            memory_total: 2,
            gpus: vec![
                GpuMetrics {
                    name: "A".into(),
                    utilization: 10.0,
                    memory_used: 1,
                    memory_total: 2,
                    temperature: 40.0,
                },
                GpuMetrics {
                    name: "B".into(),
                    utilization: 20.0,
                    memory_used: 1,
                    memory_total: 2,
                    temperature: 41.0,
                },
                GpuMetrics {
                    name: "C".into(),
                    utilization: 30.0,
                    memory_used: 1,
                    memory_total: 2,
                    temperature: 42.0,
                },
            ],
        });

        terminal
            .draw(|frame| render(frame, frame.area(), &app))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let content = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer.cell((x, y)).unwrap().symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<String>>()
            .join("\n");

        assert!(content.contains("+1 more GPUs"));
    }

    #[test]
    fn test_system_tab_highlights_gpu_outlier() {
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default());
        app.push_system(SystemMetrics {
            cpu_usage: 30.0,
            memory_used: 1,
            memory_total: 2,
            gpus: vec![
                GpuMetrics {
                    name: "A".into(),
                    utilization: 95.0,
                    memory_used: 1,
                    memory_total: 2,
                    temperature: 40.0,
                },
                GpuMetrics {
                    name: "B".into(),
                    utilization: 10.0,
                    memory_used: 1,
                    memory_total: 2,
                    temperature: 41.0,
                },
            ],
        });

        terminal
            .draw(|frame| render(frame, frame.area(), &app))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let content = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer.cell((x, y)).unwrap().symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<String>>()
            .join("\n");

        assert!(content.contains("OUTLIER"));
    }
}
