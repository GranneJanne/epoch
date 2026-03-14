use ratatui::Frame;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Sparkline};

pub struct LineGraphOptions<'a> {
    pub name: &'a str,
    pub series: &'a [u64],
    pub color: Color,
    pub comparison_name: Option<&'a str>,
    pub comparison_series: Option<&'a [u64]>,
    pub comparison_color: Color,
    pub dense: bool,
}

pub struct MetricGraph<'a> {
    pub title: &'a str,
    pub data: &'a [u64],
    pub color: Color,
    pub graph_mode: &'a str,
    pub focused: bool,
    pub focus_index: Option<u8>,
    pub empty_message: &'a str,
    pub accent: Color,
    pub muted: Color,
    pub header_fg: Color,
    pub comparison_data: Option<&'a [u64]>,
    pub comparison_label: Option<&'a str>,
    pub comparison_color: Color,
}

impl<'a> MetricGraph<'a> {
    pub fn new(title: &'a str, data: &'a [u64], color: Color) -> Self {
        Self {
            title,
            data,
            color,
            graph_mode: "sparkline",
            focused: false,
            focus_index: None,
            empty_message: "No data",
            accent: Color::Rgb(137, 180, 250),
            muted: Color::DarkGray,
            header_fg: Color::Rgb(205, 214, 244),
            comparison_data: None,
            comparison_label: None,
            comparison_color: Color::Cyan,
        }
    }

    pub fn graph_mode(mut self, mode: &'a str) -> Self {
        self.graph_mode = mode;
        self
    }

    pub fn focused(mut self, focused: bool) -> Self {
        self.focused = focused;
        self
    }

    pub fn focus_index(mut self, index: Option<u8>) -> Self {
        self.focus_index = index;
        self
    }

    pub fn empty_message(mut self, msg: &'a str) -> Self {
        self.empty_message = msg;
        self
    }

    pub fn palette(mut self, accent: Color, muted: Color, header_fg: Color) -> Self {
        self.accent = accent;
        self.muted = muted;
        self.header_fg = header_fg;
        self
    }

    pub fn comparison_series(mut self, label: &'a str, data: &'a [u64], color: Color) -> Self {
        self.comparison_data = Some(data);
        self.comparison_label = Some(label);
        self.comparison_color = color;
        self
    }

    pub fn render(self, frame: &mut Frame, area: Rect) {
        let title = if let Some(idx) = self.focus_index {
            format!("[{idx}] {}", self.title)
        } else {
            self.title.to_string()
        };

        let border_color = if self.focused {
            self.accent
        } else {
            self.muted
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_style(
                Style::default()
                    .fg(self.header_fg)
                    .add_modifier(Modifier::BOLD),
            )
            .border_style(Style::default().fg(border_color));

        let has_primary = !self.data.is_empty();
        let has_comparison = self.comparison_data.is_some_and(|data| !data.is_empty());

        if !has_primary && !has_comparison {
            let para = Paragraph::new(self.empty_message)
                .block(block)
                .alignment(Alignment::Center)
                .style(Style::default().fg(self.muted));
            frame.render_widget(para, area);
        } else if self.graph_mode == "sparkline" && !has_comparison {
            let sparkline = Sparkline::default()
                .block(block)
                .data(self.data)
                .style(Style::default().fg(self.color));
            frame.render_widget(sparkline, area);
        } else {
            render_line_graph(
                frame,
                area,
                block,
                LineGraphOptions {
                    name: self.title,
                    series: self.data,
                    color: self.color,
                    comparison_name: self.comparison_label,
                    comparison_series: self.comparison_data,
                    comparison_color: self.comparison_color,
                    dense: self.graph_mode == "dense",
                },
            );
        }
    }
}

pub fn render_line_graph(
    frame: &mut Frame,
    area: Rect,
    block: Block,
    options: LineGraphOptions<'_>,
) {
    let points = options
        .series
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as f64, *value as f64))
        .collect::<Vec<_>>();
    let comparison_points = options
        .comparison_series
        .unwrap_or(&[])
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as f64, *value as f64))
        .collect::<Vec<_>>();

    let max_y = points
        .iter()
        .chain(comparison_points.iter())
        .map(|(_, y)| *y)
        .fold(1.0_f64, |acc, y| acc.max(y));
    let max_x = points.len().max(comparison_points.len()).saturating_sub(1) as f64;
    let marker = if options.dense {
        Marker::Braille
    } else {
        Marker::Dot
    };

    let mut datasets = Vec::new();
    if !points.is_empty() {
        datasets.push(
            Dataset::default()
                .name(options.name)
                .graph_type(GraphType::Line)
                .marker(marker)
                .style(
                    Style::default()
                        .fg(options.color)
                        .add_modifier(if options.dense {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                )
                .data(&points),
        );
    }
    if !comparison_points.is_empty() {
        datasets.push(
            Dataset::default()
                .name(options.comparison_name.unwrap_or("overlay"))
                .graph_type(GraphType::Line)
                .marker(marker)
                .style(Style::default().fg(options.comparison_color))
                .data(&comparison_points),
        );
    }

    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(Axis::default().bounds([0.0, max_x.max(1.0)]))
        .y_axis(Axis::default().bounds([0.0, max_y]));
    frame.render_widget(chart, area);
}
