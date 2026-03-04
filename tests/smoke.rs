use epoch::app::App;
use epoch::config::Config;
use epoch::event::Event;
use epoch::types::{GpuMetrics, SystemMetrics};
use epoch::ui::Tab;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_app_processes_events_from_channels() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use epoch::types::TrainingMetrics;

    let mut app = App::new(Config::default());
    let (tx, mut rx) = mpsc::channel(16);

    tx.send(Event::Metrics(TrainingMetrics {
        loss: Some(0.5),
        step: Some(100),
        ..TrainingMetrics::default()
    }))
    .await
    .expect("metrics event should send");

    tx.send(Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)))
        .await
        .expect("key event should send");

    if let Some(event) = rx.recv().await {
        app.handle_event(event);
    }
    assert!(app.training.latest.is_some());
    assert_eq!(app.training.latest.as_ref().and_then(|m| m.loss), Some(0.5));

    if let Some(event) = rx.recv().await {
        app.handle_event(event);
    }
    assert_eq!(app.ui_state.selected_tab, Tab::Metrics);
}

#[tokio::test]
async fn test_training_metrics_flow_through_channel() {
    use epoch::types::TrainingMetrics;

    let (tx, mut rx) = mpsc::channel(epoch::event::METRICS_CHANNEL_CAPACITY);

    for i in 1..=5 {
        tx.send(TrainingMetrics {
            loss: Some(1.0 / i as f64),
            step: Some(i * 100),
            ..TrainingMetrics::default()
        })
        .await
        .expect("training metric should send");
    }

    let mut count = 0;
    while let Ok(metrics) = rx.try_recv() {
        assert!(metrics.loss.is_some());
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_app_new_running() {
    let app = App::new(Config::default());
    assert!(app.running);
    assert_eq!(app.ui_state.selected_tab, Tab::Dashboard);
}

#[test]
fn test_config_defaults() {
    let config = Config::default();
    assert_eq!(config.tick_rate_ms, 250);
    assert_eq!(config.history_size, 300);
    assert_eq!(config.parser, "auto");
}

#[test]
fn test_system_metrics_default() {
    let metrics = SystemMetrics::default();
    assert_eq!(metrics.cpu_usage, 0.0);
    assert_eq!(metrics.memory_used, 0);
    assert_eq!(metrics.memory_total, 0);
    assert!(metrics.gpus.is_empty());
}

#[test]
fn test_gpu_metrics_default() {
    let metrics = GpuMetrics::default();
    assert_eq!(metrics.name, "");
    assert_eq!(metrics.utilization, 0.0);
    assert_eq!(metrics.memory_used, 0);
    assert_eq!(metrics.memory_total, 0);
    assert_eq!(metrics.temperature, 0.0);
}
