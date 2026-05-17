use std::io::{self, IsTerminal, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Info,
    Warn,
    Error,
}

enum LogEvent {
    Line {
        stream: Stream,
        level: Level,
        msg: String,
    },
    Fragment {
        stream: Stream,
        msg: String,
    },
    Flush(Sender<()>),
}

struct AsyncLogger {
    tx: Sender<LogEvent>,
}

static LOGGER: OnceLock<AsyncLogger> = OnceLock::new();

fn format_level(level: Level) -> &'static str {
    match level {
        Level::Info => "INFO",
        Level::Warn => "WARN",
        Level::Error => "ERROR",
    }
}

fn classify_level(msg: &str, default: Level) -> Level {
    let trimmed = msg.trim();
    if trimmed.is_empty() {
        return default;
    }
    let upper = trimmed.to_ascii_uppercase();
    if upper.starts_with("[ERROR]") || upper.starts_with("[ERR]") {
        return Level::Error;
    }
    if upper.starts_with("[WARN]") {
        return Level::Warn;
    }

    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("panic")
        || lower.contains("fatal")
        || lower.starts_with("failed")
        || lower.contains(" error")
        || lower.contains("error:")
    {
        return Level::Error;
    }
    if lower.contains("warning") || lower.contains("warn") || lower.contains("unimplemented") {
        return Level::Warn;
    }
    default
}

fn color_wrap(enabled: bool, color: &str, text: &str) -> String {
    if enabled {
        format!("\x1b[{}m{}\x1b[0m", color, text)
    } else {
        text.to_string()
    }
}

fn write_formatted_line(
    w: &mut dyn Write,
    pretty: bool,
    color: bool,
    started: Instant,
    level: Level,
    msg: &str,
) {
    let trimmed = msg.trim_end_matches(['\n', '\r']);
    if !pretty {
        let _ = w.write_all(trimmed.as_bytes());
        let _ = w.write_all(b"\n");
        return;
    }
    let elapsed = started.elapsed();
    let secs = elapsed.as_secs();
    let millis = elapsed.subsec_millis();
    let ts = format!("+{:05}.{:03}s", secs, millis);
    let lvl = format_level(level);
    let lvl = match level {
        Level::Info => color_wrap(color, "36", lvl),
        Level::Warn => color_wrap(color, "33", lvl),
        Level::Error => color_wrap(color, "31", lvl),
    };
    let prefix = format!("[{}] [{}] ", ts, lvl);
    let _ = w.write_all(prefix.as_bytes());
    let _ = w.write_all(trimmed.as_bytes());
    let _ = w.write_all(b"\n");
}

fn flush_pending_line(
    stream: Stream,
    pending: &mut String,
    pretty: bool,
    stdout_color: bool,
    stderr_color: bool,
    started: Instant,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) {
    if pending.is_empty() {
        return;
    }
    let msg = std::mem::take(pending);
    let level = classify_level(&msg, Level::Info);
    match stream {
        Stream::Stdout => {
            write_formatted_line(stdout, pretty, stdout_color, started, level, &msg);
        }
        Stream::Stderr => {
            write_formatted_line(stderr, pretty, stderr_color, started, level, &msg);
        }
    }
}

fn drain_fragment_lines(
    stream: Stream,
    incoming: &str,
    pending: &mut String,
    pretty: bool,
    stdout_color: bool,
    stderr_color: bool,
    started: Instant,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) {
    if incoming.is_empty() {
        return;
    }
    pending.push_str(incoming);
    while let Some(pos) = pending.find('\n') {
        let line = pending[..=pos].to_string();
        let level = classify_level(&line, Level::Info);
        match stream {
            Stream::Stdout => {
                write_formatted_line(stdout, pretty, stdout_color, started, level, &line);
            }
            Stream::Stderr => {
                write_formatted_line(stderr, pretty, stderr_color, started, level, &line);
            }
        }
        pending.drain(..=pos);
    }
}

fn worker_loop(rx: Receiver<LogEvent>) {
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let started = Instant::now();
    let stdout_color = stdout.is_terminal();
    let stderr_color = stderr.is_terminal();
    let pretty = true;
    let mut pending_stdout = String::new();
    let mut pending_stderr = String::new();

    while let Ok(event) = rx.recv() {
        match event {
            LogEvent::Line { stream, level, msg } => match stream {
                Stream::Stdout => {
                    write_formatted_line(&mut stdout, pretty, stdout_color, started, level, &msg);
                }
                Stream::Stderr => {
                    write_formatted_line(&mut stderr, pretty, stderr_color, started, level, &msg);
                }
            },
            LogEvent::Fragment { stream, msg } => match stream {
                Stream::Stdout => drain_fragment_lines(
                    Stream::Stdout,
                    &msg,
                    &mut pending_stdout,
                    pretty,
                    stdout_color,
                    stderr_color,
                    started,
                    &mut stdout,
                    &mut stderr,
                ),
                Stream::Stderr => drain_fragment_lines(
                    Stream::Stderr,
                    &msg,
                    &mut pending_stderr,
                    pretty,
                    stdout_color,
                    stderr_color,
                    started,
                    &mut stdout,
                    &mut stderr,
                ),
            },
            LogEvent::Flush(done) => {
                flush_pending_line(
                    Stream::Stdout,
                    &mut pending_stdout,
                    pretty,
                    stdout_color,
                    stderr_color,
                    started,
                    &mut stdout,
                    &mut stderr,
                );
                flush_pending_line(
                    Stream::Stderr,
                    &mut pending_stderr,
                    pretty,
                    stdout_color,
                    stderr_color,
                    started,
                    &mut stdout,
                    &mut stderr,
                );
                let _ = stdout.flush();
                let _ = stderr.flush();
                let _ = done.send(());
            }
        }
    }
}

fn ensure_logger() -> &'static AsyncLogger {
    LOGGER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<LogEvent>();
        let _ = thread::Builder::new()
            .name("runtime-log".to_string())
            .spawn(move || worker_loop(rx));
        AsyncLogger { tx }
    })
}

pub fn init() {
    let _ = ensure_logger();
}

pub fn stdout_line(msg: String) {
    let logger = ensure_logger();
    let _ = logger.tx.send(LogEvent::Line {
        stream: Stream::Stdout,
        level: Level::Info,
        msg,
    });
}

pub fn stderr_line(msg: String) {
    let logger = ensure_logger();
    let level = classify_level(&msg, Level::Info);
    let _ = logger.tx.send(LogEvent::Line {
        stream: Stream::Stderr,
        level,
        msg,
    });
}

pub fn stdout_fragment(msg: String) {
    let logger = ensure_logger();
    let _ = logger.tx.send(LogEvent::Fragment {
        stream: Stream::Stdout,
        msg,
    });
}

pub fn stderr_fragment(msg: String) {
    let logger = ensure_logger();
    let _ = logger.tx.send(LogEvent::Fragment {
        stream: Stream::Stderr,
        msg,
    });
}

pub fn flush_timeout(timeout: Duration) {
    let logger = ensure_logger();
    let (tx, rx) = mpsc::channel();
    if logger.tx.send(LogEvent::Flush(tx)).is_ok() {
        let _ = rx.recv_timeout(timeout);
    }
}
