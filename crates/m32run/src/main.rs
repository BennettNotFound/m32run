//! Entry point for the m32run command line tool
//!
//! The CLI accepts the path to a 32-bit Mach-O binary followed by
//! optional arguments to pass to the guest program.  It loads the
//! binary, sets up the execution environment and runs the program
//! until the CPU encounters a halt instruction or an unimplemented
//! opcode.

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use host::graphics::{create_default_graphics_backend, HostInputEvent};
use host::{prepare_macos_ui_app, set_dock_icon_from_file};
use loader::{load, DyldError, HostUiEvent};
use plist::Value;
use shim::handle_syscall;
use x86core::{Cpu, ExecError};

macro_rules! println {
    () => {
        runtime_log::stdout_line(String::new())
    };
    ($($arg:tt)*) => {
        runtime_log::stdout_line(format!($($arg)*))
    };
}

macro_rules! eprintln {
    () => {
        runtime_log::stderr_line(String::new())
    };
    ($($arg:tt)*) => {
        runtime_log::stderr_line(format!($($arg)*))
    };
}

#[derive(Default, Debug, Clone)]
struct BundleMeta {
    app_name: Option<String>,
    icon_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct GuestThreadContext {
    tid: u32,
    cpu: Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TraceLevel {
    Off = 0,
    Basic = 1,
    GuiDbg = 2,
    Cpu = 3,
}

impl TraceLevel {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Off),
            1 => Some(Self::Basic),
            2 => Some(Self::GuiDbg),
            3 => Some(Self::Cpu),
            _ => None,
        }
    }

    fn enabled(self) -> bool {
        self != Self::Off
    }

    fn basic(self) -> bool {
        self >= Self::Basic
    }

    fn guidbg(self) -> bool {
        self >= Self::GuiDbg
    }

    fn cpu(self) -> bool {
        self >= Self::Cpu
    }
}

fn bundle_string(dict: &plist::Dictionary, key: &str) -> Option<String> {
    dict.get(key)
        .and_then(Value::as_string)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
}

fn detect_bundle_meta(exec_path: &str) -> BundleMeta {
    let exec = Path::new(exec_path);
    let app_dir = exec.ancestors().find(|p| {
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("app"))
            .unwrap_or(false)
    });
    let Some(app_dir) = app_dir else {
        return BundleMeta::default();
    };

    let plist_path = app_dir.join("Contents/Info.plist");
    let Ok(plist) = Value::from_file(&plist_path) else {
        return BundleMeta::default();
    };
    let Some(dict) = plist.as_dictionary() else {
        return BundleMeta::default();
    };

    let app_name = bundle_string(dict, "CFBundleDisplayName")
        .or_else(|| bundle_string(dict, "CFBundleName"))
        .or_else(|| bundle_string(dict, "CFBundleExecutable"));

    let icon_name = bundle_string(dict, "CFBundleIconFile")
        .or_else(|| bundle_string(dict, "CFBundleIconName"))
        .or_else(|| {
            dict.get("CFBundleIconFiles")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(Value::as_string)
                .map(str::to_string)
        });

    let icon_path = icon_name.and_then(|raw| {
        let resources = app_dir.join("Contents/Resources");
        let direct = resources.join(&raw);
        if direct.exists() {
            return Some(direct);
        }
        let with_icns = resources.join(format!("{}.icns", raw));
        if with_icns.exists() {
            return Some(with_icns);
        }
        let with_png = resources.join(format!("{}.png", raw));
        if with_png.exists() {
            return Some(with_png);
        }
        None
    });

    BundleMeta {
        app_name,
        icon_path,
    }
}

fn print_usage(exit_code: i32) -> ! {
    let usage = "Usage: m32run [options] <macho32-file> [args...]\n\
                 \n\
                 Options:\n\
                 -h, --help                 Show this help\n\
                 --trace LEVEL              Trace level (0..3)\n\
                 --max-instructions N       Stop after N total guest instructions\n\
                 --                         End of options\n\
                 \n\
                 Trace levels:\n\
                 0 = off\n\
                 1 = basic run/import/gui logs\n\
                 2 = gui debug logs\n\
                 3 = cpu instruction trace";
    if exit_code == 0 {
        println!("{}", usage);
    } else {
        eprintln!("{}", usage);
    }
    process::exit(exit_code);
}

fn main() {
    runtime_log::init();

    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut trace_level = TraceLevel::Off;
    let mut max_total_instructions: Option<u64> = None;
    let mut first_non_flag = 0usize;

    while first_non_flag < args.len() {
        match args[first_non_flag].as_str() {
            "-h" | "--help" => print_usage(0),
            "--" => {
                first_non_flag += 1;
                break;
            }
            "--trace" => {
                if first_non_flag + 1 >= args.len() {
                    print_usage(1);
                }
                let raw = &args[first_non_flag + 1];
                let parsed = raw.parse::<u32>().ok().and_then(TraceLevel::from_u32);
                let Some(level) = parsed else {
                    eprintln!("Invalid --trace value: {} (expected 0..3)", raw);
                    process::exit(1);
                };
                trace_level = level;
                first_non_flag += 2;
            }
            "--max-instructions" => {
                if first_non_flag + 1 >= args.len() {
                    print_usage(1);
                }
                let raw = &args[first_non_flag + 1];
                let parsed = raw.parse::<u64>().unwrap_or(0);
                if parsed == 0 {
                    eprintln!("Invalid --max-instructions value: {}", raw);
                    process::exit(1);
                }
                max_total_instructions = Some(parsed);
                first_non_flag += 2;
            }
            other if other.starts_with('-') => {
                eprintln!("Unknown option: {}", other);
                print_usage(1);
            }
            _ => break,
        }
    }

    if first_non_flag >= args.len() {
        print_usage(1);
    }

    let prog_path = args[first_non_flag].clone();
    let prog_args = args.split_off(first_non_flag + 1);

    let ctrlc_count = Arc::new(AtomicU8::new(0));
    let graceful_stop_requested = Arc::new(AtomicBool::new(false));
    {
        let count = Arc::clone(&ctrlc_count);
        let graceful = Arc::clone(&graceful_stop_requested);
        if let Err(err) = ctrlc::set_handler(move || {
            let hit = count.fetch_add(1, Ordering::SeqCst).saturating_add(1);
            if hit == 1 {
                graceful.store(true, Ordering::SeqCst);
                runtime_log::stderr_line(
                    "[SIGNAL] Ctrl+C received, trying graceful shutdown (press Ctrl+C again to force quit)"
                        .to_string(),
                );
            } else {
                runtime_log::stderr_line(
                    "[SIGNAL] Ctrl+C received again, forcing exit now".to_string(),
                );
                runtime_log::flush_timeout(Duration::from_millis(150));
                process::exit(130);
            }
        }) {
            eprintln!("[SIGNAL] failed to install Ctrl+C handler: {}", err);
        }
    }

    let bundle_meta = detect_bundle_meta(&prog_path);
    let mut default_window_title = bundle_meta
        .app_name
        .clone()
        .unwrap_or_else(|| "m32run Guest GUI".to_string());
    if default_window_title.trim().is_empty() {
        default_window_title = "m32run Guest GUI".to_string();
    }

    match load(&prog_path, &prog_args, trace_level.enabled()) {
        Ok(mut loaded) => {
            loaded.cpu.trace = trace_level.cpu();

            let step_instructions = 100_000;
            let mut total_instructions: u64 = 0;
            let mut traced_unresolved_stubs: HashSet<(u32, u32)> = HashSet::new();

            let mut host_window_pixels: Vec<u32> = Vec::new();
            let mut host_window_row: Vec<u8> = Vec::new();
            let mut host_window_title = default_window_title.clone();
            let mut host_ui_bootstrapped = false;
            let mut gfx_backend: Option<Box<dyn host::graphics::GraphicsBackend>> = None;
            let mut gfx_disabled = false;
            let mut gfx_first_present_logged = false;
            let mut host_last_present = Instant::now()
                .checked_sub(Duration::from_millis(33))
                .unwrap_or_else(Instant::now);

            let mut threads = vec![GuestThreadContext {
                tid: loaded.dyld.main_thread_tid(),
                cpu: loaded.cpu.clone(),
            }];
            let mut thread_cursor = 0usize;
            let mut cpu_template = loaded.cpu.clone();

            let mut last_guidbg_report_instr = 0u64;
            let guidbg_interval = 1_000_000u64;
            let mut last_fb_signature: Option<(u32, u32, u32, u32, u32)> = None;
            let mut logged_window_fb_signature: Option<(u32, u32, u32, u32, u32)> = None;
            let mut last_title_report: Option<String> = None;
            let mut last_window_state_open = true;
            let mut last_thread_pc_report: Option<Vec<(u32, u32, u32)>> = None;

            loop {
                if graceful_stop_requested.load(Ordering::SeqCst) {
                    eprintln!("[RUN] graceful stop requested, leaving guest loop...");
                    break;
                }
                if threads.is_empty() {
                    println!("\nAll guest threads exited.");
                    break;
                }
                if thread_cursor >= threads.len() {
                    thread_cursor = 0;
                }

                let (current_tid, run_result, executed) = {
                    let thread = &mut threads[thread_cursor];
                    loaded.dyld.set_current_thread_tid(thread.tid);
                    let before = thread.cpu.instr_counter;
                    let run_result = thread.cpu.run(&mut loaded.mem, step_instructions);
                    let executed = thread.cpu.instr_counter.saturating_sub(before);
                    (thread.tid, run_result, executed)
                };
                total_instructions = total_instructions.saturating_add(executed);

                if let Some(title) = loaded.dyld.guest_window_title() {
                    let t = title.trim();
                    if !t.is_empty() && t != host_window_title {
                        if trace_level.basic() {
                            eprintln!(
                                "[GUI] guest window title changed: {:?} -> {:?}",
                                host_window_title, t
                            );
                        }
                        host_window_title = t.to_string();
                        if !gfx_disabled {
                            if let Some(backend) = gfx_backend.as_mut() {
                                if let Err(err) = backend.set_title(&host_window_title) {
                                    gfx_disabled = true;
                                    eprintln!(
                                        "[WINDOW] set title failed, disable graphics: {}",
                                        err
                                    );
                                }
                            }
                        }
                    }
                }

                if !gfx_disabled && !host_ui_bootstrapped && loaded.dyld.graphics_requested() {
                    host_ui_bootstrapped = true;

                    if let Err(err) = prepare_macos_ui_app() {
                        if trace_level.basic() {
                            eprintln!("[GUI] prepare macOS ui app failed: {}", err);
                        }
                    }

                    if let Some(icon_path) = bundle_meta.icon_path.as_ref() {
                        if let Err(err) = set_dock_icon_from_file(icon_path) {
                            if trace_level.basic() {
                                eprintln!(
                                    "[GUI] failed to set dock icon from '{}': {}",
                                    icon_path.display(),
                                    err
                                );
                            }
                        } else if trace_level.basic() {
                            eprintln!("[GUI] dock icon set from '{}'", icon_path.display());
                        }
                    }

                    match create_default_graphics_backend() {
                        Ok(backend) => {
                            if trace_level.basic() {
                                eprintln!("[GFX] backend selected: {}", backend.backend_name());
                            }
                            gfx_backend = Some(backend);
                        }
                        Err(err) => {
                            gfx_disabled = true;
                            eprintln!("[GFX] backend unavailable, continue headless: {}", err);
                        }
                    }
                }

                if trace_level.guidbg()
                    && total_instructions.saturating_sub(last_guidbg_report_instr)
                        >= guidbg_interval
                {
                    last_guidbg_report_instr = total_instructions;

                    let title_now = loaded.dyld.guest_window_title().map(|s| s.to_string());
                    if title_now != last_title_report {
                        eprintln!("[GUIDBG] guest_window_title = {:?}", title_now);
                        last_title_report = title_now;
                    }

                    match loaded.dyld.guest_framebuffer_info() {
                        Some(fb) => {
                            let sig = (
                                fb.plane0,
                                fb.width,
                                fb.height,
                                fb.bytes_per_row,
                                fb.bits_per_pixel,
                            );
                            if last_fb_signature != Some(sig) {
                                eprintln!(
                                    "[GUIDBG] framebuffer plane0={:#010x} {}x{} stride={} bpp={}",
                                    fb.plane0,
                                    fb.width,
                                    fb.height,
                                    fb.bytes_per_row,
                                    fb.bits_per_pixel
                                );
                                last_fb_signature = Some(sig);
                            } else {
                                eprintln!(
                                    "[GUIDBG] framebuffer unchanged plane0={:#010x} {}x{} stride={} bpp={} @ {} instr",
                                    fb.plane0,
                                    fb.width,
                                    fb.height,
                                    fb.bytes_per_row,
                                    fb.bits_per_pixel,
                                    total_instructions
                                );
                            }
                        }
                        None => {
                            let window_open =
                                gfx_backend.as_ref().map(|b| b.is_open()).unwrap_or(false);
                            eprintln!(
                                "[GUIDBG] no framebuffer yet @ {} instr (tid={}, window_open={}, gfx_disabled={})",
                                total_instructions,
                                current_tid,
                                window_open,
                                gfx_disabled
                            );
                        }
                    }

                    let mut pcs: Vec<(u32, u32, u32)> = threads
                        .iter()
                        .map(|t| (t.tid, t.cpu.eip, t.cpu.esp))
                        .collect();
                    pcs.sort_by_key(|(tid, _, _)| *tid);
                    if pcs.len() > 4 {
                        pcs.truncate(4);
                    }
                    if last_thread_pc_report.as_ref() != Some(&pcs) {
                        let parts: Vec<String> = pcs
                            .iter()
                            .map(|(tid, eip, esp)| {
                                format!("tid={} eip={:#010x} esp={:#010x}", tid, eip, esp)
                            })
                            .collect();
                        eprintln!("[GUIDBG] thread-pc {}", parts.join(" | "));
                        last_thread_pc_report = Some(pcs);
                    }
                }

                if !gfx_disabled {
                    if let Some(backend) = gfx_backend.as_mut() {
                        if let Err(err) = backend.poll_events() {
                            gfx_disabled = true;
                            eprintln!("[WINDOW] event pump failed, disable graphics: {}", err);
                        } else {
                            for ev in backend.drain_events() {
                                match ev {
                                    HostInputEvent::Quit => {
                                        loaded.dyld.push_host_ui_event(HostUiEvent::Quit)
                                    }
                                    HostInputEvent::KeyDown { keycode } => loaded
                                        .dyld
                                        .push_host_ui_event(HostUiEvent::KeyDown { keycode }),
                                    HostInputEvent::KeyUp { keycode } => loaded
                                        .dyld
                                        .push_host_ui_event(HostUiEvent::KeyUp { keycode }),
                                    HostInputEvent::MouseMove { x, y } => loaded
                                        .dyld
                                        .push_host_ui_event(HostUiEvent::MouseMove { x, y }),
                                    HostInputEvent::MouseDown { button, x, y } => loaded
                                        .dyld
                                        .push_host_ui_event(HostUiEvent::MouseDown {
                                            button,
                                            x,
                                            y,
                                        }),
                                    HostInputEvent::MouseUp { button, x, y } => loaded
                                        .dyld
                                        .push_host_ui_event(HostUiEvent::MouseUp {
                                            button,
                                            x,
                                            y,
                                        }),
                                    HostInputEvent::MouseWheel { x, y } => loaded
                                        .dyld
                                        .push_host_ui_event(HostUiEvent::MouseWheel { x, y }),
                                    HostInputEvent::TextInput { text } => loaded
                                        .dyld
                                        .push_host_ui_event(HostUiEvent::TextInput { text }),
                                }
                            }

                            if let Some(text) = backend.clipboard_get() {
                                if !text.is_empty() {
                                    loaded.dyld.set_host_clipboard_text(text);
                                }
                            }
                            if let Some(text) = loaded.dyld.take_guest_clipboard_update() {
                                let _ = backend.clipboard_set(&text);
                            }

                            if trace_level.guidbg() {
                                let is_open = backend.is_open();
                                if is_open != last_window_state_open {
                                    eprintln!("[GUIDBG] host window presence changed: {}", is_open);
                                    last_window_state_open = is_open;
                                }
                            }
                        }
                    }
                }

                if host_ui_bootstrapped
                    && !gfx_disabled
                    && host_last_present.elapsed() >= Duration::from_millis(16)
                {
                    host_last_present = Instant::now();

                    if let Some(fb) = loaded.dyld.guest_framebuffer_info() {
                        let width = fb.width as usize;
                        let height = fb.height as usize;

                        if width > 0
                            && height > 0
                            && width >= 160
                            && height >= 120
                            && width <= 4096
                            && height <= 4096
                            && fb.plane0 != 0
                        {
                            if let Some(backend) = gfx_backend.as_mut() {
                                if let Err(err) =
                                    backend.ensure_window(&host_window_title, fb.width, fb.height)
                                {
                                    gfx_disabled = true;
                                    eprintln!("[WINDOW] create/resize failed: {}", err);
                                } else if trace_level.basic() {
                                    let sig = (
                                        fb.plane0,
                                        fb.width,
                                        fb.height,
                                        fb.bytes_per_row,
                                        fb.bits_per_pixel,
                                    );
                                    if logged_window_fb_signature != Some(sig) {
                                        let caps = backend.caps();
                                        eprintln!(
                                            "[WINDOW] ready {}x{} title='{}'",
                                            fb.width, fb.height, host_window_title
                                        );
                                        eprintln!(
                                            "[GFX] context {}",
                                            if caps.context_created {
                                                "created"
                                            } else {
                                                "not created (pixel fallback)"
                                            }
                                        );
                                        eprintln!(
                                            "[FB] framebuffer ready plane0={:#010x} {}x{} stride={} bpp={}",
                                            fb.plane0,
                                            fb.width,
                                            fb.height,
                                            fb.bytes_per_row,
                                            fb.bits_per_pixel
                                        );
                                        logged_window_fb_signature = Some(sig);
                                    }
                                }
                            }

                            if !gfx_disabled {
                                let bytes_per_pixel = ((fb.bits_per_pixel.saturating_add(7)) / 8)
                                    .clamp(1, 4)
                                    as usize;
                                let src_stride =
                                    fb.bytes_per_row.max(bytes_per_pixel as u32) as usize;
                                let palette = if bytes_per_pixel == 1 {
                                    loaded.dyld.guest_palette_snapshot()
                                } else {
                                    None
                                };

                                host_window_pixels.resize(width.saturating_mul(height), 0);
                                host_window_row.resize(src_stride, 0);

                                let mut draw_ok = true;
                                for y in 0..height {
                                    let row_addr = fb.plane0.wrapping_add(
                                        (y as u32).wrapping_mul(fb.bytes_per_row.max(1)),
                                    );
                                    if loaded.mem.read(row_addr, &mut host_window_row).is_err() {
                                        draw_ok = false;
                                        if trace_level.guidbg() {
                                            eprintln!(
                                                "[GUIDBG] framebuffer row read failed at y={} addr={:#010x}",
                                                y, row_addr
                                            );
                                        }
                                        break;
                                    }

                                    let row_out_base = y.saturating_mul(width);
                                    for x in 0..width {
                                        let src = x.saturating_mul(bytes_per_pixel);
                                        let color =
                                            if src + bytes_per_pixel <= host_window_row.len() {
                                                match bytes_per_pixel {
                                                    4 => {
                                                        let b = host_window_row[src] as u32;
                                                        let g = host_window_row[src + 1] as u32;
                                                        let r = host_window_row[src + 2] as u32;
                                                        (r << 16) | (g << 8) | b
                                                    }
                                                    3 => {
                                                        let b = host_window_row[src] as u32;
                                                        let g = host_window_row[src + 1] as u32;
                                                        let r = host_window_row[src + 2] as u32;
                                                        (r << 16) | (g << 8) | b
                                                    }
                                                    2 => {
                                                        let lo = host_window_row[src] as u16;
                                                        let hi = host_window_row[src + 1] as u16;
                                                        let v = lo | (hi << 8);
                                                        let r = ((v >> 11) & 0x1f) as u32;
                                                        let g = ((v >> 5) & 0x3f) as u32;
                                                        let b = (v & 0x1f) as u32;
                                                        let r = (r * 255) / 31;
                                                        let g = (g * 255) / 63;
                                                        let b = (b * 255) / 31;
                                                        (r << 16) | (g << 8) | b
                                                    }
                                                    _ => {
                                                        let idx = host_window_row[src] as usize;
                                                        if let Some(p) = palette.as_ref() {
                                                            p[idx.min(255)]
                                                        } else {
                                                            let v = idx as u32;
                                                            (v << 16) | (v << 8) | v
                                                        }
                                                    }
                                                }
                                            } else {
                                                0
                                            };
                                        host_window_pixels[row_out_base + x] = color;
                                    }
                                }

                                if draw_ok {
                                    if let Some(backend) = gfx_backend.as_mut() {
                                        if let Err(err) = backend.present_rgb(
                                            &host_window_pixels,
                                            fb.width,
                                            fb.height,
                                        ) {
                                            gfx_disabled = true;
                                            eprintln!(
                                                "[PRESENT] failed, disable graphics: {}",
                                                err
                                            );
                                        } else if !gfx_first_present_logged {
                                            eprintln!(
                                                "[PRESENT] first frame presented {}x{} (plane0={:#010x})",
                                                fb.width, fb.height, fb.plane0
                                            );
                                            gfx_first_present_logged = true;
                                        }
                                    }
                                }
                            }
                        } else if trace_level.guidbg() {
                            eprintln!(
                                "[GUIDBG] framebuffer rejected plane0={:#010x} {}x{} stride={} bpp={} (constraints failed)",
                                fb.plane0,
                                fb.width,
                                fb.height,
                                fb.bytes_per_row,
                                fb.bits_per_pixel
                            );
                        }
                    } else {
                        // 避免在“等待 guest framebuffer 建立”阶段占满宿主 CPU。
                        thread::sleep(Duration::from_millis(1));
                    }
                }

                if let Some(limit) = max_total_instructions {
                    if total_instructions >= limit {
                        println!(
                            "Execution reached configured limit of {} instructions",
                            limit
                        );
                        break;
                    }
                }

                let mut remove_tid: Option<u32> = None;
                let mut terminate_process = false;

                match run_result {
                    Ok(()) => {
                        if trace_level.basic() && total_instructions % 1_000_000 == 0 {
                            eprintln!(
                                "[RUN] executed {} instructions across {} thread(s)",
                                total_instructions,
                                threads.len()
                            );
                        }
                    }
                    Err(ExecError::Syscall) => {
                        let Some(thread) = threads.get_mut(thread_cursor) else {
                            continue;
                        };
                        if trace_level.basic() {
                            eprintln!(
                                "[SYSCALL] tid={} eax={:#010x} eip={:#010x}",
                                thread.tid, thread.cpu.eax, thread.cpu.eip
                            );
                        }
                        if handle_syscall(&mut thread.cpu, &mut loaded.mem).is_err() {
                            eprintln!("Execution aborted due to unhandled syscall.");
                            terminate_process = true;
                        }
                    }
                    Err(ExecError::Halt) => {
                        if current_tid == loaded.dyld.main_thread_tid() {
                            println!("\nProgram halted normally.");
                            terminate_process = true;
                        } else {
                            if trace_level.basic() {
                                eprintln!("[THREAD] tid={} halted", current_tid);
                            }
                            remove_tid = Some(current_tid);
                        }
                    }
                    Err(ExecError::UnimplementedOpcode(op, addr)) => {
                        println!(
                            "\nEncountered unimplemented opcode 0x{:02x} at 0x{:08x} (tid={})",
                            op, addr, current_tid
                        );
                        terminate_process = true;
                    }
                    Err(ExecError::UnresolvedImportStub {
                        eip,
                        stub_index,
                        indirect_symbol_index,
                    }) => {
                        if traced_unresolved_stubs.insert((eip, indirect_symbol_index)) {
                            let import_desc = loaded
                                .dyld
                                .describe_import(eip, indirect_symbol_index)
                                .unwrap_or_else(|| "<unknown import>".to_string());
                            eprintln!(
                                "[IMPORT] eip={:#010x}, stub_index={}, indirect_symbol_index={}, import={}",
                                eip, stub_index, indirect_symbol_index, import_desc
                            );
                        }

                        let Some(thread) = threads.get_mut(thread_cursor) else {
                            continue;
                        };
                        if let Err(err) = loaded.dyld.handle_unresolved_import(
                            &mut thread.cpu,
                            &mut loaded.mem,
                            eip,
                            indirect_symbol_index,
                        ) {
                            match err {
                                DyldError::GuestExit(status) => {
                                    eprintln!("Guest exited with status {}", status);
                                    terminate_process = true;
                                }
                                DyldError::GuestThreadExit { tid, status } => {
                                    if trace_level.basic() {
                                        eprintln!(
                                            "[THREAD] guest thread tid={} exited with status {:#010x}",
                                            tid, status
                                        );
                                    }
                                    remove_tid = Some(tid);
                                }
                                other => {
                                    eprintln!("dyld import handling failed: {}", other);
                                    terminate_process = true;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("\nExecution error: {}", e);
                        terminate_process = true;
                    }
                }

                let spawn_requests = loaded.dyld.take_pthread_spawn_requests();
                if !spawn_requests.is_empty() {
                    if let Some(thread) = threads.get(thread_cursor) {
                        cpu_template = thread.cpu.clone();
                    }
                    for req in spawn_requests {
                        match loaded.dyld.build_guest_thread_cpu(
                            &cpu_template,
                            &mut loaded.mem,
                            req,
                        ) {
                            Ok(mut cpu) => {
                                cpu.trace = trace_level.cpu();
                                if trace_level.basic() {
                                    eprintln!(
                                        "[THREAD] spawned tid={} start={:#010x} arg={:#010x} return_stub={:#010x}",
                                        req.tid, req.start_routine, req.arg, req.return_stub
                                    );
                                }
                                threads.push(GuestThreadContext { tid: req.tid, cpu });
                            }
                            Err(err) => {
                                eprintln!(
                                    "Failed to start guest thread tid={} ({:#010x}): {}",
                                    req.tid, req.start_routine, err
                                );
                                terminate_process = true;
                                break;
                            }
                        }
                    }
                }

                if let Some(src) = threads.get(thread_cursor).map(|t| t.cpu.clone()) {
                    let import_map = src.import_stub_indirect_map.clone();
                    let jump_table = (
                        src.import_jump_table_addr,
                        src.import_jump_table_size,
                        src.import_jump_table_stub_size,
                        src.import_jump_table_reserved1,
                    );
                    for t in &mut threads {
                        t.cpu.import_jump_table_addr = jump_table.0;
                        t.cpu.import_jump_table_size = jump_table.1;
                        t.cpu.import_jump_table_stub_size = jump_table.2;
                        t.cpu.import_jump_table_reserved1 = jump_table.3;
                        t.cpu.import_stub_indirect_map = import_map.clone();
                        t.cpu.trace = trace_level.cpu();
                    }
                    cpu_template = src.clone();
                }

                if let Some(tid) = remove_tid {
                    if let Some(pos) = threads.iter().position(|t| t.tid == tid) {
                        threads.remove(pos);
                        if pos < thread_cursor && thread_cursor > 0 {
                            thread_cursor -= 1;
                        } else if pos == thread_cursor && thread_cursor >= threads.len() {
                            thread_cursor = 0;
                        }
                    }
                } else if !threads.is_empty() {
                    thread_cursor = (thread_cursor + 1) % threads.len();
                }

                if terminate_process {
                    break;
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to load program: {}", e);
            runtime_log::flush_timeout(Duration::from_millis(200));
            process::exit(1);
        }
    }
    let force_exit = ctrlc_count.load(Ordering::SeqCst) >= 2;
    runtime_log::flush_timeout(Duration::from_millis(200));
    if force_exit {
        process::exit(130);
    }
}
