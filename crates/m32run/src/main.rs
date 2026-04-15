//! Entry point for the m32run command line tool
//!
//! The CLI accepts the path to a 32‑bit Mach‑O binary followed by
//! optional arguments to pass to the guest program.  It loads the
//! binary, sets up the execution environment and runs the program
//! until the CPU encounters a halt instruction or an unimplemented
//! opcode.

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

use host::{prepare_macos_ui_app, set_dock_icon_from_file};
use loader::{load, DyldError};
use minifb::{Window, WindowOptions};
use plist::Value;
use shim::handle_syscall;
use x86core::{Cpu, ExecError}; // 引入 shim 层的系统调用处理函数

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

fn main() {
    // 1. 先解析工具参数
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut trace_enabled = false;
    let mut max_total_instructions: Option<u64> = None;
    let mut first_non_flag = 0usize;
    while first_non_flag < args.len() {
        match args[first_non_flag].as_str() {
            "--trace" => {
                trace_enabled = true;
                first_non_flag += 1;
            }
            "--max-instructions" => {
                if first_non_flag + 1 >= args.len() {
                    eprintln!(
                        "Usage: m32run [--trace] [--max-instructions N] <macho32-file> [args...]"
                    );
                    process::exit(1);
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
            _ => break,
        }
    }

    if first_non_flag >= args.len() {
        eprintln!("Usage: m32run [--trace] [--max-instructions N] <macho32-file> [args...]");
        process::exit(1);
    }
    let prog_path = args[first_non_flag].clone();
    let prog_args = args.split_off(first_non_flag + 1);
    let bundle_meta = detect_bundle_meta(&prog_path);
    let mut default_window_title = bundle_meta
        .app_name
        .clone()
        .unwrap_or_else(|| "m32run Guest GUI".to_string());
    if default_window_title.trim().is_empty() {
        default_window_title = "m32run Guest GUI".to_string();
    }

    if let Err(err) = prepare_macos_ui_app() {
        if trace_enabled {
            eprintln!("[GUI] prepare macOS ui app failed: {}", err);
        }
    }
    if let Some(icon_path) = bundle_meta.icon_path.as_ref() {
        if let Err(err) = set_dock_icon_from_file(icon_path) {
            if trace_enabled {
                eprintln!(
                    "[GUI] failed to set dock icon from '{}': {}",
                    icon_path.display(),
                    err
                );
            }
        } else if trace_enabled {
            eprintln!("[GUI] dock icon set from '{}'", icon_path.display());
        }
    }

    // 2. 加载并运行
    match load(&prog_path, &prog_args, trace_enabled) {
        Ok(mut loaded) => {
            loaded.cpu.trace = trace_enabled; // 设置追踪开关

            let step_instructions = 100_000;
            let mut total_instructions: u64 = 0;
            let mut traced_unresolved_stubs: HashSet<(u32, u32)> = HashSet::new();
            let mut host_window: Option<Window> = None;
            let mut host_window_size: (usize, usize) = (0, 0);
            let mut host_window_pixels: Vec<u32> = Vec::new();
            let mut host_window_row: Vec<u8> = Vec::new();
            let mut host_window_disabled = false;
            let mut host_window_title = default_window_title.clone();
            let mut host_last_present = Instant::now()
                .checked_sub(Duration::from_millis(33))
                .unwrap_or_else(Instant::now);
            let mut threads = vec![GuestThreadContext {
                tid: loaded.dyld.main_thread_tid(),
                cpu: loaded.cpu.clone(),
            }];
            let mut thread_cursor = 0usize;
            let mut cpu_template = loaded.cpu.clone();

            loop {
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
                        host_window_title = t.to_string();
                        if let Some(window) = host_window.as_mut() {
                            window.set_title(&host_window_title);
                        }
                    }
                }

                if !host_window_disabled && host_last_present.elapsed() >= Duration::from_millis(16)
                {
                    host_last_present = Instant::now();
                    if let Some(window) = host_window.as_ref() {
                        if !window.is_open() {
                            host_window = None;
                            host_window_size = (0, 0);
                        }
                    }
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
                            if host_window.is_none() || host_window_size != (width, height) {
                                match Window::new(
                                    &host_window_title,
                                    width,
                                    height,
                                    WindowOptions {
                                        resize: true,
                                        ..WindowOptions::default()
                                    },
                                ) {
                                    Ok(mut window) => {
                                        window.set_target_fps(60);
                                        host_window = Some(window);
                                        host_window_size = (width, height);
                                        host_window_pixels.resize(width.saturating_mul(height), 0);
                                        if trace_enabled {
                                            eprintln!(
                                                "[GUI] created host window {}x{} for guest framebuffer (bpp={}, stride={})",
                                                width,
                                                height,
                                                fb.bits_per_pixel,
                                                fb.bytes_per_row
                                            );
                                        }
                                    }
                                    Err(err) => {
                                        host_window_disabled = true;
                                        eprintln!("[GUI] failed to create host window: {}", err);
                                    }
                                }
                            }

                            if let Some(window) = host_window.as_mut() {
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
                                    let _ = window.update_with_buffer(
                                        &host_window_pixels,
                                        width,
                                        height,
                                    );
                                } else {
                                    window.update();
                                }
                            }
                        }
                    } else if let Some(window) = host_window.as_mut() {
                        window.update();
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
                        if max_total_instructions.is_none()
                            && trace_enabled
                            && total_instructions % 1_000_000 == 0
                        {
                            eprintln!("[RUN] executed {} instructions", total_instructions);
                        }
                    }
                    Err(ExecError::Syscall) => {
                        let Some(thread) = threads.get_mut(thread_cursor) else {
                            continue;
                        };
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
                        if trace_enabled
                            && traced_unresolved_stubs.insert((eip, indirect_symbol_index))
                        {
                            let import_desc = loaded
                                .dyld
                                .describe_import(eip, indirect_symbol_index)
                                .unwrap_or_else(|| "<unknown import>".to_string());
                            eprintln!(
                                "\nHit unresolved import stub: eip={:#010x}, stub_index={}, indirect_symbol_index={}, import={}",
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
                                    if trace_enabled {
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
                        match loaded
                            .dyld
                            .build_guest_thread_cpu(&cpu_template, &mut loaded.mem, req)
                        {
                            Ok(mut cpu) => {
                                cpu.trace = trace_enabled;
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

                // 同步 import stub 元数据到所有线程，保证动态加载后每个线程都能识别新 stub。
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
                        t.cpu.trace = trace_enabled;
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
            process::exit(1);
        }
    }
}
