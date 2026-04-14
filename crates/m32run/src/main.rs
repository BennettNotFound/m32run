//! Entry point for the m32run command line tool
//!
//! The CLI accepts the path to a 32‑bit Mach‑O binary followed by
//! optional arguments to pass to the guest program.  It loads the
//! binary, sets up the execution environment and runs the program
//! until the CPU encounters a halt instruction or an unimplemented
//! opcode.

use std::env;
use std::process;

use loader::{load, DyldError};
use shim::handle_syscall;
use x86core::ExecError; // 引入 shim 层的系统调用处理函数

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
                    eprintln!("Usage: m32run [--trace] [--max-instructions N] <macho32-file> [args...]");
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

    // 2. 加载并运行
    match load(&prog_path, &prog_args, trace_enabled) {
        Ok(mut loaded) => {
            loaded.cpu.trace = trace_enabled; // 设置追踪开关

            let step_instructions = 100_000;
            let mut total_instructions: u64 = 0;

            loop {
                match loaded.cpu.run(&mut loaded.mem, step_instructions) {
                    Ok(()) => {
                        total_instructions = total_instructions.saturating_add(step_instructions as u64);
                        if let Some(limit) = max_total_instructions {
                            if total_instructions >= limit {
                                println!("Execution reached configured limit of {} instructions", limit);
                                break;
                            }
                        } else if trace_enabled && total_instructions % 1_000_000 == 0 {
                            eprintln!("[RUN] executed {} instructions", total_instructions);
                        }
                    }
                    Err(ExecError::Syscall) => {
                        if handle_syscall(&mut loaded.cpu, &mut loaded.mem).is_err() {
                            eprintln!("Execution aborted due to unhandled syscall.");
                            break;
                        }
                    }
                    Err(ExecError::Halt) => {
                        println!("\nProgram halted normally.");
                        break;
                    }
                    Err(ExecError::UnimplementedOpcode(op, addr)) => {
                        println!(
                            "\nEncountered unimplemented opcode 0x{:02x} at 0x{:08x}",
                            op, addr
                        );
                        break;
                    }
                    Err(ExecError::UnresolvedImportStub {
                        eip,
                        stub_index,
                        indirect_symbol_index,
                    }) => {
                        if trace_enabled {
                            let import_desc = loaded
                                .dyld
                                .describe_import(eip, indirect_symbol_index)
                                .unwrap_or_else(|| "<unknown import>".to_string());
                            eprintln!(
                                "\nHit unresolved import stub: eip={:#010x}, stub_index={}, indirect_symbol_index={}, import={}",
                                eip, stub_index, indirect_symbol_index, import_desc
                            );
                        }
                        if let Err(err) = loaded.dyld.handle_unresolved_import(
                            &mut loaded.cpu,
                            &mut loaded.mem,
                            eip,
                            indirect_symbol_index,
                        ) {
                            match err {
                                DyldError::GuestExit(status) => {
                                    eprintln!("Guest exited with status {}", status);
                                    break;
                                }
                                other => {
                                    eprintln!("dyld import handling failed: {}", other);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("\nExecution error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to load program: {}", e);
            process::exit(1);
        }
    }
}
