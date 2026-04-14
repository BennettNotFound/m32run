//! Entry point for the m32run command line tool
//!
//! The CLI accepts the path to a 32‑bit Mach‑O binary followed by
//! optional arguments to pass to the guest program.  It loads the
//! binary, sets up the execution environment and runs the program
//! until the CPU encounters a halt instruction or an unimplemented
//! opcode.

use std::env;
use std::process;

use loader::load;
use x86core::ExecError;
use shim::handle_syscall; // 引入 shim 层的系统调用处理函数

fn main() {
    // 1. 获取原始参数并转为迭代器，跳过第一个（程序路径）
    let mut args_iter = env::args().skip(1).peekable();

    // 2. 检查是否有 --trace 参数
    let mut trace_enabled = false;
    if let Some(arg) = args_iter.peek() {
        if arg == "--trace" {
            trace_enabled = true;
            args_iter.next(); // 真正消耗掉这个参数
        }
    }

    // 3. 获取 Mach-O 文件路径
    let prog_path = match args_iter.next() {
        Some(path) => path,
        None => {
            eprintln!("Usage: m32run [--trace] <macho32-file> [args...]");
            process::exit(1);
        }
    };

    // 4. 剩下的全部作为程序参数
    let prog_args: Vec<String> = args_iter.collect();

    // 5. 加载并运行
    match load(&prog_path, &prog_args) {
        Ok(mut loaded) => {
            loaded.cpu.trace = trace_enabled; // 设置追踪开关

            let max_instructions = 100000;
            let mut total_instructions = 0;

            loop {
                match loaded.cpu.run(&mut loaded.mem, max_instructions) {
                    Ok(()) => {
                        total_instructions += max_instructions;
                        println!("Execution reached limit of {} instructions", total_instructions);
                        break;
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
                        println!("\nEncountered unimplemented opcode 0x{:02x} at 0x{:08x}", op, addr);
                        break;
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