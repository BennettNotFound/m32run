//! System call shim layer
//!
//! The shim layer intercepts privileged instructions such as software
//! interrupts and dispatches them to host functionality.

use guestmem::GuestMemory;
use x86core::Cpu;
use std::io::{self, Write};
use std::process;

/// 辅助函数：从 GuestMemory 中读取 32 位整数
fn read_u32(mem: &GuestMemory, addr: u32) -> Result<u32, ()> {
    let mut buf = [0u8; 4];
    mem.read(addr, &mut buf).map_err(|_| {
        eprintln!("shim: failed to read memory at {:#010x}", addr);
        ()
    })?;
    Ok(u32::from_le_bytes(buf))
}

/// Handle a system call from the guest.
pub fn handle_syscall(cpu: &mut Cpu, mem: &mut GuestMemory) -> Result<(), ()> {
    let syscall_num = cpu.eax;

    match syscall_num {
        1 => {
            // SYS_exit (1)
            // macOS 32-bit stack layout for exit:
            // [ESP]   = padding
            // [ESP+4] = status (int)
            let status = read_u32(mem, cpu.esp.wrapping_add(4))?;
            println!("\n[Shim] Guest exited normally with status: {}", status);

            // 直接退出宿主机进程（完美模拟程序结束）
            process::exit(status as i32);
        }

        4 => {
            // SYS_write (4)
            // macOS 32-bit stack layout for write:
            // [ESP]    = padding
            // [ESP+4]  = fd (int)
            // [ESP+8]  = buf (pointer)
            // [ESP+12] = count (size_t)
            let fd = read_u32(mem, cpu.esp.wrapping_add(4))?;
            let buf_addr = read_u32(mem, cpu.esp.wrapping_add(8))?;
            let count = read_u32(mem, cpu.esp.wrapping_add(12))?;

            // 从 GuestMemory 中读取要打印的字符串
            let mut string_buf = vec![0u8; count as usize];
            if mem.read(buf_addr, &mut string_buf).is_ok() {
                // 根据文件描述符打印到宿主机
                match fd {
                    1 => {
                        io::stdout().write_all(&string_buf).unwrap();
                        io::stdout().flush().unwrap();
                    }
                    2 => {
                        io::stderr().write_all(&string_buf).unwrap();
                        io::stderr().flush().unwrap();
                    }
                    _ => {
                        eprintln!("[Shim] Write to unsupported fd: {}", fd);
                    }
                }

                // 系统调用成功，将写入的字节数返回给 EAX
                cpu.eax = count;
                // macOS ABI: 成功时清空 EFLAGS 的 Carry Flag (CF) 位
                cpu.eflags &= !1;

                Ok(())
            } else {
                eprintln!("[Shim] SYS_write invalid memory access at {:#010x}", buf_addr);
                // 失败时设置 Carry Flag，并返回错误码
                cpu.eflags |= 1;
                cpu.eax = 14; // EFAULT
                Ok(()) // 仍然返回 Ok(())，因为系统调用本身被成功处理了，只是结果是失败
            }
        }

        _ => {
            eprintln!("shim: unimplemented syscall {} invoked", syscall_num);
            Err(())
        }
    }
}