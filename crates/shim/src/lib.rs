//! System call shim layer
//!
//! The shim layer intercepts privileged instructions such as software
//! interrupts and dispatches them to host functionality.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::process;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::sync::{Mutex, OnceLock};

use guestmem::{GuestMemory, Prot};
use x86core::Cpu;

macro_rules! eprintln {
    () => {
        runtime_log::stderr_line(String::new())
    };
    ($($arg:tt)*) => {
        runtime_log::stderr_line(format!($($arg)*))
    };
}

const SYS_READ: u32 = 3;
const SYS_WRITE: u32 = 4;
const SYS_OPEN: u32 = 5;
const SYS_CLOSE: u32 = 6;
const SYS_GETPID: u32 = 20;
const SYS_IOCTL: u32 = 54;
const SYS_SELECT: u32 = 93;
const SYS_GETTIMEOFDAY: u32 = 116;
const SYS_MPROTECT: u32 = 74;
const SYS_MUNMAP: u32 = 73;
const SYS_PREAD: u32 = 153;
const SYS_MMAP: u32 = 197;
const SYS_LSEEK: u32 = 199;
const SYS_NANOSLEEP: u32 = 240;
const SYS_POLL: u32 = 230;
const SYS_SIGACTION: u32 = 46;
const SYS_READ_NOCANCEL: u32 = 396;
const SYS_OPEN_NOCANCEL: u32 = 398;
const SYS_CLOSE_NOCANCEL: u32 = 399;

const O_ACCMODE: u32 = 0x0003;
const O_WRONLY: u32 = 0x0001;
const O_RDWR: u32 = 0x0002;
const O_APPEND: u32 = 0x0008;
const O_CREAT: u32 = 0x0200;
const O_TRUNC: u32 = 0x0400;
const O_EXCL: u32 = 0x0800;

const MAP_ANON: u32 = 0x1000;
const MAP_FIXED: u32 = 0x0010;

const SEEK_SET: u32 = 0;
const SEEK_CUR: u32 = 1;
const SEEK_END: u32 = 2;

const ENOMEM: u32 = 12;
const EIO: u32 = 5;
const EBADF: u32 = 9;
const EFAULT: u32 = 14;
const EINVAL: u32 = 22;
const ENOENT: u32 = 2;
const ENOSYS: u32 = 78;

#[derive(Debug)]
struct HostState {
    next_fd: u32,
    files: HashMap<u32, File>,
    next_mmap_addr: u32,
}

impl Default for HostState {
    fn default() -> Self {
        Self {
            next_fd: 3,
            files: HashMap::new(),
            next_mmap_addr: 0x2000_0000,
        }
    }
}

fn host_state() -> &'static Mutex<HostState> {
    static STATE: OnceLock<Mutex<HostState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(HostState::default()))
}

fn read_u32(mem: &GuestMemory, addr: u32) -> Result<u32, ()> {
    let mut buf = [0u8; 4];
    mem.read(addr, &mut buf).map_err(|_| ())?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(mem: &GuestMemory, addr: u32) -> Result<i32, ()> {
    let mut buf = [0u8; 4];
    mem.read(addr, &mut buf).map_err(|_| ())?;
    Ok(i32::from_le_bytes(buf))
}

fn syscall_arg(mem: &GuestMemory, cpu: &Cpu, arg_index: u32) -> Result<u32, ()> {
    let addr = cpu
        .esp
        .wrapping_add(4)
        .wrapping_add(arg_index.wrapping_mul(4));
    read_u32(mem, addr)
}

fn read_c_string(mem: &GuestMemory, addr: u32, max_len: usize) -> Result<String, ()> {
    let mut out = Vec::new();
    for i in 0..max_len {
        let mut b = [0u8; 1];
        mem.read(addr.wrapping_add(i as u32), &mut b)
            .map_err(|_| ())?;
        if b[0] == 0 {
            break;
        }
        out.push(b[0]);
    }
    Ok(String::from_utf8_lossy(&out).to_string())
}

fn set_ok(cpu: &mut Cpu, value: u32) {
    cpu.eax = value;
    cpu.eflags &= !1;
}

fn set_err(cpu: &mut Cpu, errno: u32) {
    cpu.eax = errno;
    cpu.eflags |= 1;
}

fn align_up(value: u32, align: u32) -> u32 {
    (value.wrapping_add(align - 1)) & !(align - 1)
}

fn is_page_aligned(value: u32) -> bool {
    (value & 0xFFF) == 0
}

fn now_timeval32() -> (i32, i32) {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    let secs = (dur.as_secs().min(i32::MAX as u64)) as i32;
    let usec = dur.subsec_micros() as i32;
    (secs, usec)
}

fn mmap_prot_to_guest(prot: u32) -> Prot {
    let mut out = Prot::READ;
    if (prot & 0x2) != 0 {
        out = out | Prot::WRITE;
    }
    if (prot & 0x4) != 0 {
        out = out | Prot::EXEC;
    }
    out
}

fn open_options_from_flags(flags: u32) -> OpenOptions {
    let mut opts = OpenOptions::new();
    match flags & O_ACCMODE {
        O_WRONLY => {
            opts.write(true);
        }
        O_RDWR => {
            opts.read(true).write(true);
        }
        _ => {
            opts.read(true);
        }
    }
    if (flags & O_CREAT) != 0 {
        opts.create(true);
    }
    if (flags & O_TRUNC) != 0 {
        opts.truncate(true);
    }
    if (flags & O_EXCL) != 0 {
        opts.create_new(true);
    }
    if (flags & O_APPEND) != 0 {
        opts.append(true);
    }
    opts
}

fn handle_read_like(
    cpu: &mut Cpu,
    mem: &mut GuestMemory,
    fd: u32,
    buf_addr: u32,
    count: u32,
    offset: Option<u32>,
) {
    let count = count.min(16 * 1024 * 1024);
    let mut data = vec![0u8; count as usize];

    if fd == 0 {
        set_ok(cpu, 0);
        return;
    }

    let mut state = host_state().lock().expect("host state poisoned");
    let file = match state.files.get_mut(&fd) {
        Some(f) => f,
        None => {
            set_err(cpu, EBADF);
            return;
        }
    };

    if let Some(off) = offset {
        if file.seek(SeekFrom::Start(off as u64)).is_err() {
            set_err(cpu, EIO);
            return;
        }
    }

    match file.read(&mut data) {
        Ok(n) => {
            if mem.write(buf_addr, &data[..n]).is_err() {
                set_err(cpu, EFAULT);
                return;
            }
            set_ok(cpu, n as u32);
        }
        Err(_) => set_err(cpu, EIO),
    }
}

/// Handle a system call from the guest.
pub fn handle_syscall(cpu: &mut Cpu, mem: &mut GuestMemory) -> Result<(), ()> {
    let syscall_num = cpu.eax;

    match syscall_num {
        1 => {
            let status = syscall_arg(mem, cpu, 0)?;
            if cpu.trace {
                eprintln!("[SHIM] guest exit({})", status);
            }
            process::exit(status as i32);
        }

        SYS_GETPID => {
            set_ok(cpu, process::id());
            Ok(())
        }

        SYS_READ | SYS_READ_NOCANCEL => {
            let fd = syscall_arg(mem, cpu, 0)?;
            let buf_addr = syscall_arg(mem, cpu, 1)?;
            let count = syscall_arg(mem, cpu, 2)?;
            handle_read_like(cpu, mem, fd, buf_addr, count, None);
            Ok(())
        }

        SYS_PREAD => {
            let fd = syscall_arg(mem, cpu, 0)?;
            let buf_addr = syscall_arg(mem, cpu, 1)?;
            let count = syscall_arg(mem, cpu, 2)?;
            let offset = syscall_arg(mem, cpu, 3)?;
            handle_read_like(cpu, mem, fd, buf_addr, count, Some(offset));
            Ok(())
        }

        SYS_WRITE => {
            let fd = syscall_arg(mem, cpu, 0)?;
            let buf_addr = syscall_arg(mem, cpu, 1)?;
            let count = syscall_arg(mem, cpu, 2)?.min(16 * 1024 * 1024);
            let mut buf = vec![0u8; count as usize];
            if mem.read(buf_addr, &mut buf).is_err() {
                set_err(cpu, EFAULT);
                return Ok(());
            }

            match fd {
                1 => {
                    if io::stdout().write_all(&buf).is_err() || io::stdout().flush().is_err() {
                        set_err(cpu, EIO);
                    } else {
                        set_ok(cpu, count);
                    }
                }
                2 => {
                    if io::stderr().write_all(&buf).is_err() || io::stderr().flush().is_err() {
                        set_err(cpu, EIO);
                    } else {
                        set_ok(cpu, count);
                    }
                }
                _ => {
                    let mut state = host_state().lock().expect("host state poisoned");
                    if let Some(file) = state.files.get_mut(&fd) {
                        match file.write(&buf) {
                            Ok(n) => set_ok(cpu, n as u32),
                            Err(_) => set_err(cpu, EIO),
                        }
                    } else {
                        set_err(cpu, EBADF);
                    }
                }
            }
            Ok(())
        }

        SYS_OPEN | SYS_OPEN_NOCANCEL => {
            let path_ptr = syscall_arg(mem, cpu, 0)?;
            let flags = syscall_arg(mem, cpu, 1)?;
            let _mode = syscall_arg(mem, cpu, 2)?;
            let path = match read_c_string(mem, path_ptr, 4096) {
                Ok(p) => p,
                Err(_) => {
                    set_err(cpu, EFAULT);
                    return Ok(());
                }
            };

            let options = open_options_from_flags(flags);
            match options.open(&path) {
                Ok(file) => {
                    let mut state = host_state().lock().expect("host state poisoned");
                    let fd = state.next_fd;
                    state.next_fd = state.next_fd.wrapping_add(1).max(3);
                    state.files.insert(fd, file);
                    set_ok(cpu, fd);
                }
                Err(err) => {
                    let errno = if err.kind() == io::ErrorKind::NotFound {
                        ENOENT
                    } else {
                        EIO
                    };
                    set_err(cpu, errno);
                }
            }
            Ok(())
        }

        SYS_CLOSE | SYS_CLOSE_NOCANCEL => {
            let fd = syscall_arg(mem, cpu, 0)?;
            if fd <= 2 {
                set_ok(cpu, 0);
                return Ok(());
            }
            let mut state = host_state().lock().expect("host state poisoned");
            if state.files.remove(&fd).is_some() {
                set_ok(cpu, 0);
            } else {
                set_err(cpu, EBADF);
            }
            Ok(())
        }

        SYS_IOCTL => {
            let _fd = syscall_arg(mem, cpu, 0)?;
            let _req = syscall_arg(mem, cpu, 1)?;
            let _argp = syscall_arg(mem, cpu, 2)?;
            // 先走“成功且无副作用”路径，避免大量 GUI/终端初始化因 ioctl 直接失败。
            set_ok(cpu, 0);
            Ok(())
        }

        SYS_LSEEK => {
            let fd = syscall_arg(mem, cpu, 0)?;
            let offset = syscall_arg(mem, cpu, 1)? as i32 as i64;
            let whence = syscall_arg(mem, cpu, 2)?;
            let pos = match whence {
                SEEK_SET => SeekFrom::Start(offset.max(0) as u64),
                SEEK_CUR => SeekFrom::Current(offset),
                SEEK_END => SeekFrom::End(offset),
                _ => {
                    set_err(cpu, EINVAL);
                    return Ok(());
                }
            };

            let mut state = host_state().lock().expect("host state poisoned");
            if let Some(file) = state.files.get_mut(&fd) {
                match file.seek(pos) {
                    Ok(new_pos) => set_ok(cpu, new_pos as u32),
                    Err(_) => set_err(cpu, EIO),
                }
            } else {
                set_err(cpu, EBADF);
            }
            Ok(())
        }

        SYS_GETTIMEOFDAY => {
            let tv_ptr = syscall_arg(mem, cpu, 0)?;
            let tz_ptr = syscall_arg(mem, cpu, 1)?;
            let (sec, usec) = now_timeval32();
            if tv_ptr != 0 {
                if mem.write(tv_ptr, &sec.to_le_bytes()).is_err()
                    || mem.write(tv_ptr.wrapping_add(4), &usec.to_le_bytes()).is_err()
                {
                    set_err(cpu, EFAULT);
                    return Ok(());
                }
            }
            if tz_ptr != 0 {
                let zero = 0i32.to_le_bytes();
                if mem.write(tz_ptr, &zero).is_err()
                    || mem.write(tz_ptr.wrapping_add(4), &zero).is_err()
                {
                    set_err(cpu, EFAULT);
                    return Ok(());
                }
            }
            set_ok(cpu, 0);
            Ok(())
        }

        SYS_MMAP => {
            let req_addr = syscall_arg(mem, cpu, 0)?;
            let len = syscall_arg(mem, cpu, 1)?;
            let prot = syscall_arg(mem, cpu, 2)?;
            let flags = syscall_arg(mem, cpu, 3)?;
            let fd = syscall_arg(mem, cpu, 4)?;
            let offset = syscall_arg(mem, cpu, 5)?;

            if len == 0 {
                set_err(cpu, EINVAL);
                return Ok(());
            }
            if req_addr != 0 && !is_page_aligned(req_addr) {
                set_err(cpu, EINVAL);
                return Ok(());
            }
            if (flags & MAP_FIXED) != 0 && req_addr == 0 {
                set_err(cpu, EINVAL);
                return Ok(());
            }

            let len_aligned = align_up(len, 0x1000);
            let mut state = host_state().lock().expect("host state poisoned");
            let mut map_addr = if req_addr != 0 {
                align_up(req_addr, 0x1000)
            } else {
                let addr = align_up(state.next_mmap_addr, 0x1000);
                addr
            };

            let runtime_prot = mmap_prot_to_guest(prot);
            let load_prot = runtime_prot | Prot::WRITE;
            if (flags & MAP_FIXED) != 0 {
                if mem.map_fixed(map_addr, len_aligned, load_prot).is_err() {
                    set_err(cpu, EINVAL);
                    return Ok(());
                }
            } else {
                let mut mapped = false;
                for _ in 0..4096 {
                    if mem.map(map_addr, len_aligned, load_prot).is_ok() {
                        mapped = true;
                        break;
                    }
                    map_addr = align_up(
                        map_addr.wrapping_add(len_aligned).wrapping_add(0x1000),
                        0x1000,
                    );
                }
                if !mapped {
                    set_err(cpu, ENOMEM);
                    return Ok(());
                }
            }
            state.next_mmap_addr = map_addr.wrapping_add(len_aligned).wrapping_add(0x1000);

            if (flags & MAP_ANON) == 0 {
                let file = match state.files.get_mut(&fd) {
                    Some(f) => f,
                    None => {
                        set_err(cpu, EBADF);
                        return Ok(());
                    }
                };
                if file.seek(SeekFrom::Start(offset as u64)).is_err() {
                    set_err(cpu, EIO);
                    return Ok(());
                }
                let mut file_data = vec![0u8; len as usize];
                match file.read(&mut file_data) {
                    Ok(n) => {
                        if n > 0 && mem.write(map_addr, &file_data[..n]).is_err() {
                            set_err(cpu, EFAULT);
                            return Ok(());
                        }
                    }
                    Err(_) => {
                        set_err(cpu, EIO);
                        return Ok(());
                    }
                }
            }

            if mem.protect(map_addr, len_aligned, runtime_prot).is_err() {
                set_err(cpu, EINVAL);
                return Ok(());
            }
            set_ok(cpu, map_addr);
            Ok(())
        }

        SYS_SELECT => {
            let _nfds = syscall_arg(mem, cpu, 0)?;
            let _readfds = syscall_arg(mem, cpu, 1)?;
            let _writefds = syscall_arg(mem, cpu, 2)?;
            let _exceptfds = syscall_arg(mem, cpu, 3)?;
            let timeout_ptr = syscall_arg(mem, cpu, 4)?;

            if timeout_ptr != 0 {
                let sec = read_i32(mem, timeout_ptr).unwrap_or(0).max(0) as u64;
                let usec = read_i32(mem, timeout_ptr.wrapping_add(4))
                    .unwrap_or(0)
                    .clamp(0, 999_999) as u64;
                let total = Duration::from_secs(sec).saturating_add(Duration::from_micros(usec));
                if !total.is_zero() {
                    std::thread::sleep(total.min(Duration::from_millis(16)));
                }
            }
            // 简化：当前不追踪 fd 就绪，统一返回超时/无事件。
            set_ok(cpu, 0);
            Ok(())
        }

        SYS_POLL => {
            let _fds = syscall_arg(mem, cpu, 0)?;
            let _nfds = syscall_arg(mem, cpu, 1)?;
            let timeout_ms = syscall_arg(mem, cpu, 2)? as i32;
            if timeout_ms > 0 {
                std::thread::sleep(Duration::from_millis((timeout_ms as u64).min(16)));
            }
            set_ok(cpu, 0);
            Ok(())
        }

        SYS_NANOSLEEP => {
            let req_ptr = syscall_arg(mem, cpu, 0)?;
            let rem_ptr = syscall_arg(mem, cpu, 1)?;
            if req_ptr == 0 {
                set_err(cpu, EFAULT);
                return Ok(());
            }
            let sec = read_i32(mem, req_ptr).unwrap_or(0).max(0) as u64;
            let nsec = read_i32(mem, req_ptr.wrapping_add(4))
                .unwrap_or(0)
                .clamp(0, 999_999_999) as u64;
            let dur = Duration::from_secs(sec).saturating_add(Duration::from_nanos(nsec));
            if !dur.is_zero() {
                std::thread::sleep(dur.min(Duration::from_millis(32)));
            }
            if rem_ptr != 0 {
                let zero = 0i32.to_le_bytes();
                let _ = mem.write(rem_ptr, &zero);
                let _ = mem.write(rem_ptr.wrapping_add(4), &zero);
            }
            set_ok(cpu, 0);
            Ok(())
        }

        SYS_SIGACTION => {
            let _signum = syscall_arg(mem, cpu, 0)?;
            let _act = syscall_arg(mem, cpu, 1)?;
            let oldact = syscall_arg(mem, cpu, 2)?;
            if oldact != 0 {
                let zero = [0u8; 16];
                let _ = mem.write(oldact, &zero);
            }
            set_ok(cpu, 0);
            Ok(())
        }

        SYS_MPROTECT => {
            let addr = syscall_arg(mem, cpu, 0)?;
            let len = syscall_arg(mem, cpu, 1)?;
            let prot = syscall_arg(mem, cpu, 2)?;
            if len == 0 || !is_page_aligned(addr) {
                set_err(cpu, EINVAL);
                return Ok(());
            }
            let len_aligned = align_up(len, 0x1000);
            let guest_prot = mmap_prot_to_guest(prot);
            if mem.protect(addr, len_aligned, guest_prot).is_err() {
                set_err(cpu, EINVAL);
                return Ok(());
            }
            set_ok(cpu, 0);
            Ok(())
        }

        SYS_MUNMAP => {
            let addr = syscall_arg(mem, cpu, 0)?;
            let len = syscall_arg(mem, cpu, 1)?;
            if len == 0 || !is_page_aligned(addr) {
                set_err(cpu, EINVAL);
                return Ok(());
            }
            let len_aligned = align_up(len, 0x1000);
            if mem.unmap(addr, len_aligned).is_err() {
                set_err(cpu, EINVAL);
                return Ok(());
            }
            set_ok(cpu, 0);
            Ok(())
        }

        _ => {
            if cpu.trace {
                eprintln!("[SHIM] unimplemented syscall {}", syscall_num);
            }
            set_err(cpu, ENOSYS);
            Ok(())
        }
    }
}
