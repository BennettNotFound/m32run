//! Program loader for 32‑bit Mach‑O binaries
//!
//! This crate ties together the Mach‑O parser, guest memory manager,
//! ABI setup and CPU core to create a runnable environment for a
//! 32‑bit x86 binary.  The loader reads the binary from disk, maps
//! its segments into memory, sets up an initial stack with
//! command‑line arguments and returns an initialised CPU and memory
//! ready for execution.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use abi::{prepare_stack, STACK_BASE, STACK_SIZE};
use guestmem::{GuestMemory, Prot};
use macho32::{parse, Segment32};
use thiserror::Error;
use x86core::Cpu;

/// Errors that can occur while loading a binary.
#[derive(Error, Debug)]
pub enum LoadError {
    #[error(transparent)]
    Io(#[from] io::Error),
    // 如果 Parse 是 Mach-O 解析错误，它应该对应 macho32 的错误类型
    // 或者如果你想区分不同的 IO 场景，手动实现或改名
    #[error("Mach-O parse error: {0}")]
    Macho(String),

    // 必须添加这个，因为报错信息显示你还在用 ? 处理 guestmem::Error
    #[error("Memory error: {0}")]
    Memory(#[from] guestmem::Error),
}

/// Represents a loaded program: guest memory, initial CPU state and
/// optional program name.  The CPU has its instruction pointer and
/// stack pointer initialised; other registers are cleared.
#[derive(Debug)]
pub struct LoadedProgram {
    pub mem: GuestMemory,
    pub cpu: Cpu,
}

/// Loads the program at `path` with the given command‑line arguments.
/// Returns a [`LoadedProgram`] on success.  Note that the loader
/// assumes the binary is a valid 32‑bit Mach‑O and will map its
/// segments verbatim into the guest address space; improper binaries
/// may crash the interpreter.
pub fn load<P: AsRef<Path>>(path: P, args: &[String]) -> Result<LoadedProgram, LoadError> {
    // Parse the Mach‑O file header and segment table.
    let macho = parse(&path).map_err(LoadError::Io)?;
    let mut file = File::open(path)?;

    let mut mem = GuestMemory::new();

    // Map each segment into guest memory and copy its contents from the file.
    for seg in &macho.segments {
        // Determine permissions: for this proof of concept we map
        // executable segments as RX and data segments as RWX.  In a
        // real loader you would honour the maxprot/initprot fields.
        let mut prot = Prot::READ;
        if (seg.maxprot & 0x1) != 0 {
            prot = prot | Prot::EXEC;
        }
        if (seg.maxprot & 0x2) != 0 {
            prot = prot | Prot::WRITE;
        }
        // Map region
        mem.map(seg.vmaddr, seg.vmsize, prot)?;
        // Read segment contents from file if filesize > 0
        if seg.filesize > 0 {
            file.seek(SeekFrom::Start(seg.fileoff as u64))?;
            let mut buf = vec![0u8; seg.filesize as usize];
            file.read_exact(&mut buf)?;
            mem.write(seg.vmaddr, &buf)?;
        }
        // Zero initialised BSS (remaining bytes) is already zero due to map
    }

    // Set up the stack with arguments and obtain initial ESP.
    let sp = prepare_stack(&mut mem, args)?;

    // Create a new CPU and initialise registers.
    let mut cpu = Cpu::new();
    cpu.eip = macho.entry_point;
    cpu.esp = sp;

    Ok(LoadedProgram { mem, cpu })
}