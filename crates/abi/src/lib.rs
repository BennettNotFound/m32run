//! System call and ABI support
//!
//! This crate defines constants and helper routines related to the
//! 32‑bit Darwin Application Binary Interface (ABI).  For the
//! purposes of this example we only set up the initial stack with
//! command‑line arguments.  Real implementations would also need to
//! populate the environment pointer array, auxiliary vector and other
//! process metadata.

use guestmem::{GuestMemory, Prot};
use guestmem::Error as GuestError;

/// Address at which the guest stack will be mapped.  We reserve the
/// top 64 KiB of the 32‑bit address space for the stack.  In a real
/// system the stack would be much larger and would grow downwards as
/// needed.
pub const STACK_BASE: u32 = 0x7fff_0000;
pub const STACK_SIZE: u32 = 0x0001_0000; // 64 KiB

/// Sets up the initial stack for the program.  The function maps a
/// new stack region in the guest memory, copies the argument strings
/// into the stack, writes pointers to the strings and returns the
/// initial value of the guest ESP register.  The stack grows
/// downwards: data is placed at decreasing addresses.  The return
/// value points to the argc word at the top of the stack frame.
pub fn prepare_stack(mem: &mut GuestMemory, args: &[String]) -> Result<u32, GuestError> {
    // Map the stack region with read/write permissions.
    mem.map(STACK_BASE - STACK_SIZE, STACK_SIZE, Prot::READ | Prot::WRITE)?;

    // We will build the stack from the top down.
    let mut sp: u32 = STACK_BASE;

    // Copy argument strings into the stack and record their addresses.
    // Arguments are pushed in reverse order so that argv[0] ends up at
    // the lowest address.
    let mut arg_ptrs = Vec::new();
    for arg in args.iter().rev() {
        let bytes = arg.as_bytes();
        let len = bytes.len() as u32 + 1; // include null terminator
        sp = sp.wrapping_sub(len);
        let addr = sp;
        // Write string
        let mut buf = Vec::with_capacity(len as usize);
        buf.extend_from_slice(bytes);
        buf.push(0); // null terminator
        mem.write(addr, &buf)?;
        arg_ptrs.push(addr);
    }
    // Word align
    sp &= !0x3;
    // Now push a null pointer terminator for argv
    sp = sp.wrapping_sub(4);
    mem.write(sp, &0u32.to_le_bytes())?;
    // Push pointers to arguments in ascending order
    for &ptr in arg_ptrs.iter().rev() {
        sp = sp.wrapping_sub(4);
        mem.write(sp, &ptr.to_le_bytes())?;
    }
    // Push argc
    let argc = args.len() as u32;
    sp = sp.wrapping_sub(4);
    mem.write(sp, &argc.to_le_bytes())?;
    Ok(sp)
}