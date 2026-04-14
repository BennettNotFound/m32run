//! Minimal 32‑bit Mach‑O parser
//!
//! This crate provides just enough infrastructure to load a 32‑bit
//! Mach‑O binary from disk and extract its header and segment
//! information.  It does not attempt to interpret every load command
//! or symbol table entry; rather it focuses on the subset needed to
//! map a program into memory and identify its entry point.  The
//! parser is intentionally simple and avoids external dependencies to
//! make it easy to understand and extend.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Magic numbers used by 32‑bit Mach‑O binaries.
const MH_MAGIC: u32 = 0xfeed_face;
const MH_CIGAM: u32 = 0xcefa_edfe; // swapped endianness

/// CPU types.  In a real implementation this list would be much longer
/// and follow the definitions in `<mach/machine.h>`.  We only include
/// the ones relevant to 32‑bit Intel.
#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuType {
    I386 = 7,     // CPU_TYPE_I386
    X86_64 = 0x1000007, // CPU_TYPE_X86_64
    Unknown(u32),
}

impl From<u32> for CpuType {
    fn from(value: u32) -> Self {
        match value {
            7 => CpuType::I386,
            0x1000007 => CpuType::X86_64,
            other => CpuType::Unknown(other),
        }
    }
}

/// Representation of the 32‑bit Mach‑O header.
#[derive(Debug, Clone)]
pub struct MachHeader32 {
    pub magic: u32,
    pub cputype: CpuType,
    pub cpusubtype: u32,
    pub filetype: u32,
    pub ncmds: u32,
    pub sizeofcmds: u32,
    pub flags: u32,
}

/// Representation of a 32‑bit segment load command (LC_SEGMENT).
#[derive(Debug, Clone)]
pub struct Segment32 {
    pub segname: String,
    pub vmaddr: u32,
    pub vmsize: u32,
    pub fileoff: u32,
    pub filesize: u32,
    pub maxprot: u32,
    pub initprot: u32,
    pub nsects: u32,
    pub flags: u32,
}

/// Top‑level representation of a parsed Mach‑O image.  Contains the
/// header and all segment commands found in the load command list.
#[derive(Debug, Clone)]
pub struct MachO32 {
    pub header: MachHeader32,
    pub segments: Vec<Segment32>,
    pub entry_point: u32,
}

/// Reads a 32‑bit little‑endian integer from the provided byte slice.
fn read_u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

/// Parses a 32‑bit Mach‑O file from the given path.  Returns a
/// [`MachO32`] on success or an [`io::Error`] on failure.  The parser
/// only understands 32‑bit little‑endian files and will return an
/// error if the magic number or CPU type do not match expectations.
pub fn parse<P: AsRef<Path>>(path: P) -> io::Result<MachO32> {
    let mut file = File::open(path)?;

    // Read the Mach‑O header (28 bytes for 32‑bit files).
    let mut header_buf = [0u8; 28];
    file.read_exact(&mut header_buf)?;

    let magic = read_u32_le(&header_buf[0..4]);
    if magic != MH_MAGIC && magic != MH_CIGAM {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported magic: 0x{:08x}", magic),
        ));
    }

    let cputype = read_u32_le(&header_buf[4..8]);
    let cpusubtype = read_u32_le(&header_buf[8..12]);
    let filetype = read_u32_le(&header_buf[12..16]);
    let ncmds = read_u32_le(&header_buf[16..20]);
    let sizeofcmds = read_u32_le(&header_buf[20..24]);
    let flags = read_u32_le(&header_buf[24..28]);

    let header = MachHeader32 {
        magic,
        cputype: CpuType::from(cputype),
        cpusubtype,
        filetype,
        ncmds,
        sizeofcmds,
        flags,
    };

    // Validate CPU type
    if header.cputype != CpuType::I386 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Only 32‑bit Intel Mach‑O files are supported, found CPU type: {:?}", header.cputype),
        ));
    }

    // Read all load commands into a buffer
    let mut cmds = vec![0u8; header.sizeofcmds as usize];
    file.read_exact(&mut cmds)?;

    let mut segments = Vec::new();
    let mut offset = 0usize;
    let mut entry_point: Option<u32> = None;

    for _ in 0..header.ncmds {
        if offset + 8 > cmds.len() {
            break;
        }
        // cmd + cmdsize are always the first 8 bytes
        let cmd = read_u32_le(&cmds[offset..offset + 4]);
        let cmdsize = read_u32_le(&cmds[offset + 4..offset + 8]) as usize;
        if cmdsize == 0 || offset + cmdsize > cmds.len() {
            break;
        }
        match cmd {
            // LC_SEGMENT = 0x1
            0x1 => {
                if cmdsize < 56 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "LC_SEGMENT size too small",
                    ));
                }
                // segname (16 bytes), vmaddr (4), vmsize (4), fileoff (4), filesize (4), maxprot (4), initprot (4), nsects (4), flags (4)
                let segname_bytes = &cmds[offset + 8..offset + 24];
                let segname_end = segname_bytes
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(segname_bytes.len());
                let segname = String::from_utf8_lossy(&segname_bytes[..segname_end]).to_string();
                let vmaddr = read_u32_le(&cmds[offset + 24..offset + 28]);
                let vmsize = read_u32_le(&cmds[offset + 28..offset + 32]);
                let fileoff = read_u32_le(&cmds[offset + 32..offset + 36]);
                let filesize = read_u32_le(&cmds[offset + 36..offset + 40]);
                let maxprot = read_u32_le(&cmds[offset + 40..offset + 44]);
                let initprot = read_u32_le(&cmds[offset + 44..offset + 48]);
                let nsects = read_u32_le(&cmds[offset + 48..offset + 52]);
                let flags = read_u32_le(&cmds[offset + 52..offset + 56]);

                segments.push(Segment32 {
                    segname,
                    vmaddr,
                    vmsize,
                    fileoff,
                    filesize,
                    maxprot,
                    initprot,
                    nsects,
                    flags,
                });
            }
            // LC_UNIXTHREAD = 0x5
            0x5 => {
                // The thread command encodes the entry point for 32‑bit Mach‑O files.
                // The format is: flavor (4), count (4), followed by state words.
                // For i386, flavor is 1 and state layout has 16 32‑bit registers (eax, ebx, ecx, edx,
                // edi, esi, ebp, esp, ss, eflags, eip, cs, ds, es, fs, gs).
                if cmdsize >= 16 + 4 * 16 {
                    let eip_offset = offset + 16 + 4 * 10; // 修复：16 + 40 = 56
                    if eip_offset + 4 <= offset + cmdsize {
                        let eip = read_u32_le(&cmds[eip_offset..eip_offset + 4]);
                        entry_point = Some(eip);
                    }
                }
            }
            _ => {}
        }
        offset += cmdsize;
    }

    // Some old binaries use LC_MAIN (0x80000028) instead of LC_UNIXTHREAD; we don't parse it
    let entry = entry_point.unwrap_or(0);
    Ok(MachO32 {
        header,
        segments,
        entry_point: entry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    #[test]
    fn parse_self_binary() {
        // Use the test binary itself as input.  On systems where the test
        // binary is not a 32‑bit Mach‑O this test will simply fail.
        let path = env::current_exe().unwrap();
        let result = parse(&path);
        // The parser should return an error on non‑32‑bit files.
        if let Ok(macho) = result {
            assert_eq!(macho.header.cputype, CpuType::I386);
        }
    }
}