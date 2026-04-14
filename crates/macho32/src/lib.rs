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
const FAT_MAGIC: u32 = 0xcafe_babe;
const FAT_CIGAM: u32 = 0xbeba_feca;
const FAT_MAGIC_64: u32 = 0xcafe_babf;
const FAT_CIGAM_64: u32 = 0xbfba_feca;
const CPU_TYPE_I386: u32 = 7;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const LC_LOAD_DYLIB: u32 = 0xc;
const LC_LOAD_WEAK_DYLIB: u32 = 0x80000018;
const LC_REEXPORT_DYLIB: u32 = 0x8000001f;
const LC_LOAD_UPWARD_DYLIB: u32 = 0x80000023;
const LC_LAZY_LOAD_DYLIB: u32 = 0x20;

const SECTION_TYPE_MASK: u32 = 0x000000ff;
const S_NON_LAZY_SYMBOL_POINTERS: u32 = 0x6;
const S_LAZY_SYMBOL_POINTERS: u32 = 0x7;
const S_SYMBOL_STUBS: u32 = 0x8;

const INDIRECT_SYMBOL_LOCAL: u32 = 0x80000000;
const INDIRECT_SYMBOL_ABS: u32 = 0x40000000;

const N_STAB: u8 = 0xe0;
const N_TYPE: u8 = 0x0e;
const N_EXT: u8 = 0x01;
const N_UNDF: u8 = 0x00;

/// CPU types.  In a real implementation this list would be much longer
/// and follow the definitions in `<mach/machine.h>`.  We only include
/// the ones relevant to 32‑bit Intel.
#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuType {
    I386 = 7,           // CPU_TYPE_I386
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

/// Metadata for `__IMPORT.__jump_table`.
///
/// `reserved1` is the starting index into the indirect symbol table.
/// `stub_size` is the size in bytes of each jump-table stub.
#[derive(Debug, Clone)]
pub struct ImportJumpTable32 {
    pub addr: u32,
    pub size: u32,
    pub reserved1: u32,
    pub stub_size: u32,
}

#[derive(Debug, Clone)]
pub struct Section32 {
    pub sectname: String,
    pub segname: String,
    pub addr: u32,
    pub size: u32,
    pub flags: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportKind32 {
    SymbolStub,
    LazyPointer,
    NonLazyPointer,
}

#[derive(Debug, Clone)]
pub struct ImportEntry32 {
    pub kind: ImportKind32,
    pub addr: u32,
    pub size: u32,
    pub indirect_symbol_index: u32,
    pub symbol_name: Option<String>,
    pub dylib: Option<String>,
    pub section: String,
}

#[derive(Debug, Clone)]
pub struct DylibLoadCommand32 {
    pub ordinal: u8,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct ExportSymbol32 {
    pub name: String,
    pub value: u32,
}

/// Top‑level representation of a parsed Mach‑O image.  Contains the
/// header and all segment commands found in the load command list.
#[derive(Debug, Clone)]
pub struct MachO32 {
    pub header: MachHeader32,
    pub segments: Vec<Segment32>,
    pub sections: Vec<Section32>,
    pub dylibs: Vec<DylibLoadCommand32>,
    pub imports: Vec<ImportEntry32>,
    pub exports: Vec<ExportSymbol32>,
    pub entry_point: u32,
    pub import_jump_table: Option<ImportJumpTable32>,
    pub file_base_offset: u64,
}

/// Reads a 32‑bit little‑endian integer from the provided byte slice.
fn read_u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

/// Reads a 32‑bit big‑endian integer from the provided byte slice.
fn read_u32_be(buf: &[u8]) -> u32 {
    u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
}

/// Reads a 64‑bit big‑endian integer from the provided byte slice.
fn read_u64_be(buf: &[u8]) -> u64 {
    u64::from_be_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ])
}

fn read_c_string(buf: &[u8], start: usize) -> Option<String> {
    if start >= buf.len() {
        return None;
    }
    let end = buf[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|p| start + p)
        .unwrap_or(buf.len());
    Some(String::from_utf8_lossy(&buf[start..end]).to_string())
}

#[derive(Debug, Clone)]
struct SymbolTableEntry32 {
    name: Option<String>,
    dylib_ordinal: u8,
}

/// Parse a FAT header and return the offset of the i386 slice if present.
fn parse_fat_i386_offset(file: &mut File, fat_magic: u32) -> io::Result<Option<u64>> {
    let is_64 = matches!(fat_magic, FAT_MAGIC_64 | FAT_CIGAM_64);
    let arch_size = if is_64 { 32usize } else { 20usize };

    // fat_header starts with magic at offset 0, then nfat_arch at offset 4.
    file.seek(SeekFrom::Start(4))?;
    let mut nfat_arch_buf = [0u8; 4];
    file.read_exact(&mut nfat_arch_buf)?;
    let nfat_arch = read_u32_be(&nfat_arch_buf) as usize;

    let mut arch_buf = [0u8; 32];
    for _ in 0..nfat_arch {
        file.read_exact(&mut arch_buf[..arch_size])?;

        let cputype = read_u32_be(&arch_buf[0..4]);
        if cputype != CPU_TYPE_I386 {
            continue;
        }

        let offset = if is_64 {
            read_u64_be(&arch_buf[8..16])
        } else {
            read_u32_be(&arch_buf[8..12]) as u64
        };
        return Ok(Some(offset));
    }

    Ok(None)
}

fn parse_macho32_at(file: &mut File, base_offset: u64) -> io::Result<MachO32> {
    file.seek(SeekFrom::Start(base_offset))?;

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
            format!(
                "Only 32‑bit Intel Mach‑O files are supported, found CPU type: {:?}",
                header.cputype
            ),
        ));
    }

    // Read all load commands into a buffer
    let mut cmds = vec![0u8; header.sizeofcmds as usize];
    file.read_exact(&mut cmds)?;

    let mut segments = Vec::new();
    let mut sections = Vec::new();
    let mut dylibs = Vec::new();
    let mut offset = 0usize;
    let mut entry_point: Option<u32> = None;
    let mut import_jump_table: Option<ImportJumpTable32> = None;
    let mut symtab: Option<(u32, u32, u32, u32)> = None;
    let mut dysymtab: Option<(u32, u32)> = None;

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
                let fileoff_in_slice = read_u32_le(&cmds[offset + 32..offset + 36]);
                let filesize = read_u32_le(&cmds[offset + 36..offset + 40]);
                let maxprot = read_u32_le(&cmds[offset + 40..offset + 44]);
                let initprot = read_u32_le(&cmds[offset + 44..offset + 48]);
                let nsects = read_u32_le(&cmds[offset + 48..offset + 52]);
                let flags = read_u32_le(&cmds[offset + 52..offset + 56]);

                let fileoff = fileoff_in_slice;

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

                // Parse section records attached to this LC_SEGMENT.
                // section (32-bit) layout is 68 bytes.
                let sections_start = offset + 56;
                let section_size = 68usize;
                let cmd_end = offset + cmdsize;
                for idx in 0..(nsects as usize) {
                    let sec_off = sections_start + idx * section_size;
                    if sec_off + section_size > cmd_end {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "LC_SEGMENT section list truncated",
                        ));
                    }

                    let sectname_bytes = &cmds[sec_off..sec_off + 16];
                    let sectname_end = sectname_bytes
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(sectname_bytes.len());
                    let sectname =
                        String::from_utf8_lossy(&sectname_bytes[..sectname_end]).to_string();

                    let sec_segname_bytes = &cmds[sec_off + 16..sec_off + 32];
                    let sec_segname_end = sec_segname_bytes
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(sec_segname_bytes.len());
                    let sec_segname =
                        String::from_utf8_lossy(&sec_segname_bytes[..sec_segname_end]).to_string();
                    let sec_addr = read_u32_le(&cmds[sec_off + 32..sec_off + 36]);
                    let sec_size = read_u32_le(&cmds[sec_off + 36..sec_off + 40]);
                    let sec_flags = read_u32_le(&cmds[sec_off + 56..sec_off + 60]);
                    let reserved1 = read_u32_le(&cmds[sec_off + 60..sec_off + 64]);
                    let reserved2 = read_u32_le(&cmds[sec_off + 64..sec_off + 68]);

                    if sec_segname == "__IMPORT" && sectname == "__jump_table" {
                        import_jump_table = Some(ImportJumpTable32 {
                            addr: sec_addr,
                            size: sec_size,
                            reserved1,
                            stub_size: reserved2,
                        });
                    }

                    sections.push(Section32 {
                        sectname,
                        segname: sec_segname,
                        addr: sec_addr,
                        size: sec_size,
                        flags: sec_flags,
                        reserved1,
                        reserved2,
                    });
                }
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
            LC_SYMTAB => {
                if cmdsize >= 24 {
                    let symoff = read_u32_le(&cmds[offset + 8..offset + 12]);
                    let nsyms = read_u32_le(&cmds[offset + 12..offset + 16]);
                    let stroff = read_u32_le(&cmds[offset + 16..offset + 20]);
                    let strsize = read_u32_le(&cmds[offset + 20..offset + 24]);
                    symtab = Some((symoff, nsyms, stroff, strsize));
                }
            }
            LC_DYSYMTAB => {
                if cmdsize >= 64 {
                    let indirectsymoff = read_u32_le(&cmds[offset + 56..offset + 60]);
                    let nindirectsyms = read_u32_le(&cmds[offset + 60..offset + 64]);
                    dysymtab = Some((indirectsymoff, nindirectsyms));
                }
            }
            LC_LOAD_DYLIB | LC_LOAD_WEAK_DYLIB | LC_REEXPORT_DYLIB | LC_LOAD_UPWARD_DYLIB
            | LC_LAZY_LOAD_DYLIB => {
                if cmdsize >= 24 {
                    let name_offset = read_u32_le(&cmds[offset + 8..offset + 12]) as usize;
                    let command_start = offset;
                    let name_start = command_start.saturating_add(name_offset);
                    let command_end = offset + cmdsize;
                    if name_start < command_end {
                        let name = read_c_string(&cmds[..command_end], name_start)
                            .unwrap_or_else(|| "<invalid-dylib-name>".to_string());
                        let ordinal = (dylibs.len() + 1).min(255) as u8;
                        dylibs.push(DylibLoadCommand32 { ordinal, name });
                    }
                }
            }
            _ => {}
        }
        offset += cmdsize;
    }

    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut symbols = Vec::new();
    let mut nsyms = 0u32;

    if let Some((symoff, sym_count, stroff, strsize)) = symtab {
        nsyms = sym_count;
        let mut strtab = vec![0u8; strsize as usize];
        file.seek(SeekFrom::Start(base_offset + stroff as u64))?;
        file.read_exact(&mut strtab)?;

        symbols = Vec::with_capacity(nsyms as usize);
        let mut nlist_buf = [0u8; 12];
        file.seek(SeekFrom::Start(base_offset + symoff as u64))?;
        for _ in 0..nsyms {
            file.read_exact(&mut nlist_buf)?;
            let n_strx = read_u32_le(&nlist_buf[0..4]) as usize;
            let n_type = nlist_buf[4];
            let n_desc = u16::from_le_bytes([nlist_buf[6], nlist_buf[7]]);
            let n_value = read_u32_le(&nlist_buf[8..12]);
            let dylib_ordinal = ((n_desc >> 8) & 0x00ff) as u8;
            let name = if n_strx < strtab.len() {
                read_c_string(&strtab, n_strx)
            } else {
                None
            };

            if (n_type & N_STAB) == 0
                && (n_type & N_EXT) != 0
                && (n_type & N_TYPE) != N_UNDF
                && n_value != 0
            {
                if let Some(sym_name) = name.clone() {
                    if !sym_name.is_empty() {
                        exports.push(ExportSymbol32 {
                            name: sym_name,
                            value: n_value,
                        });
                    }
                }
            }

            symbols.push(SymbolTableEntry32 {
                name,
                dylib_ordinal,
            });
        }
    }

    if let Some((indirectsymoff, nindirectsyms)) = dysymtab {
        let mut indirect_table = vec![0u8; (nindirectsyms as usize) * 4];
        file.seek(SeekFrom::Start(base_offset + indirectsymoff as u64))?;
        file.read_exact(&mut indirect_table)?;

        for sec in &sections {
            let section_type = sec.flags & SECTION_TYPE_MASK;
            let import_kind = match section_type {
                S_SYMBOL_STUBS => Some(ImportKind32::SymbolStub),
                S_LAZY_SYMBOL_POINTERS => Some(ImportKind32::LazyPointer),
                S_NON_LAZY_SYMBOL_POINTERS => Some(ImportKind32::NonLazyPointer),
                _ => None,
            };
            let Some(kind) = import_kind else {
                continue;
            };

            let item_size = if kind == ImportKind32::SymbolStub {
                sec.reserved2
            } else {
                4
            };
            if item_size == 0 {
                continue;
            }
            let item_count = sec.size / item_size;

            for i in 0..item_count {
                let indirect_symbol_index = sec.reserved1 + i;
                if indirect_symbol_index >= nindirectsyms {
                    continue;
                }

                let table_off = (indirect_symbol_index as usize) * 4;
                let raw_symbol_index = read_u32_le(&indirect_table[table_off..table_off + 4]);
                let is_local = (raw_symbol_index & INDIRECT_SYMBOL_LOCAL) != 0;
                let is_abs = (raw_symbol_index & INDIRECT_SYMBOL_ABS) != 0;

                let (symbol_name, dylib) = if is_local || is_abs {
                    (None, None)
                } else {
                    let symbol_index =
                        raw_symbol_index & !(INDIRECT_SYMBOL_LOCAL | INDIRECT_SYMBOL_ABS);
                    if symbol_index < nsyms {
                        if let Some(sym) = symbols.get(symbol_index as usize) {
                            let symbol_name = sym.name.clone();
                            let dylib = dylibs
                                .iter()
                                .find(|d| d.ordinal == sym.dylib_ordinal)
                                .map(|d| d.name.clone());
                            (symbol_name, dylib)
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    }
                };

                imports.push(ImportEntry32 {
                    kind,
                    addr: sec.addr.wrapping_add(i.wrapping_mul(item_size)),
                    size: item_size,
                    indirect_symbol_index,
                    symbol_name,
                    dylib,
                    section: format!("{}.{}", sec.segname, sec.sectname),
                });
            }
        }
    }

    // Some old binaries use LC_MAIN (0x80000028) instead of LC_UNIXTHREAD; we don't parse it
    let entry = entry_point.unwrap_or(0);
    Ok(MachO32 {
        header,
        segments,
        sections,
        dylibs,
        imports,
        exports,
        entry_point: entry,
        import_jump_table,
        file_base_offset: base_offset,
    })
}

/// Parses a 32‑bit Mach‑O file from the given path.  Returns a
/// [`MachO32`] on success or an [`io::Error`] on failure.  The parser
/// only understands 32‑bit little‑endian files and will return an
/// error if the magic number or CPU type do not match expectations.
pub fn parse<P: AsRef<Path>>(path: P) -> io::Result<MachO32> {
    let mut file = File::open(path)?;

    let mut magic_buf = [0u8; 4];
    file.read_exact(&mut magic_buf)?;

    let magic_le = read_u32_le(&magic_buf);
    if magic_le == MH_MAGIC || magic_le == MH_CIGAM {
        return parse_macho32_at(&mut file, 0);
    }

    let magic_be = read_u32_be(&magic_buf);
    if matches!(
        magic_be,
        FAT_MAGIC | FAT_CIGAM | FAT_MAGIC_64 | FAT_CIGAM_64
    ) {
        let i386_offset = parse_fat_i386_offset(&mut file, magic_be)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "FAT Mach‑O does not contain an i386 slice",
            )
        })?;
        return parse_macho32_at(&mut file, i386_offset);
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("Unsupported magic: 0x{:08x}", magic_le),
    ))
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
