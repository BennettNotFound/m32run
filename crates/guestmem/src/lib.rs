//! Simple guest memory implementation
//!
//! The goal of this module is to provide a basic model of a 32‑bit
//! address space.  Memory is organised into regions which are
//! individually allocated and tracked.  Each region records its
//! starting virtual address, size and access permissions.  Reads and
//! writes locate the appropriate region by address and perform the
//! operation in the corresponding host buffer.  Overlaps and holes
//! are not handled; regions should be mapped such that they do not
//! conflict.

use std::fmt;
use std::ops::Range;

/// Memory protection flags.  These correspond loosely to the POSIX
/// `PROT_READ`, `PROT_WRITE` and `PROT_EXEC` constants but are
/// deliberately defined here to avoid pulling in libc.  The flags
/// can be combined with the bitwise OR operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prot(u8);

impl Prot {
    pub const READ: Prot = Prot(0b001);
    pub const WRITE: Prot = Prot(0b010);
    pub const EXEC: Prot = Prot(0b100);

    pub fn contains(self, other: Prot) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for Prot {
    type Output = Prot;
    fn bitor(self, rhs: Prot) -> Prot {
        Prot(self.0 | rhs.0)
    }
}

/// Errors that can occur when accessing guest memory.
#[derive(Debug)]
pub enum Error {
    AddressNotMapped(u32),
    InvalidRange { addr: u32, size: u32 },
    WriteToReadOnly(u32),
    ReadFromWriteOnly(u32),
    ExecutePermissionDenied(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AddressNotMapped(addr) => write!(f, "address {addr:#x} not mapped"),
            Error::InvalidRange { addr, size } => {
                write!(f, "invalid range addr={addr:#x} size={size:#x}")
            }
            Error::WriteToReadOnly(addr) => {
                write!(f, "write attempted on non-writable region at {addr:#x}")
            }
            Error::ReadFromWriteOnly(addr) => {
                write!(f, "read attempted on non-readable region at {addr:#x}")
            }
            Error::ExecutePermissionDenied(addr) => {
                write!(f, "execute attempted on non-executable region at {addr:#x}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Represents a contiguous region of guest memory.  The region has a
/// starting virtual address, a length and a protection mask.  The
/// region owns a host buffer into which data is stored.
#[derive(Clone)]
struct Region {
    range: Range<u32>,
    prot: Prot,
    data: Vec<u8>,
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Region")
            .field("range", &format!("{:#x?}", self.range.clone()))
            .field("prot", &self.prot)
            .finish()
    }
}

/// A 32‑bit address space composed of non‑overlapping regions.  Each
/// region must be mapped explicitly before it can be accessed.  The
/// implementation performs linear searches over the region list; this
/// is sufficient for small prototypes but is not optimised for
/// production use.
#[derive(Debug, Default)]
pub struct GuestMemory {
    regions: Vec<Region>,
}

impl GuestMemory {
    /// Creates a new empty guest memory.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Maps a new region at the given virtual address with the given
    /// size and protection.  The host buffer is initialised to zero.
    /// It is an error to map over an existing region.
    pub fn map(&mut self, addr: u32, size: u32, prot: Prot) -> Result<(), Error> {
        let end = self.validate_range(addr, size)?;
        let new_range = addr..end;
        // Check for overlap with existing regions
        for reg in &self.regions {
            if ranges_overlap(&new_range, &reg.range) {
                return Err(Error::AddressNotMapped(addr));
            }
        }
        let data = vec![0u8; size as usize];
        self.regions.push(Region {
            range: new_range,
            prot,
            data,
        });
        Ok(())
    }

    /// Maps a region at a fixed address, replacing overlapping mappings.
    pub fn map_fixed(&mut self, addr: u32, size: u32, prot: Prot) -> Result<(), Error> {
        self.validate_range(addr, size)?;
        self.unmap_inner(addr, size, false)?;
        self.map(addr, size, prot)
    }

    /// Unmaps an existing range. The range must be fully mapped.
    pub fn unmap(&mut self, addr: u32, size: u32) -> Result<(), Error> {
        self.unmap_inner(addr, size, true)
    }

    /// Changes memory protection for an existing range.
    /// The range must be fully mapped.
    pub fn protect(&mut self, addr: u32, size: u32, prot: Prot) -> Result<(), Error> {
        let end = self.validate_range(addr, size)?;
        if !self.is_range_fully_mapped(addr, end) {
            return Err(Error::AddressNotMapped(addr));
        }

        let mut out = Vec::with_capacity(self.regions.len() + 2);
        for reg in self.regions.drain(..) {
            let ov_start = reg.range.start.max(addr);
            let ov_end = reg.range.end.min(end);
            if ov_start >= ov_end {
                out.push(reg);
                continue;
            }

            if reg.range.start < ov_start {
                let left_len = (ov_start - reg.range.start) as usize;
                out.push(Region {
                    range: reg.range.start..ov_start,
                    prot: reg.prot,
                    data: reg.data[..left_len].to_vec(),
                });
            }

            let mid_start = (ov_start - reg.range.start) as usize;
            let mid_end = (ov_end - reg.range.start) as usize;
            out.push(Region {
                range: ov_start..ov_end,
                prot,
                data: reg.data[mid_start..mid_end].to_vec(),
            });

            if ov_end < reg.range.end {
                let right_start = (ov_end - reg.range.start) as usize;
                out.push(Region {
                    range: ov_end..reg.range.end,
                    prot: reg.prot,
                    data: reg.data[right_start..].to_vec(),
                });
            }
        }
        self.regions = out;
        Ok(())
    }

    /// Writes data into the guest memory at the specified address.
    /// Returns an error if the address is unmapped or if the region
    /// does not have write permission.  Partial writes across region
    /// boundaries are not supported.
    pub fn write(&mut self, addr: u32, buf: &[u8]) -> Result<(), Error> {
        let mut cur = addr;
        let mut src_off = 0usize;
        while src_off < buf.len() {
            let (reg_idx, offset) = self.find_region(cur, Prot::WRITE)?;
            let region_end = self.regions[reg_idx].range.end;
            let can_write = (region_end - cur) as usize;
            let n = can_write.min(buf.len() - src_off);
            let region = &mut self.regions[reg_idx];
            region.data[offset..offset + n].copy_from_slice(&buf[src_off..src_off + n]);
            cur = cur.wrapping_add(n as u32);
            src_off += n;
        }
        Ok(())
    }

    /// Reads data from the guest memory at the specified address into
    /// the provided buffer.  Returns an error if the address is
    /// unmapped or if the region does not have read permission.
    pub fn read(&self, addr: u32, buf: &mut [u8]) -> Result<(), Error> {
        self.read_inner(addr, buf, Prot::READ)
    }

    /// Reads instruction bytes and enforces execute permission.
    pub fn read_exec(&self, addr: u32, buf: &mut [u8]) -> Result<(), Error> {
        self.read_inner(addr, buf, Prot::EXEC)
    }

    /// Finds the region containing the given address and checks that
    /// it has the requested permission.  Returns the region index and
    /// the offset into its data buffer.
    fn find_region(&self, addr: u32, required_prot: Prot) -> Result<(usize, usize), Error> {
        let Some(idx) = self.find_region_index(addr) else {
            return Err(Error::AddressNotMapped(addr));
        };
        let reg = &self.regions[idx];
        if !reg.prot.contains(required_prot) {
            match required_prot {
                Prot::READ => return Err(Error::ReadFromWriteOnly(addr)),
                Prot::WRITE => return Err(Error::WriteToReadOnly(addr)),
                Prot::EXEC => return Err(Error::ExecutePermissionDenied(addr)),
                _ => return Err(Error::AddressNotMapped(addr)),
            }
        }
        let offset = (addr - reg.range.start) as usize;
        Ok((idx, offset))
    }

    fn find_region_index(&self, addr: u32) -> Option<usize> {
        self.regions
            .iter()
            .position(|reg| addr >= reg.range.start && addr < reg.range.end)
    }

    fn validate_range(&self, addr: u32, size: u32) -> Result<u32, Error> {
        if size == 0 {
            return Err(Error::InvalidRange { addr, size });
        }
        addr.checked_add(size)
            .ok_or(Error::InvalidRange { addr, size })
    }

    fn is_range_fully_mapped(&self, start: u32, end: u32) -> bool {
        let mut cur = start;
        while cur < end {
            let Some(idx) = self.find_region_index(cur) else {
                return false;
            };
            let reg_end = self.regions[idx].range.end;
            if reg_end <= cur {
                return false;
            }
            cur = reg_end.min(end);
        }
        true
    }

    fn unmap_inner(&mut self, addr: u32, size: u32, require_full: bool) -> Result<(), Error> {
        let end = self.validate_range(addr, size)?;
        if require_full && !self.is_range_fully_mapped(addr, end) {
            return Err(Error::AddressNotMapped(addr));
        }

        let mut out = Vec::with_capacity(self.regions.len() + 2);
        for reg in self.regions.drain(..) {
            let ov_start = reg.range.start.max(addr);
            let ov_end = reg.range.end.min(end);
            if ov_start >= ov_end {
                out.push(reg);
                continue;
            }

            if reg.range.start < ov_start {
                let left_len = (ov_start - reg.range.start) as usize;
                out.push(Region {
                    range: reg.range.start..ov_start,
                    prot: reg.prot,
                    data: reg.data[..left_len].to_vec(),
                });
            }

            if ov_end < reg.range.end {
                let right_start = (ov_end - reg.range.start) as usize;
                out.push(Region {
                    range: ov_end..reg.range.end,
                    prot: reg.prot,
                    data: reg.data[right_start..].to_vec(),
                });
            }
        }
        self.regions = out;
        Ok(())
    }

    fn read_inner(&self, addr: u32, buf: &mut [u8], required: Prot) -> Result<(), Error> {
        let mut cur = addr;
        let mut dst_off = 0usize;
        while dst_off < buf.len() {
            let (reg_idx, offset) = self.find_region(cur, required)?;
            let region = &self.regions[reg_idx];
            let can_read = (region.range.end - cur) as usize;
            let n = can_read.min(buf.len() - dst_off);
            buf[dst_off..dst_off + n].copy_from_slice(&region.data[offset..offset + n]);
            cur = cur.wrapping_add(n as u32);
            dst_off += n;
        }
        Ok(())
    }
}

fn ranges_overlap(a: &Range<u32>, b: &Range<u32>) -> bool {
    a.start < b.end && b.start < a.end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_and_access() {
        let mut gm = GuestMemory::new();
        gm.map(0x1000, 0x100, Prot::READ | Prot::WRITE).unwrap();
        gm.write(0x1000, &[1, 2, 3, 4]).unwrap();
        let mut buf = [0u8; 4];
        gm.read(0x1000, &mut buf).unwrap();
        assert_eq!(&buf, &[1, 2, 3, 4]);
    }

    #[test]
    fn cross_region_rw() {
        let mut gm = GuestMemory::new();
        gm.map(0x1000, 0x100, Prot::READ | Prot::WRITE).unwrap();
        gm.map(0x1100, 0x100, Prot::READ | Prot::WRITE).unwrap();
        let bytes = vec![0xAB; 0x120];
        gm.write(0x1080, &bytes).unwrap();
        let mut out = vec![0u8; 0x120];
        gm.read(0x1080, &mut out).unwrap();
        assert_eq!(out, bytes);
    }

    #[test]
    fn mprotect_and_exec_check() {
        let mut gm = GuestMemory::new();
        gm.map(0x2000, 0x100, Prot::READ | Prot::WRITE).unwrap();
        let mut one = [0u8; 1];
        assert!(gm.read_exec(0x2000, &mut one).is_err());
        gm.protect(0x2000, 0x100, Prot::READ | Prot::EXEC).unwrap();
        assert!(gm.write(0x2000, &[1]).is_err());
        gm.read_exec(0x2000, &mut one).unwrap();
    }

    #[test]
    fn unmap_middle_split() {
        let mut gm = GuestMemory::new();
        gm.map(0x3000, 0x300, Prot::READ | Prot::WRITE).unwrap();
        gm.unmap(0x3100, 0x100).unwrap();
        let mut b = [0u8; 1];
        assert!(gm.read(0x3050, &mut b).is_ok());
        assert!(gm.read(0x3150, &mut b).is_err());
        assert!(gm.read(0x3250, &mut b).is_ok());
    }
}
