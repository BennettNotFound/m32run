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

use thiserror::Error;

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
#[derive(Error, Debug)]
pub enum Error {
    #[error("address {0:#x} not mapped")]
    AddressNotMapped(u32),
    #[error("write attempted on non‑writable region at {0:#x}")]
    WriteToReadOnly(u32),
    #[error("read attempted on non‑readable region at {0:#x}")]
    ReadFromWriteOnly(u32),
}

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
        Self { regions: Vec::new() }
    }

    /// Maps a new region at the given virtual address with the given
    /// size and protection.  The host buffer is initialised to zero.
    /// It is an error to map over an existing region.
    pub fn map(&mut self, addr: u32, size: u32, prot: Prot) -> Result<(), Error> {
        let end = addr.checked_add(size).expect("overflow in region size");
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

    /// Writes data into the guest memory at the specified address.
    /// Returns an error if the address is unmapped or if the region
    /// does not have write permission.  Partial writes across region
    /// boundaries are not supported.
    pub fn write(&mut self, addr: u32, buf: &[u8]) -> Result<(), Error> {
        let (reg_idx, offset) = self.find_region(addr, Prot::WRITE)?;
        let region = &mut self.regions[reg_idx];
        if offset + buf.len() > region.data.len() {
            return Err(Error::WriteToReadOnly(addr));
        }
        region.data[offset..offset + buf.len()].copy_from_slice(buf);
        Ok(())
    }

    /// Reads data from the guest memory at the specified address into
    /// the provided buffer.  Returns an error if the address is
    /// unmapped or if the region does not have read permission.
    pub fn read(&self, addr: u32, buf: &mut [u8]) -> Result<(), Error> {
        let (reg_idx, offset) = self.find_region(addr, Prot::READ)?;
        let region = &self.regions[reg_idx];
        if offset + buf.len() > region.data.len() {
            return Err(Error::ReadFromWriteOnly(addr));
        }
        buf.copy_from_slice(&region.data[offset..offset + buf.len()]);
        Ok(())
    }

    /// Finds the region containing the given address and checks that
    /// it has the requested permission.  Returns the region index and
    /// the offset into its data buffer.
    fn find_region(&self, addr: u32, required_prot: Prot) -> Result<(usize, usize), Error> {
        for (idx, reg) in self.regions.iter().enumerate() {
            if addr >= reg.range.start && addr < reg.range.end {
                if !reg.prot.contains(required_prot) {
                    match required_prot {
                        Prot::READ => return Err(Error::ReadFromWriteOnly(addr)),
                        Prot::WRITE => return Err(Error::WriteToReadOnly(addr)),
                        _ => return Err(Error::AddressNotMapped(addr)),
                    }
                }
                let offset = (addr - reg.range.start) as usize;
                return Ok((idx, offset));
            }
        }
        Err(Error::AddressNotMapped(addr))
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
}