use crate::exec::ExecError;
use guestmem::GuestMemory;

#[derive(Debug, Clone, Copy)]
pub struct ModRm {
    pub mod_: u8,
    pub reg: u8,
    pub rm: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct Sib {
    pub scale: u8,
    pub index: u8,
    pub base: u8,
}

#[derive(Debug, Clone, Copy)]
pub enum Operand32 {
    Reg(u8),
    Mem(u32),
}

pub struct InstructionStream<'a> {
    mem: &'a GuestMemory,
    pub ip: u32,
    pub start_eip: u32,
}

impl<'a> InstructionStream<'a> {
    pub fn new(mem: &'a GuestMemory, ip: u32) -> Self {
        Self {
            mem,
            ip,
            start_eip: ip,
        }
    }

    pub fn fetch_u8(&mut self) -> Result<u8, ExecError> {
        let mut buf = [0u8; 1];
        self.mem.read(self.ip, &mut buf)?;
        self.ip = self.ip.wrapping_add(1);
        Ok(buf[0])
    }

    pub fn fetch_i8(&mut self) -> Result<i8, ExecError> {
        Ok(self.fetch_u8()? as i8)
    }

    pub fn fetch_u16(&mut self) -> Result<u16, ExecError> {
        let mut buf = [0u8; 2];
        self.mem.read(self.ip, &mut buf)?;
        self.ip = self.ip.wrapping_add(2);
        Ok(u16::from_le_bytes(buf))
    }

    pub fn fetch_u32(&mut self) -> Result<u32, ExecError> {
        let mut buf = [0u8; 4];
        self.mem.read(self.ip, &mut buf)?;
        self.ip = self.ip.wrapping_add(4);
        Ok(u32::from_le_bytes(buf))
    }

    pub fn fetch_i32(&mut self) -> Result<i32, ExecError> {
        Ok(self.fetch_u32()? as i32)
    }

    pub fn fetch_modrm(&mut self) -> Result<ModRm, ExecError> {
        let b = self.fetch_u8()?;
        Ok(ModRm {
            mod_: (b >> 6) & 0b11,
            reg: (b >> 3) & 0b111,
            rm: b & 0b111,
        })
    }

    pub fn fetch_sib(&mut self) -> Result<Sib, ExecError> {
        let b = self.fetch_u8()?;
        Ok(Sib {
            scale: (b >> 6) & 0b11,
            index: (b >> 3) & 0b111,
            base: b & 0b111,
        })
    }
}
