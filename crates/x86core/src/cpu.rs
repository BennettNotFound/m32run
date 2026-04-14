use crate::decode::{InstructionStream, ModRm, Operand32};
use crate::exec::ExecError;
use crate::flags::Flags;
use guestmem::GuestMemory;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Cpu {
    pub eax: u32,
    pub ebx: u32,
    pub ecx: u32,
    pub edx: u32,
    pub esi: u32,
    pub edi: u32,
    pub ebp: u32,
    pub esp: u32,
    pub eip: u32,
    pub eflags: u32,
    pub tsc: u64,
    pub xmm: [[u8; 16]; 8],
    pub fpu_stack: Vec<f64>,
    // __IMPORT.__jump_table 元数据
    pub import_jump_table_addr: u32,
    pub import_jump_table_size: u32,
    pub import_jump_table_stub_size: u32,
    pub import_jump_table_reserved1: u32,
    // 任意导入 stub 地址 -> indirect symbol index
    pub import_stub_indirect_map: HashMap<u32, u32>,
    pub trace: bool, // 新增：是否开启追踪
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
            esi: 0,
            edi: 0,
            ebp: 0,
            esp: 0,
            eip: 0,
            eflags: 0,
            tsc: 0,
            xmm: [[0u8; 16]; 8],
            fpu_stack: Vec::new(),
            import_jump_table_addr: 0,
            import_jump_table_size: 0,
            import_jump_table_stub_size: 0,
            import_jump_table_reserved1: 0,
            import_stub_indirect_map: HashMap::new(),
            trace: false, // 默认关闭
        }
    }

    pub fn trace_state(&self, mem: &GuestMemory) {
        if !self.trace {
            return;
        }

        let mut bytes = Vec::new();
        for i in 0..8u32 {
            match self.read_u8(mem, self.eip.wrapping_add(i)) {
                Ok(b) => bytes.push(format!("{:02x}", b)),
                Err(_) => {
                    bytes.push("??".to_string());
                    break;
                }
            }
        }

        eprintln!(
            "TRACE: EIP={:#010x} BYTES=[{}] | EAX={:#010x} EBX={:#010x} ECX={:#010x} EDX={:#010x} | ESP={:#010x} EBP={:#010x}",
            self.eip,
            bytes.join(" "),
            self.eax,
            self.ebx,
            self.ecx,
            self.edx,
            self.esp,
            self.ebp,
        );
    }
    pub fn set_import_jump_table(&mut self, addr: u32, size: u32, stub_size: u32, reserved1: u32) {
        self.import_jump_table_addr = addr;
        self.import_jump_table_size = size;
        self.import_jump_table_stub_size = stub_size;
        self.import_jump_table_reserved1 = reserved1;
    }

    pub fn register_import_stub(&mut self, addr: u32, indirect_symbol_index: u32) {
        self.import_stub_indirect_map
            .insert(addr, indirect_symbol_index);
    }

    pub fn import_indirect_symbol_index(&self, eip: u32) -> Option<u32> {
        if let Some(&indirect_symbol_index) = self.import_stub_indirect_map.get(&eip) {
            return Some(indirect_symbol_index);
        }
        self.import_stub_index(eip)
            .map(|stub_index| self.import_jump_table_reserved1 + stub_index)
    }

    pub fn import_stub_index(&self, eip: u32) -> Option<u32> {
        if self.import_jump_table_addr == 0
            || self.import_jump_table_size == 0
            || self.import_jump_table_stub_size == 0
        {
            return None;
        }

        let start = self.import_jump_table_addr;
        let end = start.wrapping_add(self.import_jump_table_size);

        if eip < start || eip >= end {
            return None;
        }

        let offset = eip - start;
        if offset % self.import_jump_table_stub_size != 0 {
            return None;
        }

        Some(offset / self.import_jump_table_stub_size)
    }

    pub fn is_in_import_jump_table(&self, eip: u32) -> bool {
        self.import_stub_index(eip).is_some()
    }

    pub fn reg32(&self, idx: u8) -> u32 {
        match idx & 7 {
            0 => self.eax,
            1 => self.ecx,
            2 => self.edx,
            3 => self.ebx,
            4 => self.esp,
            5 => self.ebp,
            6 => self.esi,
            7 => self.edi,
            _ => unreachable!(),
        }
    }

    pub fn reg8(&self, idx: u8) -> u8 {
        let idx = idx & 7;
        let base = idx & 3;
        let val = self.reg32(base);
        if idx < 4 {
            (val & 0xFF) as u8
        } else {
            ((val >> 8) & 0xFF) as u8
        }
    }

    pub fn set_reg32(&mut self, idx: u8, value: u32) {
        match idx & 7 {
            0 => self.eax = value,
            1 => self.ecx = value,
            2 => self.edx = value,
            3 => self.ebx = value,
            4 => self.esp = value,
            5 => self.ebp = value,
            6 => self.esi = value,
            7 => self.edi = value,
            _ => unreachable!(),
        }
    }

    pub fn set_reg8(&mut self, idx: u8, value: u8) {
        let idx = idx & 7;
        let base = idx & 3;
        let mut val32 = self.reg32(base);
        if idx < 4 {
            val32 = (val32 & 0xFFFF_FF00) | value as u32;
        } else {
            val32 = (val32 & 0xFFFF_00FF) | ((value as u32) << 8);
        }
        self.set_reg32(base, val32);
    }

    pub fn xmm_reg(&self, idx: u8) -> [u8; 16] {
        self.xmm[(idx & 7) as usize]
    }

    pub fn set_xmm_reg(&mut self, idx: u8, value: [u8; 16]) {
        self.xmm[(idx & 7) as usize] = value;
    }

    pub fn fpu_push(&mut self, value: f64) {
        if self.fpu_stack.len() >= 8 {
            self.fpu_stack.remove(0);
        }
        self.fpu_stack.push(value);
    }

    pub fn fpu_peek(&self) -> f64 {
        *self.fpu_stack.last().unwrap_or(&0.0)
    }

    pub fn fpu_pop(&mut self) -> f64 {
        self.fpu_stack.pop().unwrap_or(0.0)
    }

    pub fn read_u8(&self, mem: &GuestMemory, addr: u32) -> Result<u8, ExecError> {
        let mut buf = [0u8; 1];
        mem.read(addr, &mut buf)?;
        Ok(buf[0])
    }

    pub fn read_u16(&self, mem: &GuestMemory, addr: u32) -> Result<u16, ExecError> {
        let mut buf = [0u8; 2];
        mem.read(addr, &mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    pub fn read_u32(&self, mem: &GuestMemory, addr: u32) -> Result<u32, ExecError> {
        let mut buf = [0u8; 4];
        mem.read(addr, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn write_u8(&self, mem: &mut GuestMemory, addr: u32, value: u8) -> Result<(), ExecError> {
        mem.write(addr, &[value])?;
        Ok(())
    }

    pub fn write_u16(&self, mem: &mut GuestMemory, addr: u32, value: u16) -> Result<(), ExecError> {
        mem.write(addr, &value.to_le_bytes())?;
        Ok(())
    }

    pub fn write_u32(&self, mem: &mut GuestMemory, addr: u32, value: u32) -> Result<(), ExecError> {
        mem.write(addr, &value.to_le_bytes())?;
        Ok(())
    }

    pub fn push32(&mut self, mem: &mut GuestMemory, value: u32) -> Result<(), ExecError> {
        self.esp = self.esp.wrapping_sub(4);
        self.write_u32(mem, self.esp, value)
    }

    pub fn pop32(&mut self, mem: &GuestMemory) -> Result<u32, ExecError> {
        let value = self.read_u32(mem, self.esp)?;
        self.esp = self.esp.wrapping_add(4);
        Ok(value)
    }

    pub fn read_operand32(&self, mem: &GuestMemory, operand: Operand32) -> Result<u32, ExecError> {
        match operand {
            Operand32::Reg(r) => Ok(self.reg32(r)),
            Operand32::Mem(addr) => self.read_u32(mem, addr),
        }
    }

    pub fn read_operand8(&self, mem: &GuestMemory, operand: Operand32) -> Result<u8, ExecError> {
        match operand {
            Operand32::Reg(r) => Ok(self.reg8(r)),
            Operand32::Mem(addr) => self.read_u8(mem, addr),
        }
    }

    pub fn write_operand32(
        &mut self,
        mem: &mut GuestMemory,
        operand: Operand32,
        value: u32,
    ) -> Result<(), ExecError> {
        match operand {
            Operand32::Reg(r) => {
                self.set_reg32(r, value);
                Ok(())
            }
            Operand32::Mem(addr) => self.write_u32(mem, addr, value),
        }
    }

    pub fn write_operand8(
        &mut self,
        mem: &mut GuestMemory,
        operand: Operand32,
        value: u8,
    ) -> Result<(), ExecError> {
        match operand {
            Operand32::Reg(r) => {
                self.set_reg8(r, value);
                Ok(())
            }
            Operand32::Mem(addr) => self.write_u8(mem, addr, value),
        }
    }

    pub fn decode_rm32_operand(
        &mut self,
        mem: &GuestMemory,
        stream: &mut InstructionStream,
        modrm: ModRm,
    ) -> Result<Operand32, ExecError> {
        if modrm.mod_ == 0b11 {
            Ok(Operand32::Reg(modrm.rm))
        } else {
            let addr = self.calc_effective_addr(mem, stream, modrm)?;
            Ok(Operand32::Mem(addr))
        }
    }

    pub fn calc_effective_addr(
        &self,
        _mem: &GuestMemory,
        s: &mut InstructionStream,
        modrm: ModRm,
    ) -> Result<u32, ExecError> {
        // 如果 mod == 11，它是寄存器寻址，不应该调用这个函数计算内存地址
        if modrm.mod_ == 0b11 {
            return Err(ExecError::InvalidInstruction(
                s.ip,
                "calc_effective_addr called with mod == 11",
            ));
        }

        let mut has_sib = false;
        let mut sib = 0u8;

        // 核心修复：当 rm == 4 时，必定跟随着一个 SIB 字节
        if modrm.rm == 4 {
            has_sib = true;
            sib = s.fetch_u8()?;
        }

        // 读取位移 (Displacement)
        let mut disp: u32 = 0;
        if modrm.mod_ == 0b01 {
            disp = s.fetch_i8()? as i32 as u32; // 8位符号扩展
        } else if modrm.mod_ == 0b10 {
            disp = s.fetch_u32()?; // 32位直接读取
        } else if modrm.mod_ == 0b00 && modrm.rm == 5 {
            // 特殊情况：mod=00 且 rm=5 (EBP) 时，没有基址寄存器，直接跟 32位 disp
            disp = s.fetch_u32()?;
        }

        let mut addr: u32 = 0;

        // 核心修复：解析 SIB 字节
        if has_sib {
            let scale = sib >> 6;
            let index = (sib >> 3) & 7;
            let base = sib & 7;

            // 处理 Index (index == 4 表示没有 index)
            if index != 4 {
                let index_val = self.reg32(index);
                addr = addr.wrapping_add(index_val << scale);
            }

            // 处理 Base
            if base == 5 && modrm.mod_ == 0b00 {
                // SIB 特殊情况：base=5 且 mod=00 时，直接读取 32位 base，不用 EBP
                let base_disp = s.fetch_u32()?;
                addr = addr.wrapping_add(base_disp);
            } else {
                addr = addr.wrapping_add(self.reg32(base));
            }
        } else {
            // 没有 SIB 的普通情况
            if modrm.mod_ == 0b00 && modrm.rm == 5 {
                // 前面已经把 disp 读出来了，这里什么都不用加
            } else {
                addr = addr.wrapping_add(self.reg32(modrm.rm));
            }
        }

        addr = addr.wrapping_add(disp);
        Ok(addr)
    }

    pub fn alu_add32(&mut self, lhs: u32, rhs: u32) -> u32 {
        let result = lhs.wrapping_add(rhs);
        Flags::set_add32(&mut self.eflags, lhs, rhs, result);
        result
    }

    pub fn alu_sub32(&mut self, lhs: u32, rhs: u32) -> u32 {
        let result = lhs.wrapping_sub(rhs);
        Flags::set_sub32(&mut self.eflags, lhs, rhs, result);
        result
    }

    pub fn alu_adc32(&mut self, lhs: u32, rhs: u32) -> u32 {
        let carry = if Flags::get(self.eflags, Flags::CF) {
            1u32
        } else {
            0u32
        };
        let sum = lhs as u64 + rhs as u64 + carry as u64;
        let result = sum as u32;
        let rhs_with_carry = rhs.wrapping_add(carry);
        let of = ((lhs ^ result) & (rhs_with_carry ^ result) & 0x8000_0000) != 0;
        Flags::set(&mut self.eflags, Flags::CF, sum > 0xFFFF_FFFF);
        Flags::set(&mut self.eflags, Flags::OF, of);
        Flags::set_zf_sf(&mut self.eflags, result);
        result
    }

    pub fn alu_sbb32(&mut self, lhs: u32, rhs: u32) -> u32 {
        let carry = if Flags::get(self.eflags, Flags::CF) {
            1u32
        } else {
            0u32
        };
        let rhs_with_carry = rhs.wrapping_add(carry);
        let result = lhs.wrapping_sub(rhs_with_carry);
        let cf = (lhs as u64) < (rhs as u64 + carry as u64);
        let of = ((lhs ^ rhs_with_carry) & (lhs ^ result) & 0x8000_0000) != 0;
        Flags::set(&mut self.eflags, Flags::CF, cf);
        Flags::set(&mut self.eflags, Flags::OF, of);
        Flags::set_zf_sf(&mut self.eflags, result);
        result
    }

    pub fn alu_xor32(&mut self, lhs: u32, rhs: u32) -> u32 {
        let result = lhs ^ rhs;
        Flags::set_logic(&mut self.eflags, result);
        result
    }

    pub fn alu_or32(&mut self, lhs: u32, rhs: u32) -> u32 {
        let result = lhs | rhs;
        Flags::set_logic(&mut self.eflags, result);
        result
    }

    pub fn alu_and32(&mut self, lhs: u32, rhs: u32) -> u32 {
        let result = lhs & rhs;
        Flags::set_logic(&mut self.eflags, result);
        result
    }

    pub fn condition_true(&self, cc: u8) -> bool {
        let cf = Flags::get(self.eflags, Flags::CF);
        let pf = Flags::get(self.eflags, Flags::PF);
        let zf = Flags::get(self.eflags, Flags::ZF);
        let sf = Flags::get(self.eflags, Flags::SF);
        let of = Flags::get(self.eflags, Flags::OF);

        match cc & 0x0F {
            0x0 => of,
            0x1 => !of,
            0x2 => cf,
            0x3 => !cf,
            0x4 => zf,
            0x5 => !zf,
            0x6 => cf || zf,
            0x7 => !cf && !zf,
            0x8 => sf,
            0x9 => !sf,
            0xA => pf,
            0xB => !pf,
            0xC => sf != of,
            0xD => sf == of,
            0xE => zf || (sf != of),
            0xF => !zf && (sf == of),
            _ => unreachable!(),
        }
    }
}
