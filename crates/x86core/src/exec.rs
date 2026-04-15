use crate::cpu::Cpu;
use crate::decode::InstructionStream;
use crate::flags::Flags;
use guestmem::GuestMemory;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecError {
    #[error(transparent)]
    Memory(#[from] guestmem::Error),

    #[error("unimplemented opcode {0:#04x} at {1:#010x}")]
    UnimplementedOpcode(u8, u32),

    #[error("invalid instruction at {0:#010x}: {1}")]
    InvalidInstruction(u32, &'static str),

    #[error("execution halted")]
    Halt,

    #[error("syscall requested")]
    Syscall,

    #[error(
        "unresolved import stub at {eip:#010x} (stub_index={stub_index}, indirect_symbol_index={indirect_symbol_index})"
    )]
    UnresolvedImportStub {
        eip: u32,
        stub_index: u32,
        indirect_symbol_index: u32,
    },
}

impl Cpu {
    pub fn step(&mut self, mem: &mut GuestMemory) -> Result<(), ExecError> {
        if self.trace {
            // 打印关键寄存器状态
            self.trace_state(mem);
        }
        self.instr_counter = self.instr_counter.wrapping_add(1);
        let start = self.eip;
        let mut s = InstructionStream::new(mem, self.eip);

        let mut opcode = s.fetch_u8()?;
        let mut is_16bit = false;
        let mut prefix_66 = false;
        let mut prefix_f2 = false;
        let mut prefix_f3 = false;

        // 1. 过滤掉指令前缀 (比如 0x66)
        loop {
            match opcode {
                0x66 => {
                    prefix_66 = true;
                    is_16bit = true;
                    opcode = s.fetch_u8()?;
                }
                // CS, DS, ES, SS 段覆盖前缀（在平坦内存模型中可以直接忽略）
                0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 => {
                    opcode = s.fetch_u8()?;
                }
                // 地址大小前缀：当前解释器仅实现 32 位地址模式，先忽略
                0x67 => {
                    opcode = s.fetch_u8()?;
                }
                // F2, F3 是 REP 前缀，有时也用于对齐或分支预测提示
                0xF2 => {
                    prefix_f2 = true;
                    opcode = s.fetch_u8()?;
                }
                0xF3 => {
                    prefix_f3 = true;
                    opcode = s.fetch_u8()?;
                }
                // LOCK 前缀：当前单线程解释执行，原子语义可退化为普通操作
                0xF0 => {
                    opcode = s.fetch_u8()?;
                }
                _ => break, // 遇到真正的操作码，跳出循环
            }
        }

        match opcode {
            0x90 => {
                // 原本的 0x90 NOP，直接放行
            }
            0x9B => {
                // FWAIT/WAIT：在无异步 x87 异常模型下可视为 NOP。
                self.eip = s.ip;
                return Ok(());
            }
            0x91..=0x97 => {
                // XCHG EAX, r32
                let reg = opcode - 0x90;
                let other = self.reg32(reg);
                self.set_reg32(reg, self.eax);
                self.eax = other;
                self.eip = s.ip;
                return Ok(());
            }
            0xF4 => {
                if let Some(indirect_symbol_index) = self.import_indirect_symbol_index(start) {
                    let stub_index = self.import_stub_index(start).unwrap_or(0);
                    return Err(ExecError::UnresolvedImportStub {
                        eip: start,
                        stub_index,
                        indirect_symbol_index,
                    });
                } else {
                    return Err(ExecError::Halt);
                }
            }
            0x00 | 0x08 | 0x20 | 0x28 | 0x30 | 0x38 => {
                // r/m8 <- r8 (ADD/OR/AND/SUB/XOR/CMP)
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand8(mem, dst)?;
                let rhs = self.reg8(modrm.reg);
                let result = match opcode {
                    0x00 => self.alu_add32(lhs as u32, rhs as u32) as u8,
                    0x08 => self.alu_or32(lhs as u32, rhs as u32) as u8,
                    0x20 => self.alu_and32(lhs as u32, rhs as u32) as u8,
                    0x28 => self.alu_sub32(lhs as u32, rhs as u32) as u8,
                    0x30 => self.alu_xor32(lhs as u32, rhs as u32) as u8,
                    0x38 => self.alu_sub32(lhs as u32, rhs as u32) as u8, // CMP
                    _ => unreachable!(),
                };
                self.eip = s.ip;
                if opcode != 0x38 {
                    self.write_operand8(mem, dst, result)?;
                }
                return Ok(());
            }
            0x02 | 0x0A | 0x22 | 0x2A | 0x32 | 0x3A => {
                // r8 <- r/m8 (ADD/OR/AND/SUB/XOR/CMP)
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.reg8(modrm.reg);
                let rhs = self.read_operand8(mem, src)?;
                let result = match opcode {
                    0x02 => self.alu_add32(lhs as u32, rhs as u32) as u8,
                    0x0A => self.alu_or32(lhs as u32, rhs as u32) as u8,
                    0x22 => self.alu_and32(lhs as u32, rhs as u32) as u8,
                    0x2A => self.alu_sub32(lhs as u32, rhs as u32) as u8,
                    0x32 => self.alu_xor32(lhs as u32, rhs as u32) as u8,
                    0x3A => self.alu_sub32(lhs as u32, rhs as u32) as u8, // CMP
                    _ => unreachable!(),
                };
                self.eip = s.ip;
                if opcode != 0x3A {
                    self.set_reg8(modrm.reg, result);
                }
                return Ok(());
            }
            0x04 | 0x0C | 0x24 | 0x2C | 0x34 | 0x3C => {
                // AL, imm8 (ADD/OR/AND/SUB/XOR/CMP)
                let imm = s.fetch_u8()?;
                let lhs = self.reg8(0);
                let result = match opcode {
                    0x04 => self.alu_add32(lhs as u32, imm as u32) as u8,
                    0x0C => self.alu_or32(lhs as u32, imm as u32) as u8,
                    0x24 => self.alu_and32(lhs as u32, imm as u32) as u8,
                    0x2C => self.alu_sub32(lhs as u32, imm as u32) as u8,
                    0x34 => self.alu_xor32(lhs as u32, imm as u32) as u8,
                    0x3C => self.alu_sub32(lhs as u32, imm as u32) as u8, // CMP
                    _ => unreachable!(),
                };
                self.eip = s.ip;
                if opcode != 0x3C {
                    self.set_reg8(0, result);
                }
                return Ok(());
            }
            0x05 | 0x0D | 0x25 | 0x2D | 0x35 | 0x3D => {
                // EAX/AX, imm (ADD/OR/AND/SUB/XOR/CMP)
                let imm = if is_16bit {
                    s.fetch_u16()? as u32
                } else {
                    s.fetch_u32()?
                };
                let lhs = if is_16bit {
                    self.eax & 0xFFFF
                } else {
                    self.eax
                };
                let result = match opcode {
                    0x05 => self.alu_add32(lhs, imm),
                    0x0D => self.alu_or32(lhs, imm),
                    0x25 => self.alu_and32(lhs, imm),
                    0x2D => self.alu_sub32(lhs, imm),
                    0x35 => self.alu_xor32(lhs, imm),
                    0x3D => self.alu_sub32(lhs, imm), // CMP
                    _ => unreachable!(),
                };
                self.eip = s.ip;
                if opcode != 0x3D {
                    if is_16bit {
                        self.eax = (self.eax & 0xFFFF_0000) | (result & 0xFFFF);
                    } else {
                        self.eax = result;
                    }
                }
                return Ok(());
            }
            0x10 | 0x18 => {
                // ADC/SBB r/m8, r8
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand8(mem, dst)?;
                let rhs = self.reg8(modrm.reg);
                let result = if opcode == 0x10 {
                    self.alu_adc32(lhs as u32, rhs as u32) as u8
                } else {
                    self.alu_sbb32(lhs as u32, rhs as u32) as u8
                };
                self.eip = s.ip;
                self.write_operand8(mem, dst, result)?;
                return Ok(());
            }
            0x12 | 0x1A => {
                // ADC/SBB r8, r/m8
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.reg8(modrm.reg);
                let rhs = self.read_operand8(mem, src)?;
                let result = if opcode == 0x12 {
                    self.alu_adc32(lhs as u32, rhs as u32) as u8
                } else {
                    self.alu_sbb32(lhs as u32, rhs as u32) as u8
                };
                self.eip = s.ip;
                self.set_reg8(modrm.reg, result);
                return Ok(());
            }
            0x14 | 0x1C => {
                // ADC/SBB AL, imm8
                let imm = s.fetch_u8()?;
                let lhs = self.reg8(0);
                let result = if opcode == 0x14 {
                    self.alu_adc32(lhs as u32, imm as u32) as u8
                } else {
                    self.alu_sbb32(lhs as u32, imm as u32) as u8
                };
                self.eip = s.ip;
                self.set_reg8(0, result);
                return Ok(());
            }
            0x11 | 0x19 => {
                // ADC/SBB r/m16|32, r16|32
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                self.eip = s.ip;
                if is_16bit {
                    let lhs = match dst {
                        crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)? as u32,
                        crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                    };
                    let rhs = self.reg32(modrm.reg) & 0xFFFF;
                    let result = if opcode == 0x11 {
                        self.alu_adc32(lhs, rhs)
                    } else {
                        self.alu_sbb32(lhs, rhs)
                    } & 0xFFFF;
                    match dst {
                        crate::decode::Operand32::Mem(addr) => {
                            mem.write(addr, &(result as u16).to_le_bytes())?
                        }
                        crate::decode::Operand32::Reg(r) => {
                            let old = self.reg32(r);
                            self.set_reg32(r, (old & 0xFFFF_0000) | result);
                        }
                    }
                } else {
                    let lhs = self.read_operand32(mem, dst)?;
                    let rhs = self.reg32(modrm.reg);
                    let result = if opcode == 0x11 {
                        self.alu_adc32(lhs, rhs)
                    } else {
                        self.alu_sbb32(lhs, rhs)
                    };
                    self.write_operand32(mem, dst, result)?;
                }
                return Ok(());
            }
            0x13 | 0x1B => {
                // ADC/SBB r16|32, r/m16|32
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                if is_16bit {
                    let lhs = self.reg32(modrm.reg) & 0xFFFF;
                    let rhs = match src {
                        crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)? as u32,
                        crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                    };
                    let result = if opcode == 0x13 {
                        self.alu_adc32(lhs, rhs)
                    } else {
                        self.alu_sbb32(lhs, rhs)
                    } & 0xFFFF;
                    let old = self.reg32(modrm.reg);
                    self.set_reg32(modrm.reg, (old & 0xFFFF_0000) | result);
                } else {
                    let lhs = self.reg32(modrm.reg);
                    let rhs = self.read_operand32(mem, src)?;
                    let result = if opcode == 0x13 {
                        self.alu_adc32(lhs, rhs)
                    } else {
                        self.alu_sbb32(lhs, rhs)
                    };
                    self.set_reg32(modrm.reg, result);
                }
                self.eip = s.ip;
                return Ok(());
            }
            0x15 | 0x1D => {
                // ADC/SBB AX/EAX, imm16|32
                let imm = if is_16bit {
                    s.fetch_u16()? as u32
                } else {
                    s.fetch_u32()?
                };
                self.eip = s.ip;
                if is_16bit {
                    let lhs = self.eax & 0xFFFF;
                    let result = if opcode == 0x15 {
                        self.alu_adc32(lhs, imm)
                    } else {
                        self.alu_sbb32(lhs, imm)
                    } & 0xFFFF;
                    self.eax = (self.eax & 0xFFFF_0000) | result;
                } else {
                    self.eax = if opcode == 0x15 {
                        self.alu_adc32(self.eax, imm)
                    } else {
                        self.alu_sbb32(self.eax, imm)
                    };
                }
                return Ok(());
            }

            0x68 => {
                // 致命修复：16位模式下 push 的是 2 字节立即数
                let imm = if is_16bit {
                    s.fetch_u16()? as u32
                } else {
                    s.fetch_u32()?
                };
                self.eip = s.ip;

                if is_16bit {
                    self.esp = self.esp.wrapping_sub(2);
                    mem.write(self.esp, &(imm as u16).to_le_bytes())?;
                } else {
                    self.push32(mem, imm)?;
                }
                return Ok(());
            }
            0x6A => {
                let imm = s.fetch_i8()? as i32 as u32;
                self.eip = s.ip;
                self.push32(mem, imm)?;
                return Ok(());
            }
            0x69 => {
                // IMUL r16|32, r/m16|32, imm16|32
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = if is_16bit {
                    match src {
                        crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)? as u32,
                        crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                    }
                } else {
                    self.read_operand32(mem, src)?
                };
                let imm = if is_16bit {
                    s.fetch_u16()? as u32
                } else {
                    s.fetch_u32()?
                };
                self.eip = s.ip;

                if is_16bit {
                    let res = ((lhs as i16) as i32).wrapping_mul((imm as i16) as i32) as u32;
                    let old = self.reg32(modrm.reg);
                    self.set_reg32(modrm.reg, (old & 0xFFFF_0000) | (res & 0xFFFF));
                } else {
                    let res = (lhs as i32).wrapping_mul(imm as i32) as u32;
                    self.set_reg32(modrm.reg, res);
                }
                return Ok(());
            }
            0x6B => {
                // IMUL r16|32, r/m16|32, imm8
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = if is_16bit {
                    match src {
                        crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)? as u32,
                        crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                    }
                } else {
                    self.read_operand32(mem, src)?
                };
                let imm = s.fetch_i8()? as i32 as u32;
                self.eip = s.ip;

                if is_16bit {
                    let res = ((lhs as i16) as i32).wrapping_mul((imm as i8) as i32) as u32;
                    let old = self.reg32(modrm.reg);
                    self.set_reg32(modrm.reg, (old & 0xFFFF_0000) | (res & 0xFFFF));
                } else {
                    let res = (lhs as i32).wrapping_mul(imm as i32) as u32;
                    self.set_reg32(modrm.reg, res);
                }
                return Ok(());
            }
            0x60 => {
                // PUSHAD
                let next_ip = s.ip;
                self.eip = next_ip;
                let original_esp = self.esp;
                self.push32(mem, self.eax)?;
                self.push32(mem, self.ecx)?;
                self.push32(mem, self.edx)?;
                self.push32(mem, self.ebx)?;
                self.push32(mem, original_esp)?;
                self.push32(mem, self.ebp)?;
                self.push32(mem, self.esi)?;
                self.push32(mem, self.edi)?;
                return Ok(());
            }
            0x61 => {
                // POPAD
                self.edi = self.pop32(mem)?;
                self.esi = self.pop32(mem)?;
                self.ebp = self.pop32(mem)?;
                let _ignored_esp = self.pop32(mem)?;
                self.ebx = self.pop32(mem)?;
                self.edx = self.pop32(mem)?;
                self.ecx = self.pop32(mem)?;
                self.eax = self.pop32(mem)?;
                self.eip = s.ip;
                return Ok(());
            }
            0x6C => {
                // INSB: [EDI] <- IN8(DX)
                let next_eip = s.ip;
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        let value = self.io_in8(self.edx as u16);
                        self.write_u8(mem, self.edi, value)?;
                        if Flags::get(self.eflags, Flags::DF) {
                            self.edi = self.edi.wrapping_sub(1);
                        } else {
                            self.edi = self.edi.wrapping_add(1);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    let value = self.io_in8(self.edx as u16);
                    self.write_u8(mem, self.edi, value)?;
                    if Flags::get(self.eflags, Flags::DF) {
                        self.edi = self.edi.wrapping_sub(1);
                    } else {
                        self.edi = self.edi.wrapping_add(1);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0x6D => {
                // INSW/INSD: [EDI] <- IN16/IN32(DX)
                let next_eip = s.ip;
                let size = if is_16bit { 2 } else { 4 };
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        if is_16bit {
                            let value = self.io_in16(self.edx as u16);
                            self.write_u16(mem, self.edi, value)?;
                        } else {
                            let value = self.io_in32(self.edx as u16);
                            self.write_u32(mem, self.edi, value)?;
                        }
                        if Flags::get(self.eflags, Flags::DF) {
                            self.edi = self.edi.wrapping_sub(size);
                        } else {
                            self.edi = self.edi.wrapping_add(size);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    if is_16bit {
                        let value = self.io_in16(self.edx as u16);
                        self.write_u16(mem, self.edi, value)?;
                    } else {
                        let value = self.io_in32(self.edx as u16);
                        self.write_u32(mem, self.edi, value)?;
                    }
                    if Flags::get(self.eflags, Flags::DF) {
                        self.edi = self.edi.wrapping_sub(size);
                    } else {
                        self.edi = self.edi.wrapping_add(size);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0x6E => {
                // OUTSB: OUT8(DX, [ESI])
                let next_eip = s.ip;
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        let value = self.read_u8(mem, self.esi)?;
                        self.io_out8(self.edx as u16, value);
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(1);
                        } else {
                            self.esi = self.esi.wrapping_add(1);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    let value = self.read_u8(mem, self.esi)?;
                    self.io_out8(self.edx as u16, value);
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(1);
                    } else {
                        self.esi = self.esi.wrapping_add(1);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0x6F => {
                // OUTSW/OUTSD: OUT16/OUT32(DX, [ESI])
                let next_eip = s.ip;
                let size = if is_16bit { 2 } else { 4 };
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        if is_16bit {
                            let value = self.read_u16(mem, self.esi)?;
                            self.io_out16(self.edx as u16, value);
                        } else {
                            let value = self.read_u32(mem, self.esi)?;
                            self.io_out32(self.edx as u16, value);
                        }
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(size);
                        } else {
                            self.esi = self.esi.wrapping_add(size);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    if is_16bit {
                        let value = self.read_u16(mem, self.esi)?;
                        self.io_out16(self.edx as u16, value);
                    } else {
                        let value = self.read_u32(mem, self.esi)?;
                        self.io_out32(self.edx as u16, value);
                    }
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(size);
                    } else {
                        self.esi = self.esi.wrapping_add(size);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0x50..=0x57 => {
                let reg = opcode - 0x50;
                let val = self.reg32(reg);
                self.eip = s.ip;
                self.push32(mem, val)?;
                return Ok(());
            }
            0x58..=0x5F => {
                let reg = opcode - 0x58;
                self.eip = s.ip;
                let value = self.pop32(mem)?;
                self.set_reg32(reg, value);
                return Ok(());
            }
            0x8F => {
                // POP r/m16|32
                let modrm = s.fetch_modrm()?;
                if modrm.reg != 0 {
                    return Err(ExecError::UnimplementedOpcode(opcode, start));
                }
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                self.eip = s.ip;

                if is_16bit {
                    let value = self.read_u16(mem, self.esp)? as u32;
                    self.esp = self.esp.wrapping_add(2);
                    match dst {
                        crate::decode::Operand32::Mem(addr) => {
                            mem.write(addr, &(value as u16).to_le_bytes())?
                        }
                        crate::decode::Operand32::Reg(r) => {
                            let old = self.reg32(r);
                            self.set_reg32(r, (old & 0xFFFF_0000) | (value & 0xFFFF));
                        }
                    }
                } else {
                    let value = self.pop32(mem)?;
                    self.write_operand32(mem, dst, value)?;
                }
                return Ok(());
            }

            0x40..=0x47 => {
                let reg = opcode - 0x40;
                let old = self.reg32(reg);
                let old_cf = Flags::get(self.eflags, Flags::CF);
                let new = self.alu_add32(old, 1);
                self.set_reg32(reg, new);
                Flags::set(&mut self.eflags, Flags::CF, old_cf);
            }
            0x48..=0x4F => {
                let reg = opcode - 0x48;
                let old = self.reg32(reg);
                let old_cf = Flags::get(self.eflags, Flags::CF);
                let new = self.alu_sub32(old, 1);
                self.set_reg32(reg, new);
                Flags::set(&mut self.eflags, Flags::CF, old_cf);
            }

            0xB8..=0xBF => {
                let reg = opcode - 0xB8;
                // 致命修复：如果是 16 位模式，只能读 2 字节！否则 EIP 会错位！
                let imm = if is_16bit {
                    s.fetch_u16()? as u32
                } else {
                    s.fetch_u32()?
                };
                self.eip = s.ip;

                if is_16bit {
                    let old = self.reg32(reg);
                    self.set_reg32(reg, (old & 0xFFFF0000) | (imm & 0xFFFF));
                } else {
                    self.set_reg32(reg, imm);
                }
                return Ok(());
            }
            0xB0..=0xB7 => {
                // MOV r8, imm8
                let reg = opcode - 0xB0;
                let imm = s.fetch_u8()?;
                self.eip = s.ip;
                self.set_reg8(reg, imm);
                return Ok(());
            }
            0x89 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let src = self.reg32(modrm.reg);
                self.eip = s.ip;

                if is_16bit {
                    match &dst {
                        crate::decode::Operand32::Mem(addr) => {
                            mem.write(*addr, &(src as u16).to_le_bytes())?
                        }
                        crate::decode::Operand32::Reg(r) => {
                            let old = self.reg32(*r);
                            self.set_reg32(*r, (old & 0xFFFF0000) | (src & 0xFFFF));
                        }
                    }
                } else {
                    self.write_operand32(mem, dst, src)?;
                }
                return Ok(());
            }
            0x8B => {
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;

                let value = if is_16bit {
                    match src {
                        crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)? as u32,
                        crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                    }
                } else {
                    self.read_operand32(mem, src)?
                };

                if is_16bit {
                    let old = self.reg32(modrm.reg);
                    self.set_reg32(modrm.reg, (old & 0xFFFF0000) | (value & 0xFFFF));
                } else {
                    self.set_reg32(modrm.reg, value);
                }
            }
            0x87 => {
                // XCHG r/m16|32, r16|32
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                self.eip = s.ip;

                if is_16bit {
                    let lhs16 = match dst {
                        crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)? as u32,
                        crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                    };
                    let rhs16 = self.reg32(modrm.reg) & 0xFFFF;

                    match dst {
                        crate::decode::Operand32::Mem(addr) => {
                            mem.write(addr, &(rhs16 as u16).to_le_bytes())?
                        }
                        crate::decode::Operand32::Reg(r) => {
                            let old = self.reg32(r);
                            self.set_reg32(r, (old & 0xFFFF_0000) | rhs16);
                        }
                    }
                    let old_reg = self.reg32(modrm.reg);
                    self.set_reg32(modrm.reg, (old_reg & 0xFFFF_0000) | lhs16);
                } else {
                    let lhs = self.read_operand32(mem, dst)?;
                    let rhs = self.reg32(modrm.reg);
                    self.write_operand32(mem, dst, rhs)?;
                    self.set_reg32(modrm.reg, lhs);
                }
                return Ok(());
            }
            0x8D => {
                // LEA
                let modrm = s.fetch_modrm()?;
                if modrm.mod_ == 0b11 {
                    return Err(ExecError::InvalidInstruction(
                        start,
                        "lea requires memory operand",
                    ));
                }
                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                self.eip = s.ip; // 必须加上这行！
                self.set_reg32(modrm.reg, addr);
                return Ok(()); // 必须提前返回！
            }
            0xA1 => {
                let addr = s.fetch_u32()?;
                self.eax = self.read_u32(mem, addr)?;
            }
            0xA0 => {
                // MOV AL, moffs8
                let addr = s.fetch_u32()?;
                self.eip = s.ip;
                let value = self.read_u8(mem, addr)?;
                self.set_reg8(0, value);
                return Ok(());
            }
            0xA3 => {
                let addr = s.fetch_u32()?;
                self.eip = s.ip;
                self.write_u32(mem, addr, self.eax)?;
                return Ok(());
            }
            0xA2 => {
                // MOV moffs8, AL
                let addr = s.fetch_u32()?;
                self.eip = s.ip;
                self.write_u8(mem, addr, self.reg8(0))?;
                return Ok(());
            }
            0xA8 => {
                // TEST AL, imm8
                let imm = s.fetch_u8()?;
                self.eip = s.ip;
                let _ = self.alu_and32(self.reg8(0) as u32, imm as u32);
                return Ok(());
            }
            0xA9 => {
                // TEST AX/EAX, imm
                let imm = if is_16bit {
                    s.fetch_u16()? as u32
                } else {
                    s.fetch_u32()?
                };
                self.eip = s.ip;
                let lhs = if is_16bit {
                    self.eax & 0xFFFF
                } else {
                    self.eax
                };
                let _ = self.alu_and32(lhs, imm);
                return Ok(());
            }
            0xA4 => {
                // MOVSB: [EDI] <- [ESI]
                let next_eip = s.ip;
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        let value = self.read_u8(mem, self.esi)?;
                        self.write_u8(mem, self.edi, value)?;
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(1);
                            self.edi = self.edi.wrapping_sub(1);
                        } else {
                            self.esi = self.esi.wrapping_add(1);
                            self.edi = self.edi.wrapping_add(1);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    let value = self.read_u8(mem, self.esi)?;
                    self.write_u8(mem, self.edi, value)?;
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(1);
                        self.edi = self.edi.wrapping_sub(1);
                    } else {
                        self.esi = self.esi.wrapping_add(1);
                        self.edi = self.edi.wrapping_add(1);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xA5 => {
                // MOVSW/MOVSD: [EDI] <- [ESI]
                let next_eip = s.ip;
                let size = if is_16bit { 2 } else { 4 };
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        if is_16bit {
                            let value = self.read_u16(mem, self.esi)?;
                            self.write_u16(mem, self.edi, value)?;
                        } else {
                            let value = self.read_u32(mem, self.esi)?;
                            self.write_u32(mem, self.edi, value)?;
                        }
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(size);
                            self.edi = self.edi.wrapping_sub(size);
                        } else {
                            self.esi = self.esi.wrapping_add(size);
                            self.edi = self.edi.wrapping_add(size);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    if is_16bit {
                        let value = self.read_u16(mem, self.esi)?;
                        self.write_u16(mem, self.edi, value)?;
                    } else {
                        let value = self.read_u32(mem, self.esi)?;
                        self.write_u32(mem, self.edi, value)?;
                    }
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(size);
                        self.edi = self.edi.wrapping_sub(size);
                    } else {
                        self.esi = self.esi.wrapping_add(size);
                        self.edi = self.edi.wrapping_add(size);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xA6 => {
                // CMPSB: 比较 [ESI] 与 [EDI]
                let next_eip = s.ip;
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        let lhs = self.read_u8(mem, self.esi)?;
                        let rhs = self.read_u8(mem, self.edi)?;
                        let _ = self.alu_sub8(lhs, rhs);
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(1);
                            self.edi = self.edi.wrapping_sub(1);
                        } else {
                            self.esi = self.esi.wrapping_add(1);
                            self.edi = self.edi.wrapping_add(1);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                        if prefix_f2 {
                            if Flags::get(self.eflags, Flags::ZF) {
                                break;
                            }
                        } else if prefix_f3 && !Flags::get(self.eflags, Flags::ZF) {
                            break;
                        }
                    }
                } else {
                    let lhs = self.read_u8(mem, self.esi)?;
                    let rhs = self.read_u8(mem, self.edi)?;
                    let _ = self.alu_sub8(lhs, rhs);
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(1);
                        self.edi = self.edi.wrapping_sub(1);
                    } else {
                        self.esi = self.esi.wrapping_add(1);
                        self.edi = self.edi.wrapping_add(1);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xA7 => {
                // CMPSW/CMPSD: 比较 [ESI] 与 [EDI]
                let next_eip = s.ip;
                let size = if is_16bit { 2 } else { 4 };
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        if is_16bit {
                            let lhs = self.read_u16(mem, self.esi)?;
                            let rhs = self.read_u16(mem, self.edi)?;
                            let _ = self.alu_sub16(lhs, rhs);
                        } else {
                            let lhs = self.read_u32(mem, self.esi)?;
                            let rhs = self.read_u32(mem, self.edi)?;
                            let _ = self.alu_sub32(lhs, rhs);
                        }
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(size);
                            self.edi = self.edi.wrapping_sub(size);
                        } else {
                            self.esi = self.esi.wrapping_add(size);
                            self.edi = self.edi.wrapping_add(size);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                        if prefix_f2 {
                            if Flags::get(self.eflags, Flags::ZF) {
                                break;
                            }
                        } else if prefix_f3 && !Flags::get(self.eflags, Flags::ZF) {
                            break;
                        }
                    }
                } else {
                    if is_16bit {
                        let lhs = self.read_u16(mem, self.esi)?;
                        let rhs = self.read_u16(mem, self.edi)?;
                        let _ = self.alu_sub16(lhs, rhs);
                    } else {
                        let lhs = self.read_u32(mem, self.esi)?;
                        let rhs = self.read_u32(mem, self.edi)?;
                        let _ = self.alu_sub32(lhs, rhs);
                    }
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(size);
                        self.edi = self.edi.wrapping_sub(size);
                    } else {
                        self.esi = self.esi.wrapping_add(size);
                        self.edi = self.edi.wrapping_add(size);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xAA => {
                // STOSB: [EDI] <- AL
                let next_eip = s.ip;
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        self.write_u8(mem, self.edi, self.reg8(0))?;
                        if Flags::get(self.eflags, Flags::DF) {
                            self.edi = self.edi.wrapping_sub(1);
                        } else {
                            self.edi = self.edi.wrapping_add(1);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    self.write_u8(mem, self.edi, self.reg8(0))?;
                    if Flags::get(self.eflags, Flags::DF) {
                        self.edi = self.edi.wrapping_sub(1);
                    } else {
                        self.edi = self.edi.wrapping_add(1);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xAB => {
                // STOSW/STOSD: [EDI] <- AX/EAX
                let next_eip = s.ip;
                let size = if is_16bit { 2 } else { 4 };
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        if is_16bit {
                            self.write_u16(mem, self.edi, (self.eax & 0xFFFF) as u16)?;
                        } else {
                            self.write_u32(mem, self.edi, self.eax)?;
                        }
                        if Flags::get(self.eflags, Flags::DF) {
                            self.edi = self.edi.wrapping_sub(size);
                        } else {
                            self.edi = self.edi.wrapping_add(size);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    if is_16bit {
                        self.write_u16(mem, self.edi, (self.eax & 0xFFFF) as u16)?;
                    } else {
                        self.write_u32(mem, self.edi, self.eax)?;
                    }
                    if Flags::get(self.eflags, Flags::DF) {
                        self.edi = self.edi.wrapping_sub(size);
                    } else {
                        self.edi = self.edi.wrapping_add(size);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xAC => {
                // LODSB: AL <- [ESI]
                let next_eip = s.ip;
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        let value = self.read_u8(mem, self.esi)?;
                        self.set_reg8(0, value);
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(1);
                        } else {
                            self.esi = self.esi.wrapping_add(1);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    let value = self.read_u8(mem, self.esi)?;
                    self.set_reg8(0, value);
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(1);
                    } else {
                        self.esi = self.esi.wrapping_add(1);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xAD => {
                // LODSW/LODSD: AX/EAX <- [ESI]
                let next_eip = s.ip;
                let size = if is_16bit { 2 } else { 4 };
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        if is_16bit {
                            let value = self.read_u16(mem, self.esi)? as u32;
                            self.eax = (self.eax & 0xFFFF_0000) | value;
                        } else {
                            let value = self.read_u32(mem, self.esi)?;
                            self.eax = value;
                        }
                        if Flags::get(self.eflags, Flags::DF) {
                            self.esi = self.esi.wrapping_sub(size);
                        } else {
                            self.esi = self.esi.wrapping_add(size);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                    }
                } else {
                    if is_16bit {
                        let value = self.read_u16(mem, self.esi)? as u32;
                        self.eax = (self.eax & 0xFFFF_0000) | value;
                    } else {
                        let value = self.read_u32(mem, self.esi)?;
                        self.eax = value;
                    }
                    if Flags::get(self.eflags, Flags::DF) {
                        self.esi = self.esi.wrapping_sub(size);
                    } else {
                        self.esi = self.esi.wrapping_add(size);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xAE => {
                // SCASB: 比较 AL 与 [EDI]
                let next_eip = s.ip;
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        let rhs = self.read_u8(mem, self.edi)?;
                        let _ = self.alu_sub8(self.reg8(0), rhs);
                        if Flags::get(self.eflags, Flags::DF) {
                            self.edi = self.edi.wrapping_sub(1);
                        } else {
                            self.edi = self.edi.wrapping_add(1);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                        if prefix_f2 {
                            if Flags::get(self.eflags, Flags::ZF) {
                                break;
                            }
                        } else if prefix_f3 && !Flags::get(self.eflags, Flags::ZF) {
                            break;
                        }
                    }
                } else {
                    let rhs = self.read_u8(mem, self.edi)?;
                    let _ = self.alu_sub8(self.reg8(0), rhs);
                    if Flags::get(self.eflags, Flags::DF) {
                        self.edi = self.edi.wrapping_sub(1);
                    } else {
                        self.edi = self.edi.wrapping_add(1);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xAF => {
                // SCASW/SCASD: 比较 AX/EAX 与 [EDI]
                let next_eip = s.ip;
                let size = if is_16bit { 2 } else { 4 };
                let repeat = prefix_f2 || prefix_f3;
                if repeat {
                    while self.ecx != 0 {
                        if is_16bit {
                            let rhs = self.read_u16(mem, self.edi)?;
                            let lhs = (self.eax & 0xFFFF) as u16;
                            let _ = self.alu_sub16(lhs, rhs);
                        } else {
                            let rhs = self.read_u32(mem, self.edi)?;
                            let _ = self.alu_sub32(self.eax, rhs);
                        }
                        if Flags::get(self.eflags, Flags::DF) {
                            self.edi = self.edi.wrapping_sub(size);
                        } else {
                            self.edi = self.edi.wrapping_add(size);
                        }
                        self.ecx = self.ecx.wrapping_sub(1);
                        if prefix_f2 {
                            if Flags::get(self.eflags, Flags::ZF) {
                                break;
                            }
                        } else if prefix_f3 && !Flags::get(self.eflags, Flags::ZF) {
                            break;
                        }
                    }
                } else {
                    if is_16bit {
                        let rhs = self.read_u16(mem, self.edi)?;
                        let lhs = (self.eax & 0xFFFF) as u16;
                        let _ = self.alu_sub16(lhs, rhs);
                    } else {
                        let rhs = self.read_u32(mem, self.edi)?;
                        let _ = self.alu_sub32(self.eax, rhs);
                    }
                    if Flags::get(self.eflags, Flags::DF) {
                        self.edi = self.edi.wrapping_sub(size);
                    } else {
                        self.edi = self.edi.wrapping_add(size);
                    }
                }
                self.eip = next_eip;
                return Ok(());
            }
            0xC6 => {
                // MOV r/m8, imm8
                let modrm = s.fetch_modrm()?;
                // 对于 MOV r/m8, imm8，扩展的 reg 字段必须是 0
                if modrm.reg != 0 {
                    return Err(ExecError::UnimplementedOpcode(opcode, start));
                }

                // 寻址方式和 32 位一模一样，复用 decode_rm32_operand
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let imm = s.fetch_u8()?; // 获取 8 位立即数 '\n'

                self.eip = s.ip;

                match dst {
                    crate::decode::Operand32::Mem(addr) => {
                        // 如果是内存操作，直接调 guestmem 写 1 个字节
                        mem.write(addr, &[imm])?;
                    }
                    crate::decode::Operand32::Reg(r) => {
                        // 如果是 8 位寄存器 (AL, CL, DL, BL, AH, CH, DH, BH)
                        // r: 0-3 是低 8 位，4-7 是高 8 位
                        let base_reg = r % 4;
                        let mut val32 = self.reg32(base_reg);
                        if r < 4 {
                            val32 = (val32 & 0xFFFFFF00) | (imm as u32); // 替换 AL/CL/DL/BL
                        } else {
                            val32 = (val32 & 0xFFFF00FF) | ((imm as u32) << 8); // 替换 AH/CH/DH/BH
                        }
                        self.set_reg32(base_reg, val32);
                    }
                }
                return Ok(());
            }
            0xC7 => {
                let modrm = s.fetch_modrm()?;
                if modrm.reg != 0 {
                    return Err(ExecError::UnimplementedOpcode(opcode, start));
                }
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                // 16位前缀时，只读取 2 字节的立即数
                let imm = if is_16bit {
                    s.fetch_u16()? as u32
                } else {
                    s.fetch_u32()?
                };
                self.eip = s.ip;

                if is_16bit {
                    match &dst {
                        crate::decode::Operand32::Mem(addr) => {
                            mem.write(*addr, &(imm as u16).to_le_bytes())?
                        }
                        crate::decode::Operand32::Reg(r) => {
                            let old = self.reg32(*r);
                            self.set_reg32(*r, (old & 0xFFFF0000) | (imm & 0xFFFF));
                        }
                    }
                } else {
                    self.write_operand32(mem, dst, imm)?;
                }
                return Ok(());
            }

            0x01 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand32(mem, dst)?;
                let rhs = self.reg32(modrm.reg);
                let result = self.alu_add32(lhs, rhs);
                self.eip = s.ip;
                self.write_operand32(mem, dst, result)?;
                return Ok(());
            }
            0x03 => {
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.reg32(modrm.reg);
                let rhs = self.read_operand32(mem, src)?;
                let result = self.alu_add32(lhs, rhs);
                self.set_reg32(modrm.reg, result);
            }
            0x29 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand32(mem, dst)?;
                let rhs = self.reg32(modrm.reg);
                let result = self.alu_sub32(lhs, rhs);
                self.eip = s.ip;
                self.write_operand32(mem, dst, result)?;
                return Ok(());
            }
            0x2B => {
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.reg32(modrm.reg);
                let rhs = self.read_operand32(mem, src)?;
                let result = self.alu_sub32(lhs, rhs);
                self.set_reg32(modrm.reg, result);
            }
            0x09 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand32(mem, dst)?;
                let rhs = self.reg32(modrm.reg);
                let result = self.alu_or32(lhs, rhs);
                self.eip = s.ip;
                self.write_operand32(mem, dst, result)?;
                return Ok(());
            }
            0x0B => {
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.reg32(modrm.reg);
                let rhs = self.read_operand32(mem, src)?;
                let result = self.alu_or32(lhs, rhs);
                self.set_reg32(modrm.reg, result);
            }
            0x21 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand32(mem, dst)?;
                let rhs = self.reg32(modrm.reg);
                let result = self.alu_and32(lhs, rhs);
                self.eip = s.ip;
                self.write_operand32(mem, dst, result)?;
                return Ok(());
            }
            0x23 => {
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.reg32(modrm.reg);
                let rhs = self.read_operand32(mem, src)?;
                let result = self.alu_and32(lhs, rhs);
                self.set_reg32(modrm.reg, result);
            }
            0x31 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand32(mem, dst)?;
                let rhs = self.reg32(modrm.reg);
                let result = self.alu_xor32(lhs, rhs);
                self.eip = s.ip;
                self.write_operand32(mem, dst, result)?;
                return Ok(());
            }
            0x33 => {
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.reg32(modrm.reg);
                let rhs = self.read_operand32(mem, src)?;
                let result = self.alu_xor32(lhs, rhs);
                self.set_reg32(modrm.reg, result);
            }

            // ============================================
            // 以下三个修复了 *self 的嵌套借用冲突
            // ============================================
            0x39 => {
                let modrm = s.fetch_modrm()?;
                let op = self.decode_rm32_operand(mem, &mut s, modrm)?; // 拆分1
                let lhs = self.read_operand32(mem, op)?; // 拆分2
                let rhs = self.reg32(modrm.reg);
                let _ = self.alu_sub32(lhs, rhs);
            }
            0x3B => {
                let modrm = s.fetch_modrm()?;
                let lhs = self.reg32(modrm.reg);
                let op = self.decode_rm32_operand(mem, &mut s, modrm)?; // 拆分1
                let rhs = self.read_operand32(mem, op)?; // 拆分2
                let _ = self.alu_sub32(lhs, rhs);
            }

            0x80 | 0x82 => {
                // Group 1: 8位 ALU 运算 (ADD, OR, AND, SUB, XOR, CMP)
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let imm = s.fetch_u8()?;

                self.eip = s.ip; // 提前释放 mem 的不可变借用

                // 1. 读取 8 位左操作数
                let lhs = match &dst {
                    crate::decode::Operand32::Mem(addr) => self.read_u8(mem, *addr)?,
                    crate::decode::Operand32::Reg(r) => {
                        let base = r % 4;
                        if *r < 4 {
                            (self.reg32(base) & 0xFF) as u8
                        } else {
                            ((self.reg32(base) >> 8) & 0xFF) as u8
                        }
                    }
                };

                // 2. 计算 (对于字符转换，直接复用 32位 ALU 跑一下即可)
                let res32 = match modrm.reg {
                    0 => self.alu_add32(lhs as u32, imm as u32),
                    1 => self.alu_or32(lhs as u32, imm as u32),
                    4 => self.alu_and32(lhs as u32, imm as u32),
                    5 => self.alu_sub32(lhs as u32, imm as u32),
                    6 => self.alu_xor32(lhs as u32, imm as u32),
                    7 => self.alu_sub32(lhs as u32, imm as u32), // CMP (只算标志位)
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                };
                let res8 = res32 as u8;

                // 3. 写回 8 位结果 (如果是 CMP 则不需要写回)
                if modrm.reg != 7 {
                    match &dst {
                        crate::decode::Operand32::Mem(addr) => {
                            mem.write(*addr, &[res8])?;
                        }
                        crate::decode::Operand32::Reg(r) => {
                            let base = r % 4;
                            let mut val32 = self.reg32(base);
                            if *r < 4 {
                                val32 = (val32 & 0xFFFFFF00) | (res8 as u32);
                            } else {
                                val32 = (val32 & 0xFFFF00FF) | ((res8 as u32) << 8);
                            }
                            self.set_reg32(base, val32);
                        }
                    }
                }
                return Ok(());
            }
            0xFE => {
                // Group 4: INC/DEC r/m8
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand8(mem, dst)? as u32;
                let old_cf = Flags::get(self.eflags, Flags::CF);
                let result = match modrm.reg {
                    0 => self.alu_add32(lhs, 1) as u8,
                    1 => self.alu_sub32(lhs, 1) as u8,
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                };
                Flags::set(&mut self.eflags, Flags::CF, old_cf);
                self.eip = s.ip;
                self.write_operand8(mem, dst, result)?;
                return Ok(());
            }
            0x88 => {
                // MOV r/m8, r8 (写 8 位寄存器到内存/寄存器)
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                // src 必定是 8 位寄存器
                let val8 = self.reg8(modrm.reg);

                self.eip = s.ip;
                self.write_operand8(mem, dst, val8)?;
                return Ok(());
            }
            0x86 => {
                // XCHG r/m8, r8
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand8(mem, dst)?;
                let rhs = self.reg8(modrm.reg);
                self.eip = s.ip;
                self.write_operand8(mem, dst, rhs)?;
                self.set_reg8(modrm.reg, lhs);
                return Ok(());
            }

            0x8A => {
                // MOV r8, r/m8 (从内存/寄存器读到 8 位寄存器)
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;

                self.eip = s.ip;
                let val8 = self.read_operand8(mem, src)?;
                self.set_reg8(modrm.reg, val8);
                return Ok(());
            }
            0x84 => {
                // TEST r/m8, r8
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let lhs = self.read_operand8(mem, src)? as u32;
                let rhs = self.reg8(modrm.reg) as u32;
                self.eip = s.ip;
                let _ = self.alu_and32(lhs, rhs);
                return Ok(());
            }

            0x85 => {
                let modrm = s.fetch_modrm()?;
                let op = self.decode_rm32_operand(mem, &mut s, modrm)?; // 拆分1
                let lhs = self.read_operand32(mem, op)?; // 拆分2
                let rhs = self.reg32(modrm.reg);
                let _ = self.alu_and32(lhs, rhs);
            }

            0x81 | 0x83 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                let lhs = if is_16bit {
                    match &dst {
                        crate::decode::Operand32::Mem(addr) => self.read_u16(mem, *addr)? as u32,
                        crate::decode::Operand32::Reg(r) => self.reg32(*r) & 0xFFFF,
                    }
                } else {
                    self.read_operand32(mem, dst.clone())?
                };

                let imm = if opcode == 0x81 {
                    if is_16bit {
                        s.fetch_u16()? as u32
                    } else {
                        s.fetch_u32()?
                    }
                } else {
                    s.fetch_i8()? as i32 as u32
                };

                let result = match modrm.reg {
                    0 => self.alu_add32(lhs, imm),
                    1 => self.alu_or32(lhs, imm),
                    4 => self.alu_and32(lhs, imm),
                    5 => self.alu_sub32(lhs, imm),
                    6 => self.alu_xor32(lhs, imm),
                    7 => self.alu_sub32(lhs, imm),
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                };

                self.eip = s.ip;

                if modrm.reg != 7 {
                    // 如果不是 CMP，写回结果
                    if is_16bit {
                        match &dst {
                            crate::decode::Operand32::Mem(addr) => {
                                mem.write(*addr, &(result as u16).to_le_bytes())?
                            }
                            crate::decode::Operand32::Reg(r) => {
                                let old = self.reg32(*r);
                                self.set_reg32(*r, (old & 0xFFFF0000) | (result & 0xFFFF));
                            }
                        }
                    } else {
                        self.write_operand32(mem, dst, result)?;
                    }
                }
                return Ok(());
            }
            0x99 => {
                // CDQ (Convert Double to Quad) - 这个指令经常紧挨着 IDIV 出现，用于将 EAX 的符号位扩展到 EDX
                self.edx = if (self.eax & 0x80000000) != 0 {
                    0xFFFFFFFF
                } else {
                    0
                };
            }
            0x98 => {
                // CBW/CWDE
                if is_16bit {
                    // CBW: AL -> AX（符号扩展）
                    let al = (self.eax & 0xFF) as i8 as i16 as u16 as u32;
                    self.eax = (self.eax & 0xFFFF_0000) | al;
                } else {
                    // CWDE: AX -> EAX（符号扩展）
                    self.eax = (self.eax as u16 as i16 as i32) as u32;
                }
            }
            0x9C => {
                // PUSHFD
                self.eip = s.ip;
                self.push32(mem, self.eflags)?;
                return Ok(());
            }
            0x9D => {
                // POPFD
                self.eip = s.ip;
                self.eflags = self.pop32(mem)?;
                return Ok(());
            }
            0xF8 => {
                Flags::set(&mut self.eflags, Flags::CF, false); // CLC
                self.eip = s.ip;
                return Ok(());
            }
            0xF9 => {
                Flags::set(&mut self.eflags, Flags::CF, true); // STC
                self.eip = s.ip;
                return Ok(());
            }
            0xFA => {
                // CLI: user-mode emulation中不模拟中断屏蔽，按 NOP 处理。
                self.eip = s.ip;
                return Ok(());
            }
            0xFB => {
                // STI: user-mode emulation中不模拟中断屏蔽，按 NOP 处理。
                self.eip = s.ip;
                return Ok(());
            }
            0xFC => {
                Flags::set(&mut self.eflags, Flags::DF, false); // CLD
                self.eip = s.ip;
                return Ok(());
            }
            0xFD => {
                Flags::set(&mut self.eflags, Flags::DF, true); // STD
                self.eip = s.ip;
                return Ok(());
            }
            0xF6 => {
                // Group 3: 8-bit TEST/NOT/NEG/MUL/IMUL/DIV/IDIV
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                let imm = if modrm.reg == 0 { s.fetch_u8()? } else { 0 };
                let val = self.read_operand8(mem, dst)?;
                self.eip = s.ip;

                match modrm.reg {
                    0 => {
                        let _ = self.alu_and32(val as u32, imm as u32); // TEST
                    }
                    2 => {
                        self.write_operand8(mem, dst, !val)?; // NOT
                    }
                    3 => {
                        let result = self.alu_sub32(0, val as u32) as u8; // NEG
                        self.write_operand8(mem, dst, result)?;
                    }
                    4 => {
                        // MUL r/m8: AX = AL * r/m8
                        let al = self.reg8(0);
                        let res = (al as u16).wrapping_mul(val as u16);
                        self.set_reg8(0, (res & 0x00FF) as u8); // AL
                        self.set_reg8(4, (res >> 8) as u8); // AH
                    }
                    5 => {
                        // IMUL r/m8: AX = AL * r/m8 (signed)
                        let al = self.reg8(0) as i8 as i16;
                        let rhs = val as i8 as i16;
                        let res = al.wrapping_mul(rhs) as u16;
                        self.set_reg8(0, (res & 0x00FF) as u8);
                        self.set_reg8(4, (res >> 8) as u8);
                    }
                    6 => {
                        // DIV r/m8: AX / r/m8 => AL=quot, AH=rem
                        if val == 0 {
                            return Err(ExecError::InvalidInstruction(start, "division by zero"));
                        }
                        let dividend = ((self.reg8(4) as u16) << 8) | (self.reg8(0) as u16);
                        let quot = dividend / (val as u16);
                        let rem = dividend % (val as u16);
                        self.set_reg8(0, quot as u8);
                        self.set_reg8(4, rem as u8);
                    }
                    7 => {
                        // IDIV r/m8: AX / r/m8 => AL=quot, AH=rem (signed)
                        if val == 0 {
                            return Err(ExecError::InvalidInstruction(start, "division by zero"));
                        }
                        let dividend =
                            (((self.reg8(4) as u16) << 8) | (self.reg8(0) as u16)) as i16;
                        let divisor = val as i8 as i16;
                        if dividend == i16::MIN && divisor == -1 {
                            return Err(ExecError::InvalidInstruction(start, "division overflow"));
                        }
                        let quot = dividend / divisor;
                        let rem = dividend % divisor;
                        self.set_reg8(0, quot as u8);
                        self.set_reg8(4, rem as u8);
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                }
                return Ok(());
            }

            0xF7 => {
                // Group 3: MUL, IMUL, DIV, IDIV, NEG, NOT, TEST
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                // 致命修复：TEST 指令后面跟着立即数，16位模式下只有 2 字节！
                let imm = if modrm.reg == 0 {
                    if is_16bit {
                        s.fetch_u16()? as u32
                    } else {
                        s.fetch_u32()?
                    }
                } else {
                    0
                };

                let val = self.read_operand32(mem, dst)?;
                self.eip = s.ip; // 释放 mem 借用

                match modrm.reg {
                    0 => {
                        // TEST r/m32, imm32
                        let _ = self.alu_and32(val, imm);
                    }
                    2 => {
                        // NOT r/m32
                        let result = !val;
                        self.write_operand32(mem, dst, result)?;
                    }
                    3 => {
                        // NEG r/m32
                        let result = self.alu_sub32(0, val);
                        self.write_operand32(mem, dst, result)?;
                    }
                    4 => {
                        // MUL r/m32 (无符号乘法，结果存入 EDX:EAX)
                        let res = (self.eax as u64).wrapping_mul(val as u64);
                        self.eax = res as u32;
                        self.edx = (res >> 32) as u32;
                    }
                    5 => {
                        // IMUL r/m32 (有符号乘法)
                        let res = (self.eax as i32 as i64).wrapping_mul(val as i32 as i64);
                        self.eax = res as u32;
                        self.edx = (res >> 32) as u32;
                    }
                    6 => {
                        // DIV r/m32 (无符号除法，EDX:EAX / r/m32)
                        if val == 0 {
                            return Err(ExecError::InvalidInstruction(start, "division by zero"));
                        }
                        let dividend = ((self.edx as u64) << 32) | (self.eax as u64);
                        let quot = dividend / (val as u64);
                        let rem = dividend % (val as u64);
                        self.eax = quot as u32;
                        self.edx = rem as u32;
                    }
                    7 => {
                        // IDIV r/m32 (有符号除法)
                        if val == 0 {
                            return Err(ExecError::InvalidInstruction(start, "division by zero"));
                        }
                        let dividend = ((self.edx as u64) << 32) | (self.eax as u64);
                        let dividend_signed = dividend as i64;
                        let divisor_signed = (val as i32) as i64;

                        // 防止 INT_MIN / -1 导致的溢出崩溃
                        if dividend_signed == -9223372036854775808 && divisor_signed == -1 {
                            return Err(ExecError::InvalidInstruction(start, "division overflow"));
                        }

                        let quot = dividend_signed / divisor_signed;
                        let rem = dividend_signed % divisor_signed;
                        self.eax = quot as u32;
                        self.edx = rem as u32;
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                }
                return Ok(());
            }
            0xFF => {
                // Group 5: INC, DEC, CALL, JMP, PUSH
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                match modrm.reg {
                    0 => {
                        // INC r/m16/32 (CF unchanged)
                        let lhs = if is_16bit {
                            match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    self.read_u16(mem, addr)? as u32
                                }
                                crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                            }
                        } else {
                            self.read_operand32(mem, dst)?
                        };
                        let old_cf = Flags::get(self.eflags, Flags::CF);
                        let result = self.alu_add32(lhs, 1);
                        Flags::set(&mut self.eflags, Flags::CF, old_cf);
                        self.eip = s.ip;

                        if is_16bit {
                            match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    mem.write(addr, &(result as u16).to_le_bytes())?
                                }
                                crate::decode::Operand32::Reg(r) => {
                                    let old = self.reg32(r);
                                    self.set_reg32(r, (old & 0xFFFF_0000) | (result & 0xFFFF));
                                }
                            }
                        } else {
                            self.write_operand32(mem, dst, result)?;
                        }
                        return Ok(());
                    }
                    1 => {
                        // DEC r/m16/32 (CF unchanged)
                        let lhs = if is_16bit {
                            match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    self.read_u16(mem, addr)? as u32
                                }
                                crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                            }
                        } else {
                            self.read_operand32(mem, dst)?
                        };
                        let old_cf = Flags::get(self.eflags, Flags::CF);
                        let result = self.alu_sub32(lhs, 1);
                        Flags::set(&mut self.eflags, Flags::CF, old_cf);
                        self.eip = s.ip;

                        if is_16bit {
                            match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    mem.write(addr, &(result as u16).to_le_bytes())?
                                }
                                crate::decode::Operand32::Reg(r) => {
                                    let old = self.reg32(r);
                                    self.set_reg32(r, (old & 0xFFFF_0000) | (result & 0xFFFF));
                                }
                            }
                        } else {
                            self.write_operand32(mem, dst, result)?;
                        }
                        return Ok(());
                    }
                    2 => {
                        // CALL r/m16/32 (near absolute)
                        let slot_addr = match dst {
                            crate::decode::Operand32::Mem(addr) => Some(addr),
                            crate::decode::Operand32::Reg(_) => None,
                        };
                        let target = if is_16bit {
                            match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    self.read_u16(mem, addr)? as u32
                                }
                                crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                            }
                        } else {
                            self.read_operand32(mem, dst)?
                        };
                        let return_eip = s.ip;
                        // 兼容兜底：部分半实现运行时路径会出现空回调指针。
                        // 若直接 CALL 0 会把控制流带进 null page，后续只会执行垃圾字节。
                        // 这里把“空目标 call”退化为 no-op（等价于立刻返回）。
                        let self_ref_slot = slot_addr.map(|slot| target == slot).unwrap_or(false);
                        let data_like_target = self
                            .read_u32(mem, target)
                            .map(|v| v == target)
                            .unwrap_or(false);
                        // 运行时 heap 区本质是数据区；跳进去执行通常意味着函数指针坏掉。
                        let heap_like_target = (0x6500_0000..0x6f00_0000).contains(&target);
                        if target == 0 || self_ref_slot || data_like_target || heap_like_target {
                            if self.trace {
                                eprintln!(
                                    "[EXEC] skipped invalid indirect call at {:#010x}: slot={:#010x?} target={:#010x}",
                                    start, slot_addr, target
                                );
                            }
                            self.eip = return_eip;
                            return Ok(());
                        }
                        self.push32(mem, return_eip)?;
                        self.eip = if is_16bit { target & 0xFFFF } else { target };
                        return Ok(());
                    }
                    4 => {
                        // JMP r/m16/32 (near absolute)
                        let target = if is_16bit {
                            match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    self.read_u16(mem, addr)? as u32
                                }
                                crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                            }
                        } else {
                            self.read_operand32(mem, dst)?
                        };
                        self.eip = if is_16bit { target & 0xFFFF } else { target };
                        return Ok(());
                    }
                    6 => {
                        // PUSH r/m16/32
                        let value = if is_16bit {
                            match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    self.read_u16(mem, addr)? as u32
                                }
                                crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                            }
                        } else {
                            self.read_operand32(mem, dst)?
                        };
                        self.eip = s.ip;
                        if is_16bit {
                            self.esp = self.esp.wrapping_sub(2);
                            mem.write(self.esp, &(value as u16).to_le_bytes())?;
                        } else {
                            self.push32(mem, value)?;
                        }
                        return Ok(());
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                }
            }

            0xE8 => {
                let rel = s.fetch_i32()?;
                let return_eip = s.ip;

                // Call 指令需要更新 EIP 到目标地址
                self.eip = s.ip.wrapping_add(rel as u32);
                self.push32(mem, return_eip)?;
                return Ok(());
            }
            0xE4 => {
                // IN AL, imm8
                let port = s.fetch_u8()? as u16;
                let value = self.io_in8(port);
                self.set_reg8(0, value);
                self.eip = s.ip;
                return Ok(());
            }
            0xE5 => {
                // IN AX/EAX, imm8
                let port = s.fetch_u8()? as u16;
                if is_16bit {
                    let value = self.io_in16(port) as u32;
                    self.eax = (self.eax & 0xFFFF_0000) | value;
                } else {
                    self.eax = self.io_in32(port);
                }
                self.eip = s.ip;
                return Ok(());
            }
            0xE6 => {
                // OUT imm8, AL
                let port = s.fetch_u8()? as u16;
                self.io_out8(port, self.reg8(0));
                self.eip = s.ip;
                return Ok(());
            }
            0xE7 => {
                // OUT imm8, AX/EAX
                let port = s.fetch_u8()? as u16;
                if is_16bit {
                    self.io_out16(port, (self.eax & 0xFFFF) as u16);
                } else {
                    self.io_out32(port, self.eax);
                }
                self.eip = s.ip;
                return Ok(());
            }
            0xEC => {
                // IN AL, DX
                let value = self.io_in8(self.edx as u16);
                self.set_reg8(0, value);
                self.eip = s.ip;
                return Ok(());
            }
            0xED => {
                // IN AX/EAX, DX
                if is_16bit {
                    let value = self.io_in16(self.edx as u16) as u32;
                    self.eax = (self.eax & 0xFFFF_0000) | value;
                } else {
                    self.eax = self.io_in32(self.edx as u16);
                }
                self.eip = s.ip;
                return Ok(());
            }
            0xEE => {
                // OUT DX, AL
                self.io_out8(self.edx as u16, self.reg8(0));
                self.eip = s.ip;
                return Ok(());
            }
            0xEF => {
                // OUT DX, AX/EAX
                if is_16bit {
                    self.io_out16(self.edx as u16, (self.eax & 0xFFFF) as u16);
                } else {
                    self.io_out32(self.edx as u16, self.eax);
                }
                self.eip = s.ip;
                return Ok(());
            }
            0xC0 | 0xD0 | 0xD2 | 0xC1 | 0xD1 | 0xD3 => {
                // Group 2: 移位指令 (SHL, SHR, SAR, ROL, ROR)
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let is_8bit = matches!(opcode, 0xC0 | 0xD0 | 0xD2);

                // 确定要移动的位数
                let count = match opcode {
                    0xC0 | 0xC1 => s.fetch_u8()? & 0x1F,    // 立即数移位
                    0xD0 | 0xD1 => 1,                       // 固定移 1 位
                    0xD2 | 0xD3 => (self.ecx & 0x1F) as u8, // 根据 CL 寄存器移位
                    _ => 0,
                };

                self.eip = s.ip; // 释放 mem 借用

                if count > 0 {
                    if is_8bit {
                        let val = self.read_operand8(mem, dst)?;
                        let result = match modrm.reg {
                            0 => val.rotate_left(count as u32),  // ROL
                            1 => val.rotate_right(count as u32), // ROR
                            2 | 3 => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                            4 | 6 => val.wrapping_shl(count as u32), // SHL/SAL
                            5 => val >> count,                       // SHR
                            7 => ((val as i8) >> count) as u8,       // SAR
                            _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                        };
                        self.write_operand8(mem, dst, result)?;
                    } else {
                        let val = self.read_operand32(mem, dst)?;
                        let result = match modrm.reg {
                            0 => val.rotate_left(count as u32),  // ROL
                            1 => val.rotate_right(count as u32), // ROR
                            // RCL 和 RCR 涉及进位标志位，较复杂且这里用不到，暂时屏蔽
                            2 | 3 => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                            4 | 6 => val << count,             // SHL / SAL
                            5 => val >> count,                 // SHR
                            7 => (val as i32 >> count) as u32, // SAR
                            _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                        };
                        self.write_operand32(mem, dst, result)?;
                    }
                }
                return Ok(());
            }
            0xC3 => {
                // Ret 不再使用 s.ip
                self.eip = self.pop32(mem)?;
                return Ok(());
            }
            0xC2 => {
                let imm = s.fetch_u16()? as u32;
                self.eip = self.pop32(mem)?;
                self.esp = self.esp.wrapping_add(imm);
                return Ok(());
            }
            0xC9 => {
                self.esp = self.ebp;
                self.eip = s.ip; // 必须在 pop 前记录当前的 IP
                self.ebp = self.pop32(mem)?;
                return Ok(());
            }
            0xD9 => {
                // x87: FLD/FST/FSTP m32real
                let modrm = s.fetch_modrm()?;
                if modrm.mod_ == 0b11 {
                    return Err(ExecError::UnimplementedOpcode(opcode, start));
                }
                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                self.eip = s.ip;
                match modrm.reg {
                    0 => {
                        let mut buf = [0u8; 4];
                        mem.read(addr, &mut buf)?;
                        self.fpu_push(f32::from_le_bytes(buf) as f64);
                    }
                    2 => {
                        let v = self.fpu_peek() as f32;
                        mem.write(addr, &v.to_le_bytes())?;
                    }
                    3 => {
                        let v = self.fpu_pop() as f32;
                        mem.write(addr, &v.to_le_bytes())?;
                    }
                    5 => {
                        // FLDCW m2byte：当前未建模控制字，读取并忽略。
                        let mut cw = [0u8; 2];
                        mem.read(addr, &mut cw)?;
                    }
                    7 => {
                        // FNSTCW m2byte：返回默认 x87 控制字 0x037F。
                        mem.write(addr, &0x037Fu16.to_le_bytes())?;
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                }
                return Ok(());
            }
            0xDB => {
                // x87: FILD/FIST/FISTP m32int
                let modrm = s.fetch_modrm()?;
                if modrm.mod_ == 0b11 {
                    // 常见控制指令（如 FINIT/FNCLEX）走寄存器编码形式。
                    let modrm_byte = 0xC0 | ((modrm.reg & 7) << 3) | (modrm.rm & 7);
                    self.eip = s.ip;
                    match modrm_byte {
                        0xE3 => {
                            // FNINIT/FINIT：重置简化 FPU 状态
                            self.fpu_stack.clear();
                            return Ok(());
                        }
                        0xE2 => {
                            // FNCLEX：清异常标志（当前无完整异常模型，视为 no-op）
                            return Ok(());
                        }
                        _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                    }
                }
                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                self.eip = s.ip;
                match modrm.reg {
                    0 => {
                        let mut buf = [0u8; 4];
                        mem.read(addr, &mut buf)?;
                        self.fpu_push(i32::from_le_bytes(buf) as f64);
                    }
                    2 => {
                        let v = self.fpu_peek() as i32;
                        mem.write(addr, &v.to_le_bytes())?;
                    }
                    3 => {
                        let v = self.fpu_pop() as i32;
                        mem.write(addr, &v.to_le_bytes())?;
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                }
                return Ok(());
            }
            0xDD => {
                // x87: FLD/FST/FSTP m64real
                let modrm = s.fetch_modrm()?;
                if modrm.mod_ == 0b11 {
                    return Err(ExecError::UnimplementedOpcode(opcode, start));
                }
                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                self.eip = s.ip;
                match modrm.reg {
                    0 => {
                        let mut buf = [0u8; 8];
                        mem.read(addr, &mut buf)?;
                        self.fpu_push(f64::from_le_bytes(buf));
                    }
                    2 => {
                        let v = self.fpu_peek();
                        mem.write(addr, &v.to_le_bytes())?;
                    }
                    3 => {
                        let v = self.fpu_pop();
                        mem.write(addr, &v.to_le_bytes())?;
                    }
                    4 => {
                        // FRSTOR m94/108byte：当前仅做最小兼容，不恢复完整 x87 环境。
                    }
                    6 => {
                        // FNSAVE/FSAVE m94/108byte：写出零化环境并重置简化 FPU 状态。
                        let zero = [0u8; 108];
                        mem.write(addr, &zero)?;
                        self.fpu_stack.clear();
                    }
                    7 => {
                        // FNSTSW m2byte：写出一个“无异常”的状态字。
                        mem.write(addr, &0u16.to_le_bytes())?;
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                }
                return Ok(());
            }
            0xDF => {
                // x87: FILD/FISTP m16|m64int
                let modrm = s.fetch_modrm()?;
                if modrm.mod_ == 0b11 {
                    return Err(ExecError::UnimplementedOpcode(opcode, start));
                }
                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                self.eip = s.ip;
                match modrm.reg {
                    0 => {
                        let mut buf = [0u8; 2];
                        mem.read(addr, &mut buf)?;
                        self.fpu_push(i16::from_le_bytes(buf) as f64);
                    }
                    5 => {
                        let mut buf = [0u8; 8];
                        mem.read(addr, &mut buf)?;
                        self.fpu_push(i64::from_le_bytes(buf) as f64);
                    }
                    7 => {
                        let v = self.fpu_pop() as i64;
                        mem.write(addr, &v.to_le_bytes())?;
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                }
                return Ok(());
            }
            0xE9 => {
                let rel = s.fetch_i32()?;
                s.ip = s.ip.wrapping_add(rel as u32);
            }
            0xEB => {
                let rel = s.fetch_i8()? as i32 as u32;
                s.ip = s.ip.wrapping_add(rel);
            }
            0x70..=0x7F => {
                let cc = opcode & 0x0F;
                let rel = s.fetch_i8()? as i32 as u32;
                if self.condition_true(cc) {
                    s.ip = s.ip.wrapping_add(rel);
                }
            }
            0x0F => {
                let op2 = s.fetch_u8()?;
                match op2 {
                    0x1F => {
                        // Multi-byte NOP (经常配合 0x66 用作循环对齐)
                        let modrm = s.fetch_modrm()?;
                        let _ = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0x31 => {
                        // RDTSC
                        self.tsc = self.tsc.wrapping_add(1_000);
                        self.eax = self.tsc as u32;
                        self.edx = (self.tsc >> 32) as u32;
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0xA2 => {
                        // CPUID
                        let leaf = self.eax;
                        let subleaf = self.ecx;
                        let (eax, ebx, ecx, edx) = match leaf {
                            0x0000_0000 => (
                                0x0000_0001, // max basic leaf
                                0x756e6547,  // "Genu"
                                0x6c65746e,  // "ntel"
                                0x49656e69,  // "ineI"
                            ),
                            0x0000_0001 => {
                                let features_ecx = (1 << 0)   // SSE3
                                    | (1 << 9)                 // SSSE3
                                    | (1 << 13)                // CMPXCHG16B
                                    | (1 << 19)                // SSE4.1
                                    | (1 << 20)                // SSE4.2
                                    | (1 << 23); // POPCNT
                                let features_edx = (1 << 0) // FPU
                                    | (1 << 4)               // TSC
                                    | (1 << 15)              // CMOV
                                    | (1 << 23)              // MMX
                                    | (1 << 24)              // FXSR
                                    | (1 << 25)              // SSE
                                    | (1 << 26); // SSE2
                                (0x0000_0663, 0, features_ecx, features_edx)
                            }
                            0x0000_0007 => {
                                let _ = subleaf;
                                (0, 0, 0, 0)
                            }
                            0x8000_0000 => (0x8000_0001, 0, 0, 0),
                            0x8000_0001 => (0, 0, 0, 0),
                            _ => (0, 0, 0, 0),
                        };
                        self.eax = eax;
                        self.ebx = ebx;
                        self.ecx = ecx;
                        self.edx = edx;
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0xA3 => {
                        // BT r/m16|32, r16|32
                        // 语义：把选中位复制到 CF，其它标志不定义（这里保持不变）。
                        let modrm = s.fetch_modrm()?;
                        let bit_src = self.reg32(modrm.reg);

                        let bit = if modrm.mod_ == 0b11 {
                            if is_16bit {
                                let value = self.reg32(modrm.rm) & 0xFFFF;
                                (value >> (bit_src & 0x0F)) & 1
                            } else {
                                let value = self.reg32(modrm.rm);
                                (value >> (bit_src & 0x1F)) & 1
                            }
                        } else {
                            let base = self.calc_effective_addr(mem, &mut s, modrm)?;
                            if is_16bit {
                                let bit_index = bit_src as i32;
                                let word_disp = (bit_index >> 4) * 2;
                                let addr = base.wrapping_add(word_disp as u32);
                                let value = self.read_u16(mem, addr)? as u32;
                                (value >> ((bit_index as u32) & 0x0F)) & 1
                            } else {
                                let bit_index = bit_src as i32;
                                let dword_disp = (bit_index >> 5) * 4;
                                let addr = base.wrapping_add(dword_disp as u32);
                                let value = self.read_u32(mem, addr)?;
                                (value >> ((bit_index as u32) & 0x1F)) & 1
                            }
                        };

                        Flags::set(&mut self.eflags, Flags::CF, bit != 0);
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0x40..=0x4F => {
                        // CMOVcc r16|32, r/m16|32
                        let cc = op2 & 0x0F;
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        self.eip = s.ip;
                        if self.condition_true(cc) {
                            if is_16bit {
                                let value16 = match src {
                                    crate::decode::Operand32::Mem(addr) => {
                                        self.read_u16(mem, addr)? as u32
                                    }
                                    crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                                };
                                let old = self.reg32(modrm.reg);
                                self.set_reg32(modrm.reg, (old & 0xFFFF_0000) | value16);
                            } else {
                                let value = self.read_operand32(mem, src)?;
                                self.set_reg32(modrm.reg, value);
                            }
                        }
                        return Ok(());
                    }
                    0x80..=0x8F => {
                        let cc = op2 & 0x0F;
                        let rel = s.fetch_i32()? as u32;
                        if self.condition_true(cc) {
                            s.ip = s.ip.wrapping_add(rel);
                        }
                    }
                    0x90..=0x9F => {
                        // SETcc r/m8
                        let cc = op2 & 0x0F;
                        let modrm = s.fetch_modrm()?;
                        let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let value = if self.condition_true(cc) { 1 } else { 0 };
                        self.eip = s.ip;
                        self.write_operand8(mem, dst, value)?;
                        return Ok(());
                    }
                    0x10 => {
                        // MOVUPS/MOVUPD/MOVAPS/MOVAPD 或 MOVSS/MOVSD（按前缀区分）
                        let modrm = s.fetch_modrm()?;
                        self.eip = s.ip;
                        if prefix_f2 {
                            // MOVSD xmm, xmm/m64: 只覆盖低 64 位
                            let mut dst = self.xmm_reg(modrm.reg);
                            if modrm.mod_ == 0b11 {
                                let src = self.xmm_reg(modrm.rm);
                                dst[..8].copy_from_slice(&src[..8]);
                            } else {
                                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                                self.eip = s.ip;
                                let mut low = [0u8; 8];
                                mem.read(addr, &mut low)?;
                                dst[..8].copy_from_slice(&low);
                            }
                            self.set_xmm_reg(modrm.reg, dst);
                            return Ok(());
                        }
                        if prefix_f3 {
                            // MOVSS xmm, xmm/m32: 只覆盖低 32 位
                            let mut dst = self.xmm_reg(modrm.reg);
                            if modrm.mod_ == 0b11 {
                                let src = self.xmm_reg(modrm.rm);
                                dst[..4].copy_from_slice(&src[..4]);
                            } else {
                                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                                self.eip = s.ip;
                                let mut low = [0u8; 4];
                                mem.read(addr, &mut low)?;
                                dst[..4].copy_from_slice(&low);
                            }
                            self.set_xmm_reg(modrm.reg, dst);
                            return Ok(());
                        }

                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.set_xmm_reg(modrm.reg, src);
                        return Ok(());
                    }
                    0x11 => {
                        // MOVUPS/MOVUPD/MOVAPS/MOVAPD 或 MOVSS/MOVSD（按前缀区分）
                        let modrm = s.fetch_modrm()?;
                        let src = self.xmm_reg(modrm.reg);
                        self.eip = s.ip;
                        if prefix_f2 {
                            // MOVSD xmm/m64, xmm: 只写低 64 位
                            if modrm.mod_ == 0b11 {
                                let mut dst = self.xmm_reg(modrm.rm);
                                dst[..8].copy_from_slice(&src[..8]);
                                self.set_xmm_reg(modrm.rm, dst);
                            } else {
                                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                                self.eip = s.ip;
                                mem.write(addr, &src[..8])?;
                            }
                            return Ok(());
                        }
                        if prefix_f3 {
                            // MOVSS xmm/m32, xmm: 只写低 32 位
                            if modrm.mod_ == 0b11 {
                                let mut dst = self.xmm_reg(modrm.rm);
                                dst[..4].copy_from_slice(&src[..4]);
                                self.set_xmm_reg(modrm.rm, dst);
                            } else {
                                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                                self.eip = s.ip;
                                mem.write(addr, &src[..4])?;
                            }
                            return Ok(());
                        }

                        if modrm.mod_ == 0b11 {
                            self.set_xmm_reg(modrm.rm, src);
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            mem.write(addr, &src)?;
                        }
                        return Ok(());
                    }
                    0x28 => {
                        // MOVAPS/MOVAPD
                        let modrm = s.fetch_modrm()?;
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;
                        self.set_xmm_reg(modrm.reg, src);
                        return Ok(());
                    }
                    0x29 => {
                        // MOVAPS/MOVAPD
                        let modrm = s.fetch_modrm()?;
                        let src = self.xmm_reg(modrm.reg);
                        if modrm.mod_ == 0b11 {
                            self.eip = s.ip;
                            self.set_xmm_reg(modrm.rm, src);
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            mem.write(addr, &src)?;
                        }
                        return Ok(());
                    }
                    0x2C => {
                        // CVTTSS2SI / CVTTSD2SI（按 F3/F2 前缀区分），仅实现 32 位目标寄存器。
                        let modrm = s.fetch_modrm()?;
                        let value = if prefix_f3 {
                            if modrm.mod_ == 0b11 {
                                let src = self.xmm_reg(modrm.rm);
                                let mut raw = [0u8; 4];
                                raw.copy_from_slice(&src[..4]);
                                f32::from_le_bytes(raw) as f64
                            } else {
                                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                                self.eip = s.ip;
                                let mut raw = [0u8; 4];
                                mem.read(addr, &mut raw)?;
                                f32::from_le_bytes(raw) as f64
                            }
                        } else if prefix_f2 {
                            if modrm.mod_ == 0b11 {
                                let src = self.xmm_reg(modrm.rm);
                                let mut raw = [0u8; 8];
                                raw.copy_from_slice(&src[..8]);
                                f64::from_le_bytes(raw)
                            } else {
                                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                                self.eip = s.ip;
                                let mut raw = [0u8; 8];
                                mem.read(addr, &mut raw)?;
                                f64::from_le_bytes(raw)
                            }
                        } else {
                            return Err(ExecError::UnimplementedOpcode(op2, start));
                        };

                        // x86 约定：NaN/溢出时返回 0x80000000。
                        let out = if !value.is_finite()
                            || value >= 2_147_483_648.0
                            || value < -2_147_483_648.0
                        {
                            0x8000_0000u32
                        } else {
                            (value.trunc() as i32) as u32
                        };
                        self.eip = s.ip;
                        self.set_reg32(modrm.reg, out);
                        return Ok(());
                    }
                    0x2A => {
                        // CVTSI2SS / CVTSI2SD（按 F3/F2 前缀区分）
                        // 额外兼容无前缀/66 前缀的老式 MMX 变体，尽量给出可运行结果。
                        let modrm = s.fetch_modrm()?;
                        let mut dst = self.xmm_reg(modrm.reg);

                        if prefix_f3 || prefix_f2 {
                            let value_i32 = if modrm.mod_ == 0b11 {
                                self.reg32(modrm.rm) as i32
                            } else {
                                let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                                self.eip = s.ip;
                                self.read_u32(mem, addr)? as i32
                            };
                            self.eip = s.ip;
                            if prefix_f2 {
                                dst[..8].copy_from_slice(&(value_i32 as f64).to_le_bytes());
                            } else {
                                dst[..4].copy_from_slice(&(value_i32 as f32).to_le_bytes());
                            }
                            self.set_xmm_reg(modrm.reg, dst);
                            return Ok(());
                        }

                        // 无前缀 / 66 前缀：读取两个 32 位整数并转换到低两个 lane。
                        let (lo_i32, hi_i32) = if modrm.mod_ == 0b11 {
                            (self.reg32(modrm.rm) as i32, 0i32)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut raw = [0u8; 8];
                            mem.read(addr, &mut raw)?;
                            let lo = i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                            let hi = i32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
                            (lo, hi)
                        };
                        self.eip = s.ip;

                        if prefix_66 {
                            dst[0..8].copy_from_slice(&(lo_i32 as f64).to_le_bytes());
                            dst[8..16].copy_from_slice(&(hi_i32 as f64).to_le_bytes());
                        } else {
                            dst[0..4].copy_from_slice(&(lo_i32 as f32).to_le_bytes());
                            dst[4..8].copy_from_slice(&(hi_i32 as f32).to_le_bytes());
                        }
                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x2E | 0x2F => {
                        // UCOMISS/UCOMISD/COMISS/COMISD（按 66 前缀区分精度）
                        // 这里不区分 COMI/UCOMI 的异常行为，只实现标志位语义。
                        let modrm = s.fetch_modrm()?;
                        self.eip = s.ip;

                        let dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };

                        let (unordered, lt, eq) = if prefix_66 {
                            let mut a = [0u8; 8];
                            let mut b = [0u8; 8];
                            a.copy_from_slice(&dst[..8]);
                            b.copy_from_slice(&src[..8]);
                            let av = f64::from_le_bytes(a);
                            let bv = f64::from_le_bytes(b);
                            if av.is_nan() || bv.is_nan() {
                                (true, false, false)
                            } else {
                                (false, av < bv, av == bv)
                            }
                        } else {
                            let mut a = [0u8; 4];
                            let mut b = [0u8; 4];
                            a.copy_from_slice(&dst[..4]);
                            b.copy_from_slice(&src[..4]);
                            let av = f32::from_le_bytes(a);
                            let bv = f32::from_le_bytes(b);
                            if av.is_nan() || bv.is_nan() {
                                (true, false, false)
                            } else {
                                (false, av < bv, av == bv)
                            }
                        };

                        let (zf, pf, cf) = if unordered {
                            (true, true, true)
                        } else if eq {
                            (true, false, false)
                        } else if lt {
                            (false, false, true)
                        } else {
                            (false, false, false)
                        };
                        Flags::set(&mut self.eflags, Flags::ZF, zf);
                        Flags::set(&mut self.eflags, Flags::PF, pf);
                        Flags::set(&mut self.eflags, Flags::CF, cf);
                        Flags::set(&mut self.eflags, Flags::OF, false);
                        Flags::set(&mut self.eflags, Flags::SF, false);
                        Flags::set(&mut self.eflags, Flags::AF, false);
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0x54 => {
                        // ANDPS/ANDPD（按位与）
                        let modrm = s.fetch_modrm()?;
                        let lhs = self.xmm_reg(modrm.reg);
                        let rhs = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        let mut out = [0u8; 16];
                        for i in 0..16 {
                            out[i] = lhs[i] & rhs[i];
                        }
                        self.eip = s.ip;
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0x55 => {
                        // ANDNPS/ANDNPD：(~dst) & src
                        let modrm = s.fetch_modrm()?;
                        let lhs = self.xmm_reg(modrm.reg);
                        let rhs = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        let mut out = [0u8; 16];
                        for i in 0..16 {
                            out[i] = (!lhs[i]) & rhs[i];
                        }
                        self.eip = s.ip;
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0x56 => {
                        // ORPS/ORPD（按位或）
                        let modrm = s.fetch_modrm()?;
                        let lhs = self.xmm_reg(modrm.reg);
                        let rhs = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        let mut out = [0u8; 16];
                        for i in 0..16 {
                            out[i] = lhs[i] | rhs[i];
                        }
                        self.eip = s.ip;
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0x57 => {
                        // XORPS/XORPD（按位异或，前缀不影响按位语义）
                        let modrm = s.fetch_modrm()?;
                        let lhs = self.xmm_reg(modrm.reg);
                        let rhs = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        let mut out = [0u8; 16];
                        for i in 0..16 {
                            out[i] = lhs[i] ^ rhs[i];
                        }
                        self.eip = s.ip;
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0xEF => {
                        // PXOR/XOR128：按位异或（这里统一按 XMM 128 位处理）
                        let modrm = s.fetch_modrm()?;
                        let lhs = self.xmm_reg(modrm.reg);
                        let rhs = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        let mut out = [0u8; 16];
                        for i in 0..16 {
                            out[i] = lhs[i] ^ rhs[i];
                        }
                        self.eip = s.ip;
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0x58 => {
                        // ADDPS/ADDPD/ADDSS/ADDSD（按前缀区分）
                        let modrm = s.fetch_modrm()?;
                        let mut dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;

                        if prefix_f2 {
                            // ADDSD: 仅低 64 位参与计算
                            let mut a = [0u8; 8];
                            let mut b = [0u8; 8];
                            a.copy_from_slice(&dst[..8]);
                            b.copy_from_slice(&src[..8]);
                            let r = f64::from_le_bytes(a) + f64::from_le_bytes(b);
                            dst[..8].copy_from_slice(&r.to_le_bytes());
                        } else if prefix_f3 {
                            // ADDSS: 仅低 32 位参与计算
                            let mut a = [0u8; 4];
                            let mut b = [0u8; 4];
                            a.copy_from_slice(&dst[..4]);
                            b.copy_from_slice(&src[..4]);
                            let r = f32::from_le_bytes(a) + f32::from_le_bytes(b);
                            dst[..4].copy_from_slice(&r.to_le_bytes());
                        } else if prefix_66 {
                            // ADDPD
                            for lane in 0..2 {
                                let off = lane * 8;
                                let mut a = [0u8; 8];
                                let mut b = [0u8; 8];
                                a.copy_from_slice(&dst[off..off + 8]);
                                b.copy_from_slice(&src[off..off + 8]);
                                let r = f64::from_le_bytes(a) + f64::from_le_bytes(b);
                                dst[off..off + 8].copy_from_slice(&r.to_le_bytes());
                            }
                        } else {
                            // ADDPS
                            for lane in 0..4 {
                                let off = lane * 4;
                                let mut a = [0u8; 4];
                                let mut b = [0u8; 4];
                                a.copy_from_slice(&dst[off..off + 4]);
                                b.copy_from_slice(&src[off..off + 4]);
                                let r = f32::from_le_bytes(a) + f32::from_le_bytes(b);
                                dst[off..off + 4].copy_from_slice(&r.to_le_bytes());
                            }
                        }
                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x59 => {
                        // MULPS/MULPD/MULSS/MULSD（按前缀区分）
                        let modrm = s.fetch_modrm()?;
                        let mut dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;

                        if prefix_f2 {
                            // MULSD: 仅低 64 位参与计算
                            let mut a = [0u8; 8];
                            let mut b = [0u8; 8];
                            a.copy_from_slice(&dst[..8]);
                            b.copy_from_slice(&src[..8]);
                            let r = f64::from_le_bytes(a) * f64::from_le_bytes(b);
                            dst[..8].copy_from_slice(&r.to_le_bytes());
                        } else if prefix_f3 {
                            // MULSS: 仅低 32 位参与计算
                            let mut a = [0u8; 4];
                            let mut b = [0u8; 4];
                            a.copy_from_slice(&dst[..4]);
                            b.copy_from_slice(&src[..4]);
                            let r = f32::from_le_bytes(a) * f32::from_le_bytes(b);
                            dst[..4].copy_from_slice(&r.to_le_bytes());
                        } else if prefix_66 {
                            // MULPD
                            for lane in 0..2 {
                                let off = lane * 8;
                                let mut a = [0u8; 8];
                                let mut b = [0u8; 8];
                                a.copy_from_slice(&dst[off..off + 8]);
                                b.copy_from_slice(&src[off..off + 8]);
                                let r = f64::from_le_bytes(a) * f64::from_le_bytes(b);
                                dst[off..off + 8].copy_from_slice(&r.to_le_bytes());
                            }
                        } else {
                            // MULPS
                            for lane in 0..4 {
                                let off = lane * 4;
                                let mut a = [0u8; 4];
                                let mut b = [0u8; 4];
                                a.copy_from_slice(&dst[off..off + 4]);
                                b.copy_from_slice(&src[off..off + 4]);
                                let r = f32::from_le_bytes(a) * f32::from_le_bytes(b);
                                dst[off..off + 4].copy_from_slice(&r.to_le_bytes());
                            }
                        }
                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x5A => {
                        // CVTPS2PD/CVTPD2PS/CVTSS2SD/CVTSD2SS（按前缀区分）
                        let modrm = s.fetch_modrm()?;
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        let mut dst = self.xmm_reg(modrm.reg);
                        self.eip = s.ip;

                        if prefix_f3 {
                            // CVTSS2SD: 低 32 位 float -> 低 64 位 double
                            let mut f = [0u8; 4];
                            f.copy_from_slice(&src[..4]);
                            let v = f32::from_le_bytes(f) as f64;
                            dst[..8].copy_from_slice(&v.to_le_bytes());
                        } else if prefix_f2 {
                            // CVTSD2SS: 低 64 位 double -> 低 32 位 float
                            let mut d = [0u8; 8];
                            d.copy_from_slice(&src[..8]);
                            let v = f64::from_le_bytes(d) as f32;
                            dst[..4].copy_from_slice(&v.to_le_bytes());
                        } else if prefix_66 {
                            // CVTPD2PS: 2x f64 -> 2x f32（高 64 位置零）
                            let mut d0 = [0u8; 8];
                            let mut d1 = [0u8; 8];
                            d0.copy_from_slice(&src[0..8]);
                            d1.copy_from_slice(&src[8..16]);
                            let v0 = (f64::from_le_bytes(d0) as f32).to_le_bytes();
                            let v1 = (f64::from_le_bytes(d1) as f32).to_le_bytes();
                            dst[0..4].copy_from_slice(&v0);
                            dst[4..8].copy_from_slice(&v1);
                            dst[8..16].fill(0);
                        } else {
                            // CVTPS2PD: 低 64 位里的 2x f32 -> 2x f64
                            let mut f0 = [0u8; 4];
                            let mut f1 = [0u8; 4];
                            f0.copy_from_slice(&src[0..4]);
                            f1.copy_from_slice(&src[4..8]);
                            let v0 = (f32::from_le_bytes(f0) as f64).to_le_bytes();
                            let v1 = (f32::from_le_bytes(f1) as f64).to_le_bytes();
                            dst[0..8].copy_from_slice(&v0);
                            dst[8..16].copy_from_slice(&v1);
                        }
                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x5C => {
                        // SUBPS/SUBPD
                        let modrm = s.fetch_modrm()?;
                        let mut dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;
                        if prefix_66 {
                            for lane in 0..2 {
                                let off = lane * 8;
                                let mut a = [0u8; 8];
                                let mut b = [0u8; 8];
                                a.copy_from_slice(&dst[off..off + 8]);
                                b.copy_from_slice(&src[off..off + 8]);
                                let r = f64::from_le_bytes(a) - f64::from_le_bytes(b);
                                dst[off..off + 8].copy_from_slice(&r.to_le_bytes());
                            }
                        } else {
                            for lane in 0..4 {
                                let off = lane * 4;
                                let mut a = [0u8; 4];
                                let mut b = [0u8; 4];
                                a.copy_from_slice(&dst[off..off + 4]);
                                b.copy_from_slice(&src[off..off + 4]);
                                let r = f32::from_le_bytes(a) - f32::from_le_bytes(b);
                                dst[off..off + 4].copy_from_slice(&r.to_le_bytes());
                            }
                        }
                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x5D | 0x5F => {
                        // MINPS/MINPD/MINSS/MINSD 或 MAXPS/MAXPD/MAXSS/MAXSD
                        let modrm = s.fetch_modrm()?;
                        let mut dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;

                        let is_max = op2 == 0x5F;
                        let pick_f32 = |a: f32, b: f32, max_mode: bool| -> f32 {
                            if a.is_nan() || b.is_nan() {
                                b
                            } else if max_mode {
                                a.max(b)
                            } else {
                                a.min(b)
                            }
                        };
                        let pick_f64 = |a: f64, b: f64, max_mode: bool| -> f64 {
                            if a.is_nan() || b.is_nan() {
                                b
                            } else if max_mode {
                                a.max(b)
                            } else {
                                a.min(b)
                            }
                        };

                        if prefix_f3 {
                            // MINSS/MAXSS：仅低 32 位
                            let mut a = [0u8; 4];
                            let mut b = [0u8; 4];
                            a.copy_from_slice(&dst[..4]);
                            b.copy_from_slice(&src[..4]);
                            let r = pick_f32(f32::from_le_bytes(a), f32::from_le_bytes(b), is_max);
                            dst[..4].copy_from_slice(&r.to_le_bytes());
                        } else if prefix_f2 {
                            // MINSD/MAXSD：仅低 64 位
                            let mut a = [0u8; 8];
                            let mut b = [0u8; 8];
                            a.copy_from_slice(&dst[..8]);
                            b.copy_from_slice(&src[..8]);
                            let r = pick_f64(f64::from_le_bytes(a), f64::from_le_bytes(b), is_max);
                            dst[..8].copy_from_slice(&r.to_le_bytes());
                        } else if prefix_66 {
                            // MINPD/MAXPD
                            for lane in 0..2 {
                                let off = lane * 8;
                                let mut a = [0u8; 8];
                                let mut b = [0u8; 8];
                                a.copy_from_slice(&dst[off..off + 8]);
                                b.copy_from_slice(&src[off..off + 8]);
                                let r =
                                    pick_f64(f64::from_le_bytes(a), f64::from_le_bytes(b), is_max);
                                dst[off..off + 8].copy_from_slice(&r.to_le_bytes());
                            }
                        } else {
                            // MINPS/MAXPS
                            for lane in 0..4 {
                                let off = lane * 4;
                                let mut a = [0u8; 4];
                                let mut b = [0u8; 4];
                                a.copy_from_slice(&dst[off..off + 4]);
                                b.copy_from_slice(&src[off..off + 4]);
                                let r =
                                    pick_f32(f32::from_le_bytes(a), f32::from_le_bytes(b), is_max);
                                dst[off..off + 4].copy_from_slice(&r.to_le_bytes());
                            }
                        }

                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x5E => {
                        // DIVPS/DIVPD/DIVSD（按前缀区分）
                        let modrm = s.fetch_modrm()?;
                        let mut dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;

                        if prefix_f2 {
                            let mut a = [0u8; 8];
                            let mut b = [0u8; 8];
                            a.copy_from_slice(&dst[..8]);
                            b.copy_from_slice(&src[..8]);
                            let r = f64::from_le_bytes(a) / f64::from_le_bytes(b);
                            dst[..8].copy_from_slice(&r.to_le_bytes());
                        } else if prefix_66 {
                            for lane in 0..2 {
                                let off = lane * 8;
                                let mut a = [0u8; 8];
                                let mut b = [0u8; 8];
                                a.copy_from_slice(&dst[off..off + 8]);
                                b.copy_from_slice(&src[off..off + 8]);
                                let r = f64::from_le_bytes(a) / f64::from_le_bytes(b);
                                dst[off..off + 8].copy_from_slice(&r.to_le_bytes());
                            }
                        } else {
                            for lane in 0..4 {
                                let off = lane * 4;
                                let mut a = [0u8; 4];
                                let mut b = [0u8; 4];
                                a.copy_from_slice(&dst[off..off + 4]);
                                b.copy_from_slice(&src[off..off + 4]);
                                let r = f32::from_le_bytes(a) / f32::from_le_bytes(b);
                                dst[off..off + 4].copy_from_slice(&r.to_le_bytes());
                            }
                        }
                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x62 => {
                        // PUNPCKLDQ xmm, xmm/m128
                        let modrm = s.fetch_modrm()?;
                        let dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;
                        let mut out = [0u8; 16];
                        out[0..4].copy_from_slice(&dst[0..4]);
                        out[4..8].copy_from_slice(&src[0..4]);
                        out[8..12].copy_from_slice(&dst[4..8]);
                        out[12..16].copy_from_slice(&src[4..8]);
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0x6E => {
                        // MOVD xmm, r/m32（66 0F 6E）
                        let modrm = s.fetch_modrm()?;
                        let src32 = if modrm.mod_ == 0b11 {
                            self.reg32(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            self.read_u32(mem, addr)?
                        };
                        self.eip = s.ip;
                        let mut out = [0u8; 16];
                        out[..4].copy_from_slice(&src32.to_le_bytes());
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0x6F => {
                        // MOVDQA/MOVDQU load（这里统一按 128 位搬运）
                        let modrm = s.fetch_modrm()?;
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;
                        self.set_xmm_reg(modrm.reg, src);
                        return Ok(());
                    }
                    0x73 => {
                        // PSRLQ/PSLLQ/PSRLDQ/PSLLDQ (imm8 group, 常见于 SSE2)
                        let modrm = s.fetch_modrm()?;
                        let imm = s.fetch_u8()?;
                        self.eip = s.ip;
                        if modrm.mod_ != 0b11 {
                            return Err(ExecError::UnimplementedOpcode(op2, start));
                        }
                        let mut dst = self.xmm_reg(modrm.rm);
                        match modrm.reg {
                            2 => {
                                // PSRLQ xmm, imm8
                                let count = (imm as u32).min(64);
                                for lane in 0..2 {
                                    let off = lane * 8;
                                    let mut q = [0u8; 8];
                                    q.copy_from_slice(&dst[off..off + 8]);
                                    let v = u64::from_le_bytes(q);
                                    let r = if count >= 64 { 0 } else { v >> count };
                                    dst[off..off + 8].copy_from_slice(&r.to_le_bytes());
                                }
                            }
                            6 => {
                                // PSLLQ xmm, imm8
                                let count = (imm as u32).min(64);
                                for lane in 0..2 {
                                    let off = lane * 8;
                                    let mut q = [0u8; 8];
                                    q.copy_from_slice(&dst[off..off + 8]);
                                    let v = u64::from_le_bytes(q);
                                    let r = if count >= 64 { 0 } else { v << count };
                                    dst[off..off + 8].copy_from_slice(&r.to_le_bytes());
                                }
                            }
                            3 => {
                                // PSRLDQ xmm, imm8（按字节右移整个 128 位向量）
                                let count = (imm as usize).min(16);
                                if count >= 16 {
                                    dst.fill(0);
                                } else if count > 0 {
                                    let mut out = [0u8; 16];
                                    out[..(16 - count)].copy_from_slice(&dst[count..]);
                                    dst = out;
                                }
                            }
                            7 => {
                                // PSLLDQ xmm, imm8（按字节左移整个 128 位向量）
                                let count = (imm as usize).min(16);
                                if count >= 16 {
                                    dst.fill(0);
                                } else if count > 0 {
                                    let mut out = [0u8; 16];
                                    out[count..].copy_from_slice(&dst[..(16 - count)]);
                                    dst = out;
                                }
                            }
                            _ => return Err(ExecError::UnimplementedOpcode(op2, start)),
                        }
                        self.set_xmm_reg(modrm.rm, dst);
                        return Ok(());
                    }
                    0x7C => {
                        // HADDPD/HADDPS（按前缀区分）
                        let modrm = s.fetch_modrm()?;
                        let mut dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;
                        if prefix_66 {
                            let mut d0 = [0u8; 8];
                            let mut d1 = [0u8; 8];
                            let mut s0 = [0u8; 8];
                            let mut s1 = [0u8; 8];
                            d0.copy_from_slice(&dst[0..8]);
                            d1.copy_from_slice(&dst[8..16]);
                            s0.copy_from_slice(&src[0..8]);
                            s1.copy_from_slice(&src[8..16]);
                            let r0 = f64::from_le_bytes(d0) + f64::from_le_bytes(d1);
                            let r1 = f64::from_le_bytes(s0) + f64::from_le_bytes(s1);
                            dst[0..8].copy_from_slice(&r0.to_le_bytes());
                            dst[8..16].copy_from_slice(&r1.to_le_bytes());
                        } else {
                            let mut out = [0u8; 16];
                            let f = |buf: &[u8]| {
                                let mut a = [0u8; 4];
                                a.copy_from_slice(buf);
                                f32::from_le_bytes(a)
                            };
                            let d0 = f(&dst[0..4]);
                            let d1 = f(&dst[4..8]);
                            let d2 = f(&dst[8..12]);
                            let d3 = f(&dst[12..16]);
                            let s0 = f(&src[0..4]);
                            let s1 = f(&src[4..8]);
                            let s2 = f(&src[8..12]);
                            let s3 = f(&src[12..16]);
                            out[0..4].copy_from_slice(&(d0 + d1).to_le_bytes());
                            out[4..8].copy_from_slice(&(d2 + d3).to_le_bytes());
                            out[8..12].copy_from_slice(&(s0 + s1).to_le_bytes());
                            out[12..16].copy_from_slice(&(s2 + s3).to_le_bytes());
                            dst = out;
                        }
                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0x7E => {
                        // MOVD r/m32, xmm（66 0F 7E）
                        let modrm = s.fetch_modrm()?;
                        let src = self.xmm_reg(modrm.reg);
                        let mut low = [0u8; 4];
                        low.copy_from_slice(&src[..4]);
                        let value = u32::from_le_bytes(low);
                        self.eip = s.ip;
                        if modrm.mod_ == 0b11 {
                            self.set_reg32(modrm.rm, value);
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            self.write_u32(mem, addr, value)?;
                        }
                        return Ok(());
                    }
                    0x7F => {
                        // MOVDQA/MOVDQU store（这里统一按 128 位搬运）
                        let modrm = s.fetch_modrm()?;
                        let src = self.xmm_reg(modrm.reg);
                        self.eip = s.ip;
                        if modrm.mod_ == 0b11 {
                            self.set_xmm_reg(modrm.rm, src);
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            mem.write(addr, &src)?;
                        }
                        return Ok(());
                    }
                    0xAF => {
                        // IMUL r32, r/m32 (进阶乘法，编译器优化常客)
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let lhs = self.reg32(modrm.reg);
                        let rhs = self.read_operand32(mem, src)?;
                        let res = (lhs as i32 as i64).wrapping_mul(rhs as i32 as i64);
                        self.set_reg32(modrm.reg, res as u32);
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0xC2 => {
                        // CMPPS/CMPPD/CMPSS/CMPSD（按前缀区分），imm8 指定比较谓词
                        let modrm = s.fetch_modrm()?;
                        let imm8 = s.fetch_u8()?;
                        let pred = imm8 & 0x07;
                        let mut dst = self.xmm_reg(modrm.reg);
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;

                        let cmp_pred = |a: f64, b: f64, p: u8| -> bool {
                            let unordered = a.is_nan() || b.is_nan();
                            match p {
                                0 => !unordered && a == b, // EQ
                                1 => !unordered && a < b,  // LT
                                2 => !unordered && a <= b, // LE
                                3 => unordered,            // UNORD
                                4 => unordered || a != b,  // NEQ
                                5 => unordered || a >= b,  // NLT
                                6 => unordered || a > b,   // NLE
                                7 => !unordered,           // ORD
                                _ => false,
                            }
                        };

                        if prefix_f3 {
                            // CMPSS: 仅低 32 位写结果，高位保持原样
                            let mut a = [0u8; 4];
                            let mut b = [0u8; 4];
                            a.copy_from_slice(&dst[..4]);
                            b.copy_from_slice(&src[..4]);
                            let av = f32::from_le_bytes(a) as f64;
                            let bv = f32::from_le_bytes(b) as f64;
                            let out = if cmp_pred(av, bv, pred) { u32::MAX } else { 0 };
                            dst[..4].copy_from_slice(&out.to_le_bytes());
                        } else if prefix_f2 {
                            // CMPSD: 仅低 64 位写结果，高位保持原样
                            let mut a = [0u8; 8];
                            let mut b = [0u8; 8];
                            a.copy_from_slice(&dst[..8]);
                            b.copy_from_slice(&src[..8]);
                            let av = f64::from_le_bytes(a);
                            let bv = f64::from_le_bytes(b);
                            let out = if cmp_pred(av, bv, pred) { u64::MAX } else { 0 };
                            dst[..8].copy_from_slice(&out.to_le_bytes());
                        } else if prefix_66 {
                            // CMPPD: 2x f64 lane
                            for lane in 0..2 {
                                let off = lane * 8;
                                let mut a = [0u8; 8];
                                let mut b = [0u8; 8];
                                a.copy_from_slice(&dst[off..off + 8]);
                                b.copy_from_slice(&src[off..off + 8]);
                                let av = f64::from_le_bytes(a);
                                let bv = f64::from_le_bytes(b);
                                let out = if cmp_pred(av, bv, pred) { u64::MAX } else { 0 };
                                dst[off..off + 8].copy_from_slice(&out.to_le_bytes());
                            }
                        } else {
                            // CMPPS: 4x f32 lane
                            for lane in 0..4 {
                                let off = lane * 4;
                                let mut a = [0u8; 4];
                                let mut b = [0u8; 4];
                                a.copy_from_slice(&dst[off..off + 4]);
                                b.copy_from_slice(&src[off..off + 4]);
                                let av = f32::from_le_bytes(a) as f64;
                                let bv = f32::from_le_bytes(b) as f64;
                                let out = if cmp_pred(av, bv, pred) { u32::MAX } else { 0 };
                                dst[off..off + 4].copy_from_slice(&out.to_le_bytes());
                            }
                        }

                        self.set_xmm_reg(modrm.reg, dst);
                        return Ok(());
                    }
                    0xE6 => {
                        // 66 0F E6: CVTTPD2DQ（按截断把 2x f64 转为 2x i32）
                        let modrm = s.fetch_modrm()?;
                        if !prefix_66 {
                            return Err(ExecError::UnimplementedOpcode(op2, start));
                        }
                        let src = if modrm.mod_ == 0b11 {
                            self.xmm_reg(modrm.rm)
                        } else {
                            let addr = self.calc_effective_addr(mem, &mut s, modrm)?;
                            self.eip = s.ip;
                            let mut buf = [0u8; 16];
                            mem.read(addr, &mut buf)?;
                            buf
                        };
                        self.eip = s.ip;
                        let mut out = [0u8; 16];
                        for lane in 0..2 {
                            let off = lane * 8;
                            let mut d = [0u8; 8];
                            d.copy_from_slice(&src[off..off + 8]);
                            let v = f64::from_le_bytes(d);
                            let i =
                                if !v.is_finite() || v >= 2_147_483_648.0 || v < -2_147_483_648.0 {
                                    0x8000_0000u32
                                } else {
                                    (v.trunc() as i32) as u32
                                };
                            let o = lane * 4;
                            out[o..o + 4].copy_from_slice(&i.to_le_bytes());
                        }
                        self.set_xmm_reg(modrm.reg, out);
                        return Ok(());
                    }
                    0xB0 => {
                        // CMPXCHG r/m8, r8
                        let modrm = s.fetch_modrm()?;
                        let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let dst_val = self.read_operand8(mem, dst)?;
                        let src_val = self.reg8(modrm.reg);
                        let acc = self.reg8(0); // AL
                        let _ = self.alu_sub32(dst_val as u32, acc as u32);
                        self.eip = s.ip;
                        if dst_val == acc {
                            self.write_operand8(mem, dst, src_val)?;
                        } else {
                            self.set_reg8(0, dst_val);
                        }
                        return Ok(());
                    }
                    0xB1 => {
                        // CMPXCHG r/m16|32, r16|32
                        let modrm = s.fetch_modrm()?;
                        let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        self.eip = s.ip;
                        if is_16bit {
                            let dst_val = match dst {
                                crate::decode::Operand32::Mem(addr) => {
                                    self.read_u16(mem, addr)? as u32
                                }
                                crate::decode::Operand32::Reg(r) => self.reg32(r) & 0xFFFF,
                            };
                            let src_val = self.reg32(modrm.reg) & 0xFFFF;
                            let acc = self.eax & 0xFFFF;
                            let _ = self.alu_sub32(dst_val, acc);
                            if dst_val == acc {
                                match dst {
                                    crate::decode::Operand32::Mem(addr) => {
                                        mem.write(addr, &(src_val as u16).to_le_bytes())?
                                    }
                                    crate::decode::Operand32::Reg(r) => {
                                        let old = self.reg32(r);
                                        self.set_reg32(r, (old & 0xFFFF_0000) | src_val);
                                    }
                                }
                            } else {
                                self.eax = (self.eax & 0xFFFF_0000) | dst_val;
                            }
                        } else {
                            let dst_val = self.read_operand32(mem, dst)?;
                            let src_val = self.reg32(modrm.reg);
                            let acc = self.eax;
                            let _ = self.alu_sub32(dst_val, acc);
                            if dst_val == acc {
                                self.write_operand32(mem, dst, src_val)?;
                            } else {
                                self.eax = dst_val;
                            }
                        }
                        return Ok(());
                    }
                    0xB6 => {
                        // MOVZX r32, r/m8
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let byte = match src {
                            crate::decode::Operand32::Reg(r) => {
                                let base = r % 4;
                                if r < 4 {
                                    (self.reg32(base) & 0xFF) as u8
                                } else {
                                    ((self.reg32(base) >> 8) & 0xFF) as u8
                                }
                            }
                            crate::decode::Operand32::Mem(addr) => self.read_u8(mem, addr)?,
                        };
                        self.eip = s.ip;
                        self.set_reg32(modrm.reg, byte as u32);
                        return Ok(());
                    }
                    0xB7 => {
                        // MOVZX r32, r/m16
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let word = match src {
                            crate::decode::Operand32::Reg(r) => (self.reg32(r) & 0xFFFF) as u16,
                            crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)?,
                        };
                        self.eip = s.ip;
                        self.set_reg32(modrm.reg, word as u32);
                        return Ok(());
                    }
                    0xBE => {
                        // MOVSX r32, r/m8
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let byte = match src {
                            crate::decode::Operand32::Reg(r) => {
                                let base = r % 4;
                                if r < 4 {
                                    (self.reg32(base) & 0xFF) as u8
                                } else {
                                    ((self.reg32(base) >> 8) & 0xFF) as u8
                                }
                            }
                            crate::decode::Operand32::Mem(addr) => self.read_u8(mem, addr)?,
                        };
                        self.eip = s.ip;
                        // 将 8 位有符号扩展为 32 位有符号
                        self.set_reg32(modrm.reg, (byte as i8) as i32 as u32);
                        return Ok(());
                    }
                    0xBF => {
                        // MOVSX r32, r/m16
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let word = match src {
                            crate::decode::Operand32::Reg(r) => (self.reg32(r) & 0xFFFF) as u16,
                            crate::decode::Operand32::Mem(addr) => self.read_u16(mem, addr)?,
                        };
                        self.eip = s.ip;
                        self.set_reg32(modrm.reg, (word as i16) as i32 as u32);
                        return Ok(());
                    }
                    _ => return Err(ExecError::UnimplementedOpcode(op2, start)),
                }
            }
            0xCD => {
                let int_num = s.fetch_u8()?;
                if int_num == 0x80 {
                    self.eip = s.ip;
                    return Err(ExecError::Syscall);
                }
                return Err(ExecError::UnimplementedOpcode(opcode, start));
            }
            0xCE => {
                // INTO: only triggers when OF=1; otherwise behaves like NOP.
                if Flags::get(self.eflags, Flags::OF) {
                    return Err(ExecError::InvalidInstruction(start, "into overflow trap"));
                }
                self.eip = s.ip;
                return Ok(());
            }

            _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
        }

        self.eip = s.ip;
        Ok(())
    }

    pub fn run(&mut self, mem: &mut GuestMemory, max_instructions: usize) -> Result<(), ExecError> {
        for _ in 0..max_instructions {
            self.step(mem)?;
        }
        Ok(())
    }
}
