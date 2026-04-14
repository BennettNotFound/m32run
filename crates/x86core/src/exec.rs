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
        let start = self.eip;
        let mut s = InstructionStream::new(mem, self.eip);

        let mut opcode = s.fetch_u8()?;
        let mut is_16bit = false;

        // 1. 过滤掉指令前缀 (比如 0x66)
        loop {
            match opcode {
                0x66 => {
                    is_16bit = true;
                    opcode = s.fetch_u8()?;
                }
                // CS, DS, ES, SS 段覆盖前缀（在平坦内存模型中可以直接忽略）
                0x2E | 0x3E | 0x26 | 0x36 => {
                    opcode = s.fetch_u8()?;
                }
                // F2, F3 是 REP 前缀，有时也用于对齐或分支预测提示
                0xF2 | 0xF3 => {
                    opcode = s.fetch_u8()?;
                }
                _ => break, // 遇到真正的操作码，跳出循环
            }
        }

        match opcode {
            0x90 => {
                // 原本的 0x90 NOP，直接放行
            }
            0xF4 => {
                if let Some(stub_index) = self.import_stub_index(start) {
                    let indirect_symbol_index = self.import_jump_table_reserved1 + stub_index;
                    return Err(ExecError::UnresolvedImportStub {
                        eip: start,
                        stub_index,
                        indirect_symbol_index,
                    });
                } else {
                    return Err(ExecError::Halt);
                }
            }

            0x68 => {
                // 致命修复：16位模式下 push 的是 2 字节立即数
                let imm = if is_16bit { s.fetch_u16()? as u32 } else { s.fetch_u32()? };
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
                let imm = if is_16bit { s.fetch_u16()? as u32 } else { s.fetch_u32()? };
                self.eip = s.ip;

                if is_16bit {
                    let old = self.reg32(reg);
                    self.set_reg32(reg, (old & 0xFFFF0000) | (imm & 0xFFFF));
                } else {
                    self.set_reg32(reg, imm);
                }
                return Ok(());
            }
            0x89 => {
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let src = self.reg32(modrm.reg);
                self.eip = s.ip;

                if is_16bit {
                    match &dst {
                        crate::decode::Operand32::Mem(addr) => mem.write(*addr, &(src as u16).to_le_bytes())?,
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
            0x8D => { // LEA
                let modrm = s.fetch_modrm()?;
                if modrm.mod_ == 0b11 {
                    return Err(ExecError::InvalidInstruction(start, "lea requires memory operand"));
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
            0xA3 => {
                let addr = s.fetch_u32()?;
                self.eip = s.ip;
                self.write_u32(mem, addr, self.eax)?;
                return Ok(());
            }
            0xC6 => { // MOV r/m8, imm8
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
                let imm = if is_16bit { s.fetch_u16()? as u32 } else { s.fetch_u32()? };
                self.eip = s.ip;

                if is_16bit {
                    match &dst {
                        crate::decode::Operand32::Mem(addr) => mem.write(*addr, &(imm as u16).to_le_bytes())?,
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
                let lhs = self.read_operand32(mem, op)?;                // 拆分2
                let rhs = self.reg32(modrm.reg);
                let _ = self.alu_sub32(lhs, rhs);
            }
            0x3B => {
                let modrm = s.fetch_modrm()?;
                let lhs = self.reg32(modrm.reg);
                let op = self.decode_rm32_operand(mem, &mut s, modrm)?; // 拆分1
                let rhs = self.read_operand32(mem, op)?;                // 拆分2
                let _ = self.alu_sub32(lhs, rhs);
            }

            0x80 | 0x82 => { // Group 1: 8位 ALU 运算 (ADD, OR, AND, SUB, XOR, CMP)
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let imm = s.fetch_u8()?;

                self.eip = s.ip; // 提前释放 mem 的不可变借用

                // 1. 读取 8 位左操作数
                let lhs = match &dst {
                    crate::decode::Operand32::Mem(addr) => self.read_u8(mem, *addr)?,
                    crate::decode::Operand32::Reg(r) => {
                        let base = r % 4;
                        if *r < 4 { (self.reg32(base) & 0xFF) as u8 }
                        else { ((self.reg32(base) >> 8) & 0xFF) as u8 }
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
            0x88 => { // MOV r/m8, r8 (写 8 位寄存器到内存/寄存器)
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                // src 必定是 8 位寄存器
                let r = modrm.reg;
                let base = r % 4;
                let val8 = if r < 4 {
                    (self.reg32(base) & 0xFF) as u8
                } else {
                    ((self.reg32(base) >> 8) & 0xFF) as u8
                };

                self.eip = s.ip;
                match dst {
                    crate::decode::Operand32::Mem(addr) => {
                        mem.write(addr, &[val8])?;
                    }
                    crate::decode::Operand32::Reg(dst_r) => {
                        let dst_base = dst_r % 4;
                        let mut val32 = self.reg32(dst_base);
                        if dst_r < 4 {
                            val32 = (val32 & 0xFFFFFF00) | (val8 as u32);
                        } else {
                            val32 = (val32 & 0xFFFF00FF) | ((val8 as u32) << 8);
                        }
                        self.set_reg32(dst_base, val32);
                    }
                }
                return Ok(());
            }

            0x8A => { // MOV r8, r/m8 (从内存/寄存器读到 8 位寄存器)
                let modrm = s.fetch_modrm()?;
                let src = self.decode_rm32_operand(mem, &mut s, modrm)?;

                self.eip = s.ip;
                let val8 = match src {
                    crate::decode::Operand32::Mem(addr) => self.read_u8(mem, addr)?,
                    crate::decode::Operand32::Reg(r) => {
                        let base = r % 4;
                        if r < 4 {
                            (self.reg32(base) & 0xFF) as u8
                        } else {
                            ((self.reg32(base) >> 8) & 0xFF) as u8
                        }
                    }
                };

                let dst_r = modrm.reg;
                let dst_base = dst_r % 4;
                let mut val32 = self.reg32(dst_base);
                if dst_r < 4 {
                    val32 = (val32 & 0xFFFFFF00) | (val8 as u32);
                } else {
                    val32 = (val32 & 0xFFFF00FF) | ((val8 as u32) << 8);
                }
                self.set_reg32(dst_base, val32);
                return Ok(());
            }

            0x85 => {
                let modrm = s.fetch_modrm()?;
                let op = self.decode_rm32_operand(mem, &mut s, modrm)?; // 拆分1
                let lhs = self.read_operand32(mem, op)?;                // 拆分2
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
                    if is_16bit { s.fetch_u16()? as u32 } else { s.fetch_u32()? }
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

                if modrm.reg != 7 { // 如果不是 CMP，写回结果
                    if is_16bit {
                        match &dst {
                            crate::decode::Operand32::Mem(addr) => mem.write(*addr, &(result as u16).to_le_bytes())?,
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
            0x99 => { // CDQ (Convert Double to Quad) - 这个指令经常紧挨着 IDIV 出现，用于将 EAX 的符号位扩展到 EDX
                self.edx = if (self.eax & 0x80000000) != 0 { 0xFFFFFFFF } else { 0 };
            }

            0xF7 => { // Group 3: MUL, IMUL, DIV, IDIV, NEG, NOT, TEST
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;

                // 致命修复：TEST 指令后面跟着立即数，16位模式下只有 2 字节！
                let imm = if modrm.reg == 0 {
                    if is_16bit { s.fetch_u16()? as u32 } else { s.fetch_u32()? }
                } else { 0 };

                let val = self.read_operand32(mem, dst)?;
                self.eip = s.ip; // 释放 mem 借用

                match modrm.reg {
                    0 => { // TEST r/m32, imm32
                        let _ = self.alu_and32(val, imm);
                    }
                    2 => { // NOT r/m32
                        let result = !val;
                        self.write_operand32(mem, dst, result)?;
                    }
                    3 => { // NEG r/m32
                        let result = self.alu_sub32(0, val);
                        self.write_operand32(mem, dst, result)?;
                    }
                    4 => { // MUL r/m32 (无符号乘法，结果存入 EDX:EAX)
                        let res = (self.eax as u64).wrapping_mul(val as u64);
                        self.eax = res as u32;
                        self.edx = (res >> 32) as u32;
                    }
                    5 => { // IMUL r/m32 (有符号乘法)
                        let res = (self.eax as i32 as i64).wrapping_mul(val as i32 as i64);
                        self.eax = res as u32;
                        self.edx = (res >> 32) as u32;
                    }
                    6 => { // DIV r/m32 (无符号除法，EDX:EAX / r/m32)
                        if val == 0 { return Err(ExecError::InvalidInstruction(start, "division by zero")); }
                        let dividend = ((self.edx as u64) << 32) | (self.eax as u64);
                        let quot = dividend / (val as u64);
                        let rem = dividend % (val as u64);
                        self.eax = quot as u32;
                        self.edx = rem as u32;
                    }
                    7 => { // IDIV r/m32 (有符号除法)
                        if val == 0 { return Err(ExecError::InvalidInstruction(start, "division by zero")); }
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

            0xE8 => {
                let rel = s.fetch_i32()?;
                let return_eip = s.ip;

                // Call 指令需要更新 EIP 到目标地址
                self.eip = s.ip.wrapping_add(rel as u32);
                self.push32(mem, return_eip)?;
                return Ok(());
            }
            0xC1 | 0xD1 | 0xD3 => { // Group 2: 移位指令 (SHL, SHR, SAR, ROL, ROR)
                let modrm = s.fetch_modrm()?;
                let dst = self.decode_rm32_operand(mem, &mut s, modrm)?;
                let val = self.read_operand32(mem, dst)?;

                // 确定要移动的位数
                let count = match opcode {
                    0xC1 => s.fetch_u8()? & 0x1F,       // 立即数移位 (x86 规定最多移 31 位)
                    0xD1 => 1,                          // 固定移 1 位
                    0xD3 => (self.ecx & 0x1F) as u8,    // 根据 CL 寄存器移位
                    _ => 0,
                };

                self.eip = s.ip; // 释放 mem 借用

                if count > 0 {
                    let result = match modrm.reg {
                        0 => val.rotate_left(count as u32),  // ROL
                        1 => val.rotate_right(count as u32), // ROR
                        // RCL 和 RCR 涉及进位标志位，较复杂且这里用不到，暂时屏蔽
                        2 | 3 => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                        4 | 6 => val << count,               // SHL / SAL (逻辑/算术左移)
                        5 => val >> count,                   // SHR (逻辑右移，高位补 0)
                        7 => (val as i32 >> count) as u32,   // SAR (算术右移，高位补符号位)
                        _ => return Err(ExecError::UnimplementedOpcode(opcode, start)),
                    };
                    self.write_operand32(mem, dst, result)?;
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
                    0x1F => { // Multi-byte NOP (经常配合 0x66 用作循环对齐)
                        let modrm = s.fetch_modrm()?;
                        let _ = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0x80..=0x8F => {
                        let cc = op2 & 0x0F;
                        let rel = s.fetch_i32()? as u32;
                        if self.condition_true(cc) {
                            s.ip = s.ip.wrapping_add(rel);
                        }
                    }
                    0xAF => { // IMUL r32, r/m32 (进阶乘法，编译器优化常客)
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let lhs = self.reg32(modrm.reg);
                        let rhs = self.read_operand32(mem, src)?;
                        let res = (lhs as i32 as i64).wrapping_mul(rhs as i32 as i64);
                        self.set_reg32(modrm.reg, res as u32);
                        self.eip = s.ip;
                        return Ok(());
                    }
                    0xB6 => { // MOVZX r32, r/m8
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let byte = match src {
                            crate::decode::Operand32::Reg(r) => {
                                let base = r % 4;
                                if r < 4 { (self.reg32(base) & 0xFF) as u8 }
                                else { ((self.reg32(base) >> 8) & 0xFF) as u8 }
                            }
                            crate::decode::Operand32::Mem(addr) => self.read_u8(mem, addr)?,
                        };
                        self.eip = s.ip;
                        self.set_reg32(modrm.reg, byte as u32);
                        return Ok(());
                    }
                    0xB7 => { // MOVZX r32, r/m16
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
                    0xBE => { // MOVSX r32, r/m8
                        let modrm = s.fetch_modrm()?;
                        let src = self.decode_rm32_operand(mem, &mut s, modrm)?;
                        let byte = match src {
                            crate::decode::Operand32::Reg(r) => {
                                let base = r % 4;
                                if r < 4 { (self.reg32(base) & 0xFF) as u8 }
                                else { ((self.reg32(base) >> 8) & 0xFF) as u8 }
                            }
                            crate::decode::Operand32::Mem(addr) => self.read_u8(mem, addr)?,
                        };
                        self.eip = s.ip;
                        // 将 8 位有符号扩展为 32 位有符号
                        self.set_reg32(modrm.reg, (byte as i8) as i32 as u32);
                        return Ok(());
                    }
                    0xBF => { // MOVSX r32, r/m16
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