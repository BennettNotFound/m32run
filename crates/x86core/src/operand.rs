use crate::decode::Operand32;

#[derive(Debug, Clone, Copy)]
pub enum Width {
    W8,
    W16,
    W32,
}

#[derive(Debug, Clone, Copy)]
pub enum Immediate {
    Imm8(u8),
    Imm16(u16),
    Imm32(u32),
}

#[derive(Debug, Clone, Copy)]
pub enum DecodedOperand {
    O32(Operand32),
    Imm(Immediate),
}
