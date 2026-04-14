#[derive(Debug, Clone, Copy, Default)]
pub struct Flags;

impl Flags {
    pub const CF: u32 = 1 << 0;
    pub const PF: u32 = 1 << 2;
    pub const AF: u32 = 1 << 4;
    pub const ZF: u32 = 1 << 6;
    pub const SF: u32 = 1 << 7;
    pub const TF: u32 = 1 << 8;
    pub const IF: u32 = 1 << 9;
    pub const DF: u32 = 1 << 10;
    pub const OF: u32 = 1 << 11;

    pub fn get(eflags: u32, flag: u32) -> bool {
        (eflags & flag) != 0
    }

    pub fn set(eflags: &mut u32, flag: u32, value: bool) {
        if value {
            *eflags |= flag;
        } else {
            *eflags &= !flag;
        }
    }

    pub fn set_zf_sf(eflags: &mut u32, result: u32) {
        Self::set(eflags, Self::ZF, result == 0);
        Self::set(eflags, Self::SF, (result & 0x8000_0000) != 0);
        let low = (result & 0xFF) as u8;
        Self::set(eflags, Self::PF, low.count_ones() % 2 == 0);
    }

    pub fn set_logic(eflags: &mut u32, result: u32) {
        Self::set(eflags, Self::CF, false);
        Self::set(eflags, Self::OF, false);
        Self::set_zf_sf(eflags, result);
    }

    pub fn set_add32(eflags: &mut u32, lhs: u32, rhs: u32, result: u32) {
        let cf = result < lhs;
        let of = ((lhs ^ result) & (rhs ^ result) & 0x8000_0000) != 0;
        Self::set(eflags, Self::CF, cf);
        Self::set(eflags, Self::OF, of);
        Self::set_zf_sf(eflags, result);
    }

    pub fn set_sub32(eflags: &mut u32, lhs: u32, rhs: u32, result: u32) {
        let cf = lhs < rhs;
        let of = ((lhs ^ rhs) & (lhs ^ result) & 0x8000_0000) != 0;
        Self::set(eflags, Self::CF, cf);
        Self::set(eflags, Self::OF, of);
        Self::set_zf_sf(eflags, result);
    }
}
