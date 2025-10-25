use std::ffi::{c_int, c_void};

use std::{collections::HashMap, io::Write, ops::Neg};

use bitfield::{BitRange, BitRangeMut};

macro_rules! as_fn {
    ($exec:expr, ($($arg:ty),*) -> $ret:ty) => {
        unsafe { $exec.as_fn::<extern "C" fn($($arg),*) -> $ret>() }
    };
}

extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: c_int,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    #[cfg(target_os = "macos")]
    fn pthread_jit_write_protect_np(enabled: c_int) -> c_void;
}

pub struct ExecutableMem {
    addr: *mut c_void,
    len: usize,
}

impl ExecutableMem {
    pub unsafe fn as_fn<F>(&self) -> F {
        std::mem::transmute_copy(&self.addr)
    }
    pub fn from_bytes_copy(bytes: &[u8]) -> Self {
        // 7 = PROT_READ | PROT_WRITE | PROT_EXEC
        // 6146 = MAP_PRIVATE | MAP_ANON | MAP_JIT
        let addr = unsafe { mmap(std::ptr::null_mut(), bytes.len(), 7, 6146, -1, 0) };
        #[cfg(target_os = "macos")]
        unsafe {
            pthread_jit_write_protect_np(0)
        };
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
        #[cfg(target_os = "macos")]
        unsafe {
            pthread_jit_write_protect_np(1)
        };
        Self {
            addr,
            len: bytes.len(),
        }
    }
}

impl Drop for ExecutableMem {
    fn drop(&mut self) {
        unsafe { munmap(self.addr as *mut c_void, self.len) };
    }
}

#[derive(PartialEq, Clone, Copy, Eq, Debug)]
#[repr(u8)]
pub enum CC {
    Eq = 0,
    Neq = 1,
    Lt = 0b1011,
    Le = 0b1101,
    Gt = 0b1100,
    Ge = 0b1010,
    // unsigned higher
    Hi = 0b1000,
    // unsigned higher or same
    Hs = 0b0010,
    // unsigned lower
    Ls = 0b1001,
    Lo = 0b0011,
}

impl CC {
    pub fn invert(&self) -> Self {
        let inverted = match self {
            CC::Eq => CC::Neq,
            CC::Neq => CC::Eq,
            CC::Lt => CC::Ge,
            CC::Le => CC::Gt,
            CC::Gt => CC::Le,
            CC::Ge => CC::Lt,
            CC::Hi => CC::Ls,
            CC::Ls => CC::Hi,
            CC::Hs => CC::Lo,
            CC::Lo => CC::Hs,
        };
        assert_eq!(inverted.as_bits(), self.as_bits() ^ 1);
        inverted
    }
}

// callee-saved: x19-x28
// frame register: x29 (can't use as a general purpose register)
// link register: x30  (same)
#[derive(PartialEq, Clone, Copy, Eq, Debug)]
pub enum Reg {
    X0,
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    X8,
    X9,
    X10,
    X11,
    X28,
    X29,
    X30,

    W0,
    W1,
    W2,
    W3,
    W4,
    W5,
    W6,
    W7,
    W8,
    W9,
    W10,
    W11,
    W28,
    W29,
    W30,

    XZR,
    WZR,
    SP,
    WSP,
}

#[derive(PartialEq, Clone, Copy, Eq, Debug)]
pub enum Imm {
    U8(u8),
    I16(i16),
    U32(u32),
    U64(u64),
}

impl From<usize> for Imm {
    fn from(value: usize) -> Self {
        Imm::U64(value as u64)
    }
}

impl From<u64> for Imm {
    fn from(value: u64) -> Self {
        Imm::U64(value)
    }
}

impl Neg for Imm {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let value = self.as_isize();
        Imm::U64(-value as u64)
    }
}

impl From<u8> for Imm {
    fn from(value: u8) -> Self {
        Imm::U8(value)
    }
}
impl From<i32> for Imm {
    fn from(value: i32) -> Self {
        Imm::U64(value as u64)
    }
}

impl Imm {
    pub fn used_bits(&self) -> usize {
        let imm_value = self.as_usize();
        let used_bits = 64 - imm_value.leading_zeros() as usize;
        used_bits
    }

    pub fn as_isize(&self) -> isize {
        match self {
            Imm::U8(imm) => *imm as isize,
            Imm::I16(imm) => *imm as isize,
            Imm::U32(imm) => *imm as isize,
            Imm::U64(imm) => *imm as isize,
        }
    }

    pub fn as_usize(&self) -> usize {
        match self {
            Imm::U8(imm) => *imm as usize,
            Imm::I16(imm) => *imm as usize,
            Imm::U32(imm) => *imm as usize,
            Imm::U64(imm) => *imm as usize,
        }
    }
}

#[derive(PartialEq, Clone, Eq, Debug)]
pub struct Label(pub String);

impl From<String> for Label {
    fn from(value: String) -> Self {
        Label(value)
    }
}

impl From<&str> for Label {
    fn from(value: &str) -> Self {
        Label(value.to_string())
    }
}

impl AsRef<str> for Label {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[derive(PartialEq, Clone, Copy, Eq, Debug)]
pub enum ShiftKind {
    Lsl,
    Lsr,
    Asr,
    Reserved,
}

impl ShiftKind {
    pub fn as_asm(&self) -> &'static str {
        match self {
            ShiftKind::Lsl => "lsl",
            ShiftKind::Lsr => "lsr",
            ShiftKind::Asr => "asr",
            ShiftKind::Reserved => "reserved",
        }
    }
}

#[derive(Debug)]
pub enum Instr {
    Adr(Reg, Label),
    Br(Reg),

    Cset(Reg, CC),

    Cb(bool, Reg, Label),
    Jump(Label),
    Branch(CC, Label),

    StrImmOffset(Reg, Reg, Imm),
    StrImmPostIndex(Reg, Reg, Imm),
    StrImmPreIndex(Reg, Reg, Imm),
    Str(Reg, Reg, Reg),
    StrbPostIndex(Reg, Reg, Imm),

    StpPreIndex(Reg, Reg, Reg, Imm),
    Stp(Reg, Reg, Reg, Imm),

    Ldp(Reg, Reg, Reg, Imm),
    LdpPostIndex(Reg, Reg, Reg, Imm),

    Ldr(Reg, Reg, Reg),
    LdrImmPostIndex(Reg, Reg, Imm),
    LdrhImmPostIndex(Reg, Reg, Imm),

    Ldrb(Reg, Reg, Reg),
    LdrbImmPostIndex(Reg, Reg, Imm),
    Ldr32ImmPostIndex(Reg, Reg, Imm),
    Ldr64ImmPostIndex(Reg, Reg, Imm),

    Orr(Reg, Reg, Imm),

    AddShift(Reg, Reg, Reg, ShiftKind, Imm),
    AddImm(Reg, Reg, Imm, bool),

    Subs(Reg, Reg, Reg),
    SubsImm(Reg, Reg, Imm, bool),
    // not implementing extend or shift right now
    SubsExtReg(Reg, Reg, Reg),
    Sub(Reg, Reg, Reg),
    SubImm(Reg, Reg, Imm),

    Madd(Reg, Reg, Reg, Reg),

    Cmp(Reg, Reg),
    CmpImm(Reg, Imm, bool),

    Mov32(Reg, Reg),
    Mov64(Reg, Reg),
    MovImm(Reg, Imm),
    MovzImm(Reg, Imm, u8),
    MovkImm(Reg, Imm, u8),

    Ret,
}

const ADR: u32 = 0b0_00_10000_0000000000000000000_00000;

const CSET_32: u32 = 0b0001101010011111_0000_0_111111_00000;
const CSET_64: u32 = 0b1001101010011111_0000_0_111111_00000;

const CBZ_32: u32 = 0b00110100_0000000000000000000_00000;
const CBZ_64: u32 = 0b10110100_0000000000000000000_00000;

const CBNZ_32: u32 = 0b00110101_0000000000000000000_00000;
const CBNZ_64: u32 = 0b10110101_0000000000000000000_00000;

/// https://developer.arm.com/documentation/ddi0602/2025-03/Base-Instructions/B--Branch-?lang=en
const B: u32 = 0b000101_00000000000000000000000000;
const BR: u32 = 0b1101011000011111000000_00000_00000;
const B_COND: u32 = 0b01010100_0000000000000000000_0_0000;

/// https://developer.arm.com/documentation/ddi0602/2021-12/Base-Instructions/LDR--register---Load-Register--register--?lang=en
const LDR_32: u32 = 0b10111000011_00000_000_0_10_00000_00000;
const LDR_64: u32 = 0b11111000011_00000_000_0_10_00000_00000;

/// https://developer.arm.com/documentation/ddi0602/2021-12/Base-Instructions/LDR--immediate---Load-Register--immediate--?lang=en
const LDR_IMM_POST_INDEX_32: u32 = 0b10111000010_000000000_01_00000_00000;
const LDR_IMM_POST_INDEX_64: u32 = 0b11111000010_000000000_01_00000_00000;

const LDRH_IMM_POST_INDEX_32: u32 = 0b01111000010_000000000_01_00000_00000;

const LDRB_IMM_POST_INDEX: u32 = 0b00111000010_000000000_0_1_00000_00000;
// const LDR_IMM_POST_INDEX: u32 = 0b01001000010_00000000000000000000000_00000;
const LDR32_IMM_POST_INDEX: u32 = 0b10111000010_000000000_01_00000_00000;
const LDR64_IMM_POST_INDEX: u32 = 0b11111000010_000000000_01_00000_00000;

const LDP_POST_INDEX_32: u32 = 0b0010100011_0000000_00000_00000_00000;
const LDP_POST_INDEX_64: u32 = 0b1010100011_0000000_00000_00000_00000;

const LDP_32: u32 = 0b0_010100101_0000000_00000_00000_00000;
const LDP_64: u32 = 0b1_010100101_0000000_00000_00000_00000;

const ORR_32: u32 = 0b0_01100100_0_000000_000000_00000_00000;
const ORR_64: u32 = 0b1_01100100_0_000000_000000_00000_00000;

const STR_32: u32 = 0b10111000001_00000_000_0_10_00000_00000;
const STR_64: u32 = 0b11111000001_00000_000_0_10_00000_00000;

const STR_OFFSET_32: u32 = 0b1_0_111_00100_000000000000_00000_00000;
const STR_OFFSET_64: u32 = 0b1_1_111_00100_000000000000_00000_00000;

const STR_PRE_INDEX_32: u32 = 0b1_0_111_000000_000000000_11_00000_00000;
const STR_PRE_INDEX_64: u32 = 0b1_1_111_000000_000000000_11_00000_00000;

const STR_POST_INDEX_32: u32 = 0b1_0_111_000000_000000000_01_00000_00000;
const STR_POST_INDEX_64: u32 = 0b1_1_111_000000_000000000_01_00000_00000;

const STP_PRE_INDEX_32: u32 = 0b0_0_10_100_110_0000000_00000_00000_00000;
const STP_PRE_INDEX_64: u32 = 0b1_0_10_100_110_0000000_00000_00000_00000;

const STP_OFFSET_32: u32 = 0b0_010100100_0000000_00000_00000_00000;
const STP_OFFSET_64: u32 = 0b1_010100100_0000000_00000_00000_00000;

const STRB_POST_INDEX: u32 = 0b00111000000_000000000_01_00000_00000;

const ADD_EXTEND_32: u32 = 0b000100010_0_000000000000_00000_00000;
const ADD_EXTEND_64: u32 = 0b10001011001_00000_000_000_00000_00000;

const ADD_SHIFT_32: u32 = 0b00001011_00_0_00000_000000_00000_00000;
const ADD_SHIFT_64: u32 = 0b10001011_00_0_00000_000000_00000_00000;

/// https://developer.arm.com/documentation/ddi0602/2021-12/Base-Instructions/ADD--immediate---Add--immediate--?lang=en
const ADD_IMM_32: u32 = 0b000100010_0_000000000000_00000_00000;
const ADD_IMM_64: u32 = 0b100100010_0_000000000000_00000_00000;

/// https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/SUBS--shifted-register---Subtract--shifted-register---setting-flags-
const SUBS_32: u32 = 0b01101011_00_0_000_00000_0000000_000000;
const SUBS_64: u32 = 0b11101011_00_0_00000_000000_00000_00000;

/// https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/SUBS--extended-register---Subtract--extended-register---setting-flags-
///
/// Works with SP as source operand
const SUBS_EXT_REG_32: u32 = 0b0_1101011001_00000_000_000_00000_00000;
const SUBS_EXT_REG_64: u32 = 0b1_1101011001_00000_000_000_00000_00000;

/// https://developer.arm.com/documentation/ddi0602/2021-12/Base-Instructions/SUBS--immediate---Subtract--immediate---setting-flags-?lang=en
const SUBS_IMM_32: u32 = 0b011100010_0_000000000000_00000_00000;
const SUBS_IMM_64: u32 = 0b111100010_0_000000000000_00000_00000;

const SUB_IMM_32: u32 = 0b010100010_0_000000000000_00000_00000;
const SUB_IMM_64: u32 = 0b110100010_0_000000000000_00000_00000;

const MADD_32: u32 = 0b00011011000_00000_0_00000_00000_00000;
const MADD_64: u32 = 0b10011011000_00000_0_00000_00000_00000;

// https://developer.arm.com/documentation/ddi0602/2021-12/Base-Instructions/CMP--immediate---Compare--immediate---an-alias-of-SUBS--immediate--?lang=en
const CMP_IMM_32: u32 = 0b011100010_0_000000000000_00000_11111;
const CMP_IMM_64: u32 = 0b111100010_0_000000000000_00000_11111;

const MOV_32: u32 = 0b00101010000_00000_000000_11111_00000;
const MOV_64: u32 = 0b10101010000_00000_000000_11111_00000;
const MOV_IMM_32: u32 = 0b0011001000_000000_000000_11111_00000;
const MOV_IMM_64: u32 = MOVZ_IMM_64;
const MOVK_IMM_64: u32 = 0b111100101_00_0000000000000000_00000;

const MOVZ_IMM_32: u32 = 0b010100101_00_0000000000000000_00000;
const MOVZ_IMM_64: u32 = 0b110100101_00_0000000000000000_00000;

pub struct Assembler {
    pub used_registers: Vec<Reg>,
    pub instrs: Vec<Instr>,
    pub labels: HashMap<String, usize>,
    /// The code uses the stack, we have to do the following things:
    /// 1. Include function prologue:
    ///   ```asm
    ///   # store old frame pointer and LR onto stack (LR technically not needed right
    ///   now since we don't make calls)
    ///   stp   x29, x30, [sp, #-16]!
    ///   # new frame pointer
    ///   mov   x29, sp
    ///   ```
    ///
    /// 2. Include function epilogue before returning:
    /// ```asm
    /// mov sp, x29
    /// ldp x29, x30 [sp], #16
    /// ret
    /// ```
    ///
    /// 3. Ensure all stack allocations are 16 byte aligned
    pub uses_stack: bool,
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            used_registers: vec![Reg::X0, Reg::X1],
            instrs: vec![],
            labels: HashMap::new(),
            uses_stack: false,
        }
    }

    pub fn prologue(&mut self) {
        use Reg::*;
        self.str_imm_post_index(X29, SP, 0.into());
        self.mov(X29, SP);
    }

    pub fn epilogue(&mut self) {
        use Reg::*;
        self.ldr_imm_post_index(X29, SP, Imm::U32(16));
        self.ret();
    }

    pub fn new_label(&mut self, name_hint: &str) -> String {
        let label = if !name_hint.is_empty() {
            let mut label = format!("_{}", name_hint);
            let mut i = 1;
            while self.labels.contains_key(&label) {
                label = format!("_{}_{}", label, i);
                i += 1;
            }
            label
        } else {
            format!("label_{}", self.labels.len())
        };
        self.labels.insert(label.clone(), usize::MAX);
        label
    }

    pub fn label(&mut self, label: &str) {
        // Optimization: remove instructions which jump to a label which immediately follows them
        // Example:
        // ```asm
        // bne mylabel # <--- not necessary!
        // .Lmylabel:
        // mov x0, #1
        //```
        if let Some(last_instr) = self.instrs.last() {
            match last_instr {
                Instr::Jump(l) | Instr::Branch(_, l) | Instr::Cb(_, _, l) => {
                    if label == l.as_ref() {
                        self.instrs.pop();
                    }
                }
                Instr::Br(r) => {}
                Instr::Adr(_, l) => {
                    // TODO: Can we omit this? It owuld
                }
                Instr::Stp(reg, reg1, reg2, imm) => {}
                Instr::SubsExtReg(reg, reg1, reg2) => {}
                Instr::Orr(reg, reg1, imm) => {}
                Instr::StrImmOffset(reg, reg1, imm) => {}
                Instr::StrImmPostIndex(reg, reg1, imm) => {}
                Instr::StrImmPreIndex(reg, reg1, imm) => {}
                Instr::StpPreIndex(reg, reg1, reg2, imm) => {}
                Instr::Ldp(reg, reg1, reg2, imm) => {}
                Instr::LdpPostIndex(reg, reg1, reg2, imm) => {}
                Instr::Cset(reg, cc) => {}
                Instr::Str(reg, reg1, reg2) => {}
                Instr::StrbPostIndex(reg, reg1, imm) => {}
                Instr::Ldr(reg, reg1, reg2) => {}
                Instr::LdrImmPostIndex(reg, reg1, imm) => {}
                Instr::LdrhImmPostIndex(reg, reg1, imm) => {}
                Instr::Ldrb(reg, reg1, reg2) => {}
                Instr::LdrbImmPostIndex(reg, reg1, imm) => {}
                Instr::Ldr32ImmPostIndex(reg, reg1, imm) => {}
                Instr::Ldr64ImmPostIndex(reg, reg1, imm) => {}
                Instr::AddShift(reg, reg1, reg2, shift_kind, imm) => {}
                Instr::AddImm(reg, reg1, imm, _) => {}
                Instr::Subs(reg, reg1, reg2) => {}
                Instr::SubsImm(reg, reg1, imm, _) => {}
                Instr::Sub(reg, reg1, reg2) => {}
                Instr::SubImm(reg, reg1, imm) => {}
                Instr::Madd(reg, reg1, reg2, reg3) => {}
                Instr::Cmp(reg, reg1) => {}
                Instr::CmpImm(reg, imm, _) => {}
                Instr::Mov32(reg, reg1) => {}
                Instr::Mov64(reg, reg1) => {}
                Instr::MovImm(reg, imm) => {}
                Instr::MovzImm(reg, imm, _) => {}
                Instr::MovkImm(reg, imm, _) => {}
                Instr::Ret => {}
            }
        }
        self.labels.insert(label.to_string(), self.instrs.len());
    }

    pub fn alloc_reg(&mut self, size: usize) -> Option<Reg> {
        if size == 4 {
            for idx in 0..=10 {
                let reg = Reg::from_idx_32bit(idx);
                let reg64 = Reg::from_idx_64bit(idx);
                if !self.used_registers.contains(&reg) && !self.used_registers.contains(&reg64) {
                    self.used_registers.push(reg);
                    return Some(reg);
                }
            }
            return None;
        }

        if size == 8 {
            for idx in 0..=10 {
                let reg = Reg::from_idx_64bit(idx);
                let reg32 = Reg::from_idx_32bit(idx);
                if !self.used_registers.contains(&reg) && !self.used_registers.contains(&reg32) {
                    self.used_registers.push(reg);
                    return Some(reg);
                }
            }
            return None;
        }

        None
    }

    pub fn dealloc_reg(&mut self, reg: Reg) {
        self.used_registers.retain(|r| r.idx() != reg.idx());
    }

    pub fn cset(&mut self, reg: Reg, cc: CC) {
        self.instrs.push(Instr::Cset(reg, cc));
    }

    pub fn cbz(&mut self, reg: Reg, label: Label) {
        self.instrs.push(Instr::Cb(true, reg, label));
    }

    pub fn cbnz(&mut self, reg: Reg, label: Label) {
        self.instrs.push(Instr::Cb(false, reg, label));
    }

    pub fn cmp(&mut self, left: Reg, right: Reg) {
        // self.instrs.push(Instr::Cmp(left, right));
        let zero_reg = if left.is_64bit() { Reg::XZR } else { Reg::WZR };
        self.instrs.push(Instr::Subs(zero_reg, left, right));
    }

    pub fn ldr(&mut self, out: Reg, base: Reg, offset: Reg) {
        self.instrs.push(Instr::Ldr(out, base, offset));
    }

    pub fn ldr_imm_post_index(&mut self, out: Reg, base: Reg, offset: Imm) {
        assert!(offset.as_isize() < 256);
        assert!(offset.as_isize() >= -256);
        self.instrs.push(Instr::LdrImmPostIndex(out, base, offset));
    }

    pub fn ldrh_imm_post_index(&mut self, out: Reg, base: Reg, offset: Imm) {
        assert!(out.is_32bit());
        assert!(base.is_64bit());

        assert!(offset.as_isize() < 256);
        assert!(offset.as_isize() >= -256);
        self.instrs.push(Instr::LdrhImmPostIndex(out, base, offset));
    }

    // pub fn ldr_post_index(&mut self, out: Reg, base: Reg, offset: Imm) {
    //     self.instrs.push(Instr::LdrPostIndex(out, base, offset));
    // }

    pub fn ldrb_post_index(&mut self, out: Reg, base: Reg, offset: Imm) {
        assert!(offset.as_isize() < 256);
        assert!(offset.as_isize() >= -256);
        self.instrs.push(Instr::LdrbImmPostIndex(out, base, offset));
    }

    pub fn mul(&mut self, out: Reg, a: Reg, b: Reg) {
        assert_eq!(out.is_64bit(), a.is_64bit());
        assert_eq!(a.is_64bit(), b.is_64bit());

        let zero_reg = if out.is_64bit() { Reg::XZR } else { Reg::WZR };
        self.instrs.push(Instr::Madd(out, a, b, zero_reg));
    }

    pub fn add(&mut self, out: Reg, a: Reg, b: Reg) {
        assert_eq!(out.is_32bit(), a.is_32bit());
        assert_eq!(a.is_32bit(), b.is_32bit());

        self.instrs
            .push(Instr::AddShift(out, a, b, ShiftKind::Lsl, Imm::U32(0)));
    }

    pub fn add_imm(&mut self, out: Reg, a: Reg, b: Imm) {
        assert_eq!(out.is_32bit(), a.is_32bit());
        let val = b.as_usize();
        assert!(val <= 4095);
        self.instrs.push(Instr::AddImm(out, a, b, false));
    }

    pub fn ldp(&mut self, out1: Reg, out2: Reg, base: Reg, offset: Imm) {
        assert_eq!(out1.is_64bit(), out2.is_64bit());
        assert!(base.is_64bit());

        if out1.is_32bit() {
            assert!(offset.as_isize() <= 252);
            assert!(offset.as_isize() >= -256);
        } else {
            assert!(offset.as_isize() <= 504);
            assert!(offset.as_isize() >= -512);
        }

        self.instrs.push(Instr::Ldp(out1, out2, base, offset));
    }

    pub fn ldp_imm_post_index(&mut self, out1: Reg, out2: Reg, base: Reg, offset: Imm) {
        assert_eq!(out1.is_64bit(), out2.is_64bit());
        assert!(base.is_64bit());
        if out1.is_32bit() {
            assert!(offset.as_isize() <= 252);
            assert!(offset.as_isize() >= -256);
        } else {
            assert!(offset.as_isize() <= 504);
            assert!(offset.as_isize() >= -512);
        }
        self.instrs
            .push(Instr::LdpPostIndex(out1, out2, base, offset));
    }

    pub fn stp(&mut self, a: Reg, b: Reg, dest: Reg, offset: Imm) {
        assert_ne!(a, Reg::SP);
        assert_ne!(b, Reg::SP);
        assert_eq!(a.is_64bit(), b.is_64bit());
        assert!(dest.is_64bit());

        if a.is_32bit() {
            assert!(offset.as_isize() <= 252);
            assert!(offset.as_isize() >= -256);
        } else {
            assert!(offset.as_isize() <= 504);
            assert!(offset.as_isize() >= -512);
        }

        self.instrs.push(Instr::Stp(a, b, dest, offset));
    }

    pub fn stp_imm_pre_index(&mut self, out1: Reg, out2: Reg, base: Reg, offset: Imm) {
        assert_eq!(out1.is_64bit(), out2.is_64bit());
        assert!(base.is_64bit());
        let offset_val = offset.as_isize();
        if out1.is_32bit() {
            assert!(offset_val <= 252);
            assert!(offset_val >= -256);
        } else {
            assert!(offset_val <= 504);
            assert!(offset_val >= -512);
        }
        self.instrs
            .push(Instr::StpPreIndex(out1, out2, base, offset));
    }

    pub fn subs_ext_reg(&mut self, out: Reg, a: Reg, b: Reg) {
        if out.is_32bit() {
            assert_eq!(a.is_32bit(), b.is_32bit());
        } else {
            assert!(a.is_64bit());
        }
        assert_ne!(out, Reg::SP);
        assert_ne!(b, Reg::SP);
        self.instrs.push(Instr::SubsExtReg(out, a, b));
    }

    pub fn subs_imm(&mut self, out: Reg, a: Reg, b: Imm) {
        let val = b.as_usize();
        assert!(val <= 4095);
        self.instrs.push(Instr::SubsImm(out, a, b, false));
    }

    pub fn subs_imm_shifted(&mut self, out: Reg, a: Reg, b: Imm) {
        let val = b.as_usize();
        assert!(val <= 4095);
        self.instrs.push(Instr::SubsImm(out, a, b, true));
    }

    pub fn sub_imm(&mut self, out: Reg, a: Reg, b: Imm) {
        let val = b.as_usize();
        assert!(val <= 4095);
        self.instrs.push(Instr::SubImm(out, a, b));
    }

    pub fn str(&mut self, source: Reg, base: Reg, offset: Reg) {
        self.instrs.push(Instr::Str(source, base, offset));
    }

    pub fn str_imm_offset(&mut self, source: Reg, base: Reg, offset: Imm) {
        assert!(base.is_64bit());
        if source.is_32bit() {
            assert!(offset.as_usize() <= 16380);
            assert!(offset.as_usize() % 4 == 0);
        } else {
            assert!(offset.as_usize() <= 32760);
            assert!(offset.as_usize() % 8 == 0);
        }
        self.instrs.push(Instr::StrImmOffset(source, base, offset));
    }

    pub fn str_imm_post_index(&mut self, source: Reg, base: Reg, offset: Imm) {
        assert_ne!(source, Reg::SP);
        assert!(base.is_64bit());
        let offset_val = offset.as_isize();
        println!("offset_val: {}", offset_val);
        assert!(offset_val <= 255);
        assert!(offset_val >= -256);
        self.instrs.push(Instr::StrImmPostIndex(
            source,
            base,
            Imm::U64(offset_val as u64),
        ));
    }

    pub fn str_imm_pre_index(&mut self, source: Reg, base: Reg, offset: Imm) {
        assert_ne!(source, Reg::SP);
        assert!(base.is_64bit());
        let offset_val = offset.as_isize();
        println!("offset_val: {}", offset_val);
        assert!(offset_val <= 255);
        assert!(offset_val >= -256);
        self.instrs.push(Instr::StrImmPreIndex(
            source,
            base,
            Imm::U64(offset_val as u64),
        ));
    }

    pub fn strb_post_index(&mut self, out: Reg, base: Reg, offset: Imm) {
        assert!(offset.as_isize() < 256);
        assert!(offset.as_isize() >= -256);
        self.instrs.push(Instr::StrbPostIndex(out, base, offset));
    }

    pub fn adr(&mut self, out: Reg, label: Label) {
        self.instrs.push(Instr::Adr(out, label));
    }

    pub fn jump(&mut self, label: Label) {
        self.instrs.push(Instr::Jump(label));
    }

    pub fn branch_register(&mut self, reg: Reg) {
        self.instrs.push(Instr::Br(reg));
    }

    pub fn branch(&mut self, cc: CC, label: Label) {
        self.instrs.push(Instr::Branch(cc, label));
    }

    pub fn imm_needs_mov(&self, imm: Imm) -> bool {
        let imm_value = imm.as_usize();
        imm_value > 0xFFF || (imm_value >> 12) << 12 != imm_value
    }

    pub fn cmp_imm(&mut self, left: Reg, right: Imm) {
        let zero_reg = if left.is_64bit() { Reg::XZR } else { Reg::WZR };
        let imm_value = right.as_usize();
        // 0 - 4095 (0xFFF) is the range of the immediate
        if imm_value <= 0xFFF {
            // self.instrs.push(Instr::CmpImm(left, right, false));
            self.subs_imm(zero_reg, left, right);
            return;
        }
        // try the shifted form, only allows 12 bits shift
        if (imm_value >> 12) << 12 == imm_value {
            // self.instrs.push(Instr::CmpImm(
            //     left,
            //     Imm::U32((imm_value >> 12) as u32),
            //     true,
            // ));

            self.subs_imm_shifted(zero_reg, left, right);
            return;
        }
        // going to have to load it in a register 16 bits at a time
        let used_bits = right.used_bits();
        let reg = if used_bits > 32 {
            self.alloc_reg(8).unwrap()
        } else {
            self.alloc_reg(4).unwrap()
        };
        self.cmp_imm_mov(left, right, reg);
        self.dealloc_reg(reg);
    }

    pub fn cmp_imm_mov(&mut self, left: Reg, imm: Imm, temp_reg: Reg) {
        assert!(
            imm.used_bits() > 32 && temp_reg.is_64bit()
                || imm.used_bits() <= 32 && temp_reg.is_32bit()
        );
        self.mov_imm(temp_reg, imm);
        self.cmp(left, temp_reg);
    }

    pub fn orr(&mut self, dest: Reg, src: Reg, imm: Imm) {
        assert_eq!(dest.is_64bit(), src.is_64bit());
        self.instrs.push(Instr::Orr(dest, src, imm));
    }

    pub fn mov(&mut self, out: Reg, inp: Reg) {
        // SP and ZR have the same encoding and is not allowed here.
        assert_ne!(out, Reg::SP);
        assert_ne!(inp, Reg::SP);
        if out.is_64bit() {
            assert!(inp.is_64bit());
            self.instrs.push(Instr::Mov64(out, inp));
        } else {
            assert!(!inp.is_64bit());
            self.instrs.push(Instr::Mov32(out, inp));
        }
    }

    pub fn mov_imm(&mut self, reg: Reg, imm: Imm) {
        let mut val = imm.as_usize();
        if val < 0x10000 {
            self.instrs.push(Instr::MovImm(reg, imm));
            return;
        }

        let mut hw = 0;
        while val != 0 {
            println!("val: {}", val);
            assert!(hw < 4);
            let instr = if hw == 0 {
                Instr::MovzImm(reg, Imm::U32((val & 0xFFFF) as u32), hw)
            } else {
                Instr::MovkImm(reg, Imm::U32((val & 0xFFFF) as u32), hw)
            };
            self.instrs.push(instr);
            val = val >> 16;
            hw += 1;
        }
    }

    pub fn ret(&mut self) {
        self.instrs.push(Instr::Ret);
    }

    pub fn emit(&mut self) -> Vec<u8> {
        println!("instrs:\n{}", self.as_asm());
        for (label, idx) in self.labels.iter() {
            if *idx == usize::MAX {
                panic!("Label not patched: {}", label);
            }
        }
        let mut outbuf = vec![];
        for (idx, instr) in self.instrs.iter().enumerate() {
            self.emit_binary(instr, idx, &mut outbuf).unwrap();
        }
        outbuf
    }

    pub fn as_asm(&self) -> String {
        let mut output = String::new();

        for (idx, instr) in self.instrs.iter().enumerate() {
            // Check if this instruction position has a label
            for (label_name, &label_idx) in &self.labels {
                if label_idx == idx {
                    output.push_str(&format!(".L{}:\n", label_name));
                }
            }

            // Add the instruction with proper indentation
            output.push_str(&format!("\t{}\n", instr.as_asm()));
        }

        output
    }

    pub fn emit_binary(
        &self,
        instr: &Instr,
        idx: usize,
        mut outbuf: impl Write,
    ) -> std::io::Result<()> {
        match instr {
            Instr::Br(reg) => {
                assert!(reg.is_64bit());

                let mut instr_bits = BR;
                instr_bits.set_bit_range(9, 5, reg.as_bits());
                outbuf.write_all(&instr_bits.to_le_bytes())
            }
            Instr::Adr(reg, l) => {
                assert!(reg.is_64bit());
                let mut instr_bits = ADR;

                let label_index = l.as_bits(idx, &self.labels) as i64;
                println!("label_index: {}", label_index);

                let label_bits_signed: i64 = label_index * 4;
                // Must be in the range of +/-1mb
                assert!(label_bits_signed <= (1 << 20));
                assert!(label_bits_signed >= -(1 << 20));

                let label_bits = (label_index * 4) as u64;
                instr_bits.set_bit_range(4, 0, reg.as_bits());
                let low_two_bits: u32 = label_bits.bit_range(1, 0);
                instr_bits.set_bit_range(30, 29, low_two_bits);
                let rest_of_bits: u32 = label_bits.bit_range(20, 2);
                instr_bits.set_bit_range(23, 5, rest_of_bits);

                outbuf.write_all(&instr_bits.to_le_bytes())
            }
            Instr::Cset(reg, cc) => {
                let mut instr_bits = if reg.is_64bit() { CSET_64 } else { CSET_32 };
                instr_bits.set_bit_range(4, 0, reg.as_bits());
                let cc_bits = cc.invert().as_bits();
                // let cc_bits = cc.as_bits();
                instr_bits.set_bit_range(15, 12, cc_bits);

                outbuf.write_all(&instr_bits.to_le_bytes())
            }
            Instr::Cb(zero, reg, label) => {
                let mut instr_bits = if *zero {
                    if reg.is_64bit() {
                        CBZ_64
                    } else {
                        CBZ_32
                    }
                } else {
                    if reg.is_64bit() {
                        CBNZ_64
                    } else {
                        CBNZ_32
                    }
                };
                instr_bits.set_bit_range(4, 0, reg.as_bits());
                let label_bits = label.as_bits(idx, &self.labels);
                instr_bits.set_bit_range(23, 5, label_bits);
                outbuf.write_all(&instr_bits.to_le_bytes())?;
                Ok(())
            }
            Instr::Jump(label) => {
                let mut instrbits = B;
                instrbits.set_bit_range(25, 0, label.as_bits(idx, &self.labels));
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::Branch(cc, label) => {
                let mut instrbits = B_COND;
                instrbits.set_bit_range(3, 0, cc.as_bits());
                instrbits.set_bit_range(23, 5, label.as_bits(idx, &self.labels));

                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::StrImmOffset(reg, reg1, imm) => {
                let mut instrbits = if reg.is_64bit() {
                    STR_OFFSET_64
                } else {
                    STR_OFFSET_32
                };

                let mut imm_value = imm.as_usize();
                if reg.is_32bit() {
                    assert!(imm_value <= 16380);
                    imm_value = imm_value / 4;
                } else {
                    assert!(imm_value <= 32760);
                    imm_value = imm_value / 8;
                }

                instrbits.set_bit_range(4, 0, reg.as_bits());
                instrbits.set_bit_range(9, 5, reg1.as_bits());
                instrbits.set_bit_range(21, 10, imm_value as u32);
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::StrImmPreIndex(src, base, offset) => {
                let mut instrbits = if base.is_64bit() {
                    STR_PRE_INDEX_64
                } else {
                    STR_PRE_INDEX_32
                };

                let imm_value = offset.as_isize();
                assert!(imm_value <= 255);
                assert!(imm_value >= -256);

                instrbits.set_bit_range(4, 0, src.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 12, imm_value as u32);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::StrImmPostIndex(src, base, imm) => {
                assert!(base.is_64bit());

                let mut instrbits = if base.is_64bit() {
                    STR_POST_INDEX_64
                } else {
                    STR_POST_INDEX_32
                };

                let value = imm.as_isize();
                assert!(value <= 255);
                assert!(value >= -256);

                instrbits.set_bit_range(4, 0, src.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 12, value as u32);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Ldp(a, b, src, imm) => {
                assert_eq!(a.is_64bit(), b.is_64bit());
                assert!(src.is_64bit());

                let mut imm_value = imm.as_isize();

                if a.is_64bit() {
                    assert_eq!(imm_value % 8, 0);
                    assert!(imm_value <= 504);
                    assert!(imm_value >= -512);
                    imm_value = imm_value / 8;
                } else {
                    assert_eq!(imm_value % 4, 0);
                    assert!(imm_value <= 252);
                    assert!(imm_value >= -256);
                    imm_value = imm_value / 4;
                }

                let mut instrbits = if a.is_64bit() { LDP_64 } else { LDP_32 };
                instrbits.set_bit_range(4, 0, a.as_bits());
                instrbits.set_bit_range(9, 5, src.as_bits());
                instrbits.set_bit_range(14, 10, b.as_bits());
                instrbits.set_bit_range(21, 15, imm_value as u64);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::LdpPostIndex(a, b, dest, imm) => {
                assert_eq!(a.is_64bit(), b.is_64bit());
                assert!(dest.is_64bit());

                let mut imm_value = imm.as_isize();
                if a.is_64bit() {
                    assert_eq!(imm_value % 8, 0);
                    assert!(imm_value <= 504);
                    assert!(imm_value >= -512);
                    imm_value = imm_value / 8;
                } else {
                    assert_eq!(imm_value % 4, 0);
                    assert!(imm_value <= 252);
                    assert!(imm_value >= -256);
                    imm_value = imm_value / 4;
                }

                let mut instrbits = if a.is_64bit() {
                    LDP_POST_INDEX_64
                } else {
                    LDP_POST_INDEX_32
                };

                instrbits.set_bit_range(4, 0, a.as_bits());
                instrbits.set_bit_range(9, 5, dest.as_bits());
                instrbits.set_bit_range(14, 10, b.as_bits());
                instrbits.set_bit_range(21, 15, imm_value as u64);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Orr(dest, src, imm) => {
                assert_eq!(src.is_64bit(), dest.is_64bit());
                let mut instrbits = if dest.is_64bit() { ORR_64 } else { ORR_32 };

                let imm_value = imm.as_usize();
                assert_eq!(imm_value & 0b1111111111111, imm_value);

                instrbits.set_bit_range(4, 0, dest.as_bits());
                instrbits.set_bit_range(9, 5, src.as_bits());

                // instrbits.set_bit_range(21, 10, imm.as_usize() as u32);
                // instrbits.set_bit_range(22, 22, imm.as_usize() as u32);
                instrbits.set_bit_range(21, 10, u32::MAX);
                if src.is_64bit() {
                    instrbits.set_bit_range(22, 22, u32::MAX);
                }

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Stp(a, b, dest, imm) => {
                assert_ne!(*a, Reg::SP);
                assert_ne!(*b, Reg::SP);
                assert_eq!(a.is_64bit(), b.is_64bit());
                assert!(dest.is_64bit());

                let imm_value = imm.as_isize();
                let imm_value = if a.is_64bit() {
                    assert_eq!(imm_value % 8, 0);
                    assert!(imm_value <= 504);
                    assert!(imm_value >= -512);
                    imm_value / 8
                } else {
                    assert_eq!(imm_value % 4, 0);
                    assert!(imm_value <= 252);
                    assert!(imm_value >= -256);
                    imm_value / 4
                };

                let mut instrbits = if a.is_64bit() {
                    STP_OFFSET_64
                } else {
                    STP_OFFSET_32
                };

                instrbits.set_bit_range(4, 0, a.as_bits());
                instrbits.set_bit_range(9, 5, dest.as_bits());
                instrbits.set_bit_range(14, 10, b.as_bits());
                instrbits.set_bit_range(21, 15, imm_value as u64);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::StpPreIndex(a, b, dest, imm) => {
                assert_eq!(a.is_64bit(), b.is_64bit());
                assert!(dest.is_64bit());

                let mut imm_value = imm.as_isize();

                if a.is_32bit() {
                    assert_eq!(imm_value % 4, 0);
                    assert!(imm_value <= 252);
                    assert!(imm_value >= -256);
                    imm_value = imm_value / 4;
                } else {
                    assert_eq!(imm_value % 8, 0);
                    assert!(imm_value <= 504);
                    assert!(imm_value >= -512);
                    imm_value = imm_value / 8;
                }

                let mut instrbits = if a.is_64bit() {
                    STP_PRE_INDEX_64
                } else {
                    STP_PRE_INDEX_32
                };
                instrbits.set_bit_range(4, 0, a.as_bits());
                instrbits.set_bit_range(9, 5, dest.as_bits());
                instrbits.set_bit_range(14, 10, b.as_bits());
                instrbits.set_bit_range(21, 15, imm_value as u64);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Str(source, base, offset) => {
                let mut instrbits = if source.is_64bit() { STR_64 } else { STR_32 };
                instrbits.set_bit_range(4, 0, source.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 16, offset.as_bits());
                instrbits.set_bit_range(15, 13, 0b011);
                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::StrbPostIndex(out, base, offset) => {
                let value = offset.as_isize();
                assert!(value < 256);
                assert!(value >= -256);
                let mut instrbits = STRB_POST_INDEX;
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 12, value as u32);
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::LdrhImmPostIndex(out, base, offset) => {
                assert!(out.is_32bit());
                assert!(base.is_64bit());

                let mut instrbits = LDRH_IMM_POST_INDEX_32;
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 12, offset.as_isize() as u32);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::LdrImmPostIndex(out, base, offset) => {
                let mut instrbits = if out.is_64bit() {
                    LDR_IMM_POST_INDEX_64
                } else {
                    LDR_IMM_POST_INDEX_32
                };

                let offset_value = offset.as_isize();
                assert!(offset_value < 256);
                assert!(offset_value >= -256);

                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 12, offset_value as u32);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Ldr(out, base, offset) => {
                assert!(base.is_64bit());

                let mut instrbits = if out.is_64bit() { LDR_64 } else { LDR_32 };

                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 16, offset.as_bits());
                instrbits.set_bit_range(12, 12, 0b0);
                instrbits.set_bit_range(15, 13, 0b011);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Ldrb(out, base, offset) => {
                todo!()
            }
            Instr::LdrbImmPostIndex(out, base, offset) => {
                assert!(offset.as_isize() < 256);
                assert!(offset.as_isize() >= -256);
                let out_bits = out.as_bits();
                let base_bits = base.as_bits();
                let mut instrbits = LDRB_IMM_POST_INDEX;
                instrbits.set_bit_range(4, 0, out_bits);
                instrbits.set_bit_range(9, 5, base_bits);
                instrbits.set_bit_range(20, 12, offset.as_isize() as u32);

                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::AddImm(out, a, b, is_shifted) => {
                assert_eq!(out.is_32bit(), a.is_32bit());
                let mut instrbits = if out.is_32bit() {
                    ADD_IMM_32
                } else {
                    ADD_IMM_64
                };
                let imm_value = b.as_usize();
                assert!(imm_value <= 4095);
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, a.as_bits());
                instrbits.set_bit_range(21, 10, b.as_usize() as u32);
                if *is_shifted {
                    instrbits.set_bit_range(22, 22, 1);
                }
                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::AddShift(out, a, b, shift, imm) => {
                assert!(
                    (out.is_64bit() && a.is_64bit() && b.is_64bit())
                        || (out.is_32bit() && a.is_32bit() && b.is_32bit())
                );

                let val = imm.as_usize();
                if out.is_64bit() {
                    assert!(val <= 63);
                } else {
                    assert!(val <= 31);
                }

                let mut instrbits = if out.is_64bit() {
                    ADD_SHIFT_64
                } else {
                    ADD_SHIFT_32
                };

                use ShiftKind::*;
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, a.as_bits());
                instrbits.set_bit_range(20, 16, b.as_bits());
                instrbits.set_bit_range(
                    23,
                    22,
                    match *shift {
                        Lsl => 0b00,
                        Lsr => 0b01,
                        Asr => 0b10,
                        Reserved => 0b11,
                    },
                );
                instrbits.set_bit_range(15, 10, val as u32);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::SubsExtReg(out, a, b) => {
                if out.is_32bit() {
                    assert_eq!(a.is_32bit(), b.is_32bit());
                } else {
                    assert!(a.is_64bit());
                }

                let mut instrbits = if out.is_64bit() {
                    SUBS_EXT_REG_64
                } else {
                    SUBS_EXT_REG_32
                };

                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, a.as_bits());
                instrbits.set_bit_range(20, 16, b.as_bits());
                if b.is_64bit() {
                    instrbits.set_bit_range(15, 13, 0b011);
                } else {
                    instrbits.set_bit_range(15, 13, 0b010);
                }

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Subs(out, a, b) => {
                assert!(
                    (out.is_64bit() && a.is_64bit() && b.is_64bit())
                        || (out.is_32bit() && a.is_32bit() && b.is_32bit())
                );
                assert_ne!(*out, Reg::SP);
                assert_ne!(*a, Reg::SP);
                assert_ne!(*b, Reg::SP);
                let mut instrbits = if out.is_64bit() { SUBS_64 } else { SUBS_32 };
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, a.as_bits());
                instrbits.set_bit_range(20, 16, b.as_bits());
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::SubsImm(out, a, b, is_shifted) => {
                assert!((out.is_64bit() && a.is_64bit()) || (out.is_32bit() && a.is_32bit()));
                let mut instrbits = if out.is_64bit() {
                    SUBS_IMM_64
                } else {
                    SUBS_IMM_32
                };

                let b = b.as_usize();
                assert!(b <= 4095);

                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, a.as_bits());
                instrbits.set_bit_range(21, 10, b as u32);
                if *is_shifted {
                    instrbits.set_bit_range(22, 22, 1);
                }

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            // cmp x0, x1 is aliased of subs xzr, x0, x1
            Instr::Cmp(a, b) => {
                assert!((a.is_32bit() && b.is_32bit()) || (a.is_64bit() && b.is_64bit()));
                let mut instrbits = if a.is_64bit() { SUBS_64 } else { SUBS_32 };
                instrbits.set_bit_range(
                    4,
                    0,
                    if a.is_64bit() {
                        Reg::XZR.as_bits()
                    } else {
                        Reg::WZR.as_bits()
                    },
                );
                instrbits.set_bit_range(9, 5, a.as_bits());
                instrbits.set_bit_range(20, 16, b.as_bits());
                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Sub(reg, reg1, reg2) => todo!(),
            Instr::SubImm(out, left, right) => {
                assert!((out.is_32bit() && left.is_32bit()) || (out.is_64bit() && left.is_64bit()));
                let mut instrbits = if out.is_32bit() {
                    SUB_IMM_32
                } else {
                    SUB_IMM_64
                };
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, left.as_bits());
                instrbits.set_bit_range(21, 10, right.as_isize() as u32);
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::Madd(out, a, b, c) => {
                assert_eq!(out.is_64bit(), a.is_64bit());
                assert_eq!(a.is_64bit(), b.is_64bit());
                assert_eq!(b.is_64bit(), c.is_64bit());

                let mut instrbits = if out.is_64bit() { MADD_64 } else { MADD_32 };

                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, a.as_bits());
                instrbits.set_bit_range(14, 10, c.as_bits());
                instrbits.set_bit_range(20, 16, b.as_bits());

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::CmpImm(left, right, is_shifted) => {
                let mut instrbits = if left.is_64bit() {
                    CMP_IMM_64
                } else {
                    CMP_IMM_32
                };
                instrbits.set_bit_range(4, 0, left.as_bits());
                instrbits.set_bit_range(21, 10, right.as_isize() as u32);
                if *is_shifted {
                    instrbits.set_bit_range(22, 22, 1);
                }
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::Mov32(out, inp) => {
                assert_ne!(*out, Reg::SP);
                assert_ne!(*inp, Reg::SP);

                let mut instrbits = MOV_32;
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(20, 16, inp.as_bits());
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::Mov64(out, inp) => {
                assert_ne!(*out, Reg::SP);
                assert_ne!(*inp, Reg::SP);

                let mut instrbits = MOV_64;
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(20, 16, inp.as_bits());
                outbuf.write_all(&instrbits.to_le_bytes())?;
                Ok(())
            }
            Instr::MovImm(reg, imm) => {
                assert_ne!(*reg, Reg::SP);

                let val = imm.as_usize();
                assert!(val < 0x10000);

                // MOV (wide immediate) is an aliaz of MOVZ
                let mut instrbits = if reg.is_64bit() {
                    MOVZ_IMM_64
                } else {
                    MOVZ_IMM_32
                };
                instrbits.set_bit_range(4, 0, reg.as_bits());
                instrbits.set_bit_range(20, 5, val as u32);
                instrbits.set_bit_range(22, 21, 0);

                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::MovzImm(reg, imm, hw) => {
                assert_ne!(*reg, Reg::SP);

                let mut instrbits = MOVZ_IMM_64;
                instrbits.set_bit_range(4, 0, reg.as_bits());
                instrbits.set_bit_range(20, 5, imm.as_usize() as u32);
                instrbits.set_bit_range(22, 21, *hw as u32);
                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::MovkImm(reg, imm, hw) => {
                assert_ne!(*reg, Reg::SP);

                let mut instrbits = MOVK_IMM_64;
                instrbits.set_bit_range(4, 0, reg.as_bits());
                instrbits.set_bit_range(20, 5, imm.as_usize() as u32);
                instrbits.set_bit_range(22, 21, *hw as u32);
                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Ldr32ImmPostIndex(out, base, offset) => {
                let value = offset.as_isize();
                assert!(value < 256);
                assert!(value >= -256);
                let mut instrbits = LDR32_IMM_POST_INDEX;
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 12, base.as_bits());
                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Ldr64ImmPostIndex(out, base, offset) => {
                let value = offset.as_isize();
                assert!(value < 256);
                assert!(value >= -256);
                let mut instrbits = LDR64_IMM_POST_INDEX;
                instrbits.set_bit_range(4, 0, out.as_bits());
                instrbits.set_bit_range(9, 5, base.as_bits());
                instrbits.set_bit_range(20, 12, base.as_bits());
                outbuf.write_all(&instrbits.to_le_bytes())
            }
            Instr::Ret => {
                let mut instrbits: u32 = 0b1101011001011111000000_00000_00000;
                instrbits.set_bit_range(9, 5, 30);
                outbuf.write_all(&instrbits.to_le_bytes())
            }
        }
    }
}

impl Instr {
    pub fn as_asm(&self) -> String {
        match self {
            Instr::Orr(dest, src, imm) => {
                format!(
                    "orr\t{}, {}, #{}",
                    dest.as_asm(),
                    src.as_asm(),
                    imm.as_usize()
                )
            }
            Instr::Ldp(a, b, dest, imm) => {
                format!(
                    "ldp\t{}, {}, [{}, #{}]",
                    a.as_asm(),
                    b.as_asm(),
                    dest.as_asm(),
                    imm.as_isize()
                )
            }
            Instr::StrImmPreIndex(source, base, offset) => {
                format!(
                    "str\t{}, [{}, #{}]!",
                    source.as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::StrImmPostIndex(source, base, offset) => {
                format!(
                    "str\t{}, [{}], #{}",
                    source.as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::StrImmOffset(reg, reg1, imm) => {
                format!(
                    "str\t{}, [{}, {}]",
                    reg.as_asm(),
                    reg1.as_asm(),
                    imm.as_isize()
                )
            }
            Instr::Br(reg) => {
                format!("br\t{}", reg.as_asm())
            }
            Instr::Adr(reg, label) => {
                format!("adr\t{}, .L{}", reg.as_asm(), label.as_ref())
            }
            Instr::Cset(reg, cc) => {
                format!("cset\t{}, {}", reg.as_asm(), cc.as_asm())
            }
            Instr::Cb(zero, reg, label) => {
                if *zero {
                    format!("cbz\t{}, .L{}", reg.as_asm(), label.as_ref())
                } else {
                    format!("cbnz\t{}, .L{}", reg.as_asm(), label.as_ref())
                }
            }
            Instr::Jump(label) => {
                format!("b\t.L{}", label.as_ref())
            }
            Instr::Branch(cc, label) => {
                format!("b.{}\t.L{}", cc.as_asm(), label.as_ref())
            }
            Instr::LdpPostIndex(a, b, dest, imm) => {
                format!(
                    "ldp\t{}, {}, [{}], #{}",
                    a.as_asm(),
                    b.as_asm(),
                    dest.as_asm(),
                    imm.as_isize()
                )
            }
            Instr::StpPreIndex(a, b, dest, imm) => {
                format!(
                    "stp\t{}, {}, [{}, #{}]!",
                    a.as_asm(),
                    b.as_asm(),
                    dest.as_asm(),
                    imm.as_isize()
                )
            }
            Instr::Stp(a, b, dest, imm) => {
                format!(
                    "stp\t{}, {}, [{}, #{}]",
                    a.as_asm(),
                    b.as_asm(),
                    dest.as_asm(),
                    imm.as_isize()
                )
            }
            Instr::Str(source, base, offset) => {
                format!(
                    "str\t{}, [{}, {}]",
                    source.as_asm(),
                    base.as_asm(),
                    offset.as_asm()
                )
            }
            Instr::StrbPostIndex(out, base, offset) => {
                format!(
                    "strb\t{}, [{}], #{}",
                    out.as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::LdrhImmPostIndex(out, base, offset) => {
                format!(
                    "ldrh\t{}, [{}], #{}",
                    out.as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::LdrImmPostIndex(out, base, offset) => {
                format!(
                    "ldr\t{}, [{}], #{}",
                    out.as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::Ldr(out, base, offset) => {
                format!(
                    "ldr\t{}, [{}, {}]",
                    out.as_asm(),
                    base.as_asm(),
                    offset.as_asm()
                )
            }
            Instr::Ldrb(out, base, offset) => {
                format!(
                    "ldrb\t{}, [{}, {}]",
                    out.as_32bit().as_asm(),
                    base.as_asm(),
                    offset.as_asm()
                )
            }
            Instr::LdrbImmPostIndex(out, base, offset) => {
                format!(
                    "ldrb\t{}, [{}], #{}",
                    out.as_32bit().as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::Ldr32ImmPostIndex(out, base, offset) => {
                format!(
                    "ldr\t{}, [{}], #{}",
                    out.as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::Ldr64ImmPostIndex(out, base, offset) => {
                format!(
                    "ldr\t{}, [{}], #{}",
                    out.as_asm(),
                    base.as_asm(),
                    offset.as_isize()
                )
            }
            Instr::AddShift(out, a, b, shift, imm) => {
                format!(
                    "add\t{}, {}, {}, {} #{}",
                    out.as_asm(),
                    a.as_asm(),
                    b.as_asm(),
                    shift.as_asm(),
                    imm.as_usize()
                )
            }
            Instr::AddImm(out, a, b, is_shifted) => {
                if *is_shifted {
                    format!(
                        "add\t{}, {}, #{}, lsl #12",
                        out.as_asm(),
                        a.as_asm(),
                        b.as_usize()
                    )
                } else {
                    format!("add\t{}, {}, #{}", out.as_asm(), a.as_asm(), b.as_usize())
                }
            }
            Instr::SubsExtReg(out, a, b) => {
                format!("subs\t{}, {}, {}", out.as_asm(), a.as_asm(), b.as_asm())
            }
            Instr::Subs(out, a, b) => {
                format!("subs\t{}, {}, {}", out.as_asm(), a.as_asm(), b.as_asm())
            }
            Instr::SubsImm(out, a, b, is_shifted) => {
                if *is_shifted {
                    format!(
                        "subs\t{}, {}, #{}, lsl #12",
                        out.as_asm(),
                        a.as_asm(),
                        b.as_usize()
                    )
                } else {
                    format!("subs\t{}, {}, #{}", out.as_asm(), a.as_asm(), b.as_usize())
                }
            }
            Instr::Sub(out, a, b) => {
                format!("sub\t{}, {}, {}", out.as_asm(), a.as_asm(), b.as_asm())
            }
            Instr::SubImm(out, a, b) => {
                format!("sub\t{}, {}, #{}", out.as_asm(), a.as_asm(), b.as_isize())
            }
            Instr::Madd(out, a, b, c) => {
                format!(
                    "madd\t{}, {}, {}, {}",
                    out.as_asm(),
                    a.as_asm(),
                    b.as_asm(),
                    c.as_asm()
                )
            }
            Instr::Cmp(a, b) => {
                format!("cmp\t{}, {}", a.as_asm(), b.as_asm())
            }
            Instr::CmpImm(left, right, is_shifted) => {
                if *is_shifted {
                    format!("cmp\t{}, #{}, lsl #12", left.as_asm(), right.as_isize())
                } else {
                    format!("cmp\t{}, #{}", left.as_asm(), right.as_isize())
                }
            }
            Instr::Mov32(out, inp) => {
                format!("mov\t{}, {}", out.as_asm(), inp.as_asm())
            }
            Instr::Mov64(out, inp) => {
                format!("mov\t{}, {}", out.as_asm(), inp.as_asm())
            }
            Instr::MovImm(reg, imm) => {
                format!("mov\t{}, #{}", reg.as_asm(), imm.as_usize())
            }
            Instr::MovzImm(reg, imm, hw) => {
                format!(
                    "movz\t{}, #{}, lsl #{}",
                    reg.as_asm(),
                    imm.as_usize(),
                    hw * 16
                )
            }
            Instr::MovkImm(reg, imm, hw) => {
                format!(
                    "movk\t{}, #{}, lsl #{}",
                    reg.as_asm(),
                    imm.as_usize(),
                    hw * 16
                )
            }
            Instr::Ret => "ret".to_string(),
        }
    }
}

fn set_register_bits(instr_bits: &mut u32, reg: Reg) {
    let reg_bits = reg.as_bits();
    assert_eq!(reg_bits & !0b11111, 0);
    *instr_bits |= reg_bits;
}

impl Label {
    pub fn as_bits(&self, current_addr_offset: usize, labels: &HashMap<String, usize>) -> i64 {
        let dest_addr_offset = labels[&self.0];
        assert_ne!(dest_addr_offset, usize::MAX);
        // let result = (self.1 - pc) >> 2;
        let result = dest_addr_offset as isize - current_addr_offset as isize;
        println!("{} result: {}", self.0, result);
        assert!(result < (1 << 20));
        return result as i64;
    }
}

impl Reg {
    pub fn idx(&self) -> usize {
        match self {
            Reg::X0 | Reg::W0 => 0,
            Reg::X1 | Reg::W1 => 1,
            Reg::X2 | Reg::W2 => 2,
            Reg::X3 | Reg::W3 => 3,
            Reg::X4 | Reg::W4 => 4,
            Reg::X5 | Reg::W5 => 5,
            Reg::X6 | Reg::W6 => 6,
            Reg::X7 | Reg::W7 => 7,
            Reg::X8 | Reg::W8 => 8,
            Reg::X9 | Reg::W9 => 9,
            Reg::X10 | Reg::W10 => 10,
            Reg::X11 | Reg::W11 => 11,
            Reg::X28 | Reg::W28 => 28,
            Reg::X29 | Reg::W29 => 29,
            Reg::X30 | Reg::W30 => 30,
            Reg::XZR | Reg::WZR => todo!(),
            Reg::SP | Reg::WSP => todo!(),
        }
    }

    pub fn from_idx_64bit(idx: usize) -> Self {
        match idx {
            0 => Reg::X0,
            1 => Reg::X1,
            2 => Reg::X2,
            3 => Reg::X3,
            4 => Reg::X4,
            5 => Reg::X5,
            6 => Reg::X6,
            7 => Reg::X7,
            8 => Reg::X8,
            9 => Reg::X9,
            10 => Reg::X10,
            29 => Reg::X29,
            30 => Reg::X30,
            _ => panic!("Invalid index for 64-bit register"),
        }
    }

    pub fn from_idx_32bit(idx: usize) -> Self {
        match idx {
            0 => Reg::W0,
            1 => Reg::W1,
            2 => Reg::W2,
            3 => Reg::W3,
            4 => Reg::W4,
            5 => Reg::W5,
            6 => Reg::W6,
            7 => Reg::W7,
            8 => Reg::W8,
            9 => Reg::W9,
            10 => Reg::W10,
            29 => Reg::W29,
            30 => Reg::W30,
            _ => panic!("Invalid index for 32-bit register"),
        }
    }

    pub fn as_bits(&self) -> u32 {
        match self {
            Reg::X0 | Reg::W0 => 0,
            Reg::X1 | Reg::W1 => 1,
            Reg::X2 | Reg::W2 => 2,
            Reg::X3 | Reg::W3 => 3,
            Reg::X4 | Reg::W4 => 4,
            Reg::X5 | Reg::W5 => 5,
            Reg::X6 | Reg::W6 => 6,
            Reg::X7 | Reg::W7 => 7,
            Reg::X8 | Reg::W8 => 8,
            Reg::X9 | Reg::W9 => 9,
            Reg::X10 | Reg::W10 => 10,
            Reg::X11 | Reg::W11 => 11,
            Reg::X28 | Reg::W28 => 28,
            Reg::X29 | Reg::W29 => 29,
            Reg::X30 | Reg::W30 => 30,
            Reg::XZR | Reg::WZR => 0b11111,
            // XZR/WZR and SP/WSP have the same encoding
            Reg::SP | Reg::WSP => 0b11111,
        }
    }

    pub fn is_64bit(&self) -> bool {
        match self {
            Reg::X0
            | Reg::X1
            | Reg::X2
            | Reg::X3
            | Reg::X4
            | Reg::X5
            | Reg::X6
            | Reg::X7
            | Reg::X8
            | Reg::X9
            | Reg::X10
            | Reg::X11
            | Reg::X28
            | Reg::X29
            | Reg::X30
            | Reg::XZR => true,
            Reg::W0
            | Reg::W1
            | Reg::W2
            | Reg::W3
            | Reg::W4
            | Reg::W5
            | Reg::W6
            | Reg::W7
            | Reg::W8
            | Reg::W9
            | Reg::W10
            | Reg::W11
            | Reg::W28
            | Reg::W29
            | Reg::WZR
            | Reg::W30
            | Reg::WSP => false,
            Reg::SP => true,
        }
    }

    pub fn is_32bit(&self) -> bool {
        match self {
            Reg::W0
            | Reg::W1
            | Reg::W2
            | Reg::W3
            | Reg::W4
            | Reg::W5
            | Reg::W6
            | Reg::W7
            | Reg::W8
            | Reg::W9
            | Reg::W10
            | Reg::W11
            | Reg::W29
            | Reg::W30
            | Reg::WSP
            | Reg::W28
            | Reg::WZR => true,
            Reg::X0
            | Reg::X1
            | Reg::X2
            | Reg::X3
            | Reg::X4
            | Reg::X5
            | Reg::X6
            | Reg::X7
            | Reg::X8
            | Reg::X9
            | Reg::X10
            | Reg::X11
            | Reg::X28
            | Reg::X29
            | Reg::X30
            | Reg::SP
            | Reg::XZR => false,
        }
    }

    pub fn as_32bit(&self) -> Self {
        match self {
            Reg::X0 | Reg::W0 => Reg::W0,
            Reg::X1 | Reg::W1 => Reg::W1,
            Reg::X2 | Reg::W2 => Reg::W2,
            Reg::X3 | Reg::W3 => Reg::W3,
            Reg::X4 | Reg::W4 => Reg::W4,
            Reg::X5 | Reg::W5 => Reg::W5,
            Reg::X6 | Reg::W6 => Reg::W6,
            Reg::X7 | Reg::W7 => Reg::W7,
            Reg::X8 | Reg::W8 => Reg::W8,
            Reg::X9 | Reg::W9 => Reg::W9,
            Reg::X10 | Reg::W10 => Reg::W10,
            Reg::X11 | Reg::W11 => Reg::W11,
            Reg::X28 | Reg::W28 => Reg::W28,
            Reg::X29 | Reg::W29 => Reg::W29,
            Reg::XZR | Reg::WZR => Reg::WZR,
            Reg::SP | Reg::WSP => Reg::WSP,
            Reg::X30 | Reg::W30 => Reg::W30,
        }
    }

    pub fn as_asm(&self) -> &'static str {
        match self {
            Reg::X0 => "x0",
            Reg::X1 => "x1",
            Reg::X2 => "x2",
            Reg::X3 => "x3",
            Reg::X4 => "x4",
            Reg::X5 => "x5",
            Reg::X6 => "x6",
            Reg::X7 => "x7",
            Reg::X8 => "x8",
            Reg::X9 => "x9",
            Reg::X10 => "x10",
            Reg::X11 => "x11",
            Reg::X28 => "x28",
            Reg::X29 => "x29",
            Reg::X30 => "x30",
            Reg::W0 => "w0",
            Reg::W1 => "w1",
            Reg::W2 => "w2",
            Reg::W3 => "w3",
            Reg::W4 => "w4",
            Reg::W5 => "w5",
            Reg::W6 => "w6",
            Reg::W7 => "w7",
            Reg::W8 => "w8",
            Reg::W9 => "w9",
            Reg::W10 => "w10",
            Reg::W11 => "w11",
            Reg::W28 => "w28",
            Reg::W29 => "w29",
            Reg::W30 => "w30",
            Reg::XZR => "xzr",
            Reg::WZR => "wzr",
            Reg::SP => "sp",
            Reg::WSP => "wsp",
        }
    }
}

impl CC {
    pub fn as_bits(&self) -> u8 {
        *self as u8
    }

    pub fn as_asm(&self) -> &'static str {
        match self {
            CC::Eq => "eq",
            CC::Neq => "ne",
            CC::Lt => "lt",
            CC::Le => "le",
            CC::Gt => "gt",
            CC::Ge => "ge",
            CC::Hi => "hi",
            CC::Hs => "hs",
            CC::Ls => "ls",
            CC::Lo => "lo",
        }
    }
}

#[cfg(test)]
mod test {
    use std::ffi::c_void;
    use std::hint::black_box;
    use std::u32;

    use super::*;

    #[test]
    fn test_assembler_subs_ext_reg() {
        use Reg::*;

        let mut asm = Assembler::new();
        asm.mov_imm(X2, 34.into());

        asm.add_imm(X1, SP, 0.into());
        asm.sub_imm(SP, SP, 32.into());

        asm.subs_ext_reg(XZR, SP, X1);
        asm.mov_imm(X2, 69.into());
        asm.branch(CC::Eq, "OOGA".into());

        asm.add_imm(SP, SP, 32.into());
        asm.subs_ext_reg(XZR, SP, X1);
        asm.mov_imm(X2, 35.into());
        asm.branch(CC::Eq, "OOGA".into());

        asm.mov_imm(X0, 420.into());
        asm.ret();

        asm.label("OOGA");
        asm.mov(X0, X2);
        asm.ret();

        let exec = ExecutableMem::from_bytes_copy(&asm.emit());
        let func =
            unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> usize>(exec.addr) };

        assert_eq!(func(), 35);
    }

    #[test]
    fn test_assembler_add_imm_sp() {
        use Reg::*;

        {
            let mut asm = Assembler::new();

            asm.add_imm(X1, SP, 0.into());
            asm.sub_imm(SP, SP, Imm::U8(16));
            asm.add_imm(SP, X1, 0.into());
            asm.ret();

            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> usize>(exec.addr) };
            std::io::stdout().flush().unwrap();
            assert_eq!(func(), 0);
        }

        {
            let mut asm = Assembler::new();

            asm.add_imm(X29, SP, 0.into());
            asm.sub_imm(SP, SP, Imm::U8(16));
            asm.add_imm(SP, X29, 0.into());
            asm.ret();

            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> usize>(exec.addr) };
            std::io::stdout().flush().unwrap();
            assert_eq!(func(), 0);
        }
    }

    #[test]
    fn test_assembler_str_imm_post_index() {
        use Reg::*;

        let mut asm = Assembler::new();
        asm.str_imm_post_index(X0, SP, Imm::I16(-16));
        asm.mov_imm(X3, 16.into());
        asm.ldr(X0, SP, X3);
        asm.add_imm(SP, SP, Imm::I16(16));
        asm.ret();

        println!("{}", asm.as_asm());
        let exec = ExecutableMem::from_bytes_copy(&asm.emit());
        let func =
            unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr) };

        assert_eq!(func(420), 420);
    }
    #[test]
    fn test_assembler_str_imm_pre_index() {
        use Reg::*;

        let mut asm = Assembler::new();
        asm.str_imm_pre_index(X0, SP, Imm::I16(-16));
        asm.ldr_imm_post_index(X0, SP, Imm::I16(0));
        asm.add_imm(SP, SP, Imm::I16(16));
        asm.ret();

        println!("{}", asm.as_asm());
        let exec = ExecutableMem::from_bytes_copy(&asm.emit());
        let func =
            unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr) };

        assert_eq!(func(420), 420);
    }

    #[test]
    fn test_assembler_stack_manipulation() {
        use Reg::*;

        {
            let mut asm = Assembler::new();
            asm.stp_imm_pre_index(X29, X30, SP, Imm::I16(-16));
            asm.sub_imm(SP, SP, Imm::U8(16));
            asm.add_imm(SP, SP, Imm::I16(16));
            asm.ldp_imm_post_index(X29, X30, SP, Imm::I16(16));
            asm.ret();
            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };
            assert_eq!(func(420), 420);
        }

        {
            let mut asm = Assembler::new();
            asm.stp_imm_pre_index(X29, X30, SP, Imm::I16(-16));
            asm.sub_imm(SP, SP, Imm::U8(16));
            asm.str_imm_offset(X0, SP, Imm::I16(0));
            asm.ldr_imm_post_index(X0, SP, Imm::I16(0));
            asm.add_imm(SP, SP, Imm::I16(16));
            asm.ldp_imm_post_index(X29, X30, SP, Imm::I16(16));
            asm.ret();
            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };
            assert_eq!(func(420), 420);
            assert_eq!(func(69), 69);
        }
    }

    #[test]
    fn test_assembler_stp() {
        use Reg::*;

        {
            let mut asm = Assembler::new();
            asm.ldr_imm_post_index(X1, X0, 8.into());
            asm.ldr_imm_post_index(X2, X0, 0.into());
            asm.stp(X1, X2, SP, (-16).into());
            asm.ldp(X1, X2, SP, (-16).into());
            asm.add(X0, X1, X2);
            asm.ret();

            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(*const usize) -> usize>(exec.addr)
            };

            let input: [usize; 2] = [0; 2];
            assert_eq!(func(input.as_ptr()), 0);
            let input: [usize; 2] = [34, 35];
            assert_eq!(func(input.as_ptr()), 69);
        }

        // {
        //     let mut asm = Assembler::new();
        //     asm.ldr_imm_post_index(X1, X0, 0.into());
        //     asm.ldr_imm_post_index(X2, X0, 8.into());
        //     asm.sub_imm(SP, SP, Imm::U8(16));

        //     asm.stp(X1, X2, SP, 0.into());
        //     asm.ldp(X1, X2, SP, 0.into());
        //     asm.add(X0, X1, X2);
        //     asm.add_imm(SP, SP, Imm::U8(16));
        //     asm.ret();

        //     let exec = ExecutableMem::from_bytes_copy(&asm.emit());
        //     let func = unsafe {
        //         std::mem::transmute::<*mut c_void, extern "C" fn(*const usize) -> usize>(
        //             exec.addr,
        //         )
        //     };

        //     let input: [usize; 2] = [0; 2];
        //     assert_eq!(func(input.as_ptr()), 0);
        //     let input: [usize; 2] = [34, 35];
        //     assert_eq!(func(input.as_ptr()), 69);
        // }
    }

    #[test]
    fn test_assembler_stp_pre_index() {
        use Reg::*;
        {
            let mut asm = Assembler::new();
            asm.stp_imm_pre_index(X1, X2, X0, Imm::U8(8));
            asm.ret();

            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(*mut u8, usize, usize) -> usize>(
                    exec.addr,
                )
            };

            let mut input: [u8; 24] = *b"XXXXXXXX1234567812345678";
            func(input.as_mut_ptr(), 34, 35);
            let a = usize::from_le_bytes(input[8..16].try_into().unwrap());
            let b = usize::from_le_bytes(input[16..24].try_into().unwrap());
            assert_eq!(a, 34);
            assert_eq!(b, 35);
        }

        {
            let mut asm = Assembler::new();
            asm.stp_imm_pre_index(X1, X2, X0, Imm::I16(-8));
            asm.ret();

            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(*mut u8, usize, usize) -> usize>(
                    exec.addr,
                )
            };

            let mut input: [u8; 24] = *b"XXXXXXXX1234567812345678";
            func(input[8..16].as_mut_ptr(), 34, 35);
            let a = usize::from_le_bytes(input[0..8].try_into().unwrap());
            let b = usize::from_le_bytes(input[8..16].try_into().unwrap());
            assert_eq!(a, 34);
            assert_eq!(b, 35);
        }
    }

    #[test]
    fn test_asssembler_str_imm_offset() {
        use Reg::*;
        let mut asm = Assembler::new();
        asm.str_imm_offset(X1, X0, Imm::U8(0));
        asm.str_imm_offset(X2, X0, Imm::U8(8));
        asm.ret();

        let exec = ExecutableMem::from_bytes_copy(&asm.emit());
        let func = unsafe {
            std::mem::transmute::<*mut c_void, extern "C" fn(*mut usize, usize, usize) -> usize>(
                exec.addr,
            )
        };

        let mut input: [usize; 8] = [0; 8];
        func(input.as_mut_ptr(), 34, 35);
        assert_eq!(input[0], 34);
        assert_eq!(input[1], 35);
    }

    #[test]
    fn test_assembler_stack() {
        use Reg::*;
        let mut asm = Assembler::new();
        asm.stp_imm_pre_index(X29, X30, SP, Imm::I16(-16));
        asm.add_imm(X29, SP, 0.into());
        asm.sub_imm(SP, SP, Imm::U8(16));

        asm.str_imm_offset(X0, SP, Imm::U8(0));
        asm.str_imm_offset(X1, SP, Imm::U8(8));
        asm.ldp(X2, X3, SP, Imm::U8(0));
        asm.add(X0, X2, X3);
        asm.mov(X0, X3);

        asm.add_imm(SP, SP, Imm::U8(16));
        asm.ldp_imm_post_index(X29, X30, SP, Imm::I16(16));
        asm.ret();

        let exec = ExecutableMem::from_bytes_copy(&asm.emit());
        let func = unsafe {
            std::mem::transmute::<*mut c_void, extern "C" fn(usize, usize) -> usize>(exec.addr)
        };
        assert_eq!(func(34, 35), 35);
        let foo = black_box("nice");
        println!("{:?}", foo);
    }

    #[test]
    fn test_assembler_adr() {
        use Reg::*;

        {
            let mut asm = Assembler::new();
            let label = Label(asm.new_label("label"));
            asm.mov_imm(X0, Imm::U32(420));
            asm.adr(X1, label.clone());
            asm.branch_register(X1);
            asm.mov_imm(X0, Imm::U32(69));
            asm.label(label.as_ref());
            asm.ret();

            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };
            assert_eq!(func(0), 420);
        }

        {
            let mut asm = Assembler::new();
            let label = Label(asm.new_label("label"));
            let done = Label(asm.new_label("done"));

            asm.label(label.as_ref());
            asm.subs_imm(X0, X0, 1.into());
            asm.cmp(X0, X1);
            asm.branch(CC::Le, done.clone());

            asm.adr(X3, label.clone());
            asm.branch_register(X3);

            asm.label(done.as_ref());
            asm.ret();

            let exec = ExecutableMem::from_bytes_copy(&asm.emit());
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize, usize) -> usize>(exec.addr)
            };

            assert_eq!(func(10, 5), 5);
        }
    }

    #[test]
    fn test_assembler_cset() {
        let mut assembler = Assembler::new();
        assembler.cmp_imm(Reg::X0, Imm::U32(0));
        assembler.cset(Reg::X0, CC::Eq);
        assembler.ret();
        let bytes = assembler.emit();
        let exec = ExecutableMem::from_bytes_copy(&bytes);
        let func =
            unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> u32>(exec.addr) };

        assert_eq!(func(0), 1);
        assert_eq!(func(1), 0);
    }

    #[test]
    fn test_assembler_mov_imm_32() {
        {
            let mut assembler = Assembler::new();
            assembler.mov_imm(Reg::W0, Imm::U32(4206900));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> u32>(exec.addr) };
            assert_eq!(func(), 4206900);
        }

        {
            let mut assembler = Assembler::new();
            assembler.mov_imm(Reg::W0, Imm::U32(420));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> u32>(exec.addr) };
            assert_eq!(func(), 420);
        }
    }

    #[test]
    fn test_assembler_mov_imm_64() {
        {
            let mut assembler = Assembler::new();
            let value: usize = u32::MAX as usize + 1;
            assembler.mov_imm(Reg::X0, Imm::U64(value as u64));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> usize>(exec.addr) };
            assert_eq!(func(), value);
        }

        {
            let mut assembler = Assembler::new();
            assembler.mov_imm(Reg::X0, Imm::U32(420));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> usize>(exec.addr) };
            assert_eq!(func(), 420);
        }
    }

    #[test]
    fn test_assembler_cbz() {
        {
            let mut assembler = Assembler::new();
            let fail = Label(assembler.new_label("FAIL"));
            assembler.cbz(Reg::W0, fail.clone());
            assembler.ret();
            assembler.label(fail.as_ref());
            assembler.mov_imm(Reg::W0, Imm::U32(69));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(u32) -> u32>(exec.addr) };
            assert_eq!(func(0), 69);
            assert_eq!(func(1), 1);
            assert_eq!(func(2), 2);
        }

        {
            let mut assembler = Assembler::new();
            let fail = Label(assembler.new_label("FAIL"));
            assembler.cbz(Reg::X0, fail.clone());
            assembler.ret();
            assembler.label(fail.as_ref());
            assembler.mov_imm(Reg::X0, Imm::U64(69));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };
            assert_eq!(func(0), 69);
            assert_eq!(func(1), 1);
            assert_eq!(func(2), 2);
        }
    }

    #[test]
    fn test_assembler_cmp() {
        {
            let mut assembler = Assembler::new();
            assembler.cmp(Reg::X0, Reg::X1);
            let fail = Label(assembler.new_label("FAIL"));
            assembler.branch(CC::Lt, fail.clone());
            assembler.mov_imm(Reg::X0, Imm::U32(420));
            assembler.ret();
            assembler.label(fail.as_ref());
            assembler.mov_imm(Reg::X0, Imm::U32(69));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize, usize) -> usize>(exec.addr)
            };
            assert_eq!(func(0, 0), 420);
            assert_eq!(func(1, 1), 420);
            assert_eq!(func(2, 2), 420);
            assert_eq!(func(2, 420), 69);
        }

        {
            let mut assembler = Assembler::new();
            assembler.cmp(Reg::W0, Reg::W1);
            let fail = Label(assembler.new_label("FAIL"));
            assembler.branch(CC::Lt, fail.clone());
            assembler.mov_imm(Reg::W0, Imm::U32(420));
            assembler.ret();
            assembler.label(fail.as_ref());
            assembler.mov_imm(Reg::W0, Imm::U32(69));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(u32, u32) -> u32>(exec.addr)
            };
            assert_eq!(func(0, 0), 420);
            assert_eq!(func(1, 1), 420);
            assert_eq!(func(2, 2), 420);
            assert_eq!(func(2, 420), 69);
        }
    }

    #[test]
    fn test_assembler_cmp_imm() {
        use Reg::*;
        let input = b"LOLMAO!!";
        let result = usize::from_le_bytes(*input);
        let mut assembler = Assembler::new();
        assembler.ldr(X0, X0, XZR);
        assembler.cmp_imm(X0, Imm::U64(result as u64));
        let success = Label(assembler.new_label("SUCCESS"));
        assembler.branch(CC::Eq, success.clone());
        assembler.mov_imm(X0, Imm::U32(420));
        assembler.label(success.as_ref());
        assembler.ret();

        let bytes = assembler.emit();
        let exec = ExecutableMem::from_bytes_copy(&bytes);
        let func: extern "C" fn(*const u8, *mut usize) -> usize =
            unsafe { std::mem::transmute::<*mut c_void, _>(exec.addr) };

        let mut output: usize = 0;
        assert_eq!(func(input.as_ptr(), &mut output), result as usize);
    }

    #[test]
    fn test_assembler_str() {
        use Reg::*;
        let mut assembler = Assembler::new();
        let ptr = X0;
        let offset = X1;
        let source = X2;
        assembler.mov_imm(offset, Imm::U32(0));
        for char in b"HELLO".iter() {
            assembler.mov_imm(source, Imm::U8(*char));
            assembler.str(source, ptr, offset);
            assembler.add_imm(offset, offset, Imm::U32(1));
        }
        assembler.ret();

        let bytes = assembler.emit();
        let exec = ExecutableMem::from_bytes_copy(&bytes);
        let func =
            unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(*mut u8) -> u32>(exec.addr) };

        let mut input = vec![0; 5];
        func(input.as_mut_ptr());
        assert_eq!(input.as_slice(), b"HELLO");
    }

    #[test]
    fn test_assembler_ldrb() {
        use Reg::*;
        let mut assembler = Assembler::new();
        let fail = Label(assembler.new_label("FAIL"));
        assembler.cbz(X2, fail.clone());
        let loop_start = Label(assembler.new_label("LOOP"));
        assembler.label(loop_start.as_ref());
        assembler.ldrb_post_index(X3, X0, Imm::U32(1));
        assembler.strb_post_index(X3, X1, Imm::U32(1));
        assembler.subs_imm(X2, X2, Imm::U32(1));
        assembler.cbnz(X2, loop_start.clone());
        assembler.mov_imm(X0, Imm::U32(0));
        assembler.ret();
        assembler.label(fail.as_ref());
        assembler.mov_imm(X0, Imm::U32(1));
        assembler.ret();

        let bytes = assembler.emit();
        let exec = ExecutableMem::from_bytes_copy(&bytes);
        let func = unsafe {
            std::mem::transmute::<*mut c_void, extern "C" fn(*const u8, *mut u8, usize) -> u32>(
                exec.addr,
            )
        };

        let input = b"HELLO";
        let mut output: [u8; 5] = [0; 5];
        assert_eq!(func(input.as_ptr(), output.as_mut_ptr(), input.len()), 0);
        assert_eq!(input, &output);

        let input = b"HELLO";
        let mut output: [u8; 5] = [0; 5];
        assert_eq!(func(input.as_ptr(), output.as_mut_ptr(), 0), 1);
    }

    #[test]
    fn test_assembler_ldr() {
        use Reg::*;

        {
            let input = b"1234";
            let result = u32::from_le_bytes(*input);
            let mut assembler = Assembler::new();
            assembler.ldr(W0, X0, XZR);
            assembler.cmp_imm(W0, Imm::U32(result as u32));
            assembler.cset(W0, CC::Eq);
            assembler.ret();

            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func: extern "C" fn(*const u8) -> usize =
                unsafe { std::mem::transmute::<*mut c_void, _>(exec.addr) };

            assert_eq!(func(input.as_ptr()), 1);
            assert_eq!(func(b"01234".as_ptr()), 0);
        }
    }

    #[test]
    fn test_assembler_ldr_offset() {
        use Reg::*;
        let mut assembler = Assembler::new();
        assembler.mov_imm(X1, Imm::U32(8));
        assembler.ldr(X0, X0, X1);
        assembler.ret();

        let bytes = assembler.emit();
        let exec = ExecutableMem::from_bytes_copy(&bytes);
        let func: extern "C" fn(*const u8) -> usize =
            unsafe { std::mem::transmute::<*mut c_void, _>(exec.addr) };

        let input = b"00000000LOLMAO!!";
        let result = usize::from_le_bytes(input[8..].try_into().unwrap());
        assert_eq!(func(input.as_ptr()), result as usize);
    }

    #[test]
    fn test_assembler_add() {
        use Reg::*;
        {
            let mut assembler = Assembler::new();
            assembler.add(X0, X0, X1);
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize, usize) -> usize>(exec.addr)
            };
            assert_eq!(func(0, 0), 0);
            assert_eq!(func(1, 1), 2);
            assert_eq!(func(2, 2), 4);
            assert_eq!(func(34, 35), 69);
        }

        {
            let mut assembler = Assembler::new();
            assembler.add(X0, X1, XZR);
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize, usize) -> usize>(exec.addr)
            };
            assert_eq!(func(0, 0), 0);
            assert_eq!(func(0, 1), 1);
            assert_eq!(func(0, 2), 2);
            assert_eq!(func(0, 35), 35);
        }

        {
            let mut assembler = Assembler::new();
            assembler.add_imm(X1, SP, 0.into());
            assembler.add_imm(SP, X1, 0.into());
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };
            assert_eq!(func(0), 0);
        }
    }

    #[test]
    fn test_assembler_sub_imm() {
        use Reg::*;
        {
            let mut assembler = Assembler::new();
            assembler.sub_imm(X0, X0, Imm::U32(1));
            assembler.ret();
            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(isize) -> isize>(exec.addr)
            };
            assert_eq!(func(0), -1);
            assert_eq!(func(1), 0);
            assert_eq!(func(2), 1);
            assert_eq!(func(3), 2);
            assert_eq!(func(34), 33);
        }
    }

    #[test]
    fn test_assembler_loop() {
        use Reg::*;
        {
            let mut assembler = Assembler::new();
            let loop_end = Label(assembler.new_label("LOOP_END"));
            assembler.cbz(X0, loop_end.clone());
            let count = X1;
            assembler.mov(count, X0);
            assembler.mov_imm(X0, Imm::U32(1));
            assembler.subs_imm(count, count, Imm::U32(1));
            assembler.cbz(count, loop_end.clone());
            let loop_start = Label(assembler.new_label("LOOP"));
            assembler.label(loop_start.as_ref());
            assembler.mov_imm(X3, Imm::U32(1));
            assembler.add(X0, X0, X3);
            assembler.subs_imm(count, count, Imm::U32(1));
            assembler.cbnz(count, loop_start.clone());
            assembler.label(loop_end.as_ref());
            assembler.ret();

            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };

            assert_eq!(func(0), 0);
            assert_eq!(func(1), 1);
            assert_eq!(func(2), 2);
            assert_eq!(func(3), 3);
        }
    }

    #[test]
    fn test_assembler_fibbonacci_2() {
        use Reg::*;
        println!("RUNNING");
        let mut assembler = Assembler::new();

        let jump_ret = Label(assembler.new_label("JUMP_RET"));
        assembler.cmp_imm(X0, Imm::U8(2));
        assembler.branch(CC::Lt, jump_ret.clone());
        assembler.mov_imm(X0, Imm::U32(69));

        assembler.label(jump_ret.as_ref());
        assembler.ret();

        let bytes = assembler.emit();
        let exec = ExecutableMem::from_bytes_copy(&bytes);
        let func =
            unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr) };

        // print!("RUNNIN!");
        assert_eq!(func(1), 1);
        // assert_eq!(func(2), 0);

        // assert_eq!(func(1), 1);
        // assert_eq!(func(2), 1);
        // assert_eq!(func(3), 2);
    }

    #[test]
    fn test_assembler_branch() {
        use Reg::*;
        {
            let mut assembler = Assembler::new();
            let jump_ret = Label(assembler.new_label("JUMP_RET"));
            assembler.cmp_imm(X0, Imm::U8(2));
            assembler.branch(CC::Gt, jump_ret.clone());
            assembler.mov_imm(X0, Imm::U32(69));
            assembler.label(jump_ret.as_ref());
            assembler.ret();

            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };

            assert_eq!(func(0), 69);
            assert_eq!(func(1), 69);
            assert_eq!(func(2), 69);
            assert_eq!(func(3), 3);
        }

        {
            let mut assembler = Assembler::new();
            let jump_ret = Label(assembler.new_label("JUMP_RET"));
            let min = 5;
            let max = 10;
            assembler.sub_imm(X1, X0, Imm::U8(min));
            assembler.cmp_imm(X1, Imm::U8(max - min));
            assembler.branch(CC::Hi, jump_ret.clone());
            assembler.mov_imm(X0, Imm::U32(69));
            assembler.label(jump_ret.as_ref());
            assembler.ret();

            let bytes = assembler.emit();
            let exec = ExecutableMem::from_bytes_copy(&bytes);
            let func = unsafe {
                std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr)
            };

            assert_ne!(func(4), 69);
            assert_eq!(func(5), 69);
            assert_eq!(func(6), 69);
            assert_eq!(func(7), 69);
            assert_eq!(func(8), 69);
            assert_eq!(func(9), 69);
            assert_eq!(func(10), 69);
            assert_ne!(func(11), 69);
        }
    }

    #[test]
    fn test_assembler_fibbonacci() {
        use Reg::*;
        println!("RUNNING");
        let mut assembler = Assembler::new();

        let prev = X1;
        let prevprev = X2;
        let cur = X0;
        let n = X3;

        let jump_ret = Label(assembler.new_label("JUMP_RET"));
        assembler.cmp_imm(X0, Imm::U8(2));
        assembler.branch(CC::Lt, jump_ret.clone());
        assembler.mov_imm(prevprev, Imm::U8(0));
        assembler.mov_imm(prev, Imm::U8(1));
        assembler.mov(n, X0);

        let loop_start = Label(assembler.new_label("LOOP"));
        assembler.label(loop_start.as_ref());
        assembler.add(cur, prev, prevprev);
        assembler.mov(prevprev, prev);
        assembler.mov(prev, cur);
        assembler.sub_imm(n, n, Imm::U8(1));
        assembler.cmp_imm(n, Imm::U8(2));
        assembler.branch(CC::Ge, loop_start.clone());

        assembler.label(jump_ret.as_ref());
        assembler.ret();

        let bytes = assembler.emit();
        let exec = ExecutableMem::from_bytes_copy(&bytes);
        let func =
            unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(usize) -> usize>(exec.addr) };

        fn expected_fib(mut n: isize) -> usize {
            if n < 2 {
                return n as usize;
            }
            return expected_fib(n - 1) + expected_fib(n - 2);
        }

        // print!("RUNNIN!");
        assert_eq!(func(0), expected_fib(0));
        assert_eq!(func(1), expected_fib(1));
        assert_eq!(func(2), expected_fib(2));
        assert_eq!(func(3), expected_fib(3));
        assert_eq!(func(4), expected_fib(4));

        // assert_eq!(func(1), 1);
        // assert_eq!(func(2), 1);
        // assert_eq!(func(3), 2);
    }
}
