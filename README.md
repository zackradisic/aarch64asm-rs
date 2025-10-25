# aarch64asm

aarch64 assembler for Rust

This is a WIP for my side projects but may be useful as a reference
implementation for those interested.

## Supported Instructions

- ADD (add immediate, add shifted register)
- SUB/SUBS (subtract immediate/register, with flags)
- MADD (multiply-add)
- ORR (logical OR immediate)
- CMP (compare register, compare immediate)
- CSET (conditional set)
- B (unconditional branch)
- BR (branch to register)
- CBZ/CBNZ (compare and branch if zero/non-zero)
- ADR (address calculation for labels)
- LDR (load register from memory)
- STR (store register to memory)
- LDP/STP (load/store pair)
- Various indexed and offset variants (immediate post-index, pre-index, etc.)
- MOV (move register, move immediate)
- MOVZ/MOVK (move wide immediate)
- RET (return)

Note: This is not an exhaustive list of all AArch64 instructions, but the ones currently implemented in this assembler.
