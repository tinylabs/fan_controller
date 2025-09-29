#!/usr/bin/env python3
# fan433_modelC.py
# Minimal encoder/decoder for AC Infinity 433.9 MHz controller using Model C (Affine GF(2))
# Matches recordings:
#   - MSB-first symbol order
#   - Preamble "00000"
#   - 12 ones + 2 zeros + 5 zeros (preamble) = 7 zeros before next '1'

from pathlib import Path
import argparse
from itertools import groupby

# ---------- Protocol constants learned from your captures ----------
BASE = {
  0:0x702e, 1:0x716d, 2:0x72a8, 3:0x73eb, 4:0x7413,
  5:0x7550, 6:0x7695, 7:0x77d6, 8:0x7854, 9:0x7917, 10:0x7a91
}
# Address deltas per speed (speed 10 differs slightly)
D0 = {**{s:0x407d for s in range(0,10)}, 10:0x4040}  # a0 (LSB)
D1 = {**{s:0x20db for s in range(0,10)}, 10:0x20e6}  # a1
D2 = {**{s:0x1088 for s in range(0,10)}, 10:0x10b5}  # a2

# ---------- Framing tuned to recordings ----------
DEFAULT_PREAMBLE = "00000"          # five zeros
DEFAULT_REPEATS  = 8
DEFAULT_IDLE1    = 13               # 12 ones between repeats
DEFAULT_IDLE0    = 2                # *** 2 zeros so 2 + 5 (preamble) = 7 zeros before next '1'
DEFAULT_GAP_ZEROS= 7                # target zeros before next '1'
MIN_ONES_SPLIT   = 10               # split segments on long '1' runs

# ---------- Bit helpers ----------
def keep_bits(s: str) -> str:
    return ''.join(c for c in s if c in '01')

def rle(bits: str):
    return [(k, len(list(g))) for k, g in groupby(bits)]

def split_by_long_ones(bits: str, min_ones: int = MIN_ONES_SPLIT):
    runs = rle(bits)
    segs, pos, last = [], 0, 0
    for v, l in runs:
        if v == '1' and l >= min_ones:
            segs.append(bits[last:pos])
            last = pos + l
        pos += l
    segs.append(bits[last:])
    return [s for s in segs if s and s.count('1') > 0]

def tail_decode_16(seg_bits: str):
    """
    Collect all (1,0) pairs with len(1)=1 and len(0) in {1,2}; return the last 16 symbols.
    """
    runs = rle(seg_bits)
    syms = []
    for i in range(len(runs)-1):
        (v1,l1),(v2,l2)=runs[i],runs[i+1]
        if v1=='1' and v2=='0' and l1==1 and l2 in (1,2):
            syms.append('1' if l2==2 else '0')
    if len(syms) >= 16:
        return ''.join(syms[-16:])
    return None

def bits_to_int(b: str) -> int:
    return int(b, 2)

def int_to_bits(x: int, w: int) -> str:
    return format(x, f'0{w}b')

# ---------- Model C (Affine GF(2)) ----------
def int_to_bits16(x: int):
    return [(x >> (15 - k)) & 1 for k in range(16)]

def build_W(BASE, D0, D1, D2):
    """
    Build 16x48 weight matrix W from BASE/D* per speed using features:
      [1, a0,a1,a2, S0..S10, a0*S0..a0*S10, a1*S0..a1*S10, a2*S0..a2*S10]
    """
    S   = lambda s: 4 + s        # 4..14
    A0S = lambda s: 15 + s       # 15..25
    A1S = lambda s: 26 + s       # 26..36
    A2S = lambda s: 37 + s       # 37..47
    W = [[0]*48 for _ in range(16)]
    for s in range(0, 11):
        base_bits = int_to_bits16(BASE[s])
        d0_bits   = int_to_bits16(D0[s])
        d1_bits   = int_to_bits16(D1[s])
        d2_bits   = int_to_bits16(D2[s])
        for k in range(16):
            W[k][S(s)]   ^= base_bits[k]
            W[k][A0S(s)] ^= d0_bits[k]
            W[k][A1S(s)] ^= d1_bits[k]
            W[k][A2S(s)] ^= d2_bits[k]
    return W

W = build_W(BASE, D0, D1, D2)

def encode_affine(speed: int, addr: int) -> int:
    """cw16 from (speed, addr) via W @ features mod 2."""
    if speed not in range(0,11):
        raise ValueError("speed must be 0..10")
    if addr not in range(0,8):
        raise ValueError("addr must be 0..7")
    a0, a1, a2 = (addr & 1), ((addr>>1)&1), ((addr>>2)&1)
    f = [0]*48
    f[0] = 1
    f[1], f[2], f[3] = a0, a1, a2
    f[4 + speed] = 1
    f[15 + speed] = a0
    f[26 + speed] = a1
    f[37 + speed] = a2
    bits = []
    for k in range(16):
        acc = 0
        row = W[k]
        for j, w in enumerate(row):
            if w and f[j]:
                acc ^= 1
        bits.append(acc)
    x = 0
    for b in bits:
        x = (x << 1) | b
    return x

def decode_affine(cw16: int):
    """Brute-force invert: find (speed,addr) whose encoded cw16 matches. Returns (speed,addr) or None."""
    for s in range(0,11):
        for a in range(0,8):
            if encode_affine(s, a) == cw16:
                return (s, a)
    return None

# ---------- Wire mapping (MSB-first) ----------
def symbols_to_wave(bits16: str) -> str:
    """
    Map 16 symbol bits to on-air waveform MSB-first.
    Symbol mapping: 0 -> '10', 1 -> '100'
    """
    return ''.join('100' if s == '1' else '10' for s in bits16)

def leading_zeros(s: str) -> int:
    i = 0
    while i < len(s) and s[i] == '0':
        i += 1
    return i

def build_tx_bits(speed: int, addr: int,
                  preamble: str = DEFAULT_PREAMBLE,
                  idle1: int = DEFAULT_IDLE1,
                  idle0: int = DEFAULT_IDLE0,
                  repeats: int = DEFAULT_REPEATS,
                  gap_zeros: int = DEFAULT_GAP_ZEROS,
                  auto_gap: bool = True) -> str:
    """
    Build (preamble + body + '1'*idle1 + '0'*idle0) * repeats.
    If auto_gap=True, choose idle0 so that (#zeros before next '1') == gap_zeros.
    With preamble="00000" and gap_zeros=7 -> idle0 = 7 - 5 = 2 (matches recordings).
    """
    if auto_gap:
        idle0 = max(0, gap_zeros - leading_zeros(preamble))
    cw = encode_affine(speed, addr)
    bits16 = int_to_bits(cw, 16)                    # MSB..LSB
    body = symbols_to_wave(bits16)                  # expand 0->10, 1->100
    burst = f"{preamble}{body}{'1'*idle1}{'0'*idle0}"
    return burst * repeats

def decode_file_to_unique_cw16(raw_bits: str):
    """Return sorted unique 16-bit codewords from a raw capture."""
    segs = split_by_long_ones(keep_bits(raw_bits), MIN_ONES_SPLIT)
    out = []
    for seg in segs:
        b16 = tail_decode_16(seg)
        if b16:
            out.append(bits_to_int(b16))
    return sorted(set(out))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="AC Infinity 433.9 MHz (Model C) encoder/decoder")
    sub = ap.add_subparsers(dest='cmd', required=True)

    # Decode a capture to unique 16-bit words and (speed,addr)
    p_dec = sub.add_parser('decode', help='Decode raw 0/1 capture -> 16-bit codewords and (speed,addr)')
    p_dec.add_argument('-f','--file', help='capture file (default: stdin)')
    p_dec.add_argument('--show16', action='store_true')

    # Generate 16-bit codeword
    p_gen = sub.add_parser('gen', help='Encode (speed,addr) -> 16-bit codeword')
    p_gen.add_argument('--speed', type=int, required=True)
    p_gen.add_argument('--addr',  type=int, required=True)

    # Generate full on-air waveform
    p_wave = sub.add_parser('genwave', help='Encode to full on-air bitstream (preamble+repeat+idle)*repeats')
    p_wave.add_argument('--speed', type=int, required=True)
    p_wave.add_argument('--addr',  type=int, required=True)
    p_wave.add_argument('--repeats', type=int, default=DEFAULT_REPEATS)
    p_wave.add_argument('--preamble', default=DEFAULT_PREAMBLE)
    p_wave.add_argument('--idle-ones', type=int, default=DEFAULT_IDLE1)
    p_wave.add_argument('--idle-zeros', type=int, default=DEFAULT_IDLE0)
    p_wave.add_argument('--gap-zeros', type=int, default=DEFAULT_GAP_ZEROS, help='target zeros before next 1 (auto mode)')
    p_wave.add_argument('--no-auto-gap', action='store_true', help='disable automatic gap calculation')
    p_wave.add_argument('--out', help='write to file (default: stdout)')

    args = ap.parse_args()

    if args.cmd == 'decode':
        data = Path(args.file).read_text(encoding='utf-8') if args.file else input()
        cws = decode_file_to_unique_cw16(data)
        print(f"Unique codewords: {len(cws)}")
        for i, cw in enumerate(cws, 1):
            bits16 = int_to_bits(cw,16)
            dec = decode_affine(cw)
            if args.show16:
                if dec:
                    s,a = dec
                    print(f"[{i}] {bits16}  (0x{cw:04x})  -> speed={s} addr={a:03b}")
                else:
                    print(f"[{i}] {bits16}  (0x{cw:04x})  -> unknown")
            else:
                if dec:
                    print(f"0x{cw:04x}  speed={dec[0]} addr={dec[1]:03b}")
                else:
                    print(f"0x{cw:04x}  unknown")

    elif args.cmd == 'gen':
        cw = encode_affine(args.speed, args.addr)
        print(int_to_bits(cw,16))

    elif args.cmd == 'genwave':
        bits = build_tx_bits(args.speed, args.addr,
                             preamble=args.preamble,
                             idle1=args.idle_ones,
                             idle0=args.idle_zeros,
                             repeats=args.repeats,
                             gap_zeros=args.gap_zeros,
                             auto_gap=not args.no_auto_gap)
        if args.out:
            Path(args.out).write_text(bits, encoding='utf-8')
        else:
            print(bits)

if __name__ == '__main__':
    main()
