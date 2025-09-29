#!/usr/bin/env python3
# fan433_16.py
"""
AC Infinity 433.9 MHz Fan Controller (16-bit codeword model)

- Each repeat on air decodes to 16 symbols via (1,0) pair lengths:
    (1,1)->'0'   (short low)
    (1,2)->'1'   (long  low)
- We treat the **entire 16 symbols** as the codeword.

Model learned from labeled data:
  For speeds 1..10 (“run”):  cw16 = base_run[s] XOR (addr_mask applied to Δ_run)
  For speed 0   (“off”)   :  cw16 = base_off    XOR (addr_mask applied to Δ_off)

where addr is 3 bits (a2 a1 a0), and Δ_* are three 16-bit delta vectors (one per DIP).

This file provides:
  - decoding of raw bitstream -> unique 16-bit codewords
  - learning the (base, deltas) from labeled samples
  - encoding (addr,speed) -> 16-bit codeword -> bitstream
  - URH CLI command generator (USRP B210-friendly)

Persisted model path: ~/.config/fan433_model16.json
"""

from pathlib import Path
from itertools import groupby
from typing import List, Tuple, Optional, Dict, Iterable
import json
import shlex

# ------------------------------ utils ------------------------------

def bits_to_int(b: str) -> int:
    return int(''.join(c for c in b if c in '01'), 2)

def int_to_bits(x: int, width: int) -> str:
    return format(x, f'0{width}b')

def rle(bits: str) -> List[Tuple[str, int]]:
    return [(k, len(list(g))) for k, g in groupby(bits)]

def keep_bits(s: str) -> str:
    return ''.join(c for c in s if c in '01')

# ------------------------------ class ------------------------------

class Fan433Model16:
    # Framing defaults (tuneable)
    DEFAULT_PREAMBLE = "000001010101010"  # matches your captures
    DEFAULT_REPEATS  = 8
    DEFAULT_IDLE1    = 12                  # '1's between repeats
    DEFAULT_IDLE0    = 7                   # '0's after those 1s (empirical)
    MIN_ONES_IDLE    = 10                  # split repeats on long '1' runs

    # Model persistence
    MODEL_PATH = Path("~/.config/fan433_model16.json").expanduser()

    def __init__(self,
                 preamble: str = None,
                 repeats: int = None,
                 idle_ones: int = None,
                 idle_zeros: int = None,
                 min_ones_idle: int = None):
        self.preamble   = self.DEFAULT_PREAMBLE if preamble   is None else preamble
        self.repeats    = self.DEFAULT_REPEATS  if repeats    is None else repeats
        self.idle_ones  = self.DEFAULT_IDLE1    if idle_ones  is None else idle_ones
        self.idle_zeros = self.DEFAULT_IDLE0    if idle_zeros is None else idle_zeros
        self.min_ones   = self.MIN_ONES_IDLE    if min_ones_idle is None else min_ones_idle

        # Learned model (all 16-bit integers)
        self.base_run: Dict[int, int] = {}      # speed 1..10 -> 16b int
        self.base_off: Optional[int]  = None    # speed 0 -> 16b int
        self.delta_run: List[int]     = []      # [d_a0, d_a1, d_a2] (16b each)
        self.delta_off: List[int]     = []      # [d_a0, d_a1, d_a2] (16b each)

        self._load_model()

        # Precompute subset maps when available
        self._subset_run: Optional[Dict[int, int]] = self._build_subset_map(self.delta_run) if self.delta_run else None
        self._subset_off: Optional[Dict[int, int]] = self._build_subset_map(self.delta_off) if self.delta_off else None

    # ------------------ decode raw -> 16-symbols ------------------

    def decode_capture(self, raw_bits: str, dedupe: bool = True) -> List[str]:
        """
        Return unique 16-symbol strings extracted from a raw 0/1 capture.
        Uses tail-aligned (1,0) pair decoding and splits on long 1-runs.
        """
        segs = self._segments_by_long_ones(keep_bits(raw_bits), self.min_ones)
        out: List[str] = []
        for seg in segs:
            cw16 = self._decode_pairs_tail16(seg)
            if cw16:
                out.append(cw16)
        if dedupe:
            out = sorted(set(out))
        return out

    @staticmethod
    def _segments_by_long_ones(bits: str, min_ones: int) -> List[str]:
        runs = rle(bits)
        segs, pos, last = [], 0, 0
        for v, l in runs:
            if v == '1' and l >= min_ones:
                segs.append(bits[last:pos])
                last = pos + l
            pos += l
        segs.append(bits[last:])
        return [s for s in segs if s and s.count('1') > 0]

    @staticmethod
    def _decode_pairs_tail16(seg_bits: str) -> Optional[str]:
        """
        Collect ALL (1,0) pairs with len(1)=1 and len(0)∈{1,2}.
        Return the **last 16** symbols (avoids preamble pairs).
        """
        runs = rle(seg_bits)
        syms = []
        for i in range(len(runs) - 1):
            (v1, l1), (v2, l2) = runs[i], runs[i + 1]
            if v1 == '1' and v2 == '0' and l1 == 1 and l2 in (1, 2):
                syms.append('1' if l2 == 2 else '0')
        if len(syms) >= 16:
            return ''.join(syms[-16:])
        return None

    # ------------------ model: encode / decode ------------------

    @staticmethod
    def _apply_addr_mask(deltas: List[int], addr: int) -> int:
        """XOR together the delta vectors selected by addr bits (bit0=a0, bit1=a1, bit2=a2)."""
        x = 0
        for i in range(3):
            if (addr >> i) & 1:
                x ^= deltas[i]
        return x

    @staticmethod
    def _build_subset_map(deltas: List[int]) -> Dict[int, int]:
        """Map every 3-bit mask to the XOR of selected deltas."""
        return {mask: Fan433Model16._apply_addr_mask(deltas, mask) for mask in range(8)}

    def encode_cw16(self, speed: int, addr: int) -> int:
        """Build a 16-bit codeword from (speed, addr) using the learned model."""
        if speed == 0:
            if self.base_off is None or not self.delta_off:
                raise ValueError("Model incomplete (off). Learn with speed=0 samples.")
            return self.base_off ^ self._apply_addr_mask(self.delta_off, addr)
        if not (1 <= speed <= 10):
            raise ValueError("speed must be in 0..10")
        if speed not in self.base_run or not self.delta_run:
            raise ValueError(f"Model incomplete (run). Missing base for speed {speed} or deltas.")
        return self.base_run[speed] ^ self._apply_addr_mask(self.delta_run, addr)

    def decode_cw16(self, cw16: int) -> Optional[Tuple[int, int]]:
        """
        Given a 16-bit codeword, return (speed, addr) if it matches the model; else None.
        """
        # Try run speeds
        if self.delta_run and self._subset_run:
            for s, b in self.base_run.items():
                delta_obs = cw16 ^ b
                # look up which address mask yields this delta
                for mask, val in self._subset_run.items():
                    if val == delta_obs:
                        return (s, mask)
        # Try off
        if self.base_off is not None and self.delta_off and self._subset_off:
            delta_obs = cw16 ^ self.base_off
            for mask, val in self._subset_off.items():
                if val == delta_obs:
                    return (0, mask)
        return None

    # --------------- learn model from labeled samples ---------------

    def learn_from_labeled(self, samples: List[Dict[str, int]]) -> None:
        """
        Learn the 16-bit model from labeled samples.
        Each sample must have: {"cw16": int, "addr": int(0..7), "speed": int(0..10)}.

        Steps:
          - Learn Δ_run by pooling **address differences** across all speeds 1..10
          - Learn Δ_off by pooling differences for speed 0
          - Compute base_run[s] and base_off from any sample via base = cw ^ (Δ · addr)
        """
        # Group samples
        run_samples = [s for s in samples if 1 <= s["speed"] <= 10]
        off_samples = [s for s in samples if s["speed"] == 0]

        # Learn deltas by difference equations (eliminate base)
        self.delta_run = self._solve_deltas_by_differences(run_samples)
        self.delta_off = self._solve_deltas_by_differences(off_samples)

        # Build subset maps
        self._subset_run = self._build_subset_map(self.delta_run) if self.delta_run else None
        self._subset_off = self._build_subset_map(self.delta_off) if self.delta_off else None

        # Compute bases
        self.base_run = {}
        for s in range(1, 11):
            # choose any sample for this speed
            cand = next((x for x in run_samples if x["speed"] == s), None)
            if cand is None:
                continue
            self.base_run[s] = cand["cw16"] ^ self._apply_addr_mask(self.delta_run, cand["addr"])

        if off_samples:
            cand0 = off_samples[0]
            self.base_off = cand0["cw16"] ^ self._apply_addr_mask(self.delta_off, cand0["addr"])

        # Save model
        self._save_model()

    @staticmethod
    def _addr_vec(addr: int) -> List[int]:
        return [(addr >> 0) & 1, (addr >> 1) & 1, (addr >> 2) & 1]

    def _solve_deltas_by_differences(self, samples: List[Dict[str, int]]) -> List[int]:
        """
        Solve for the 3 delta vectors (each 16-bit) using **address differences**:
          For same speed, cw(m_i) XOR cw(m_ref) = Δ · (m_i XOR m_ref)
        Pool rows across speeds (linear system over GF(2), 3 unknowns per bit).
        Returns [d0, d1, d2]. If not enough diversity, raises ValueError.
        """
        if not samples:
            return []

        # Build linear system rows across all speeds
        X_rows: List[List[int]] = []    # each row: 3-bit vector (addr_i XOR addr_ref)
        Y_bits: List[List[int]] = []    # same number of rows; each is 16-bit vector as list[0/1]

        # Group by speed so we only compare addresses within the same speed
        by_speed: Dict[int, List[Dict[str, int]]] = {}
        for s in samples:
            by_speed.setdefault(s["speed"], []).append(s)

        for s, group in by_speed.items():
            if len(group) < 2:
                continue
            # pick a reference address in this speed
            ref = group[0]
            a_ref = ref["addr"]
            c_ref = ref["cw16"]
            for g in group[1:]:
                a = g["addr"]
                x = a ^ a_ref
                if x == 0:
                    continue
                X_rows.append(self._addr_vec(x))
                # y = cw XOR cw_ref (16 bits -> list)
                y = g["cw16"] ^ c_ref
                Y_bits.append([ (y >> (15-k)) & 1 for k in range(16) ])  # MSB first

        # Need at least 3 independent rows
        if len(X_rows) < 3:
            raise ValueError("Not enough address diversity to solve deltas (need ≥3 distinct address differences).")

        # Solve 3 unknowns for each bit position k independently (GF(2))
        import copy
        d_bits = [[0]*16 for _ in range(3)]   # [d0_bits, d1_bits, d2_bits]
        for k in range(16):
            # Build augmented matrix A = [X | y_k]
            A = [row[:] + [Y_bits[i][k]] for i, row in enumerate(X_rows)]
            # Gaussian elimination over GF(2)
            m, n = len(A), 3
            r = c = 0
            piv = [-1]*n
            while r < m and c < n:
                pr = next((i for i in range(r, m) if A[i][c] & 1), None)
                if pr is None:
                    c += 1
                    continue
                A[r], A[pr] = A[pr], A[r]
                piv[c] = r
                for i in range(m):
                    if i != r and (A[i][c] & 1):
                        for k2 in range(c, n+1):
                            A[i][k2] ^= A[r][k2]
                r += 1
                c += 1
            # consistency check
            for i in range(r, m):
                if (A[i][0] | A[i][1] | A[i][2]) == 0 and (A[i][3] & 1):
                    raise ValueError("Inconsistent labeled data while solving deltas.")
            # back-substitute (free vars -> 0)
            xsol = [0,0,0]
            for j in range(n-1, -1, -1):
                if piv[j] != -1:
                    ssum = A[piv[j]][n]
                    for k2 in range(j+1, n):
                        ssum ^= (A[piv[j]][k2] & xsol[k2])
                    xsol[j] = ssum & 1
            # record bit k of each delta
            for j in range(3):
                d_bits[j][k] = xsol[j]

        # pack bits to ints
        deltas: List[int] = []
        for j in range(3):
            val = 0
            for k in range(16):
                val = (val << 1) | (d_bits[j][k] & 1)
            deltas.append(val)
        return deltas

    # ---------------- bitstream I/O + URH helpers ----------------

    @staticmethod
    def cw16_to_symbols(cw16: int) -> str:
        return int_to_bits(cw16, 16)

    @staticmethod
    def symbols_to_wave(bits16: str) -> str:
        """Map '0'->'10', '1'->'100' across the 16 symbols."""
        out = []
        for s in bits16:
            if s == '0':
                out.append('10')
            elif s == '1':
                out.append('100')
            else:
                raise ValueError("16-symbol string must contain only '0'/'1'")
        return ''.join(out)

    def build_tx_burst(self, cw16: int, preamble: Optional[str] = None,
                       idle_ones: Optional[int] = None, idle_zeros: Optional[int] = None) -> str:
        pre = self.preamble if preamble is None else preamble
        i1  = self.idle_ones if idle_ones is None else idle_ones
        i0  = self.idle_zeros if idle_zeros is None else idle_zeros
        body = self.symbols_to_wave(self.cw16_to_symbols(cw16))
        return f"{pre}{body}{'1'*i1}{'0'*i0}"

    def build_tx_stream(self, cw16: int, repeats: Optional[int] = None,
                        preamble: Optional[str] = None, idle_ones: Optional[int] = None,
                        idle_zeros: Optional[int] = None) -> str:
        reps = self.repeats if repeats is None else repeats
        return ''.join(self.build_tx_burst(cw16, preamble, idle_ones, idle_zeros) for _ in range(reps))

    @staticmethod
    def write_bits_file(path: str, bits: str) -> str:
        p = Path(path).expanduser().resolve()
        p.write_text(bits, encoding="utf-8")
        return str(p)

    def urh_tx_cmd(self, device: str, file_path: str,
                   fc_hz: float = 433.9e6, fs_sps: float = 1e6, sps: int = 600,
                   tx_gain: int = 35, deviation_hz: float = 30e3, pause_ms: int = 0,
                   verbose: bool = True, extra_args: str = "") -> str:
        parts = [
            "urh_cli","-d",device,"-f",f"{fc_hz}","-s",f"{fs_sps}","-g",f"{tx_gain}",
            "-mo","FSK","-sps",f"{sps}","-pm",f"{-float(deviation_hz)}",f"{float(deviation_hz)}",
            "-cf","0","-file",file_path,"-tx"
        ]
        if pause_ms: parts += ["-p", f"{pause_ms}ms"]
        if verbose: parts.append("-v")
        if extra_args: parts += shlex.split(extra_args)
        return " ".join(shlex.quote(p) for p in parts)

    def urh_tx_b210_cmd(self, file_path: str, **kw) -> str:
        return self.urh_tx_cmd("USRP", file_path, **kw)

    # ---------------- model persistence ----------------

    def _save_model(self):
        data = {
            "base_run": {str(k): int(v) for k, v in self.base_run.items()},
            "base_off": int(self.base_off) if self.base_off is not None else None,
            "delta_run": [int(x) for x in self.delta_run] if self.delta_run else [],
            "delta_off": [int(x) for x in self.delta_off] if self.delta_off else [],
        }
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.MODEL_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_model(self):
        if not self.MODEL_PATH.exists():
            return
        try:
            d = json.loads(self.MODEL_PATH.read_text(encoding="utf-8"))
            self.base_run = {int(k): int(v) for k, v in d.get("base_run", {}).items()}
            self.base_off = d.get("base_off", None)
            if self.base_off is not None:
                self.base_off = int(self.base_off)
            self.delta_run = [int(x) for x in d.get("delta_run", [])]
            self.delta_off = [int(x) for x in d.get("delta_off", [])]
        except Exception:
            # corrupt model file; ignore
            self.base_run, self.base_off, self.delta_run, self.delta_off = {}, None, [], []
