#!/usr/bin/env python3
import argparse, sys, csv, json, zipfile, tempfile, shutil
from pathlib import Path
from itertools import groupby
from collections import defaultdict, Counter

# ---------- low-level helpers ----------

def keep_bits(s: str) -> str:
    return ''.join(c for c in s if c in '01')

def rle(bits: str):
    return [(k, len(list(g))) for k, g in groupby(bits)]

def split_by_long_ones(bits: str, min_ones: int = 10):
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
    """Collect all (1,0) pairs where len(1)=1 and len(0) in {1,2}; return last 16 symbols."""
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

def a_bits(addr: int):
    return (addr & 1, (addr>>1)&1, (addr>>2)&1)

# ---------- dataset loader ----------

def load_dataset(csv_path: Path, root: Path|None, zippath: Path|None, min_ones=10, dedupe=True):
    """
    CSV columns: file,addr,speed
    If zippath is provided, files are read from inside the zip using relative paths from CSV.
    root can override base dir.
    Returns list of dicts: {'file','addr','speed','bits16','cw16'}
    """
    rows = []
    if zippath:
        zf = zipfile.ZipFile(zippath, 'r')
    else:
        zf = None

    with csv_path.open('r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            rel = Path(r['file'])
            addr = int(str(r['addr']), 0)
            speed = int(str(r['speed']), 0)
            if zf:
                with zf.open(str(rel).lstrip('./'), 'r') as fh:
                    data = fh.read().decode('utf-8', errors='ignore')
            else:
                path = (root/rel) if root else rel
                data = Path(path).read_text(encoding='utf-8')
            segs = split_by_long_ones(keep_bits(data), min_ones)
            bits16s = []
            for seg in segs:
                b16 = tail_decode_16(seg)
                if b16:
                    bits16s.append(b16)
            if not bits16s:
                print(f"[WARN] No decodable codeword in {rel}", file=sys.stderr)
                continue
            if dedupe:
                bits16s = sorted(set(bits16s))
            # choose majority if multiple
            mc = Counter(bits16s).most_common(1)[0][0]
            rows.append({'file': str(rel), 'addr': addr, 'speed': speed,
                         'bits16': mc, 'cw16': bits_to_int(mc)})
    if zf:
        zf.close()
    return rows

# ---------- diagnostics ----------

def consistency_report(rows):
    """
    For each speed, for any two addresses i,j: mask = ai XOR aj, y = cw_i XOR cw_j.
    For a given mask at a given speed, all y should be identical if a separable
    address-delta model holds. Return dict of conflicts.
    """
    by_speed = defaultdict(list)
    for r in rows:
        by_speed[r['speed']].append((r['addr'], r['cw16']))
    problems = {}
    for spd, lst in by_speed.items():
        if len(lst) < 2: continue
        bucket = defaultdict(set)
        n = len(lst)
        for i in range(n):
            ai, ci = lst[i]
            for j in range(i+1, n):
                aj, cj = lst[j]
                m = ai ^ aj
                if m == 0: continue
                y = ci ^ cj
                bucket[m].add(y)
        bad = {m: {int_to_bits(y,16) for y in ys} for m, ys in bucket.items() if len(ys) > 1}
        if bad:
            problems[spd] = bad
    return problems

# ---------- model tests ----------

def separable_fit(rows):
    """
    Try cw16 = base_s XOR Δ(mask) with Δ independent of speed.
    Return (ok, base_run, base_off, deltas_run, deltas_off)
    """
    # Learn deltas for run using pairwise equations across all run speeds
    run = [r for r in rows if 1 <= r['speed'] <= 10]
    off = [r for r in rows if r['speed'] == 0]

    def solve_deltas(group):
        # Build rows: for fixed speed pairs: cw_i XOR cw_j = Δ · (ai XOR aj)
        X, Ybits = [], []
        by_s = defaultdict(list)
        for r in group:
            by_s[r['speed']].append(r)
        for s, lst in by_s.items():
            if len(lst) < 2: continue
            ref = lst[0]
            ar, cr = ref['addr'], ref['cw16']
            for g in lst[1:]:
                a, c = g['addr'], g['cw16']
                m = a ^ ar
                if m == 0: continue
                X.append([m & 1, (m>>1)&1, (m>>2)&1])  # a0,a1,a2
                y = c ^ cr
                Ybits.append([(y >> (15-k)) & 1 for k in range(16)])
        if len(X) < 3: return None
        # Solve each bit; require consistency
        deltas = [0,0,0]
        for k in range(16):
            # 3 unknowns, Gaussian elim GF(2)
            A = [X[i][:] + [Ybits[i][k]] for i in range(len(X))]
            m, n = len(A), 3
            r = c = 0; piv=[-1]*n
            while r<m and c<n:
                pr = next((i for i in range(r,m) if A[i][c]&1), None)
                if pr is None: c+=1; continue
                A[r],A[pr]=A[pr],A[r]; piv[c]=r
                for i in range(m):
                    if i!=r and (A[i][c]&1):
                        for kk in range(c, n+1):
                            A[i][kk] ^= A[r][kk]
                r+=1; c+=1
            # contradiction?
            for i in range(r,m):
                if (A[i][0]|A[i][1]|A[i][2])==0 and (A[i][3]&1):
                    return None
            # back-sub
            x=[0,0,0]
            for j in range(n-1,-1,-1):
                if piv[j]!=-1:
                    ssum=A[piv[j]][n]
                    for kk in range(j+1,n):
                        ssum ^= (A[piv[j]][kk] & x[kk])
                    x[j]=ssum&1
            # set bit k into deltas
            for j in range(3):
                deltas[j] = (deltas[j] << 1) | x[j]
        return deltas

    d_run = solve_deltas(run)
    d_off = solve_deltas(off)

    if d_run is None:
        return (False, {}, None, None, None)

    # Build bases: base = cw ^ (Δ · addr)
    def addr_mask(deltas, addr):
        out=0
        for i in range(3):
            if (addr>>i)&1: out ^= deltas[i]
        return out

    base_run = {}
    for s in range(1,11):
        rr = [r for r in run if r['speed']==s]
        if not rr: continue
        b = rr[0]['cw16'] ^ addr_mask(d_run, rr[0]['addr'])
        base_run[s] = b
        # verify all others
        for r in rr[1:]:
            if (b ^ addr_mask(d_run, r['addr'])) != r['cw16']:
                return (False, {}, None, None, None)

    base_off = None
    if off:
        if d_off is None:
            return (False, {}, None, None, None)
        base_off = off[0]['cw16'] ^ addr_mask(d_off, off[0]['addr'])
        for r in off[1:]:
            if (base_off ^ addr_mask(d_off, r['addr'])) != r['cw16']:
                return (False, {}, None, None, None)

    return (True, base_run, base_off, d_run, d_off)

def speed_dep_fit(rows):
    """
    cw16 = base[s] XOR (a0·D0[s] ⊕ a1·D1[s] ⊕ a2·D2[s])  independently per speed.
    """
    ok_all = True
    bases, deltas = {}, {}
    for s in sorted({r['speed'] for r in rows}):
        grp = [r for r in rows if r['speed']==s]
        if len(grp) < 1: continue
        # solve deltas for this speed
        # build X,Y as before but within this speed
        X,Ybits = [], []
        if len(grp)>=2:
            ref = grp[0]
            ar,cr = ref['addr'], ref['cw16']
            for g in grp[1:]:
                a,c = g['addr'], g['cw16']
                m = a ^ ar
                if m==0: continue
                X.append([m & 1, (m>>1)&1, (m>>2)&1])
                y = c ^ cr
                Ybits.append([(y>>(15-k))&1 for k in range(16)])
        # if <3 rows, we can't solve uniquely; assume zeros
        if len(X) < 3:
            D=[0,0,0]
        else:
            D=[0,0,0]
            for k in range(16):
                A=[X[i][:]+[Ybits[i][k]] for i in range(len(X))]
                m,n=len(A),3; r=c=0; piv=[-1]*n
                while r<m and c<n:
                    pr=next((i for i in range(r,m) if A[i][c]&1), None)
                    if pr is None: c+=1; continue
                    A[r],A[pr]=A[pr],A[r]; piv[c]=r
                    for i in range(m):
                        if i!=r and (A[i][c]&1):
                            for kk in range(c,n+1):
                                A[i][kk]^=A[r][kk]
                    r+=1; c+=1
                for i in range(r,m):
                    if (A[i][0]|A[i][1]|A[i][2])==0 and (A[i][3]&1):
                        ok_all=False
                        break
                if not ok_all: break
                x=[0,0,0]
                for j in range(n-1,-1,-1):
                    if piv[j]!=-1:
                        ssum=A[piv[j]][n]
                        for kk in range(j+1,n):
                            ssum ^= (A[piv[j]][kk] & x[kk])
                        x[j]=ssum&1
                for j in range(3):
                    D[j]=(D[j]<<1)|x[j]
            if not ok_all: break
        deltas[s]=D
        # base from first sample
        def mask(D, a):
            out=0
            for i in range(3):
                if (a>>i)&1: out ^= D[i]
            return out
        b = grp[0]['cw16'] ^ mask(D, grp[0]['addr'])
        bases[s]=b
        # verify
        for g in grp[1:]:
            if (b ^ mask(D, g['addr'])) != g['cw16']:
                ok_all=False
                break
        if not ok_all: break

    return ok_all, bases, deltas

def affine_with_interactions_fit(rows):
    """
    Solve per bit a linear system over GF(2):
      cw_bit = w · feat   (mod 2)
    feat = [1, a0, a1, a2, S0..S10, a0*S0..a2*S10]  (total 1+3+11+33 = 48 features)
    Requires enough samples; reports exact fit or bit error counts.
    """
    # build design matrix
    speeds = sorted({r['speed'] for r in rows})
    s2idx = {s:i for i,s in enumerate(speeds)}  # usually 0..10
    def feats(addr, speed):
        a0,a1,a2 = a_bits(addr)
        S = [1 if i==s2idx[speed] else 0 for i in range(len(speeds))]
        F = [1, a0, a1, a2] + S
        # interactions
        for a in (a0,a1,a2):
            F += [a*Si for Si in S]
        return F

    X = [feats(r['addr'], r['speed']) for r in rows]
    Y = [r['cw16'] for r in rows]
    m, p = len(X), len(X[0])
    # per-bit solve
    W = [[0]*p for _ in range(16)]
    errs = [0]*16

    def gf2_solve(A):
        m = len(A); n = len(A[0])-1
        r=c=0; piv=[-1]*n
        while r<m and c<n:
            pr=next((i for i in range(r,m) if A[i][c]&1), None)
            if pr is None: c+=1; continue
            A[r],A[pr] = A[pr],A[r]; piv[c]=r
            for i in range(m):
                if i!=r and (A[i][c]&1):
                    for k in range(c,n+1):
                        A[i][k]^=A[r][k]
            r+=1; c+=1
        # back-sub with consistency; if inconsistent, return None
        for i in range(r,m):
            if all(A[i][j]==0 for j in range(n)) and A[i][n]==1:
                return None, piv
        x=[0]*n
        for j in range(n-1,-1,-1):
            if piv[j]!=-1:
                ssum=A[piv[j]][n]
                for k in range(j+1,n):
                    ssum ^= (A[piv[j]][k] & x[k])
                x[j]=ssum&1
        return x, piv

    # try solve each bit exactly; if no exact solution, compute residuals by picking a null-space solution (x=0) and measure error
    for k in range(16):
        A = []
        for i in range(m):
            yk = (Y[i] >> (15-k)) & 1
            A.append([*(xi & 1 for xi in X[i]), yk])
        sol, _ = gf2_solve([row[:] for row in A])
        if sol is None:
            # cannot fit exactly; leave weights zero, compute residuals
            pred = []
            for i in range(m):
                dot = 0
                # zero weights -> predict 0
                pred.append(0)
            err = sum(((Y[i]>>(15-k)) & 1) ^ pred[i] for i in range(m))
            errs[k] = err
        else:
            W[k] = sol
            # check residuals
            err = 0
            for i in range(m):
                dot = 0
                xi = X[i]
                for j in range(p):
                    dot ^= (xi[j] & sol[j])
                err += ((Y[i]>>(15-k)) & 1) ^ (dot & 1)
            errs[k] = err

    exact = all(e==0 for e in errs)
    return exact, errs, W, p, speeds

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Analyze 16-bit codewords & test models")
    ap.add_argument('--csv', required=True, help='CSV with columns: file,addr,speed')
    ap.add_argument('--root', help='optional root dir for files')
    ap.add_argument('--zip', dest='zipfile', help='optional zip file containing captures')
    ap.add_argument('--dedupe', action='store_true', help='dedupe per file')
    ap.add_argument('--min-ones', type=int, default=10)
    ap.add_argument('--export-json', help='if a model fits (separable or speed-dep), write JSON here')
    args = ap.parse_args()

    rows = load_dataset(Path(args.csv), Path(args.root).expanduser() if args.root else None,
                        Path(args.zipfile).expanduser() if args.zipfile else None,
                        min_ones=args.min_ones, dedupe=args.dedupe)
    if not rows:
        print("No rows loaded.", file=sys.stderr); sys.exit(2)

    # Summary table
    print(f"Loaded {len(rows)} labeled samples.")
    table = defaultdict(lambda: {})
    for r in rows:
        table[(r['speed'], r['addr'])] = r['bits16']
    for s in sorted({r['speed'] for r in rows}):
        addrs = sorted({r['addr'] for r in rows if r['speed']==s})
        line = [f"speed {s:>2}:"]
        for a in addrs:
            line.append(f"a{a:03b}:{table[(s,a)]}")
        print("  " + "  ".join(line))

    # Consistency diagnostics
    probs = consistency_report(rows)
    if probs:
        print("\n[Diag] Separable model contradictions:")
        for spd in sorted(probs):
            print(f"  speed {spd}:")
            for m, ys in sorted(probs[spd].items()):
                print(f"    addr_xor={m:03b} -> cw_xor in {{{', '.join(sorted(ys))}}}")
    else:
        print("\n[Diag] No contradictions found for separable address deltas (necessary condition holds).")

    # Try separable fit
    ok_sep, base_run, base_off, d_run, d_off = separable_fit(rows)
    print("\n[Model A] Separable deltas:", "OK" if ok_sep else "FAIL")
    if ok_sep:
        print("  delta_run:", [f"0x{d:04x}" for d in d_run])
        if d_off: print("  delta_off:", [f"0x{d:04x}" for d in d_off])
        for s in sorted(base_run): print(f"  base_run[{s}]=0x{base_run[s]:04x}")
        if base_off is not None: print(f"  base_off   =0x{base_off:04x}")

    # Try speed-dependent deltas
    ok_spd, bases, deltas = speed_dep_fit(rows)
    print("\n[Model B] Speed-dependent deltas:", "OK" if ok_spd else "FAIL")
    if ok_spd:
        for s in sorted(bases):
            print(f"  s={s:>2} base=0x{bases[s]:04x}  D=[{', '.join('0x%04x'%d for d in deltas[s])}]")

    # Try general affine with interactions
    exact, errs, W, p, speeds = affine_with_interactions_fit(rows)
    print("\n[Model C] Affine GF(2) with interactions:", "EXACT" if exact else "NOT exact")
    if not exact:
        print("  bit errors per position (MSB..LSB):", errs)

    # Export a working model JSON if possible
    if args.export_json and (ok_sep or ok_spd):
        out = {"mode": "separable" if ok_sep else "speed_dep"}
        if ok_sep:
            out.update({
                "base_run": {str(k): base_run[k] for k in base_run},
                "base_off": base_off,
                "delta_run": d_run,
                "delta_off": d_off if d_off else [],
            })
        else:
            out.update({
                "bases": {str(k): bases[k] for k in bases},
                "deltas": {str(k): deltas[k] for k in deltas},
            })
        Path(args.export_json).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f"\nExported model to {args.export_json}")

if __name__ == '__main__':
    main()
