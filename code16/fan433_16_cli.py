#!/usr/bin/env python3
# fan433_16_cli.py
import argparse, sys, csv
from pathlib import Path
from fan433_16 import Fan433Model16, bits_to_int, int_to_bits
import os
from datetime import datetime
from collections import Counter

def main():
    ap = argparse.ArgumentParser(description="AC Infinity 433.9 MHz — 16-bit codeword tools")
    sub = ap.add_subparsers(dest='cmd', required=True)

    # NEW: auto-extract one file per command from a long capture
    p_ext = sub.add_parser('extract-series', help='Split a long capture into one file per command (+optional labels CSV)')
    p_ext.add_argument('--file', required=True, help='path to long 0/1 capture (stdout of URH rx)')
    p_ext.add_argument('--outdir', help='output directory (default: ./series_<timestamp>)')
    p_ext.add_argument('--csv', help='optional path to write labels CSV (file,addr,speed)')
    p_ext.add_argument('--addr', type=int, help='force address label for all commands (0..7)')
    p_ext.add_argument('--start-speed', type=int, help='if given, assign sequential speeds from here (wrap 1..10->0)')
    p_ext.add_argument('--expect', type=int, default=11, help='expected number of distinct commands (default 11)')
    p_ext.add_argument('--repeats', type=int, default=Fan433Model16.DEFAULT_REPEATS, help='repeats to synthesize per file')
    p_ext.add_argument('--preamble', default=Fan433Model16.DEFAULT_PREAMBLE)
    p_ext.add_argument('--idle-ones', type=int, default=Fan433Model16.DEFAULT_IDLE1)
    p_ext.add_argument('--idle-zeros', type=int, default=Fan433Model16.DEFAULT_IDLE0)

    # Decode a raw capture to unique 16-bit codewords
    p_dec = sub.add_parser('decode', help='Decode raw bitstream into unique 16-bit codewords')
    p_dec.add_argument('-f','--file', help='0/1 capture (default: stdin)')
    p_dec.add_argument('--show', action='store_true', help='print 16-bit strings')

    # Learn model from labeled CSV
    p_lrn = sub.add_parser('learn16', help='Learn model from labeled CSV: file,addr,speed')
    p_lrn.add_argument('csv', help='CSV with columns: file,addr,speed')
    p_lrn.add_argument('--dedupe', action='store_true', help='dedupe per file (default: keep all)')

    # Show current model
    p_show = sub.add_parser('show-model', help='Print the learned 16-bit model')

    # Generate one 16-bit codeword
    p_gen = sub.add_parser('gen16', help='Generate 16-bit codeword for speed/address')
    p_gen.add_argument('--speed', type=int, required=True)
    p_gen.add_argument('--addr',  type=int, required=True)

    # Generate full TX waveform (preamble + repeats + idle)
    p_wave = sub.add_parser('genwave', help='Generate full TX waveform (0/1)')
    p_wave.add_argument('--speed', type=int, required=True)
    p_wave.add_argument('--addr',  type=int, required=True)
    p_wave.add_argument('--repeats', type=int, default=Fan433Model16.DEFAULT_REPEATS)
    p_wave.add_argument('--preamble', default=Fan433Model16.DEFAULT_PREAMBLE)
    p_wave.add_argument('--idle-ones', type=int, default=Fan433Model16.DEFAULT_IDLE1)
    p_wave.add_argument('--idle-zeros', type=int, default=Fan433Model16.DEFAULT_IDLE0)
    p_wave.add_argument('--out', help='write to file (default: stdout)')

    # Self-test (encode -> decode)
    p_st = sub.add_parser('selftest', help='Round-trip check with current model')
    p_st.add_argument('--speed', type=int, required=True)
    p_st.add_argument('--addr',  type=int, required=True)
    p_st.add_argument('--show', action='store_true')

    # URH command
    p_urh = sub.add_parser('urh', help='Print urh_cli command to TX a bits file')
    p_urh.add_argument('--file', required=True)
    p_urh.add_argument('--device', default='USRP')
    p_urh.add_argument('--fc', type=float, default=433.9e6)
    p_urh.add_argument('--fs', type=float, default=1e6)
    p_urh.add_argument('--sps', type=int, default=600)
    p_urh.add_argument('--gain', type=int, default=35)
    p_urh.add_argument('--dev', type=float, default=30e3)
    p_urh.add_argument('--pause', type=int, default=0)
    p_urh.add_argument('--extra', default='')

    args = ap.parse_args()
    fan = Fan433Model16()

    if args.cmd == 'decode':
        data = Path(args.file).read_text(encoding='utf-8') if args.file else sys.stdin.read()
        cw16_list = fan.decode_capture(data, dedupe=True)
        print(f"Unique codewords: {len(cw16_list)}")
        for i, b in enumerate(cw16_list, 1):
            cw = bits_to_int(b)
            dec = fan.decode_cw16(cw)
            if args.show:
                if dec:
                    s, a = dec
                    print(f"[{i}] {b}  (0x{cw:04x})  -> speed={s} addr={a:03b}")
                else:
                    print(f"[{i}] {b}  (0x{cw:04x})  -> unknown (model not fit?)")
            else:
                if dec:
                    print(f"0x{cw:04x}  speed={dec[0]} addr={dec[1]:03b}")
                else:
                    print(f"0x{cw:04x}  unknown")

    elif args.cmd == 'learn16':
        rows = []
        csv_path = Path(args.csv).expanduser()
        with csv_path.open('r', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for r in rd:
                file = Path(r['file']).expanduser()
                addr = int(r['addr'], 0) if isinstance(r['addr'], str) else int(r['addr'])
                speed = int(r['speed'], 0) if isinstance(r['speed'], str) else int(r['speed'])
                raw = file.read_text(encoding='utf-8')
                cw16s = fan.decode_capture(raw, dedupe=True if args.dedupe else False)
                if not cw16s:
                    print(f"[WARN] no decodable codewords in {file}", file=sys.stderr)
                    continue
                # In a well-labeled CSV each file corresponds to one speed; if multiple found, take majority.
                from collections import Counter
                cnt = Counter(cw16s)
                cw16_bits = cnt.most_common(1)[0][0]
                rows.append({"cw16": bits_to_int(cw16_bits), "addr": addr, "speed": speed})
        if not rows:
            print("No usable labeled samples.", file=sys.stderr); sys.exit(2)
        fan.learn_from_labeled(rows)
        print("Model learned & saved to", fan.MODEL_PATH)

    elif args.cmd == 'show-model':
        print("Model file:", fan.MODEL_PATH)
        if not fan.base_run and fan.base_off is None:
            print("  (no model learned yet)")
        else:
            for s in sorted(fan.base_run):
                print(f"base_run[{s}]= {int_to_bits(fan.base_run[s],16)}  (0x{fan.base_run[s]:04x})")
            if fan.base_off is not None:
                print(f"base_off   = {int_to_bits(fan.base_off,16)}  (0x{fan.base_off:04x})")
            if fan.delta_run:
                print("delta_run [a0,a1,a2]:")
                for i,d in enumerate(fan.delta_run):
                    print(f"  d{i} = {int_to_bits(d,16)}  (0x{d:04x})")
            if fan.delta_off:
                print("delta_off [a0,a1,a2]:")
                for i,d in enumerate(fan.delta_off):
                    print(f"  d{i} = {int_to_bits(d,16)}  (0x{d:04x})")

    elif args.cmd == 'gen16':
        cw = fan.encode_cw16(args.speed, args.addr)
        print(int_to_bits(cw,16))

    elif args.cmd == 'genwave':
        cw = fan.encode_cw16(args.speed, args.addr)
        wave = fan.build_tx_stream(cw, repeats=args.repeats,
                                   preamble=args.preamble,
                                   idle_ones=args.idle_ones,
                                   idle_zeros=args.idle_zeros)
        if args.out:
            Path(args.out).expanduser().write_text(wave, encoding='utf-8')
        else:
            print(wave)

    elif args.cmd == 'selftest':
        cw = fan.encode_cw16(args.speed, args.addr)
        bits16 = int_to_bits(cw, 16)
        wave = fan.build_tx_stream(cw)
        # decode our own wave
        got_list = fan.decode_capture(wave, dedupe=True)
        if not got_list:
            print("SELFTEST: FAIL (no decodes)"); sys.exit(1)
        got = got_list[0]
        cw_got = bits_to_int(got)
        ok_encode = (cw_got == cw)
        dec = fan.decode_cw16(cw_got)
        ok_decode = (dec == (args.speed, args.addr))
        print("SELFTEST:", "PASS" if (ok_encode and ok_decode) else "FAIL")
        if args.show:
            print(" expected 16:", bits16)
            print(" decoded 16 :", got)
            print(" model decode:", dec)

        sys.exit(0 if (ok_encode and ok_decode) else 1)

    elif args.cmd == 'urh':
        file_path = str(Path(args.file).expanduser().resolve())
        cmd = fan.urh_tx_cmd(args.device, file_path, fc_hz=args.fc, fs_sps=args.fs,
                             sps=args.sps, tx_gain=args.gain, deviation_hz=args.dev,
                             pause_ms=args.pause, extra_args=args.extra)
        print(cmd)

    elif args.cmd == 'extract-series':
        raw = Path(args.file).expanduser().read_text(encoding='utf-8')

        # Split the capture into segments by long idle 1-runs
        segs = fan._segments_by_long_ones(raw, fan.min_ones)

        # Decode each segment tail to 16 symbols; collapse repeats keeping order
        ordered_bits16 = []
        for seg in segs:
            b16 = fan._decode_pairs_tail16(seg)
            if not b16:
                continue
            if not ordered_bits16 or b16 != ordered_bits16[-1]:
                ordered_bits16.append(b16)

        if not ordered_bits16:
            print("No decodable commands found.", file=sys.stderr)
            sys.exit(2)

        if args.expect and len(ordered_bits16) != args.expect:
            print(f"[WARN] Found {len(ordered_bits16)} commands (expected {args.expect}).", file=sys.stderr)

        # Prepare output directory
        outdir = Path(args.outdir) if args.outdir else Path(f"series_{datetime.now():%Y%m%d_%H%M%S}")
        outdir.mkdir(parents=True, exist_ok=True)

        # Helper to wrap speeds 1..10 -> 0
        def wrap_next_speed(s: int) -> int:
            if s == 0:
                return 1  # next after 0 goes back to 1
            return 0 if s == 10 else (s + 1)

        # Process and write each command
        rows = []  # for CSV
        next_speed = args.start_speed if args.start_speed is not None else None

        for idx, b16 in enumerate(ordered_bits16, 1):
            cw = bits_to_int(b16)

            # Try to decode with learned model (if available)
            lab_speed, lab_addr = None, None
            dec = fan.decode_cw16(cw)
            if dec is not None:
                lab_speed, lab_addr = dec

            # If model didn't decode, fall back to --addr/--start-speed scheme
            if lab_addr is None and args.addr is not None:
                lab_addr = args.addr
            if lab_speed is None and next_speed is not None:
                lab_speed = next_speed
                next_speed = wrap_next_speed(next_speed)

            # Synthesize a normalized waveform (preamble + 1 repeat + idle) for this command
            burst = fan.build_tx_burst(cw,
                                       preamble=args.preamble,
                                       idle_ones=args.idle_ones,
                                       idle_zeros=args.idle_zeros)
            # If you prefer multiple repeats per file, you can use build_tx_stream(cw, repeats=args.repeats, ...)

            # File naming with available labels (keeps things sortable)
            name_bits = [f"{idx:02d}"]
            if lab_speed is not None: name_bits.append(f"s{lab_speed}")
            if lab_addr  is not None: name_bits.append(f"a{lab_addr:03b}")
            fname = "_".join(name_bits) + ".txt"

            out_path = outdir / fname
            out_path.write_text(burst, encoding='utf-8')

            # CSV row (only if requested)
            if args.csv:
                rows.append({"file": str(out_path), "addr": "" if lab_addr is None else lab_addr,
                             "speed": "" if lab_speed is None else lab_speed})

        # Write CSV if requested
        if args.csv:
            csv_path = Path(args.csv).expanduser()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open('w', encoding='utf-8') as f:
                f.write("file,addr,speed\n")
                for r in rows:
                    f.write(f"{r['file']},{r['addr']},{r['speed']}\n")
            print(f"Wrote CSV: {csv_path}")

        print(f"Wrote {len(ordered_bits16)} command files to {outdir.resolve()}")

if __name__ == '__main__':
    main()
