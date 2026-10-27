#!/usr/bin/env python3
"""Measure real audio at the length being generated, for use as a control.

Generated takes in this repository are 4 s, and the "real corpus" row in
reports/CUDA.md was one 60 s excerpt of a single clipped master. Comparing a 4 s
take with a 60 s master exaggerates the corpus's darkness, its level and its
low-band weight, so descriptor gaps measured that way are not like-for-like.

This cuts windows of the same length out of every file in a corpus directory,
runs `feral analyze` on each, and reports the median and interquartile range.
Pass `--output` a path inside an experiment directory; the JSON is the control
row to quote next to generated takes.

Example (the measurement behind arm K):

  python3 scripts/matched_corpus_control.py \
      --corpus experiments/audio-corpus-64/train --seconds 4 \
      --output experiments/big-1m4-c4000/corpus-control-4s.json
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile


def dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=pathlib.Path, help="directory of WAV files")
    ap.add_argument("--seconds", type=float, default=4.0, help="window length, matching generated takes")
    ap.add_argument("--fractions", default="0.25,0.5,0.75",
                    help="window centres as fractions of each file's duration")
    ap.add_argument("--reference", type=pathlib.Path,
                    help="reference file for the copying screen (default: first corpus file)")
    ap.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("target/release/feral"))
    ap.add_argument("--output", required=True, type=pathlib.Path)
    a = ap.parse_args()

    files = sorted(p for p in a.corpus.iterdir() if p.suffix.lower() == ".wav")
    if not files:
        sys.exit(f"no .wav files in {a.corpus}")
    reference = (a.reference or files[0]).resolve()
    fractions = [float(x) for x in a.fractions.split(",")]
    binary = a.binary.resolve()
    rows, skipped = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for src in files:
            try:
                length = dur(src)
            except (subprocess.CalledProcessError, ValueError):
                skipped.append((src.name, "unreadable duration"))
                continue
            if length < a.seconds + 1:
                skipped.append((src.name, f"{length:.2f} s shorter than {a.seconds + 1} s"))
                continue
            for frac in fractions:
                start = max(0.0, length * frac - a.seconds / 2)
                tag = f"{src.stem}@{frac:g}"
                wav = tmp / (tag + ".wav")
                cut = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-ss", f"{start:.3f}",
                                      "-i", str(src), "-t", f"{a.seconds:g}", "-ac", "1", "-ar", "16000",
                                      "-c:a", "pcm_f32le", str(wav)], capture_output=True, text=True)
                if cut.returncode != 0:
                    skipped.append((tag, "ffmpeg decode failed"))
                    continue
                ana = tmp / (tag + ".json")
                res = subprocess.run([str(binary), "analyze", "--input", str(wav),
                                      "--reference", str(reference), "--output", str(ana)],
                                     capture_output=True, text=True)
                if res.returncode != 0:
                    skipped.append((tag, res.stderr.strip()[-120:]))
                    continue
                sig = json.loads(ana.read_text())["signal"]
                rows.append(dict(file=src.name, fraction=frac, rms=sig["rms"],
                                 flatness=sig["spectral_flatness"], lag1=sig["lag1_uncentered"],
                                 bands=sig["spectral_band_fractions"]))
    if not rows:
        sys.exit("no windows measured")

    def m(key):
        return statistics.median(r[key] for r in rows)

    def iqr(key):
        return [round(x, 4) for x in statistics.quantiles([r[key] for r in rows], n=4)]

    def band(i):
        return statistics.median(r["bands"][i] for r in rows)

    def band_iqr(i):
        return [round(x, 4) for x in statistics.quantiles([r["bands"][i] for r in rows], n=4)]

    summary = dict(corpus=str(a.corpus), seconds=a.seconds, fractions=fractions,
                   windows=len(rows), files=len({r["file"] for r in rows}), skipped=skipped,
                   reference=str(reference),
                   median=dict(rms=m("rms"), flatness=m("flatness"), lag1=m("lag1"),
                               bands=[band(i) for i in range(4)]),
                   iqr=dict(rms=iqr("rms"), flatness=iqr("flatness"), lag1=iqr("lag1"),
                            bands=[band_iqr(i) for i in range(4)]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(summary=summary, windows=rows), indent=1) + "\n")

    md = summary["median"]
    print(f"{len(rows)} windows of {a.seconds:g} s from {summary['files']} files "
          f"({len(skipped)} skipped)")
    print(f"  median  rms {md['rms']:.3f}  flatness {md['flatness']:.3f}  lag-1 {md['lag1']:.3f}  "
          f"bands {' / '.join(f'{b:.2f}' for b in md['bands'])}")
    print(f"  IQR     rms {summary['iqr']['rms']}  flatness {summary['iqr']['flatness']}  "
          f"lag-1 {summary['iqr']['lag1']}")
    for name, why in skipped[:5]:
        print(f"  skipped {name}: {why}")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
