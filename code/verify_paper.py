"""checks every number in the manuscript traces back to a result file"""
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\MetaSeg_data")
DEFAULT = Path(r"C:\Users\Irwin\OneDrive\Desktop\NLST_Reconstruction_Paper\manuscript_RadAI.md")
SOURCES = ["results/eval/summary.json", "results/posthoc/d7_summary.json", "results/posthoc/d8_summary.json",
           "results/posthoc/d9_summary.json", "results/posthoc/d10_summary.json", "results/posthoc/d10b_summary.json",
           "results/reader/reader_summary.json", "results/gate2_v2.json", "results/posthoc/d12_summary.json", "results/posthoc/d13_summary.json"]
# design constants and counts from the protocol/qc files, not from results
DESIGN = {"601": "NLSTSeg participants (QC_REPORT)", "575": "Gate 1 usable", "19": "off-grid masks", "5": "unpublished CT / ±5 slices",
          "1": "single exclusions", "128": "Aim-1 participants", "91": "GE standard→bone pairs / % GE", "25": "lung→standard pairs / threshold",
          "8": "Philips D→C", "4": "Philips B→D / levels / k", "421": "participants with prompt", "107": "LIDC readers patients",
          "290": "train", "44": "val / % data loss", "112": "test", "42": "split seed", "1.25": "grid mm", "320": "grid px",
          "1000": "window", "400": "window", "160": "window", "240": "window", "100": "epochs / percent", "10": "mm dilation / 10 000",
          "000": "10 000 resamples", "0.99": "pair correlation", "0.01": "grid tolerance mm", "2.77": "RC constant", "95": "CI",
          "2.5": "2.5D", "3": "seeds / %", "9": "% Philips", "63": "% prompted test (71/112)", "103": "D10a common set", "99": "D13 common set",
          "110": "Aim-1 with prompt", "71": "prompted test", "181": "held-out with prompt", "43": "reader held-out", "35": "reader+prompt",
          "41": "test without prompt", "31": "U-Net >25% count", "15": "D7 count / %", "108": "D7 evaluable", "14": "D7", "104": "D7 nnU-Net evaluable",
          "124": "nnU-Net V_A>0", "3.11": "Python version", "255": "MedSAM 0-255 scaling", "0.85": "aug scale", "1.15": "aug scale", "512": "nnU-Net patch", "0.66": "nnU-Net spacing", "250": "nnU-Net epochs", "0.67": "fg prob", "209": "prompt offset donors", "50": "box epochs", "4000": "MAML iters", "12.5": "percentile rule", "37.5": "percentile rule", "62.5": "percentile rule", "87.5": "percentile rule", "5.18": "transformers version", "2.5": "PyTorch 2.5.1 / 2.5D", "2.8": "nnU-Net 2.8.1", "16": "%", "6": "channels", "7": "AJCC 7th", "12": "Philips pairs", "116": "GE pairs", "13": "%", "22": "%", "24": "%"}


def flatten(x, out):
    if isinstance(x, dict):
        for v in x.values():
            flatten(v, out)
    elif isinstance(x, list):
        for v in x:
            flatten(v, out)
    elif isinstance(x, (int, float)) and x == x:
        out.append(float(x))


def allowed():
    vals = []
    for s in SOURCES:
        f = ROOT / s
        if f.exists():
            flatten(json.loads(f.read_text()), vals)
    t1 = (ROOT / "results" / "cohort" / "table1.csv").read_text(encoding="utf-8")
    vals += [float(v) for v in re.findall(r"\d+(?:\.\d+)?", t1)]
    ok = set()
    for v in vals:
        for x in (v, abs(v), 100 * v, 100 * abs(v)):
            for d in range(0, 4):
                q = Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
                for s in (f"{x:.{d}f}", f"{q:f}"):   # python rounds half-even, papers round half-up, allow both
                    ok.add(s)
                    if s.startswith("0."):
                        ok.add(s[1:])  # rsna writes p values like .49
    return ok


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    text = path.read_text(encoding="utf-8")
    text = text.split("## References")[0] + text.split("## Tables")[1] if "## Tables" in text else text.split("## References")[0]
    text = text.split("## Summary Statement", 1)[-1]
    clean = re.sub(r"\((\d+(?:[,–]\d+)*)\)", " ", text)          # citation markers (3,4)
    clean = re.sub(r"\b(Figs?|Table|Tables)\s+\d+(,\s*\d+)*", " ", clean)  # figure/table numbers
    clean = re.sub(r"`[^`]*`", " ", clean)                         # code paths
    ok = allowed()
    bad = []
    for m in re.finditer(r"(?<![\w.])[−-]?(?:\d+(?:\.\d+)?|\.\d+)", clean):
        tok = m.group().replace("−", "-").lstrip("-")
        if tok in ok or tok in DESIGN:
            continue
        bad.append((tok, clean[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")))
    print(f"numeric tokens checked; untraced: {len(bad)}")
    for tok, ctx in bad:
        print(f"  UNTRACED {tok!r}: ...{ctx}...")
    print("RESULT:", "ALL TRACED" if not bad else "REVIEW THE LIST ABOVE")


if __name__ == "__main__":
    main()
