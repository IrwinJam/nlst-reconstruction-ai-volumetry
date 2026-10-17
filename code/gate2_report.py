"""gate 2 report, best val dice per run"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\MetaSeg_data") / "results"
THRESH = 0.60
PREFIX = sys.argv[1] if len(sys.argv) > 1 else ""  # "" = round 0, "v2_" = round 1


def best(run, pat):
    f = ROOT / run / "log.txt"
    if not f.exists():
        return None
    # skip resume lines, they also say "val Dice"
    text = "\n".join(l for l in f.read_text(encoding="utf-8").splitlines() if "resume" not in l)
    vals = [float(m.group(1)) for m in re.finditer(pat, text)]
    return {"best": max(vals), "last": vals[-1], "n_evals": len(vals)} if vals else None


rows = {}
for s in range(3):
    rows[f"{PREFIX}conv_s{s}"] = best(f"{PREFIX}conv_s{s}", r"val Dice ([0-9.]+)")
    rows[f"{PREFIX}maml_s{s}"] = best(f"{PREFIX}maml_s{s}", r"val Dice@k4 ([0-9.]+)")
for v in ["GE_MEDICAL_SYSTEMS", "SIEMENS", "TOSHIBA", "Philips"]:
    rows[f"{PREFIX}lovo_{v}_conv"] = best(f"{PREFIX}lovo_{v}", r"CONV epoch .* val Dice ([0-9.]+)")
    rows[f"{PREFIX}lovo_{v}_maml"] = best(f"{PREFIX}lovo_{v}", r"val Dice@k4 ([0-9.]+)")

conv = [rows[f"{PREFIX}conv_s{s}"]["best"] for s in range(3) if rows[f"{PREFIX}conv_s{s}"]]
verdict = "PENDING" if len(conv) < 3 else ("PASS" if np.mean(conv) >= THRESH else "FAIL")
out = {"threshold": THRESH, "criterion": "mean best validation Dice of conventional models (3 seeds) >= 0.60",
       "conv_mean": round(float(np.mean(conv)), 4) if conv else None,
       "conv_sd": round(float(np.std(conv, ddof=1)), 4) if len(conv) > 1 else None,
       "verdict": verdict, "runs": rows}
(ROOT / f"gate2{PREFIX.rstrip('_') and '_' + PREFIX.rstrip('_')}.json").write_text(json.dumps(out, indent=2))
lines = [f"# Gate 2: model quality (validation only)\n", f"**Verdict: {verdict}**. {out['criterion']}.\n",
         f"Conventional mean best val Dice: {out['conv_mean']} (SD {out['conv_sd']})\n",
         "| Run | Best val Dice | Last | Evaluations |", "|---|---|---|---|"]
for k, v in rows.items():
    lines.append(f"| {k} | {v['best']:.4f} | {v['last']:.4f} | {v['n_evals']} |" if v else f"| {k} | NA | NA | NA |")
lines.append("\nMAML values are validation Dice after adapting on k=4 annotated slices; conventional values are"
             " without adaptation, so the two columns are not directly comparable (that comparison is Aim 3).")
(ROOT / f"gate2{PREFIX.rstrip('_') and '_' + PREFIX.rstrip('_')}.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
