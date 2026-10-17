"""cropping control, rc of automatic volumes cropped to the box vs the box guided model (D13)"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"C:\MetaSeg_data")
PH, EV = ROOT / "results" / "posthoc", ROOT / "results" / "eval"
NBOOT = 10_000


def rc(va, vb):
    d = np.log(vb) - np.log(va)
    return 100 * (np.exp(2.77 * np.std(d, ddof=1) / np.sqrt(2)) - 1)


def main():
    d7 = pd.read_csv(PH / "d7_rows.csv", dtype={"patient_id": str})
    unet = d7[d7.model == "unet"].groupby("patient_id")[["va_in", "vb_in"]].mean()
    nn = d7[d7.model == "nnunet"].set_index("patient_id")[["va_in", "vb_in"]]
    bx = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    box = bx[bx.set == "aim1"].groupby("patient_id")[["vol_box_a", "vol_box_b"]].mean()
    vols = {"U-Net cropped to prompt": unet.values.T, "nnU-Net cropped to prompt": nn.values.T, "Box-guided U-Net": box.values.T}
    frames = {"U-Net cropped to prompt": unet, "nnU-Net cropped to prompt": nn, "Box-guided U-Net": box}
    common = None
    for f in frames.values():
        ok = set(f.index[(f.iloc[:, 0] > 0) & (f.iloc[:, 1] > 0)])
        common = ok if common is None else common & ok
    ids = sorted(common)
    arr = {k: (f.loc[ids].iloc[:, 0].values, f.loc[ids].iloc[:, 1].values) for k, f in frames.items()}
    point = {k: round(float(rc(*a)), 2) for k, a in arr.items()}
    # paired bootstrap, same patients for every model
    rng = np.random.default_rng(0)
    boots = {k: [] for k in arr}
    for _ in range(NBOOT):
        ix = rng.integers(0, len(ids), len(ids))
        for k, (a, b) in arr.items():
            boots[k].append(rc(a[ix], b[ix]))
    boots = {k: np.array(v) for k, v in boots.items()}
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]
    ref = "Box-guided U-Net"
    summ = {"label": "POST-HOC (D13)", "n_common": len(ids), "RC_pct": point, "RC_CI": {k: ci(v) for k, v in boots.items()},
            "ratio_to_box": {k: {"ratio": round(point[k] / point[ref], 2), "CI": ci(boots[k] / boots[ref])} for k in arr if k != ref}}
    (PH / "d13_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
