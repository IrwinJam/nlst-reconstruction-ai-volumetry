"""volume change inside vs outside the prompt region (D7)"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, SimpleITK as sitk
import nlst_seg as S
import box_mode as B
import evaluate as E
from prepare_nlst import to_grid

ROOT = S.ROOT
OUT = ROOT / "results" / "posthoc"
S.CACHE = ROOT / "cache" / "NLST_v2"


def nn_pred(case, n):
    f = ROOT / "nnunet" / "pred_ts" / f"{case}.nii.gz"
    img = sitk.ReadImage(str(f)); arr = sitk.GetArrayFromImage(img); sp = img.GetSpacing()
    if len(arr) != n:
        return None
    return np.stack([to_grid(s_.astype(np.float32), (sp[1], sp[0]), True) for s_ in arr]) >= 0.5


def summarize(df, model):
    d = df[(df.model == model) & (df.va_tot > 0)].groupby("patient_id").mean(numeric_only=True)
    pct = lambda a, b: 100 * (d[b] - d[a]).abs() / d[a]
    tot = pct("va_tot", "vb_tot") > 25
    din = d[d.va_in > 0]
    ins = (100 * (din.vb_in - din.va_in).abs() / din.va_in) > 25
    abs_out = (d.vb_out - d.va_out).abs().sum()
    abs_in = (d.vb_in - d.va_in).abs().sum()
    return {"n": len(d), "frac_total_gt25": round(float(tot.mean()), 4), "n_total_gt25": int(tot.sum()),
            "n_inside_evaluable": len(din), "frac_inside_gt25": round(float(ins.mean()), 4), "n_inside_gt25": int(ins.sum()),
            "share_of_abs_change_outside_region": round(float(abs_out / (abs_out + abs_in)), 4),
            "patients_with_any_outside_volume_A_or_B": int(((d.va_out > 0) | (d.vb_out > 0)).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp, aim1p, pts, ptype = E.split_lists()
    _, real = B.real_prompts()
    pids = [p for p in aim1p if p in real]
    st = S.Store(pids)
    bst = E.box_view(st, {p: real[p] for p in pids})
    rows = []
    models = [(f"unet_s{s}", E.load(ROOT / "results" / f"v2_conv_s{s}" / "conv_best.pt", st.cin)) for s in E.SEEDS]
    for p in pids:
        reg = B.region(bst, p)
        ml = 1.25 * 1.25 * st.d[p]["dz"] / 1000
        preds = {}
        for name, m in models:
            preds[name] = {k: S.predict(m, E.params_of(m), st, p, pair=(k == "B")) >= 0.5 for k in "AB"}
        pa, pb = nn_pred(f"NLST_{p}", st.n(p)), nn_pred(f"NLST_{p}p", st.n(p))
        if pa is not None and pb is not None:
            preds["nnunet"] = {"A": pa, "B": pb}
        for name, pr in preds.items():
            r = {"patient_id": p, "model": "unet" if name.startswith("unet") else "nnunet", "run": name}
            for k in "AB":
                r[f"v{k.lower()}_tot"] = pr[k].sum() * ml
                r[f"v{k.lower()}_in"] = (pr[k] & reg).sum() * ml
                r[f"v{k.lower()}_out"] = (pr[k] & ~reg).sum() * ml
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d7_rows.csv", index=False)
    summ = {"label": "POST-HOC (D7)", "n_patients": len(pids), "unet_3seed_mean": summarize(df, "unet"),
            "nnunet": summarize(df, "nnunet")}
    (OUT / "d7_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
