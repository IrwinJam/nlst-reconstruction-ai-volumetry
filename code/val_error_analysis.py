"""val error analysis, this is what led to D3"""
import glob, json, os
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
import nlst_seg as N
import qc_gate1 as Q

OUT = N.ROOT / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
sp = N.load_split()
val = sp["val"]
pts = pd.read_csv(N.CACHE / "patients.csv", dtype={"patient_id": str}).set_index("patient_id")
store = N.Store(val)


def roi_mask(gt, margin=16):
    m = np.zeros_like(gt, bool)
    tum = gt.reshape(len(gt), -1).any(1)
    ys, xs = np.nonzero(gt.any(0))
    y0, y1, x0, x1 = max(ys.min() - margin, 0), ys.max() + margin + 1, max(xs.min() - margin, 0), xs.max() + margin + 1
    m[np.ix_(tum, np.arange(y0, min(y1, gt.shape[1])), np.arange(x0, min(x1, gt.shape[2])))] = True
    return m, tum


def dice(p, g):
    d = p.sum() + g.sum()
    return float(2 * (p & g).sum() / d) if d else 1.0


rows = []
for s in range(3):
    m = N.UNet(base=32).to(N.DEV)
    m.load_state_dict(torch.load(N.ROOT / "results" / f"conv_s{s}" / "conv_best.pt", map_location=N.DEV))
    m.eval()
    base = {n: t.detach() for n, t in m.named_parameters()}
    for p in val:
        prob = N.predict(m, base, store, p)
        gt = store.d[p]["msk"].astype(bool)
        roi, tum = roi_mask(gt)
        r = {"seed": s, "patient_id": p, "vendor": pts.vendor[p], "kernel": pts.kernel[p],
             "volume_ml": pts.expert_volume_ml_grid[p], "n_tumor_slices": int(tum.sum())}
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            r[f"dice_slab_t{thr}"] = dice(prob >= thr, gt)
        pb = prob >= 0.5
        r["dice_slab"] = r["dice_slab_t0.5"]
        r["dice_tumor_slices"] = dice(pb[tum], gt[tum])
        r["dice_roi"] = dice(pb & roi, gt)
        fp = pb & ~gt
        r["fn_vox"] = int((gt & ~pb).sum()); r["tp_vox"] = int((gt & pb).sum()); r["gt_vox"] = int(gt.sum())
        r["fp_margin_slices"] = int(fp[~tum].sum())
        r["fp_tumor_slices_outside_roi"] = int((fp[tum] & ~roi[tum]).sum())
        r["fp_inside_roi"] = int((fp & roi).sum())
        rows.append(r)

df = pd.DataFrame(rows)

# resolution ceiling, mask -> 1.25mm grid -> back, no model involved
man = pd.read_csv(N.ROOT / "nlstseg_manifest.csv", dtype={"PatientID": str}).set_index("PatientID")
ceil = {}
for p in val:
    r = man.loc[p]
    heads = Q.ct_headers(Q.find(p, "CT", r.ct_series)[0])
    masks, _ = Q.seg_masks(glob.glob(os.path.join(Q.find(p, "SEG", r.seg_series)[0], "*.dcm"))[0], heads)
    inter = tot = 0
    for sop, mk in masks.items():
        if sop not in heads or mk.sum() < Q.MIN_PX:
            continue
        sp_ = heads[sop]["sp"]; h, w = mk.shape
        nh, nw = int(round(h * sp_[0] / N.PX_MM)), int(round(w * sp_[1] / N.PX_MM))
        small = cv2.resize(mk.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) >= 0.5
        back = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
        inter += (back & mk).sum(); tot += back.sum() + mk.sum()
    ceil[p] = 2 * inter / tot
df["resolution_ceiling_dice"] = df.patient_id.map(ceil)
df.to_csv(OUT / "val_errors.csv", index=False)

# patient level means over the 3 seeds
g = df.groupby("patient_id").mean(numeric_only=True)
size_bins = pd.cut(g.volume_ml, [0, 1, 5, 20, 1e9], labels=["<1 mL", "1-5 mL", "5-20 mL", ">20 mL"])
err = df[["fn_vox", "fp_margin_slices", "fp_tumor_slices_outside_roi", "fp_inside_roi"]].sum()
summary = {
    "n_val_patients": len(val),
    "dice_slab_mean": round(g.dice_slab.mean(), 4),
    "dice_slab_median": round(g.dice_slab.median(), 4),
    "dice_tumor_slices_mean": round(g.dice_tumor_slices.mean(), 4),
    "dice_roi_mean": round(g.dice_roi.mean(), 4),
    "resolution_ceiling_mean": round(float(np.mean(list(ceil.values()))), 4),
    "resolution_ceiling_min": round(float(np.min(list(ceil.values()))), 4),
    "threshold_sweep_mean": {t: round(g[f"dice_slab_t{t}"].mean(), 4) for t in (0.3, 0.4, 0.5, 0.6, 0.7)},
    "missed_patients_dice_lt_0.1": int((g.dice_slab < 0.1).sum()),
    "error_voxel_share": {k: round(float(v / err.sum()), 3) for k, v in err.items()},
    "dice_slab_by_size": g.groupby(size_bins, observed=False).dice_slab.agg(["count", "mean"]).round(3).to_dict(),
    "dice_roi_by_size": g.groupby(size_bins, observed=False).dice_roi.agg(["count", "mean"]).round(3).to_dict(),
    "dice_slab_by_vendor": df.groupby("vendor").dice_slab.mean().round(3).to_dict(),
}
(OUT / "val_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
