"""preprocessing, tumor slabs + masks into the training cache"""
from __future__ import annotations
import glob, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, pydicom, cv2
import qc_gate1 as Q

ROOT = Path(r"C:\MetaSeg_data")
# config through env vars bc windows spawns new worker processes and they need it (NLST_CACHE_OUT, NLST_TWO_WINDOWS=1)
OUT = Path(os.environ.get("NLST_CACHE_OUT", str(ROOT / "cache" / "NLST")))
TWO_WINDOWS = os.environ.get("NLST_TWO_WINDOWS") == "1"
PX_MM, SIZE, MARGIN, WIN = 1.25, 320, 5, (-1000.0, 400.0)  # 1.25mm grid, 320px, slab = tumor +-5 slices
WIN_ST = (-160.0, 240.0)  # soft tissue window, helps split tumor from mediastinum/fat


def to_grid(arr, sp, is_mask):
    h, w = arr.shape
    nh, nw = int(round(h * sp[0] / PX_MM)), int(round(w * sp[1] / PX_MM))
    interp = cv2.INTER_AREA if nh < h else cv2.INTER_LINEAR
    out = cv2.resize(arr.astype(np.float32), (nw, nh), interpolation=interp)
    if is_mask:
        out = (out >= 0.5).astype(np.float32)
    canvas = np.zeros((SIZE, SIZE), np.float32)
    ys, xs = max((nh - SIZE) // 2, 0), max((nw - SIZE) // 2, 0)
    yd, xd = max((SIZE - nh) // 2, 0), max((SIZE - nw) // 2, 0)
    ch, cw = min(nh, SIZE), min(nw, SIZE)
    canvas[yd:yd + ch, xd:xd + cw] = out[ys:ys + ch, xs:xs + cw]
    return canvas


def window(hu, win=WIN):
    lo, hi = win
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)


def process(args):
    r, pair = args
    pid = str(r["PatientID"])
    heads = Q.ct_headers(Q.find(pid, "CT", r["ct_series"])[0])
    masks, labels = Q.seg_masks(glob.glob(os.path.join(Q.find(pid, "SEG", r["seg_series"])[0], "*.dcm"))[0], heads)
    order = sorted(heads, key=lambda s: heads[s]["pos"][2])
    zs = np.array([heads[s]["pos"][2] for s in order])
    dz = float(np.median(np.abs(np.diff(zs))))
    tum_idx = [i for i, s in enumerate(order) if s in masks and masks[s].sum() >= Q.MIN_PX]
    lo, hi = max(min(tum_idx) - MARGIN, 0), min(max(tum_idx) + MARGIN, len(order) - 1)
    sp = heads[order[tum_idx[0]]]["sp"]
    pair_heads = {}
    if pair is not None:
        pair_heads = {round(v["pos"][2], 2): v for v in Q.ct_headers(Q.find(pid, "CT", pair)[0]).values()}
    imgs, msks, imgs_b, imgs_st, imgs_b_st, native_vox = [], [], [], [], [], 0
    for i in range(lo, hi + 1):
        s = order[i]
        m = masks.get(s, np.zeros(heads[s]["rc"], bool))
        m = m if m.sum() >= Q.MIN_PX else np.zeros_like(m)
        native_vox += int(m.sum())
        hu_a = Q.hu(heads[s]["file"])
        imgs.append(to_grid(window(hu_a), sp, False))
        if TWO_WINDOWS:
            imgs_st.append(to_grid(window(hu_a, WIN_ST), sp, False))
        msks.append(to_grid(m, sp, True))
        if pair_heads:
            hu_b = Q.hu(pair_heads[round(heads[s]["pos"][2], 2)]["file"])
            imgs_b.append(to_grid(window(hu_b), sp, False))
            if TWO_WINDOWS:
                imgs_b_st.append(to_grid(window(hu_b, WIN_ST), sp, False))
    msk = np.array(msks, np.uint8)
    rec = {"img": np.array(imgs, np.float16), "msk": msk, "tumor": msk.reshape(len(msk), -1).any(1),
           "z": zs[lo:hi + 1], "dz_mm": dz}
    if imgs_b:
        rec["img_pair"] = np.array(imgs_b, np.float16)
    if imgs_st:
        rec["img_st"] = np.array(imgs_st, np.float16)
    if imgs_b_st:
        rec["img_pair_st"] = np.array(imgs_b_st, np.float16)
    np.savez_compressed(OUT / f"{pid}.npz", **rec)
    return {"patient_id": pid, "vendor": r["Manufacturer"], "scanner": r["ManufacturerModelName"],
            "kernel": r["kernel"], "slice_thickness_mm": r["slice_thickness_mm"], "dz_mm": round(dz, 3),
            "pixel_mm": sp[0], "n_slab": len(msk), "n_tumor_slices": int(rec["tumor"].sum()),
            "expert_volume_ml_native": round(native_vox * sp[0] * sp[1] * dz / 1000, 3),
            "expert_volume_ml_grid": round(float(msk.sum()) * PX_MM * PX_MM * dz / 1000, 3),
            "paired_series": pair or "", "lesion_labels": " | ".join(labels)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "nlstseg_manifest.csv", dtype={"PatientID": str})
    ok = pd.read_csv(ROOT / "results" / "qc" / "final_status.csv", dtype={"patient_id": str})
    ok = set(ok[ok.final_status == "PASS"].patient_id)
    pairs = pd.read_csv(ROOT / "results" / "qc" / "aim1_pairs_final.csv", dtype={"patient_id": str})
    pair_of = pairs.drop_duplicates("patient_id").set_index("patient_id").paired_series.to_dict()
    jobs = [(r.to_dict(), pair_of.get(r.PatientID)) for _, r in man.iterrows() if r.PatientID in ok]
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(process, jobs, chunksize=4))
    df = pd.DataFrame(rows).sort_values("patient_id")
    df.to_csv(OUT / "patients.csv", index=False)
    rel = (df.expert_volume_ml_grid / df.expert_volume_ml_native - 1).abs()
    print(f"cached {len(df)} patients ({(df.paired_series != '').sum()} with paired reconstruction) -> {OUT}")
    print(f"slab slices: median {df.n_slab.median():.0f}, total {df.n_slab.sum()}; tumor slices total {df.n_tumor_slices.sum()}")
    print(f"resampling preserves expert volume: median |error| {rel.median():.2%}, 95th pct {rel.quantile(.95):.2%}")


if __name__ == "__main__":
    main()
