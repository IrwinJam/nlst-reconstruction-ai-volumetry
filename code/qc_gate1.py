"""gate 1 data checks for the masks and the paired recons"""
from __future__ import annotations
import glob, os, random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, pydicom
from scipy import ndimage

ROOT = Path(r"C:\MetaSeg_data")
OUT = ROOT / "results" / "qc"
HU_LO, HU_HI, MIN_PX, TOL = -300.0, 200.0, 10, 0.01  # 0.01mm grid tolerance, set before looking at any data


def ct_headers(ct_dir):
    h = {}
    for f in glob.glob(os.path.join(ct_dir, "*.dcm")):
        d = pydicom.dcmread(f, stop_before_pixels=True)
        h[d.SOPInstanceUID] = {"file": f, "pos": [float(v) for v in d.ImagePositionPatient],
                               "ori": [round(float(v), 4) for v in d.ImageOrientationPatient],
                               "sp": [float(v) for v in d.PixelSpacing], "rc": (d.Rows, d.Columns)}
    return h


def hu(file):
    d = pydicom.dcmread(file)
    return d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1)) + float(d.get("RescaleIntercept", 0))


def seg_masks(seg_file, heads=None, stats=None):
    """maps each mask frame to its ct slice. uses the source ref, falls back to z position (3 files dont have refs)"""
    ds = pydicom.dcmread(seg_file)
    px = ds.pixel_array
    px = px[None] if px.ndim == 2 else px
    zmap = {round(v["pos"][2], 2): s for s, v in heads.items()} if heads else {}
    out = {}
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        try:
            sop = fg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        except (AttributeError, IndexError):
            z = round(float(fg.PlanePositionSequence[0].ImagePositionPatient[2]), 2)
            sop = zmap.get(z)
            if sop is None:
                if stats is not None:
                    stats["max_frame_z_offset_mm"] = max(stats.get("max_frame_z_offset_mm", 0.0), 999.0)
                continue
            if stats is not None:
                stats["position_matched_frames"] = stats.get("position_matched_frames", 0) + 1
        if stats is not None and heads and sop in heads:
            dz = abs(float(fg.PlanePositionSequence[0].ImagePositionPatient[2]) - heads[sop]["pos"][2])
            stats["max_frame_z_offset_mm"] = max(stats.get("max_frame_z_offset_mm", 0.0), dz)
        mk = px[i] > 0
        out[sop] = mk if sop not in out else (out[sop] | mk)
    return out, [s.SegmentLabel for s in ds.SegmentSequence]


def find(pid, prefix, uid):
    return glob.glob(str(ROOT / "raw" / "nlst" / str(pid) / "*" / f"{prefix}_{uid}"))


def check_patient(args):
    r, pairs_for_pid = args
    pid = str(r["PatientID"])
    res = {"patient_id": pid, "vendor": r["Manufacturer"], "kernel": r["kernel"],
           "thickness_mm": r["slice_thickness_mm"]}
    try:
        return _check_patient(r, pairs_for_pid, pid, res)
    except Exception as e:
        return {**res, "status": "EXCLUDE", "reason": f"read error: {type(e).__name__}: {e}"}, []


def _check_patient(r, pairs_for_pid, pid, res):
    ct_dir, seg_dir = find(pid, "CT", r["ct_series"]), find(pid, "SEG", r["seg_series"])
    if not ct_dir:
        return {**res, "status": "EXCLUDE", "reason": "CT series not available in IDC"}, []
    seg_files = glob.glob(os.path.join(seg_dir[0], "*.dcm")) if seg_dir else []
    if not seg_files:
        return {**res, "status": "EXCLUDE", "reason": "mask file missing on disk"}, []
    heads = ct_headers(ct_dir[0])
    stats = {}
    masks, labels = seg_masks(seg_files[0], heads, stats)
    res["position_matched_frames"] = stats.get("position_matched_frames", 0)
    res["max_frame_z_offset_mm"] = round(stats.get("max_frame_z_offset_mm", 0.0), 3)
    if res["max_frame_z_offset_mm"] > TOL:
        return {**res, "status": "EXCLUDE",
                "reason": f"mask drawn on a different slice grid (frames up to {res['max_frame_z_offset_mm']:.2f} mm off CT slices)"}, []
    unmatched = [s for s in masks if s not in heads]
    tum = {s: m for s, m in masks.items() if s in heads and m.sum() >= MIN_PX}
    vin, vring, vox = [], [], 0
    for s, m in tum.items():
        img = hu(heads[s]["file"])
        ring = ndimage.binary_dilation(m, iterations=3) & ~m
        vin.append(img[m]); vring.append(img[ring]); vox += int(m.sum())
    if not tum:
        return {**res, "status": "EXCLUDE", "reason": "mask has no tumor slice >= 10 px",
                "n_mask_frames": len(masks), "n_unmatched": len(unmatched)}, []
    mean_in = float(np.concatenate(vin).mean()); mean_ring = float(np.concatenate(vring).mean())
    sp = heads[next(iter(tum))]["sp"]
    zs = sorted(heads[s]["pos"][2] for s in tum)
    thick = float(r["slice_thickness_mm"]) if pd.notna(r["slice_thickness_mm"]) else np.nan
    in_range = HU_LO <= mean_in <= HU_HI
    contrast = mean_in - mean_ring
    status, reason = "PASS", ""
    if unmatched:
        status, reason = "EXCLUDE", f"{len(unmatched)} mask frames reference missing CT slices"
    elif contrast < 100:
        status, reason = "REVIEW", "low tumor-vs-surround contrast (possible misalignment)"
    elif not in_range:
        status, reason = "PASS", "outside pre-specified HU range but strong contrast (likely ground-glass)"
    res.update({"n_ct_slices": len(heads), "n_mask_frames": len(masks), "n_unmatched": len(unmatched),
                "n_tumor_slices": len(tum), "tumor_voxels": vox,
                "volume_ml": round(vox * sp[0] * sp[1] * thick / 1000, 3),
                "mean_hu_inside": round(mean_in, 1), "mean_hu_ring": round(mean_ring, 1),
                "contrast_hu": round(contrast, 1), "in_prespecified_range": in_range,
                "z_extent_mm": round(zs[-1] - zs[0] + thick, 1), "lesion_labels": " | ".join(labels),
                "status": status, "reason": reason})

    pair_rows = []
    for _, p in pairs_for_pid.iterrows():
        pd_dir = find(pid, "CT", p["paired_series"])
        pr = {"patient_id": pid, "vendor": r["Manufacturer"], "annotated_kernel": p["annotated_kernel"],
              "paired_kernel": p["paired_kernel"], "paired_series": p["paired_series"]}
        if not pd_dir:
            pair_rows.append({**pr, "status": "EXCLUDE", "reason": "paired series missing"}); continue
        ph = ct_headers(pd_dir[0])
        a = {round(v["pos"][2], 2): v for v in heads.values()}
        b = {round(v["pos"][2], 2): v for v in ph.values()}
        same_z = set(a) == set(b)
        same_geo = same_z and all(
            np.allclose(a[z]["pos"][:2], b[z]["pos"][:2], atol=TOL) and np.allclose(a[z]["sp"], b[z]["sp"], atol=1e-4)
            and a[z]["ori"] == b[z]["ori"] and a[z]["rc"] == b[z]["rc"] for z in a)
        hub = []
        if same_geo:
            for s, m in tum.items():
                zb = round(heads[s]["pos"][2], 2)
                hub.append(hu(b[zb]["file"])[m])
        pr.update({"n_slices_annot": len(a), "n_slices_paired": len(b), "same_positions": same_z,
                   "same_geometry": same_geo, "mean_hu_inside_annot": round(mean_in, 1),
                   "mean_hu_inside_paired": round(float(np.concatenate(hub).mean()), 1) if hub else None,
                   "status": "PASS" if same_geo else "EXCLUDE",
                   "reason": "" if same_geo else ("slice positions differ" if not same_z else "spacing/origin/orientation differ")})
        pair_rows.append(pr)
    return res, pair_rows


def montage(rows, fname, title, paired=False, n=24, seed=42):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = random.Random(seed)
    pick = rng.sample(rows, min(n, len(rows)))
    cols = 4 if paired else 6
    nrow = int(np.ceil(len(pick) / (cols // 2 if paired else cols)))
    fig, axes = plt.subplots(nrow, cols, figsize=(cols * 2.4, nrow * 2.5))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, item in enumerate(pick):
        panels = item["panels"]
        for j, (img, mk, lab) in enumerate(panels):
            ax = axes.ravel()[i * len(panels) + j]
            ys, xs = np.nonzero(mk)
            cy, cx = int(ys.mean()), int(xs.mean())
            h = 64
            y0, x0 = max(cy - h, 0), max(cx - h, 0)
            ax.imshow(np.clip((img[y0:y0 + 2 * h, x0:x0 + 2 * h] + 1000) / 1400, 0, 1), cmap="gray")
            ax.contour(mk[y0:y0 + 2 * h, x0:x0 + 2 * h], levels=[0.5], colors="#ff5a36", linewidths=1.2)
            ax.set_title(lab, fontsize=7)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def montage_items(qc, pairs_ok, man):
    items_m, items_p = [], []
    ok = qc[qc.status == "PASS"].patient_id.tolist()
    for pid in random.Random(42).sample(ok, min(24, len(ok))):
        r = man[man.PatientID.astype(str) == pid].iloc[0]
        heads = ct_headers(find(pid, "CT", r.ct_series)[0])
        masks, _ = seg_masks(glob.glob(os.path.join(find(pid, "SEG", r.seg_series)[0], "*.dcm"))[0], heads)
        s = max((k for k in masks if k in heads), key=lambda k: masks[k].sum())
        q = qc[qc.patient_id == pid].iloc[0]
        items_m.append({"panels": [(hu(heads[s]["file"]), masks[s],
                                    f"{pid} {q.vendor.split()[0][:7]} {q.kernel}\n{q.mean_hu_inside:.0f} HU")]})
    for _, p in pairs_ok.sample(min(8, len(pairs_ok)), random_state=42).iterrows():
        pid = p.patient_id
        r = man[man.PatientID.astype(str) == pid].iloc[0]
        ha = ct_headers(find(pid, "CT", r.ct_series)[0]); hb = ct_headers(find(pid, "CT", p.paired_series)[0])
        masks, _ = seg_masks(glob.glob(os.path.join(find(pid, "SEG", r.seg_series)[0], "*.dcm"))[0], ha)
        s = max((k for k in masks if k in ha), key=lambda k: masks[k].sum())
        zb = {round(v["pos"][2], 2): v for v in hb.values()}[round(ha[s]["pos"][2], 2)]
        items_p.append({"panels": [(hu(ha[s]["file"]), masks[s], f"{pid} {p.annotated_kernel} (annotated)"),
                                   (hu(zb["file"]), masks[s], f"{pid} {p.paired_kernel} (same scan)")]})
    montage(items_m, "montage_masks.png", "Gate 1: expert masks on CT (24 random patients, lung window)")
    montage(items_p, "montage_pairs.png", "Gate 1: same scan, two reconstructions, same expert mask", paired=True, n=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "nlstseg_manifest.csv")
    pairs = pd.read_csv(ROOT / "kernel_pairs.csv")
    jobs = [(r.to_dict(), pairs[pairs.PatientID == r.PatientID]) for _, r in man.iterrows()]
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(check_patient, jobs, chunksize=4))
    qc = pd.DataFrame([a for a, _ in results])
    pq = pd.DataFrame([p for _, ps in results for p in ps])
    qc.to_csv(OUT / "qc_masks.csv", index=False)
    pq.to_csv(OUT / "qc_pairs.csv", index=False)
    ex_rows = [{"type": "mask", "patient_id": r.patient_id, "series": "", "reason": r.reason}
               for r in qc[qc.status != "PASS"].itertuples()]
    ex_rows += [{"type": "pair", "patient_id": r.patient_id, "series": r.paired_series, "reason": r.reason}
                for r in pq[pq.status != "PASS"].itertuples()]
    pd.DataFrame(ex_rows, columns=["type", "patient_id", "series", "reason"]).to_csv(OUT / "exclusions.csv", index=False)
    montage_items(qc, pq[pq.status == "PASS"], man)

    n = len(qc); npass = (qc.status == "PASS").sum()
    avail = qc[qc.reason != "CT series not available in IDC"]
    in_rng = int(avail.in_prespecified_range.fillna(False).sum())
    pair_pts = pq[pq.status == "PASS"].patient_id.nunique()
    print(f"masks: {npass}/{n} pass | in pre-specified HU range {in_rng}/{len(avail)} "
          f"({in_rng/len(avail):.1%}) | review {(qc.status=='REVIEW').sum()} | exclude {(qc.status=='EXCLUDE').sum()}")
    print(f"pairs: {(pq.status=='PASS').sum()}/{len(pq)} series pass -> {pair_pts} patients usable for Aim 1")
    print(qc.status.value_counts().to_string()); print(qc.reason.value_counts().to_string())


if __name__ == "__main__":
    main()
