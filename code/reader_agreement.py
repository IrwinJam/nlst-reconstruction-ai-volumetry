"""radiologist agreement, readers vs nlstseg and vs each other"""
import glob, json, os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import numpy as np, pandas as pd, pydicom
import qc_gate1 as Q

ROOT = r"C:\MetaSeg_data"
BASE = os.path.join(ROOT, "raw", "lidc_annot_nlst501")
OUT = os.path.join(ROOT, "results", "reader")
Z_TOL = 0.1  # mm. seg z positions are off by ~0.01mm rounding and slices are >=0.6mm apart so 0.1 is safe


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


def reader_masks(f, zidx, shape, origin_xy):
    """manual segments from one reader SEG file, put on our (z,row,col) grid"""
    ds = pydicom.dcmread(f)
    if (ds.Rows, ds.Columns) != shape[1:]:
        return None, "grid size differs"
    manual = {s.SegmentNumber for s in ds.SegmentSequence
              if s.get("SegmentAlgorithmType") == "MANUAL" and "lung" not in str(s.SegmentLabel).lower()}
    px = ds.pixel_array
    px = px[None] if px.ndim == 2 else px
    out, unmatched = {}, 0
    zarr = np.array(sorted(zidx))
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        n = fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        if n not in manual or not px[i].any():
            continue
        pos = [float(v) for v in fg.PlanePositionSequence[0].ImagePositionPatient]
        if max(abs(pos[0] - origin_xy[0]), abs(pos[1] - origin_xy[1])) > 0.01:
            return None, "in-plane origin differs"
        j = int(np.argmin(np.abs(zarr - pos[2])))
        if abs(zarr[j] - pos[2]) > Z_TOL:
            unmatched += 1
            continue
        k = zidx[zarr[j]]
        out.setdefault(n, np.zeros(shape, bool))[k] |= px[i] > 0
    return out, (f"{unmatched} foreground frames off our z grid" if unmatched else "")


def analyse(args):
    pid, ct_series, seg_series, files, group = args
    try:
        heads = Q.ct_headers(Q.find(pid, "CT", ct_series)[0])
        masks, _ = Q.seg_masks(glob.glob(os.path.join(Q.find(pid, "SEG", seg_series)[0], "*.dcm"))[0], heads)
        zs = sorted({round(v["pos"][2], 2) for v in heads.values()})
        zidx = {z: i for i, z in enumerate(zs)}
        any_s = next(iter(heads.values()))
        shape = (len(zs),) + tuple(any_s["rc"])
        origin = any_s["pos"][:2]
        ref = np.zeros(shape, bool)
        for s, m in masks.items():
            if s in heads and m.sum() >= Q.MIN_PX:
                ref[zidx[round(heads[s]["pos"][2], 2)]] |= m
        vox_ml = any_s["sp"][0] * any_s["sp"][1] * float(np.median(np.diff(zs))) / 1000
        rows, tum = [], {}
        for f in sorted(files):
            rd = pydicom.dcmread(f, stop_before_pixels=True)
            reader = str(rd.get("ContentCreatorName", os.path.basename(os.path.dirname(f))))
            segs, note = reader_masks(f, zidx, shape, origin)
            if segs is None:
                rows.append({"patient_id": pid, "reader": reader, "status": "skipped", "note": note})
                continue
            hit = [m for m in segs.values() if (m & ref).any()]
            r = {"patient_id": pid, "group": group, "reader": reader, "n_findings": len(segs), "note": note,
                 "nlstseg_ml": round(ref.sum() * vox_ml, 3)}
            if hit:
                u = np.logical_or.reduce(hit)
                tum[reader] = u
                r.update(status="segmented", dice_vs_nlstseg=round(dice(u, ref), 4), reader_ml=round(u.sum() * vox_ml, 3))
            else:
                r.update(status="missed", dice_vs_nlstseg=0.0, reader_ml=0.0)
            rows.append(r)
        pairs = [{"patient_id": pid, "group": group, "reader_a": a, "reader_b": b,
                  "dice": round(dice(tum[a], tum[b]), 4)} for a, b in combinations(sorted(tum), 2)]
        return rows, pairs
    except Exception as e:
        return [{"patient_id": pid, "status": "error", "note": f"{type(e).__name__}: {e}"}], []


def main():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, "nlstseg_manifest.csv"), dtype={"PatientID": str}).set_index("PatientID")
    grp = {p: g for g, v in json.load(open(os.path.join(ROOT, "results", "split.json"))).items() if g != "_meta" for p in v}
    sub = pd.read_csv(os.path.join(BASE, "our_seg_subset.csv"), dtype={"patient_id": str})
    jobs = [(p, man.loc[p, "ct_series"], man.loc[p, "seg_series"],
             glob.glob(os.path.join(BASE, "seg", p, "*", "*.dcm")), grp.get(p)) for p in sorted(sub.patient_id.unique())]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("READER_WORKERS", "2"))) as ex:
        res = list(ex.map(analyse, jobs, chunksize=2))
    rv = pd.DataFrame([r for rows, _ in res for r in rows])
    rp = pd.DataFrame([p for _, ps in res for p in ps])
    rv.to_csv(os.path.join(OUT, "reader_vs_nlstseg.csv"), index=False)
    rp.to_csv(os.path.join(OUT, "reader_pairs.csv"), index=False)
    seg = rv[rv.status == "segmented"]
    per_pt = rp.groupby("patient_id").dice.mean() if len(rp) else pd.Series(dtype=float)
    s = {"patients": int(rv.patient_id.nunique()), "reads": len(rv), "status": rv.status.value_counts().to_dict(),
         "reads_with_notes": int((rv.note.fillna("") != "").sum()),
         "nlstseg_vs_reader_dice_segmented_median": round(float(seg.dice_vs_nlstseg.median()), 4),
         "nlstseg_vs_reader_dice_segmented_mean": round(float(seg.dice_vs_nlstseg.mean()), 4),
         "nlstseg_vs_reader_dice_all_reads_mean_(miss=0)": round(float(rv[rv.status.isin(["segmented", "missed"])].dice_vs_nlstseg.mean()), 4),
         "inter_reader_pairs": len(rp),
         "inter_reader_dice_median": round(float(rp.dice.median()), 4) if len(rp) else None,
         "inter_reader_dice_mean": round(float(rp.dice.mean()), 4) if len(rp) else None,
         "inter_reader_dice_per_patient_mean_then_median": round(float(per_pt.median()), 4) if len(per_pt) else None,
         "patients_where_all_readers_segmented": int((rv[rv.status != "skipped"].groupby("patient_id").status
                                                      .apply(lambda x: (x == "segmented").all())).sum())}
    json.dump(s, open(os.path.join(OUT, "reader_summary.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
