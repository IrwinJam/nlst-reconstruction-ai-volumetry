"""checks the sybil boxes actually sit on the expert masks"""
import glob, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, pydicom
import qc_gate1 as Q

ROOT = r"C:\MetaSeg_data"


def walk(items):
    for it in items:
        yield it
        if "ContentSequence" in it:
            yield from walk(it.ContentSequence)


def boxes_in(sr_file):
    """pulls the boxes out of a sybil SR, keyed by sop uid"""
    ds = pydicom.dcmread(sr_file)
    out = {}
    for it in walk(ds.ContentSequence):
        if it.get("ValueType") != "SCOORD":
            continue
        sop = None
        for sub in it.get("ContentSequence", []):
            if "ReferencedSOPSequence" in sub:
                sop = sub.ReferencedSOPSequence[0].ReferencedSOPInstanceUID
        if sop is None:
            continue
        g = np.array(it.GraphicData, float).reshape(-1, 2)
        out.setdefault(sop, []).append((g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()))
    return out


def check(args):
    pid, ct_series, seg_series, twin, sr_files = args
    try:
        heads = Q.ct_headers(Q.find(pid, "CT", ct_series)[0])
        masks, _ = Q.seg_masks(glob.glob(os.path.join(Q.find(pid, "SEG", seg_series)[0], "*.dcm"))[0], heads)
        z_of = {s: round(v["pos"][2], 2) for s, v in heads.items()}
        if twin:  # twin has the same pixel grid so map its sops to our slices by z
            tw = Q.ct_headers(Q.find(pid, "CT", twin)[0])
            z_of.update({s: round(v["pos"][2], 2) for s, v in tw.items()})
        box_by_z = {}
        for f in sr_files:
            for sop, bl in boxes_in(f).items():
                if sop in z_of:
                    box_by_z.setdefault(z_of[sop], []).extend(bl)
        mask_by_z = {round(heads[s]["pos"][2], 2): m for s, m in masks.items() if s in heads and m.sum() >= Q.MIN_PX}
        both = [z for z in mask_by_z if z in box_by_z]
        inside = total = 0
        for z in both:
            m = mask_by_z[z]
            box_mask = np.zeros_like(m)
            for x0, y0, x1, y1 in box_by_z[z]:
                box_mask[int(np.floor(y0)):int(np.ceil(y1)) + 1, int(np.floor(x0)):int(np.ceil(x1)) + 1] = True
            inside += (m & box_mask).sum(); total += m.sum()
        return {"patient_id": pid, "mask_slices": len(mask_by_z), "box_slices_on_our_grid": len(box_by_z),
                "slices_with_both": len(both),
                "slice_coverage": round(len(both) / len(mask_by_z), 3) if mask_by_z else 0.0,
                "mask_inside_box": round(inside / total, 3) if total else None}
    except Exception as e:
        return {"patient_id": pid, "error": f"{type(e).__name__}: {e}"}


def main():
    man = pd.read_csv(os.path.join(ROOT, "nlstseg_manifest.csv"), dtype={"PatientID": str}).set_index("PatientID")
    sy = pd.read_csv(os.path.join(ROOT, "results", "audit", "sybil_series_match.csv"), dtype={"patient_id": str})
    pairs = pd.read_csv(os.path.join(ROOT, "results", "qc", "aim1_pairs_final.csv"), dtype={"patient_id": str})
    twin = pairs.drop_duplicates("patient_id").set_index("patient_id").paired_series.to_dict()
    use = sy[sy.match.isin(["same series as expert mask", "same-scan twin (Aim-1 pair)"])]
    jobs = []
    for pid, g in use.groupby("patient_id"):
        # grab all of this patients sybil reports (folders are named by the SR series uid)
        # boxes only get kept if they point at our series or its twin
        files = glob.glob(os.path.join(ROOT, "raw", "nlst_sybil", pid, "*", "*.dcm"))
        jobs.append((pid, man.loc[pid, "ct_series"], man.loc[pid, "seg_series"], twin.get(pid), files))
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = pd.DataFrame(list(ex.map(check, jobs, chunksize=4)))
    res = res.merge(sy.drop_duplicates("patient_id")[["patient_id", "group"]], on="patient_id")
    res.to_csv(os.path.join(ROOT, "results", "audit", "sybil_box_check.csv"), index=False)
    ok = res.dropna(subset=["mask_inside_box"])
    good = ok[(ok.mask_inside_box >= 0.8) & (ok.slice_coverage >= 0.5)]
    s = {"checked": len(res), "errors": int(res.get("error", pd.Series(dtype=str)).notna().sum()),
         "median_mask_inside_box": float(ok.mask_inside_box.median()),
         "median_slice_coverage": float(ok.slice_coverage.median()),
         "usable_(>=80% mask inside box & >=50% slice coverage)": len(good),
         "usable_by_group": good.group.value_counts().to_dict()}
    json.dump(s, open(os.path.join(ROOT, "results", "audit", "sybil_box_summary.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
