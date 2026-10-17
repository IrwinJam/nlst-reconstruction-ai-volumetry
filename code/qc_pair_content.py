"""checks paired recons are really the same scan by pixel content (D2)"""
import glob, os
import numpy as np, pandas as pd
from scipy import ndimage
import qc_gate1 as Q

ROOT = Q.ROOT
pairs = pd.read_csv(ROOT / "results" / "qc" / "qc_pairs.csv", dtype={"patient_id": str})
final = pd.read_csv(ROOT / "results" / "qc" / "final_status.csv", dtype={"patient_id": str})
man = pd.read_csv(ROOT / "nlstseg_manifest.csv", dtype={"PatientID": str})
ok = set(final[final.final_status == "PASS"].patient_id)
rows = []
for p in pairs[(pairs.status == "PASS") & pairs.patient_id.isin(ok)].itertuples():
    r = man[man.PatientID == p.patient_id].iloc[0]
    ha = Q.ct_headers(Q.find(p.patient_id, "CT", r.ct_series)[0])
    hb = {round(v["pos"][2], 2): v for v in Q.ct_headers(Q.find(p.patient_id, "CT", p.paired_series)[0]).values()}
    masks, _ = Q.seg_masks(glob.glob(os.path.join(Q.find(p.patient_id, "SEG", r.seg_series)[0], "*.dcm"))[0], ha)
    rc, rf, md = [], [], []
    for s, mk in masks.items():
        if s not in ha or mk.sum() < Q.MIN_PX:
            continue
        A = ndimage.gaussian_filter(np.clip(Q.hu(ha[s]["file"]), -1000, 400), 3)
        B = ndimage.gaussian_filter(np.clip(Q.hu(hb[round(ha[s]["pos"][2], 2)]["file"]), -1000, 400), 3)
        ys, xs = np.nonzero(mk)
        cy, cx = int(ys.mean()), int(xs.mean())
        sl = (slice(max(cy - 64, 0), cy + 64), slice(max(cx - 64, 0), cx + 64))
        rc.append(np.corrcoef(A[sl].ravel(), B[sl].ravel())[0, 1])
        rf.append(np.corrcoef(A.ravel(), B.ravel())[0, 1])
        md.append(np.median(np.abs(A[sl] - B[sl])))
    rows.append({"patient_id": p.patient_id, "paired_series": p.paired_series,
                 "pair": f"{p.annotated_kernel}->{p.paired_kernel}",
                 "r_crop_min": round(min(rc), 4), "r_crop_median": round(float(np.median(rc)), 4),
                 "r_full_min": round(min(rf), 4), "blur_abs_diff_hu_median": round(float(np.median(md)), 1)})
out = pd.DataFrame(rows).sort_values("r_crop_min")
out.to_csv(ROOT / "results" / "qc" / "qc_pairs_content.csv", index=False)
print(out.head(12).to_string(index=False))
print("\nr_crop_min quantiles:", out.r_crop_min.quantile([0, .02, .05, .1, .25, .5]).round(4).to_dict())
