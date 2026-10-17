"""checks which sybil / lidc501 annotations land on our series"""
import glob, json, os
from collections import Counter
import pandas as pd, pydicom

ROOT = r"C:\MetaSeg_data"
man = pd.read_csv(os.path.join(ROOT, "nlstseg_manifest.csv"), dtype={"PatientID": str}).set_index("PatientID")
sp = json.load(open(os.path.join(ROOT, "results", "split.json")))
ours = {p: k for k in ("aim1", "train", "val", "test") for p in sp[k]}
pairs = pd.read_csv(os.path.join(ROOT, "results", "qc", "aim1_pairs_final.csv"), dtype={"patient_id": str})
twin = pairs.drop_duplicates("patient_id").set_index("patient_id").paired_series.to_dict()


def walk(items):
    for it in items:
        yield it
        if "ContentSequence" in it:
            yield from walk(it.ContentSequence)


rows = []
for f in glob.glob(os.path.join(ROOT, "raw", "nlst_sybil", "*", "*", "*.dcm")):
    ds = pydicom.dcmread(f)
    pid = str(ds.PatientID)
    if pid not in ours:
        continue
    ev = [s.SeriesInstanceUID for e in ds.get("CurrentRequestedProcedureEvidenceSequence", [])
          for s in e.get("ReferencedSeriesSequence", [])]
    n_boxes = sum(1 for it in walk(ds.ContentSequence) if it.get("ValueType") == "SCOORD")
    study = str(ds.StudyInstanceUID)
    ct, st = man.loc[pid, "ct_series"], man.loc[pid, "StudyInstanceUID"]
    if ct in ev:
        match = "same series as expert mask"
    elif twin.get(pid) in ev:
        match = "same-scan twin (Aim-1 pair)"
    elif study == st:
        match = "same study, different series"
    else:
        match = "different study (other screening round)"
    rows.append({"patient_id": pid, "group": ours[pid], "sybil_series": ";".join(ev), "n_boxes": n_boxes, "match": match})

sy = pd.DataFrame(rows)
# some patients have more than one sybil SR, keep the best match
rank = {"same series as expert mask": 0, "same-scan twin (Aim-1 pair)": 1, "same study, different series": 2,
        "different study (other screening round)": 3}
sy["rank"] = sy.match.map(rank)
best = sy.sort_values("rank").drop_duplicates("patient_id")
sy.to_csv(os.path.join(ROOT, "results", "audit", "sybil_series_match.csv"), index=False)

lp = pd.read_csv(os.path.join(ROOT, "raw", "lidc_annot_nlst501", "LIDC-annot-NLST501-patients.csv"), dtype={"pid": str})
ls = pd.read_csv(os.path.join(ROOT, "raw", "lidc_annot_nlst501", "LIDC-annot-NLST501-series.csv"), dtype={"pid": str})
lrows = []
for p in sorted(set(lp.pid) & set(ours)):
    ser = set(ls[ls.pid == p].series_uid)
    lrows.append({"patient_id": p, "group": ours[p],
                  "same_series_as_mask": man.loc[p, "ct_series"] in ser,
                  "same_scan_twin": twin.get(p) in ser if twin.get(p) else False,
                  "n_readers_listed": int(lp[lp.pid == p].filter(like="reader").iloc[0, :6].notna().sum())})
lo = pd.DataFrame(lrows)
lo.to_csv(os.path.join(ROOT, "results", "audit", "lidc501_overlap.csv"), index=False)

summary = {
    "sybil_best_match_per_patient": {g: best[best.group == g].match.value_counts().to_dict() for g in ("aim1", "train", "val", "test")},
    "sybil_patients_with_box_on_our_scan_or_twin": int(best["rank"].le(1).sum()),
    "lidc501_patients_overlapping": len(lo),
    "lidc501_by_group": lo.group.value_counts().to_dict() if len(lo) else {},
    "lidc501_same_series_as_mask": int(lo.same_series_as_mask.sum()) if len(lo) else 0,
    "lidc501_on_same_scan_twin": int(lo.same_scan_twin.sum()) if len(lo) else 0,
}
json.dump(summary, open(os.path.join(ROOT, "results", "audit", "overlap_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
