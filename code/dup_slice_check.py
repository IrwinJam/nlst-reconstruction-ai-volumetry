"""finds ct series with 2+ images at the same table position (D5)"""
import json, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, pydicom
import qc_gate1 as Q

ROOT = r"C:\MetaSeg_data"


def check(args):
    pid, role, series = args
    heads = Q.ct_headers(Q.find(pid, "CT", series)[0])
    byz = defaultdict(list)
    for s, v in heads.items():
        byz[round(v["pos"][2], 3)].append(s)
    dup = {z: s for z, s in byz.items() if len(s) > 1}
    exact = distinct = 0
    acq = set()
    for z, sops in dup.items():
        px = [Q.hu(heads[s]["file"]) for s in sops]
        if all(np.array_equal(px[0], p) for p in px[1:]):
            exact += 1
        else:
            distinct += 1
        acq |= {str(pydicom.dcmread(heads[s]["file"], stop_before_pixels=True).get("AcquisitionNumber")) for s in sops}
    return {"patient_id": pid, "series_role": role, "series_uid": series, "images": len(heads),
            "positions": len(byz), "dup_positions": len(dup), "dup_exact_copy": exact,
            "dup_distinct_image": distinct, "acquisition_numbers": ";".join(sorted(acq))}


def main():
    man = pd.read_csv(os.path.join(ROOT, "nlstseg_manifest.csv"), dtype={"PatientID": str}).set_index("PatientID")
    sp = json.load(open(os.path.join(ROOT, "results", "split.json")))
    pairs = pd.read_csv(os.path.join(ROOT, "results", "qc", "aim1_pairs_final.csv"), dtype={"patient_id": str})
    jobs = [(p, g, man.loc[p, "ct_series"]) for g in ("train", "val", "test", "aim1") for p in sp[g]]
    jobs += [(r.patient_id, "aim1_paired", r.paired_series) for r in pairs.itertuples()]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("DUP_WORKERS", "1"))) as ex:
        res = pd.DataFrame(list(ex.map(check, jobs, chunksize=8)))
    res.to_csv(os.path.join(ROOT, "results", "qc", "dup_slices.csv"), index=False)
    bad = res[res.dup_positions > 0]
    print(f"checked {len(res)} series; {len(bad)} with duplicated positions")
    print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
