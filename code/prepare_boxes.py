"""turns the radiologist boxes into prompts on the cache grid (D6)"""
import glob, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy import ndimage
import qc_gate1 as Q
from prepare_nlst import to_grid
from sybil_box_check import boxes_in

ROOT = r"C:\MetaSeg_data"
CACHE = os.path.join(ROOT, "cache", "NLST_v2")
OUT = os.path.join(ROOT, "cache", "NLST_v2_box")


def bbox(vol):
    z, y, x = np.nonzero(vol)
    return [int(z.min()), int(z.max()), int(y.min()), int(y.max()), int(x.min()), int(x.max())]


def prep(args):
    pid, ct_series, twin, group = args
    row = {"patient_id": pid, "group": group}
    try:
        c = np.load(os.path.join(CACHE, f"{pid}.npz"))
        msk, zslab = c["msk"].astype(bool), np.round(c["z"].astype(float), 2)
        heads = Q.ct_headers(Q.find(pid, "CT", ct_series)[0])
        sp = next(iter(heads.values()))["sp"]
        z_of = {s: round(v["pos"][2], 2) for s, v in heads.items()}
        rc = {s: v["rc"] for s, v in heads.items()}
        if twin:
            tw = Q.ct_headers(Q.find(pid, "CT", twin)[0])
            z_of.update({s: round(v["pos"][2], 2) for s, v in tw.items()})
            rc.update({s: v["rc"] for s, v in tw.items()})
        native = {}
        for f in glob.glob(os.path.join(ROOT, "raw", "nlst_sybil", pid, "*", "*.dcm")):
            for sop, bl in boxes_in(f).items():
                if sop not in z_of:
                    continue
                m = native.setdefault(z_of[sop], np.zeros(rc[sop], np.float32))
                for x0, y0, x1, y1 in bl:
                    m[int(np.floor(y0)):int(np.ceil(y1)) + 1, int(np.floor(x0)):int(np.ceil(x1)) + 1] = 1
        rbox = np.zeros(msk.shape, bool)
        for z, m in native.items():
            for i in np.nonzero(zslab == z)[0]:
                rbox[i] |= to_grid(m, sp, True) > 0
        # group boxes into lesions, keep the lesion that overlaps the expert mask
        lab, n = ndimage.label(rbox, structure=np.ones((3, 3, 3)))
        hit = sorted(set(np.unique(lab[msk & (lab > 0)])))
        row.update(box_slices_in_slab=int(rbox.any((1, 2)).sum()), lesions=int(n), target_lesions=len(hit))
        if not hit:
            row["status"] = "no box on target lesion"
            return row
        les = np.isin(lab, hit)
        pr, gt = bbox(les), bbox(msk)
        box3d = np.zeros_like(msk)
        box3d[pr[0]:pr[1] + 1, pr[2]:pr[3] + 1, pr[4]:pr[5] + 1] = True
        np.savez_compressed(os.path.join(OUT, f"{pid}.npz"), rbox=les.astype(np.uint8), prompt=np.array(pr))
        off = [gt[0] - pr[0], pr[1] - gt[1], gt[2] - pr[2], pr[3] - gt[3], gt[4] - pr[4], pr[5] - gt[5]]
        row.update(status="ok", prompt=" ".join(map(str, pr)), gt_bbox=" ".join(map(str, gt)),
                   off_z0=off[0], off_z1=off[1], off_y0=off[2], off_y1=off[3], off_x0=off[4], off_x1=off[5],
                   mask_in_prompt=round(float((msk & box3d).sum() / msk.sum()), 4))
    except Exception as e:
        row.update(status="error", note=f"{type(e).__name__}: {e}")
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, "nlstseg_manifest.csv"), dtype={"PatientID": str}).set_index("PatientID")
    sy = pd.read_csv(os.path.join(ROOT, "results", "audit", "sybil_series_match.csv"), dtype={"patient_id": str})
    pairs = pd.read_csv(os.path.join(ROOT, "results", "qc", "aim1_pairs_final.csv"), dtype={"patient_id": str})
    twin = pairs.drop_duplicates("patient_id").set_index("patient_id").paired_series.to_dict()
    grp = {p: g for g, v in json.load(open(os.path.join(ROOT, "results", "split.json"))).items() if g != "_meta" for p in v}
    use = sorted(set(sy[sy.match.isin(["same series as expert mask", "same-scan twin (Aim-1 pair)"])].patient_id))
    jobs = [(p, man.loc[p, "ct_series"], twin.get(p), grp.get(p)) for p in use if os.path.exists(os.path.join(CACHE, f"{p}.npz"))]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("BOX_WORKERS", "2"))) as ex:
        res = pd.DataFrame(list(ex.map(prep, jobs, chunksize=4)))
    res.to_csv(os.path.join(OUT, "prompts.csv"), index=False)
    print(res.groupby(["group", "status"]).size().unstack(fill_value=0))
    ok = res[res.status == "ok"]
    print("mask fraction inside prompt: median", ok.mask_in_prompt.median(), "| <0.9:", int((ok.mask_in_prompt < 0.9).sum()))
    print("train offsets (median, grid px / slices):", ok[ok.group == "train"].filter(like="off_").median().to_dict())


if __name__ == "__main__":
    main()
