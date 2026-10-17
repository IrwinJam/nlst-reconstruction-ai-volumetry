"""exports the slabs in nnunet format"""
import glob, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, SimpleITK as sitk
import qc_gate1 as Q

ROOT = r"C:\MetaSeg_data"
DS = os.path.join(ROOT, "nnunet", "raw", "Dataset501_NLSTSeg")
MARGIN = 5


def write(arr_zyx, spacing_xyz, path, dtype):
    img = sitk.GetImageFromArray(arr_zyx.astype(dtype))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    sitk.WriteImage(img, path, useCompression=True)


def dedupe(heads, masks):
    """one image per table position (D5). exact copies collapse to the one with the mask, real mixed acquisitions get flagged"""
    byz = {}
    for s in sorted(heads):
        byz.setdefault(round(heads[s]["pos"][2], 3), []).append(s)
    keep, conflicts = [], set()
    for z, sops in byz.items():
        if len(sops) > 1:
            px = [Q.hu(heads[s]["file"]) for s in sops]
            if not all(np.array_equal(px[0], p) for p in px[1:]):
                conflicts.add(z)
            sops = sorted(sops, key=lambda s: -(masks[s].sum() if s in masks else 0))
        keep.append(sops[0])
    return sorted(keep, key=lambda s: heads[s]["pos"][2]), conflicts


def convert(args):
    pid, ct_series, seg_series, pair, role = args
    case = f"NLST_{pid}"
    done = os.path.join(DS, "imagesTr" if role in ("train", "val") else "imagesTs", f"{case}_0000.nii.gz")
    heads = Q.ct_headers(Q.find(pid, "CT", ct_series)[0])
    masks, _ = Q.seg_masks(glob.glob(os.path.join(Q.find(pid, "SEG", seg_series)[0], "*.dcm"))[0], heads)
    order, conflicts = dedupe(heads, masks)
    zs = [heads[s]["pos"][2] for s in order]
    dz = float(np.median(np.abs(np.diff(zs))))
    tum = [i for i, s in enumerate(order) if s in masks and masks[s].sum() >= Q.MIN_PX]
    lo, hi = max(min(tum) - MARGIN, 0), min(max(tum) + MARGIN, len(order) - 1)
    sl = order[lo:hi + 1]
    inside = sorted(z for z in conflicts if round(zs[lo], 3) <= z <= round(zs[hi], 3))
    if inside:
        return [{"case": case, "patient_id": pid, "role": role, "recon": "annotated",
                 "excluded": f"mixed acquisitions inside tumour slab ({len(inside)} positions)"}]
    sp = heads[sl[0]]["sp"]  # [row, col] mm
    spacing = (sp[1], sp[0], dz)
    out = []
    if not os.path.exists(done):  # skip cases already exported
        vol = np.stack([Q.hu(heads[s]["file"]) for s in sl])
        write(vol, spacing, done, np.int16)
        if role in ("train", "val"):
            lab = np.stack([(masks[s] if s in masks and masks[s].sum() >= Q.MIN_PX else np.zeros(heads[s]["rc"], bool)) for s in sl])
            write(lab, spacing, os.path.join(DS, "labelsTr", f"{case}.nii.gz"), np.uint8)
    out.append({"case": case, "patient_id": pid, "role": role, "recon": "annotated", "n_slices": len(sl)})
    pfile = done.replace("_0000", "p_0000")
    if pair:
        if not os.path.exists(pfile):
            ph = {round(v["pos"][2], 2): v for v in Q.ct_headers(Q.find(pid, "CT", pair)[0]).values()}
            volb = np.stack([Q.hu(ph[round(heads[s]["pos"][2], 2)]["file"]) for s in sl])
            write(volb, spacing, pfile, np.int16)
        out.append({"case": case + "p", "patient_id": pid, "role": role, "recon": "paired", "n_slices": len(sl)})
    return out


def main():
    for d in ("imagesTr", "labelsTr", "imagesTs"):
        os.makedirs(os.path.join(DS, d), exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, "nlstseg_manifest.csv"), dtype={"PatientID": str}).set_index("PatientID")
    sp = json.load(open(os.path.join(ROOT, "results", "split.json")))
    pairs = pd.read_csv(os.path.join(ROOT, "results", "qc", "aim1_pairs_final.csv"), dtype={"patient_id": str})
    twin = pairs.drop_duplicates("patient_id").set_index("patient_id").paired_series.to_dict()
    jobs = [(p, man.loc[p, "ct_series"], man.loc[p, "seg_series"], twin.get(p) if g == "aim1" else None, g)
            for g in ("train", "val", "test", "aim1") for p in sp[g]]
    # 1 worker by default, runs on cpu while training is eating the ram
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NNUNET_PREP_WORKERS", "1"))) as ex:
        rows = [r for rs in ex.map(convert, jobs, chunksize=4) for r in rs]
    cm = pd.DataFrame(rows)
    if "excluded" not in cm:
        cm["excluded"] = None
    cm.to_csv(os.path.join(ROOT, "nnunet", "case_map.csv"), index=False)
    print(cm[cm.excluded.notna()].to_string(index=False))
    cm = cm[cm.excluded.isna()]
    kept = set(cm.patient_id)
    sp = {g: [p for p in v if p in kept] for g, v in sp.items()}
    n_tr = int(cm[cm.role.isin(["train", "val"])].shape[0])
    json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "tumor": 1},
               "numTraining": n_tr, "file_ending": ".nii.gz",
               "name": "NLSTSeg tumour slabs (NLST, NCI IDC, CC BY 4.0)"},
              open(os.path.join(DS, "dataset.json"), "w"), indent=2)
    split = [{"train": [f"NLST_{p}" for p in sp["train"]], "val": [f"NLST_{p}" for p in sp["val"]]}]
    json.dump(split, open(os.path.join(ROOT, "nnunet", "splits_final.json"), "w"), indent=1)
    print(f"nnU-Net raw dataset: {n_tr} training cases (train {len(sp['train'])} + val {len(sp['val'])}), "
          f"{(~cm.role.isin(['train','val'])).sum()} inference cases -> {DS}")


if __name__ == "__main__":
    main()
