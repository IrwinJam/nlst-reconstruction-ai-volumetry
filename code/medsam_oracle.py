"""medsam with tight boxes from the expert mask, just a sanity check on the implementation (D12)"""
import json
import numpy as np, pandas as pd, torch
import nlst_seg as S
import evaluate as E
import medsam_baseline as M

ROOT = S.ROOT
OUT = ROOT / "results" / "posthoc"


@torch.no_grad()
def main():
    box = pd.read_csv(ROOT / "results" / "eval" / "box_rows.csv", dtype={"patient_id": str})
    ids = sorted(set(box[box.set == "test"].patient_id))
    st = S.Store(ids)
    model, proc = M.load_medsam()
    rows = []
    for p in ids:
        d = st.d[p]
        msk = d["msk"].astype(bool)
        pred = np.zeros(msk.shape, bool)
        per_slice = []
        for i in np.nonzero(msk.reshape(len(msk), -1).any(1))[0]:
            ys, xs = np.nonzero(msk[i])
            prompt = [i, i, int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
            pred[i] = M.segment(model, proc, d["img"], prompt)[i]
            inter = (pred[i] & msk[i]).sum()
            per_slice.append(2 * inter / (pred[i].sum() + msk[i].sum()))
        rows.append({"patient_id": p, "slab_dice": S.slab_metrics(pred.astype(np.float32), d["msk"], d["dz"])["dice"],
                     "tumour_slice_dice": float(np.mean(per_slice))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d12_rows.csv", index=False)
    summ = {"label": "POST-HOC (D12)", "n": len(df), "slab_dice_mean": E.r4(df.slab_dice.mean()),
            "tumour_slice_dice_mean": E.r4(df.tumour_slice_dice.mean()), "tumour_slice_dice_CI": E.boot(df.tumour_slice_dice)}
    (OUT / "d12_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
