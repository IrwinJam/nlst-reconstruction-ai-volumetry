"""zero shot medsam with the box prompt (D9), --per-slice for D10b"""
import json
import numpy as np, pandas as pd, torch, cv2
from transformers import SamConfig, SamModel, SamProcessor
import nlst_seg as S
import box_mode as B
import evaluate as E
from posthoc_d8 import agreement

ROOT = S.ROOT
MODEL = ROOT / "models" / "medsam-vit-base"
OUT = ROOT / "results" / "posthoc"
EV = ROOT / "results" / "eval"
S.CACHE = ROOT / "cache" / "NLST_v2"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the weights ourselves, transformers wont load the .bin on torch 2.5
def load_medsam():
    cfg = SamConfig.from_pretrained(MODEL)
    m = SamModel(cfg)
    sd = torch.load(MODEL / "pytorch_model.bin", map_location="cpu", weights_only=True)  # weights_only so it only loads tensors
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "position" not in k], f"missing keys: {missing[:5]}"
    return m.to(DEV).eval(), SamProcessor.from_pretrained(MODEL)


@torch.no_grad()
def segment(model, proc, img, prompt):
    """img is (n,320,320) lung window 0-1, prompt is [z0,z1,y0,y1,x0,x1]. same box on every slice in range"""
    z0, z1, y0, y1, x0, x1 = prompt
    out = np.zeros(img.shape, bool)
    box = [[[float(x0), float(y0), float(x1 + 1), float(y1 + 1)]]]
    for i in range(z0, z1 + 1):
        rgb = np.repeat((np.clip(img[i].astype(np.float32), 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
        inp = proc(images=rgb, input_boxes=box, return_tensors="pt").to(DEV)
        o = model(**inp, multimask_output=False)
        pm = proc.image_processor.post_process_masks(o.pred_masks.sigmoid().cpu(), inp["original_sizes"].cpu(),
                                                    inp["reshaped_input_sizes"].cpu(), binarize=False)
        out[i] = pm[0][0, 0].numpy() > 0.5
    return out


@torch.no_grad()
def segment_perslice(model, proc, img, rbox):
    """D10b, each slice gets its own radiologist box, slices without one stay empty"""
    out = np.zeros(img.shape, bool)
    for i in np.nonzero(rbox.reshape(len(rbox), -1).any(1))[0]:
        ys, xs = np.nonzero(rbox[i])
        box = [[[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]]]
        rgb = np.repeat((np.clip(img[i].astype(np.float32), 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
        inp = proc(images=rgb, input_boxes=box, return_tensors="pt").to(DEV)
        o = model(**inp, multimask_output=False)
        pm = proc.image_processor.post_process_masks(o.pred_masks.sigmoid().cpu(), inp["original_sizes"].cpu(),
                                                    inp["reshaped_input_sizes"].cpu(), binarize=False)
        out[i] = pm[0][0, 0].numpy() > 0.5
    return out


PER_SLICE = False


def main():
    global PER_SLICE
    import sys
    PER_SLICE = "--per-slice" in sys.argv
    tag = "d10b" if PER_SLICE else "d9"
    OUT.mkdir(parents=True, exist_ok=True)
    sp, aim1p, pts, _ = E.split_lists()
    _, real = B.real_prompts()
    box = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    ids = sorted(set(box.patient_id))
    st = S.Store(ids)
    model, proc = load_medsam()
    rows = []
    for p in ids:
        d = st.d[p]
        msk, dz = d["msk"], d["dz"]
        rb = np.load(B.BOXDIR / f"{p}.npz")["rbox"] if PER_SLICE else None
        seg = (lambda im: segment_perslice(model, proc, im, rb)) if PER_SLICE else (lambda im: segment(model, proc, im, real[p]))
        ma = seg(d["img"]).astype(np.float32)
        r = {"patient_id": p, "set": "aim1" if p in set(aim1p) else "test"}
        m_a = S.slab_metrics(ma, msk, dz)
        r.update(dice_a=m_a["dice"], vol_a=m_a["vol_pred_ml"], vol_true=m_a["vol_true_ml"])
        if "img_pair" in d:
            mb = seg(d["img_pair"]).astype(np.float32)
            m_b = S.slab_metrics(mb, msk, dz)
            r.update(dice_b=m_b["dice"], vol_b=m_b["vol_pred_ml"])
        rows.append(r)
    df = pd.DataFrame(rows).set_index("patient_id")
    df.to_csv(OUT / f"{tag}_rows.csv")
    d8 = json.loads((OUT / "d8_summary.json").read_text())
    d8a = pd.read_csv(OUT / "d8a_rows.csv", dtype={"patient_id": str}).set_index("patient_id")
    rd = pd.read_csv(EV / "readers_rows.csv", dtype={"patient_id": str})
    sets = {"test_71": [p for p in df.index if df.set[p] == "test"], "heldout_181": list(df.index),
            "reader_35": [p for p in rd.dropna(subset=["box_dice"]).patient_id if p in df.index]}
    summ = {"label": f"POST-HOC ({tag.upper()})"}
    for k, ix in sets.items():
        g = df.loc[ix].join(d8a[["dice_box", "nn"]], how="left")
        summ[k] = {"n": len(g), "medsam": E.r4(g.dice_a.mean()), "box_guided": E.r4(g.dice_box.mean()), "nnunet": E.r4(g.nn.mean()),
                   "box_minus_medsam": E.r4((g.dice_box - g.dice_a).mean()), "box_minus_medsam_CI": E.boot(g.dice_box - g.dice_a),
                   "box_vs_medsam_p": E.wilcox(g.dice_box, g.dice_a)}
    a1 = df[df.set == "aim1"].dropna(subset=["vol_b"])
    ch = (100 * (a1.vol_b - a1.vol_a).abs() / a1.vol_a)[a1.vol_a > 0]
    summ["aim1_110"] = {"n": len(a1), "dice_A": E.r4(a1.dice_a.mean()), "dice_B": E.r4(a1.dice_b.mean()),
                        "frac_gt25": E.r4((ch > 25).mean()), "n_volA_positive": int((a1.vol_a > 0).sum()),
                        "agreement": agreement(a1.vol_a.values, a1.vol_b.values)}
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
