"""montages for eyeballing the flagged masks"""
import glob, os, random
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qc_gate1 as Q

ROOT = Q.ROOT
q = pd.read_csv(ROOT / "results" / "qc" / "qc_masks.csv", dtype={"patient_id": str})
man = pd.read_csv(ROOT / "nlstseg_manifest.csv", dtype={"PatientID": str})
review = q[q.status == "REVIEW"].patient_id.tolist()
ggo_pool = q[(q.status == "PASS") & q.reason.fillna("").str.contains("ground-glass")].patient_id.tolist()
ggo = random.Random(7).sample(ggo_pool, 12)

cases = [(p, "REVIEW") for p in review] + [(p, "ground-glass") for p in ggo]
cols = 7
rows = int(np.ceil(len(cases) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
for ax in axes.ravel():
    ax.axis("off")
for ax, (pid, kind) in zip(axes.ravel(), cases):
    r = man[man.PatientID == pid].iloc[0]
    heads = Q.ct_headers(Q.find(pid, "CT", r.ct_series)[0])
    masks, _ = Q.seg_masks(glob.glob(os.path.join(Q.find(pid, "SEG", r.seg_series)[0], "*.dcm"))[0], heads)
    s = max((k for k in masks if k in heads), key=lambda k: masks[k].sum())
    img, mk = Q.hu(heads[s]["file"]), masks[s]
    ys, xs = np.nonzero(mk)
    h = max(48, int(max(np.ptp(ys), np.ptp(xs)) * 0.9))
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = max(cy - h, 0), max(cx - h, 0)
    ax.imshow(np.clip((img[y0:y0 + 2 * h, x0:x0 + 2 * h] + 1000) / 1400, 0, 1), cmap="gray")
    ax.contour(mk[y0:y0 + 2 * h, x0:x0 + 2 * h], levels=[0.5], colors="#ff5a36", linewidths=1.1)
    row = q[q.patient_id == pid].iloc[0]
    ax.set_title(f"{kind}: {pid}\nin {row.mean_hu_inside:.0f} / ring {row.mean_hu_ring:.0f} HU",
                 fontsize=7, color="#b00020" if kind == "REVIEW" else "#1b5e20")
fig.suptitle("Gate 1 visual review: red titles = low contrast flags, green = random ground-glass passes",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "results" / "qc" / "review_flagged.png", dpi=150)
print("wrote review_flagged.png with", len(cases), "cases")
