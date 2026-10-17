"""paired rc comparison + d10 figs. run with the medsam env"""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nlst_seg as S
import box_mode as B
import evaluate as E

ROOT = S.ROOT
EV, PH, FIG = ROOT / "results" / "eval", ROOT / "results" / "posthoc", ROOT / "results" / "figures"
DESK = Path(r"C:\Users\Irwin\OneDrive\Desktop\NLST_Reconstruction_Paper\figures")
S.CACHE = ROOT / "cache" / "NLST_v2"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def volumes():
    u = pd.read_csv(EV / "aim1_per_seed.csv", dtype={"patient_id": str}).groupby("patient_id")[["vol_a", "vol_b"]].mean()
    n = pd.read_csv(EV / "nnunet_aim1_rows.csv", dtype={"patient_id": str}).set_index("patient_id")[["vol_a", "vol_b"]]
    bx = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    b = bx[bx.set == "aim1"].groupby("patient_id")[["vol_box_a", "vol_box_b"]].mean()
    b.columns = ["vol_a", "vol_b"]
    m = pd.read_csv(PH / "d9_rows.csv", dtype={"patient_id": str}).set_index("patient_id")
    m = m[m.set == "aim1"][["vol_a", "vol_b"]]
    out = {"U-Net": u, "nnU-Net": n, "MedSAM (3D box)": m, "Box-guided U-Net": b}
    if (PH / "d10b_rows.csv").exists():
        mp = pd.read_csv(PH / "d10b_rows.csv", dtype={"patient_id": str}).set_index("patient_id")
        out["MedSAM (per-slice boxes)"] = mp[mp.set == "aim1"][["vol_a", "vol_b"]]
    return out


def rc(va, vb):
    d = np.log(vb) - np.log(va)
    return 100 * (np.exp(2.77 * np.std(d, ddof=1) / np.sqrt(2)) - 1)


def part_a(vols):
    common = None
    for v in vols.values():
        ok = set(v.index[(v.vol_a > 0) & (v.vol_b > 0)])
        common = ok if common is None else common & ok
    ids = sorted(common)
    arr = {k: (v.loc[ids, "vol_a"].values, v.loc[ids, "vol_b"].values) for k, v in vols.items()}
    point = {k: round(float(rc(*a)), 2) for k, a in arr.items()}
    # paired bootstrap, resample patients and recompute rc for every model each time
    rng = np.random.default_rng(0)
    boots = {k: [] for k in arr}
    for _ in range(E.NBOOT):
        ix = rng.integers(0, len(ids), len(ids))
        for k, (a, b) in arr.items():
            boots[k].append(rc(a[ix], b[ix]))
    boots = {k: np.array(v) for k, v in boots.items()}
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]
    ref = "Box-guided U-Net"
    comp = {}
    for k in arr:
        if k == ref:
            continue
        comp[f"{k} vs box"] = {"ratio": round(point[k] / point[ref], 2), "ratio_CI": ci(boots[k] / boots[ref]),
                               "difference_pct_points": round(point[k] - point[ref], 2), "difference_CI": ci(boots[k] - boots[ref])}
    return {"n_common": len(ids), "RC_pct": point, "RC_CI": {k: ci(v) for k, v in boots.items()}, "paired": comp}


def fig_ba(vols, summ):
    keys = ["U-Net", "nnU-Net", "MedSAM (3D box)", "Box-guided U-Net"]
    cols = [ORANGE, AQUA, VIOLET, BLUE]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for a, k, c in zip(axs, keys, cols):
        d = vols[k][(vols[k].vol_a > 0) & (vols[k].vol_b > 0)]
        gm, pc = np.sqrt(d.vol_a * d.vol_b), 100 * (d.vol_b / d.vol_a - 1)
        dl = np.log(d.vol_b) - np.log(d.vol_a)
        bias, sd = dl.mean(), dl.std(ddof=1)
        a.axhspan(-25, 25, color="#f0efec", zorder=0)
        a.scatter(gm, pc.clip(-100, 300), s=14, color=c, alpha=0.8, lw=0)
        a.axhline(100 * (np.exp(bias) - 1), color=INK2, lw=1.2)
        for l in (bias - 1.96 * sd, bias + 1.96 * sd):
            a.axhline(100 * (np.exp(l) - 1), color=INK2, lw=1, ls="--")
        a.set_xscale("log"); a.set_ylim(-100, 300)
        a.set_title(f"{k} (n={len(d)})", loc="left", fontsize=11, color=INK)
        a.text(0.98, 0.96, f"RC {rc(d.vol_a.values, d.vol_b.values):.0f}%", transform=a.transAxes, ha="right", va="top", fontsize=10)
        a.set_xlabel("Geometric mean volume (mL)")
    axs[0].set_ylabel("Volume difference B vs A (%)")
    fig.suptitle("Agreement of AI tumor volume between reconstructions (post-hoc; dashed = 95% limits of agreement; shaded ±25%)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_bland_altman4.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_qual():
    import SimpleITK as sitk
    from prepare_nlst import to_grid
    import medsam_baseline as M
    bx = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    t = bx[bx.set == "test"].groupby("patient_id").dice_box.mean().sort_values()
    picks = []
    for q in (12.5, 37.5, 62.5, 87.5):
        target = np.percentile(t.values, q)
        picks.append(t.index[int(np.argmin(np.abs(t.values - target)))])
    _, real = B.real_prompts()
    st = S.Store(picks)
    bst = E.box_view(st, {p: real[p] for p in picks})
    S.DEV = torch.device("cpu")
    bm = E.load(ROOT / "results" / "v2_box_s0" / "box_best.pt", st.cin + 1)
    msam, proc = M.load_medsam()
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
    for a, p in zip(axs, picks):
        d = st.d[p]
        msk = d["msk"].astype(bool)
        box = B.eval_box(bm, bst, p, return_prob=True)[1] >= 0.5
        f = ROOT / "nnunet" / "pred_ts" / f"NLST_{p}.nii.gz"
        img = sitk.ReadImage(str(f)); arr = sitk.GetArrayFromImage(img); spx = img.GetSpacing()
        nn = np.stack([to_grid(x.astype(np.float32), (spx[1], spx[0]), True) for x in arr]) >= 0.5
        ms = M.segment(msam, proc, d["img"], real[p])
        i = int(msk.sum((1, 2)).argmax())
        yy, xx = np.nonzero(msk[i]); cy, cx, h = int(yy.mean()), int(xx.mean()), 36
        sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
        a.imshow(d["img"][i][sl].astype(float), cmap="gray", vmin=0, vmax=1)
        for m_, c, ls in [(msk, "white", "--"), (nn, AQUA, "-"), (ms, VIOLET, "-"), (box, BLUE, "-")]:
            if m_[i][sl].any():
                a.contour(m_[i][sl].astype(float), [0.5], colors=c, linewidths=1.6, linestyles=ls)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"Box-guided Dice {t[p]:.2f} (slice {i})", loc="left", fontsize=10.5)
    fig.text(0.01, -0.02, "White dashed: expert mask · aqua: nnU-Net · violet: MedSAM (3D box) · blue: box-guided U-Net (seed 0). "
                          "Test participants at the 12.5/37.5/62.5/87.5th percentiles of box-guided Dice (pre-set rule, D10c).",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "figS_qualitative.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    return picks


def main():
    vols = volumes()
    summ = {"label": "POST-HOC (D10)", "a_paired_RC": part_a(vols)}
    fig_ba(vols, summ)
    summ["c_qualitative_picks"] = fig_qual()
    if (PH / "d10b_summary.json").exists():
        summ["b_medsam_per_slice"] = json.loads((PH / "d10b_summary.json").read_text())
    (PH / "d10_summary.json").write_text(json.dumps(summ, indent=2))
    DESK.mkdir(parents=True, exist_ok=True)
    for f in ["fig3_bland_altman4.png", "figS_qualitative.png"]:
        shutil.copy(FIG / f, DESK / f)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
