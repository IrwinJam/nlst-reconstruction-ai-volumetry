"""figures 1-4"""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nlst_seg as S
import box_mode as B

ROOT = S.ROOT
OUT = ROOT / "results" / "figures"
DESK = Path(r"C:\Users\Irwin\OneDrive\Desktop\NLST_Reconstruction_Paper\figures")
EV = ROOT / "results" / "eval"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
s = json.loads((EV / "summary.json").read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def volpct(df, a, b):
    d = df[df[a] > 0].copy()
    d["pct"] = 100 * (d[b] - d[a]) / d[a]
    return d.groupby("patient_id").pct.mean()


# fig 2: dice unchanged, volume not
def fig_dice_vs_volume():
    u = pd.read_csv(EV / "aim1_per_seed.csv", dtype={"patient_id": str})
    n = pd.read_csv(EV / "nnunet_aim1_rows.csv", dtype={"patient_id": str})
    rows = [("2.5D U-Net (3 seeds)", u), ("nnU-Net", n)]
    fig, ax = plt.subplots(2, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})
    rng = np.random.default_rng(0)
    for r, (name, d) in enumerate(rows):
        dd = d.assign(dd=d.dice_b - d.dice_a).groupby("patient_id").dd.mean()
        pc = volpct(d, "vol_a", "vol_b")
        dv = d[d.vol_a > 0].assign(ab=lambda x: (100 * (x.vol_b - x.vol_a) / x.vol_a).abs())
        absm = dv.groupby("patient_id").ab.mean().reindex(pc.index)  # same def as the protocol, mean of |change| across seeds
        a = ax[r, 0]
        a.scatter(dd, rng.uniform(-0.35, 0.35, len(dd)), s=18, color=BLUE, alpha=0.75, lw=0)
        a.axvline(0, color=INK2, lw=1)
        a.set_xlim(-0.4, 0.4); a.set_yticks([]); a.set_ylim(-0.6, 0.6)
        a.set_ylabel(name, rotation=0, ha="right", va="center", color=INK, fontsize=11)
        a.text(0.02, 0.97, f"mean {dd.mean():+.3f}", transform=a.transAxes, va="top", color=INK2, fontsize=10)
        b = ax[r, 1]
        clip = pc.clip(-100, 150)
        out = absm > 25
        b.axvspan(-25, 25, color="#f0efec", zorder=0)
        y = rng.uniform(-0.35, 0.35, len(pc))
        b.scatter(clip[~out], y[~out.values], s=18, color=BLUE, alpha=0.75, lw=0, label="within ±25%")
        b.scatter(clip[out], y[out.values], s=22, color=ORANGE, alpha=0.9, lw=0, label="beyond ±25%")
        b.set_xlim(-105, 155); b.set_yticks([]); b.set_ylim(-0.6, 0.6)
        b.text(0.99, 0.97, f"{int(out.sum())} of {len(pc)} patients ({100 * out.mean():.0f}%) beyond ±25%",
               transform=b.transAxes, ha="right", va="top", color=INK, fontsize=10.5)
        if r == 0:
            a.set_title("Dice change (B − A): unchanged", loc="left", fontsize=12, color=INK)
            b.set_title("Tumor volume change, % (B vs A)", loc="left", fontsize=12, color=INK)
            b.legend(loc="lower right", frameon=False, fontsize=9.5)
        if r == 1:
            a.set_xlabel("Dice difference"); b.set_xlabel("Volume change (%), clipped to −100…150")
    fig.text(0.01, -0.03, "Points: per-patient mean over seeds. 'Beyond ±25%' uses the pre-specified definition "
                           "(mean absolute change across seeds), so a few points near the band edge may sit inside it.",
             fontsize=8.5, color=MUTED)
    fig.suptitle("Same acquisition, two reconstruction filters (128 NLST participants)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "fig2_dice_vs_volume.png")


# fig 3: box guidance
def fig_box():
    t, vb = s["box"]["test"], s["box"]["aim1_exploratory_volume_change"]
    d8 = json.loads((ROOT / "results" / "posthoc" / "d8_summary.json").read_text())
    nn71, rc = d8["a_same_patient"]["test_71"]["nn"], d8["c_volumetric_agreement"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    panels = [
        (ax[0], f"Test Dice, same {t['n']} patients", [("Automatic U-Net", t["mean_dice_auto"], ORANGE),
                                                       ("U-Net cropped to box", t["mean_dice_crop"], MUTED),
                                                       ("nnU-Net (no box)*", nn71, AQUA),
                                                       ("Box-guided U-Net", t["mean_dice_box"], BLUE)], (0, 1), "{:.2f}"),
        (ax[1], "Repeatability coefficient across\nreconstructions (lower is better)*",
         [("Automatic U-Net", rc["unet_all"]["RC_pct"], ORANGE), ("nnU-Net", rc["nnunet_all"]["RC_pct"], AQUA),
          ("Box-guided U-Net", rc["box_prompt110"]["RC_pct"], BLUE)], (0, 230), "{:.0f}%"),
        (ax[2], f"Patients with >25% volume change\nbetween reconstructions (n={vb['box_guided']['n_patients_with_V_A>0']})",
         [("Automatic U-Net", 100 * vb["automatic_same_patients"]["frac_abs_pct_gt25"], ORANGE),
          ("Box-guided U-Net", 100 * vb["box_guided"]["frac_abs_pct_gt25"], BLUE)], (0, 40), "{:.0f}%"),
    ]
    for a, title, bars, lim, fmt in panels:
        a.barh(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], height=0.6)
        for i, b in enumerate(bars):
            a.text(b[1] + (lim[1] - lim[0]) * 0.015, i, fmt.format(b[1]), va="center", color=INK, fontsize=11)
        a.set_yticks(range(len(bars))); a.set_yticklabels([b[0] for b in bars], color=INK)
        a.set_xlim(*lim); a.set_title(title, loc="left", fontsize=11.5, color=INK); a.tick_params(axis="y", length=0)
    ax[2].text(0.0, -0.3, "exploratory (pre-specified)", transform=ax[2].transAxes, color=MUTED, fontsize=9)
    fig.text(0.01, -0.04, "* post-hoc analyses (logged before running). Repeatability coefficient = 2.77 × within-subject SD of "
                          "log volume between reconstructions, as %.", fontsize=8.5, color=MUTED)
    fig.suptitle("One radiologist box per lesion: Dice like a strong model, far more stable volumes", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, "fig3_box_guidance.png")


# fig 4: few-shot recovery on unseen vendors
def fig_fewshot():
    p = s["aim3"]["3b"]
    ks = [1, 2, 4, 8]
    fig, a = plt.subplots(figsize=(7, 4))
    a.axhline(0, color=INK2, lw=1.2)
    a.text(-0.3, 0.003, "all-vendor model (same patients)", va="bottom", ha="left", color=INK2, fontsize=9)
    x0 = [0] + ks
    conv = [p["k=1"]["conv_unadapted"] - p["k=1"]["ref_unadapted"]] + [p[f"k={k}"]["conv_finetuned"] - p[f"k={k}"]["ref_unadapted"] for k in ks]
    maml = [p["k=1"]["maml_unadapted"] - p["k=1"]["ref_unadapted"]] + [p[f"k={k}"]["maml_adapted"] - p[f"k={k}"]["ref_unadapted"] for k in ks]
    a.plot(x0, conv, "-o", color=ORANGE, lw=2, ms=7, label="U-Net + fine-tuning")
    a.plot(x0, maml, "-o", color=BLUE, lw=2, ms=7, label="MAML + adaptation")
    a.set_xticks(x0)
    a.set_xticklabels(["0"] + [f"{k}" for k in ks])
    for k in ks:
        a.text(k, -0.112, f"n={p[f'k={k}']['n']}", ha="center", color=MUTED, fontsize=8.5)
    a.set_xlabel("Annotated slices from the new patient (k)", labelpad=14)
    a.set_ylabel("Dice minus all-vendor model")
    a.set_ylim(-0.115, 0.03); a.grid(axis="y", color=GRID, lw=0.8)
    a.legend(frameon=False, loc="lower right")
    a.set_title("Unseen scanner vendor: four annotated slices close the gap", loc="left", fontsize=12)
    fig.text(0.01, -0.04, "Leave-one-vendor-out models on each held-out vendor's test patients (seed 0). Patients need "
                          ">= k+1 tumour slices, so n falls with k; each point is compared with the all-vendor model on "
                          "the same patients and slices.", fontsize=8.5, color=MUTED, wrap=True)
    fig.tight_layout()
    save(fig, "fig4_fewshot_vendor.png")


# fig 1: example case (rule in figure_rules.md)
def fig_example():
    u = pd.read_csv(EV / "aim1_per_seed.csv", dtype={"patient_id": str})
    bx = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    pool = set(bx[bx.set == "aim1"].patient_id)
    pm = u[u.patient_id.isin(pool)].groupby("patient_id")[["vol_a", "vol_b", "vol_true"]].mean()
    pm = pm[pm.vol_a > 0]
    pm["absch"] = (100 * (pm.vol_b - pm.vol_a) / pm.vol_a).abs()
    elig = pm[pm.absch > 25].sort_values("absch")
    pid = elig.index[(len(elig) - 1) // 2]
    print(f"Fig 1 rule: {len(elig)} eligible; lower median -> {pid} (|change| {elig.loc[pid, 'absch']:.1f}%)")
    S.CACHE = ROOT / "cache" / "NLST_v2"; S.DEV = torch.device("cpu")
    st = S.Store([pid])
    _, real = B.real_prompts()
    bst = B.box_view(st, {pid: real[pid]}) if hasattr(B, "box_view") else None
    if bst is None:
        bst = B.BoxStore.__new__(B.BoxStore); bst.d, bst.windows, bst.cin, bst.prompt = st.d, st.windows, st.cin + 1, {pid: real[pid]}
    conv = S.UNet(cin=st.cin).to(S.DEV); conv.load_state_dict(torch.load(ROOT / "results/v2_conv_s0/conv_best.pt", map_location="cpu")); conv.eval()
    bm = S.UNet(cin=bst.cin).to(S.DEV); bm.load_state_dict(torch.load(ROOT / "results/v2_box_s0/box_best.pt", map_location="cpu")); bm.eval()
    pr = {n: t.detach() for n, t in conv.named_parameters()}
    auto = {k: S.predict(conv, pr, st, pid, pair=(k == "B")) >= 0.5 for k in "AB"}
    boxp = {k: B.eval_box(bm, bst, pid, pair=(k == "B"), return_prob=True)[1] >= 0.5 for k in "AB"}
    msk = st.d[pid]["msk"].astype(bool)
    i = int(msk.sum((1, 2)).argmax())
    yy, xx = np.nonzero(msk[i]); cy, cx = int(yy.mean()), int(xx.mean()); h = 40
    y0, x0 = max(cy - h, 0), max(cx - h, 0)
    pair = pd.read_csv(ROOT / "results/qc/aim1_pairs_final.csv", dtype={"patient_id": str}).drop_duplicates("patient_id").set_index("patient_id").pair[pid]
    ka, kb = pair.split("->")
    bm_ = bx[(bx.set == "aim1") & (bx.patient_id == pid)][["vol_box_a", "vol_box_b"]].mean()
    vols = {"A": (pm.loc[pid, "vol_a"], bm_.vol_box_a), "B": (pm.loc[pid, "vol_b"], bm_.vol_box_b)}
    dz = st.d[pid]["dz"]
    ml = 1.25 * 1.25 * dz / 1000
    fig = plt.figure(figsize=(10, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[4.2, 0.75, 2.6], hspace=0.12, top=0.93)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for a, k, kern in [(ax[0], "A", ka), (ax[1], "B", kb)]:
        img = st.d[pid]["img_pair" if k == "B" else "img"][i].astype(float)
        sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
        a.imshow(img[sl], cmap="gray", vmin=0, vmax=1)
        a.contour(msk[i][sl], [0.5], colors="white", linewidths=1.3, linestyles="--")
        a.contour(auto[k][i][sl], [0.5], colors=ORANGE, linewidths=1.8)
        a.contour(boxp[k][i][sl], [0.5], colors=BLUE, linewidths=1.8)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"Reconstruction {k}: {kern} filter (slice {i})", fontsize=12, color=INK, loc="left")
        tx = fig.add_subplot(gs[1, 0 if k == "A" else 1]); tx.axis("off")
        tx.text(0.0, 0.85, f"Automatic AI volume  {vols[k][0]:.2f} mL\nBox-guided AI volume  {vols[k][1]:.2f} mL",
                va="top", fontsize=10.5, color=INK, transform=tx.transAxes)
    pa = fig.add_subplot(gs[2, :])
    z = np.arange(len(msk))
    pa.fill_between(z, msk.sum((1, 2)) * ml, color="#e1e0d9", step="mid", label="expert mask")
    for k, ls in [("A", "-"), ("B", "--")]:
        pa.step(z, auto[k].sum((1, 2)) * ml, where="mid", color=ORANGE, ls=ls, lw=1.8, label=f"automatic, {k}")
        pa.step(z, boxp[k].sum((1, 2)) * ml, where="mid", color=BLUE, ls=ls, lw=1.8, label=f"box-guided, {k}")
    pa.axvline(i, color=MUTED, lw=0.8, ls=":")
    pa.set_xlabel("Slice in the evaluated slab"); pa.set_ylabel("Segmented volume\nper slice (mL)")
    pa.legend(frameon=False, ncol=1, fontsize=9, loc="upper right")
    pa.set_title("Where the volume differs (seed-0 models)", loc="left", fontsize=11.5, color=INK)
    ch_auto = 100 * (vols["B"][0] - vols["A"][0]) / vols["A"][0]
    ch_box = 100 * (vols["B"][1] - vols["A"][1]) / vols["A"][1]
    fig.suptitle(f"Same scan, two filters: automatic volume {ch_auto:+.0f}%, box-guided {ch_box:+.0f}% "
                 f"(expert {pm.loc[pid, 'vol_true']:.2f} mL)", x=0.01, y=0.995, ha="left", fontsize=12.5)
    fig.text(0.01, 0.005, f"White dashed: expert mask · orange: automatic U-Net · blue: box-guided U-Net. Contours and profile "
                          f"from seed-0 models; volumes printed are 3-seed means. Case {pid} chosen by a pre-set rule "
                          f"(lower median of {len(elig)} patients with >25% automatic change).", fontsize=8.5, color=MUTED, wrap=True)
    save(fig, "fig1_example_case.png")


if __name__ == "__main__":
    fig_dice_vs_volume(); fig_box(); fig_fewshot(); fig_example()
    DESK.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("fig*.png"):
        shutil.copy(f, DESK / f.name)
    print("copied to", DESK)
