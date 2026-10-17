"""flow diagram + bland altman figs"""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(r"C:\MetaSeg_data")
OUT = ROOT / "results" / "figures"
DESK = Path(r"C:\Users\Irwin\OneDrive\Desktop\NLST_Reconstruction_Paper\figures")
EV = ROOT / "results" / "eval"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def fig_flow():
    sp = json.loads((ROOT / "results" / "split.json").read_text())
    st = pd.read_csv(ROOT / "results" / "qc" / "final_status.csv", dtype=str)
    pairs = pd.read_csv(ROOT / "results" / "qc" / "aim1_pairs_final.csv", dtype={"patient_id": str})
    s = json.loads((EV / "summary.json").read_text())
    n_pass = int((st.final_status == "PASS").sum())
    n_aim1 = s["aim1"]["n_patients"]
    box = s["box"]["n_prompts"]
    rd = s["readers"]
    boxes = [  # x, y, text, color
        (0.5, 0.92, f"NLSTSeg expert masks: 601 NLST participants", INK2),
        (0.5, 0.76, f"Gate 1 data integrity: {n_pass} usable\n(26 excluded: 19 off-grid masks, 5 unpublished CT,\n1 missing slice reference, 1 visual review)", INK2),
        (0.17, 0.52, f"Training {len(sp['train'])}\nValidation {len(sp['val'])}", BLUE),
        (0.5, 0.52, f"Test {len(sp['test'])}\n(GE 48, Siemens 45,\nToshiba 11, Philips 8)", BLUE),
        (0.83, 0.52, f"Paired reconstructions (Aim 1)\n148 candidates → {len(pairs)} verified series\n({pairs.patient_id.nunique()} participants) → {n_aim1} after D5", ORANGE),
        (0.33, 0.24, f"Radiologist prompts (NLST-Sybil)\nbox on target lesion: test {box['test']}, Aim 1 {box['aim1']}", AQUA),
        (0.72, 0.24, f"Four-reader segmentations\n(LIDC-annot-NLST501): 107 participants,\n{rd['n_patients']} held-out (test {rd['by_set']['test']}, Aim 1 {rd['by_set']['aim1']})", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.2)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0.1, 1)
    for x, y, txt, c in boxes:
        nl = txt.count("\n") + 1
        w, h = (0.46 if y > 0.7 else 0.31), 0.03 + 0.035 * nl
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                    fc=SURF, ec=c, lw=1.6))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9.5, color=INK)
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.2)
    for (x0, y0), (x1, y1) in [((0.5, 0.885), (0.5, 0.83)), ((0.5, 0.69), (0.17, 0.59)), ((0.5, 0.69), (0.5, 0.60)),
                               ((0.5, 0.69), (0.83, 0.60)), ((0.5, 0.44), (0.36, 0.295)), ((0.83, 0.44), (0.40, 0.295)),
                               ((0.5, 0.44), (0.69, 0.31)), ((0.83, 0.44), (0.75, 0.31))]:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=arrow)
    ax.text(0.02, 0.66, "Locked patient-level split (seed 42);\nAim 1 participants held out entirely",
            ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.set_ylim(0.14, 0.97)
    fig.suptitle("Study flow", x=0.02, y=0.99, ha="left", fontsize=13)
    fig.savefig(OUT / "figS_flow.png", dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote figS_flow.png")


def fig_bland_altman():
    u = pd.read_csv(EV / "aim1_per_seed.csv", dtype={"patient_id": str}).groupby("patient_id")[["vol_a", "vol_b"]].mean()
    n = pd.read_csv(EV / "nnunet_aim1_rows.csv", dtype={"patient_id": str}).set_index("patient_id")[["vol_a", "vol_b"]]
    bx = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    b = bx[bx.set == "aim1"].groupby("patient_id")[["vol_box_a", "vol_box_b"]].mean().rename(columns={"vol_box_a": "vol_a", "vol_box_b": "vol_b"})
    d8 = json.loads((ROOT / "results" / "posthoc" / "d8_summary.json").read_text())["c_volumetric_agreement"]
    panels = [("Automatic U-Net (n=128)", u, d8["unet_all"], ORANGE), ("nnU-Net (n=120)", n, d8["nnunet_all"], AQUA),
              ("Box-guided U-Net (n=110)", b, d8["box_prompt110"], BLUE)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for a, (title, d, st, c) in zip(axs, panels):
        d = d[(d.vol_a > 0) & (d.vol_b > 0)]
        gm = np.sqrt(d.vol_a * d.vol_b)
        pc = 100 * (d.vol_b / d.vol_a - 1)
        a.scatter(gm, pc.clip(-100, 300), s=16, color=c, alpha=0.8, lw=0)
        a.axhline(st["bias_pct"], color=INK2, lw=1.2)
        for l in st["loa_pct"]:
            a.axhline(l, color=INK2, lw=1, ls="--")
        a.axhspan(-25, 25, color="#f0efec", zorder=0)
        a.set_xscale("log"); a.set_ylim(-100, 300)
        a.set_title(title, loc="left", fontsize=11.5, color=INK)
        a.text(0.98, 0.96, f"bias {st['bias_pct']:+.1f}%\n95% LoA {st['loa_pct'][0]:.0f}% to {st['loa_pct'][1]:+.0f}%\nRC {st['RC_pct']:.0f}%",
               transform=a.transAxes, ha="right", va="top", fontsize=9.5, color=INK)
        a.set_xlabel("Geometric mean volume (mL, log scale)")
    axs[0].set_ylabel("Volume difference B vs A (%)")
    fig.suptitle("Agreement of AI tumor volume between reconstructions (post-hoc, D8; shaded ±25%)", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "figS_bland_altman.png", dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote figS_bland_altman.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_flow(); fig_bland_altman()
    DESK.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("figS_*.png"):
        shutil.copy(f, DESK / f.name)
