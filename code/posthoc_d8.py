"""post hoc stuff for the paper (D8). run --part f after train_sizematched.py"""
import argparse, json
import numpy as np, pandas as pd, torch, SimpleITK as sitk
from scipy import stats
import nlst_seg as S
import box_mode as B
import evaluate as E
from prepare_nlst import to_grid

ROOT = S.ROOT
OUT = ROOT / "results" / "posthoc"
EV = ROOT / "results" / "eval"
S.CACHE = ROOT / "cache" / "NLST_v2"
r4 = E.r4


def nn_prob(case, n):
    f = ROOT / "nnunet" / "pred_ts" / f"{case}.nii.gz"
    img = sitk.ReadImage(str(f)); arr = sitk.GetArrayFromImage(img); sp = img.GetSpacing()
    return None if len(arr) != n else np.stack([to_grid(x.astype(np.float32), (sp[1], sp[0]), True) for x in arr])


# (a) same-patient comparators ------------------------------------------------------------------
def part_a(sp, aim1p):
    _, real = B.real_prompts()
    box = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    rd = pd.read_csv(EV / "readers_rows.csv", dtype={"patient_id": str})
    ids = sorted(set(box.patient_id))
    st = S.Store(ids)
    bst = E.box_view(st, {p: real[p] for p in ids})
    rows = []
    for p in ids:
        pr = nn_prob(f"NLST_{p}", st.n(p))
        if pr is None:
            continue
        msk, dz = st.d[p]["msk"], st.d[p]["dz"]
        rows.append({"patient_id": p, "nn": S.slab_metrics(pr, msk, dz)["dice"],
                     "nn_crop": S.slab_metrics(pr * B.region(bst, p), msk, dz)["dice"]})
    nn = pd.DataFrame(rows).set_index("patient_id")
    pp = box.groupby("patient_id")[["dice_auto", "dice_crop", "dice_box"]].mean().join(nn, how="inner")
    sets = {"test_71": [p for p in pp.index if p in set(box[box.set == "test"].patient_id)],
            "heldout_181": list(pp.index),
            "reader_35": [p for p in rd.dropna(subset=["box_dice"]).patient_id if p in pp.index]}
    out = {}
    for k, ids_ in sets.items():
        g = pp.loc[ids_]
        out[k] = {"n": len(g), **{c: r4(g[c].mean()) for c in g},
                  "box_minus_nn": r4((g.dice_box - g.nn).mean()), "box_minus_nn_CI": E.boot(g.dice_box - g.nn),
                  "box_vs_nn_p": E.wilcox(g.dice_box, g.nn),
                  "box_minus_nncrop": r4((g.dice_box - g.nn_crop).mean()), "box_minus_nncrop_CI": E.boot(g.dice_box - g.nn_crop),
                  "box_vs_nncrop_p": E.wilcox(g.dice_box, g.nn_crop)}
    pp.to_csv(OUT / "d8a_rows.csv")
    return out


# (b) automatic U-Net on all 112 test patients -------------------------------------------------
def part_b(sp, pts):
    test = sp["test"]
    st = S.Store(test)
    rows = []
    for s in E.SEEDS:
        m = E.load(ROOT / "results" / f"v2_conv_s{s}" / "conv_best.pt", st.cin)
        for p in test:
            rows.append({"seed": s, "patient_id": p, "vendor": pts.vendor[p], "dice": S.eval_patient(m, st, p)["dice"]})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "d8b_rows.csv", index=False)
    pp = d.groupby(["patient_id", "vendor"]).dice.mean().reset_index()
    per_seed = d.groupby("seed").dice.mean()
    return {"n": len(pp), "mean": r4(pp.dice.mean()), "CI": E.boot(pp.dice), "per_seed_mean_sd": [r4(per_seed.mean()), r4(per_seed.std())],
            "by_vendor": {v: r4(g.dice.mean()) for v, g in pp.groupby("vendor")}}, pp


# (c) volumetric agreement ---------------------------------------------------------------------
# rc = 2.77 x within subject sd of log volume (qiba style), turned back into %
def agreement(va, vb):
    ok = (va > 0) & (vb > 0)
    d = np.log(vb[ok]) - np.log(va[ok])
    def stats_(x):
        wsd = np.std(x, ddof=1) / np.sqrt(2)
        return np.mean(x), np.std(x, ddof=1), wsd
    bias, sdd, wsd = stats_(d)
    rng = np.random.default_rng(0)
    bs = np.array([stats_(d[rng.integers(0, len(d), len(d))]) for _ in range(E.NBOOT)])
    pct = lambda x: 100 * (np.exp(x) - 1)
    ci = lambda col, f: [round(float(np.percentile(f(bs[:, col]), q)), 2) for q in (2.5, 97.5)]
    return {"n": int(ok.sum()), "excluded_zero_volume": int((~ok).sum()),
            "bias_pct": round(float(pct(bias)), 2), "bias_pct_CI": ci(0, pct),
            "loa_pct": [round(float(pct(bias - 1.96 * sdd)), 2), round(float(pct(bias + 1.96 * sdd)), 2)],
            "RC_pct": round(float(pct(2.77 * wsd)), 2), "RC_pct_CI": ci(2, lambda w: pct(2.77 * w)),
            "wCV_pct": round(float(100 * np.sqrt(np.exp(wsd ** 2) - 1)), 2)}


def part_c():
    u = pd.read_csv(EV / "aim1_per_seed.csv", dtype={"patient_id": str}).groupby("patient_id")[["vol_a", "vol_b", "vol_true"]].mean()
    n = pd.read_csv(EV / "nnunet_aim1_rows.csv", dtype={"patient_id": str}).set_index("patient_id")
    bx = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    b = bx[bx.set == "aim1"].groupby("patient_id")[["vol_box_a", "vol_box_b", "vol_auto_a", "vol_auto_b"]].mean()
    return {"unet_all": agreement(u.vol_a.values, u.vol_b.values),
            "nnunet_all": agreement(n.vol_a.values, n.vol_b.values),
            "unet_prompt110": agreement(b.vol_auto_a.values, b.vol_auto_b.values),
            "box_prompt110": agreement(b.vol_box_a.values, b.vol_box_b.values)}, u, n


# (d) tumour-size strata -----------------------------------------------------------------------
def part_d(u, n):
    out = {}
    bins, labels = [0, 1, 5, np.inf], ["<1 mL", "1-5 mL", ">5 mL"]
    for name, d in [("unet", u), ("nnunet", n)]:
        d = d[d.vol_a > 0].copy()
        d["abs"] = (100 * (d.vol_b - d.vol_a) / d.vol_a).abs()
        d["stratum"] = pd.cut(d.vol_true, bins, labels=labels, right=False)
        out[name] = {str(k): {"n": len(g), "frac_gt25": r4((g["abs"] > 25).mean()), "median_abs_pct": r4(g["abs"].median())}
                     for k, g in d.groupby("stratum", observed=True)}
    return out


# (e) prompt-selection check ------------------------------------------------------------------
def part_e(sp, pts, bpp):
    _, real = B.real_prompts()
    t = pts.loc[sp["test"]].copy()
    t["prompt"] = [p in real for p in t.index]
    t = t.join(bpp.set_index("patient_id").dice)
    w, wo = t[t.prompt], t[~t.prompt]
    ct = pd.crosstab(t.vendor, t.prompt)
    return {"n_with": len(w), "n_without": len(wo),
            "expert_ml_median": {"with": r4(w.expert_volume_ml_native.median()), "without": r4(wo.expert_volume_ml_native.median())},
            "expert_ml_p": float(stats.mannwhitneyu(w.expert_volume_ml_native, wo.expert_volume_ml_native).pvalue),
            "auto_dice_mean": {"with": r4(w.dice.mean()), "without": r4(wo.dice.mean())},
            "auto_dice_p": float(stats.mannwhitneyu(w.dice, wo.dice).pvalue),
            "vendor_counts": {v: {"with": int(ct.loc[v, True]) if True in ct else 0, "without": int(ct.loc[v, False]) if False in ct else 0} for v in ct.index},
            "vendor_chi2_p": float(stats.chi2_contingency(ct)[1])}


# (g) box-guided Dice between reconstructions --------------------------------------------------
def part_g():
    bx = pd.read_csv(EV / "box_rows.csv", dtype={"patient_id": str})
    b = bx[bx.set == "aim1"].groupby("patient_id")[["dice_box", "dice_box_b"]].mean()
    d = b.dice_box_b - b.dice_box
    return {"n": len(b), "dice_A": r4(b.dice_box.mean()), "dice_B": r4(b.dice_box_b.mean()),
            "diff": r4(d.mean()), "CI": E.boot(d), "wilcoxon_p": E.wilcox(b.dice_box, b.dice_box_b)}


# (f) size-matched control ---------------------------------------------------------------------
def part_f(sp, pts):
    ge = [p for p in sp["test"] if pts.vendor[p] == "GE MEDICAL SYSTEMS"]
    st = S.Store(ge)
    res = {}
    for tag, path in [("size_matched_allvendor", "v2_sizematched_s0/conv_best.pt"),
                      ("lovo_GE", "v2_lovo_GE_MEDICAL_SYSTEMS/conv_best.pt"), ("full_allvendor_s0", "v2_conv_s0/conv_best.pt")]:
        m = E.load(ROOT / "results" / path, st.cin)
        res[tag] = [S.eval_patient(m, st, p)["dice"] for p in ge]
    d = pd.DataFrame(res, index=ge)
    d.to_csv(OUT / "d8f_rows.csv")
    return {"n_GE_test": len(ge), **{k: r4(d[k].mean()) for k in d},
            "sizematched_minus_lovo": r4((d.size_matched_allvendor - d.lovo_GE).mean()),
            "sizematched_minus_lovo_CI": E.boot(d.size_matched_allvendor - d.lovo_GE),
            "sizematched_vs_lovo_p": E.wilcox(d.size_matched_allvendor, d.lovo_GE),
            "full_minus_sizematched": r4((d.full_allvendor_s0 - d.size_matched_allvendor).mean()),
            "full_minus_sizematched_CI": E.boot(d.full_allvendor_s0 - d.size_matched_allvendor),
            "note": "gap attributable to vendor absence ~ sizematched - lovo; to training size ~ full - sizematched"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="abcdeg"); ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    S.DEV = torch.device("cpu"); torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "d8_summary.json"
    summ = json.loads(f.read_text()) if f.exists() else {"label": "POST-HOC (D8)"}
    sp, aim1p, pts, _ = E.split_lists()
    if "f" in a.part:
        summ["f_size_matched_control"] = part_f(sp, pts)
    if "a" in a.part:
        summ["a_same_patient"] = part_a(sp, aim1p)
    if "b" in a.part or "e" in a.part:
        summ["b_unet_full_test"], bpp = part_b(sp, pts)
        summ["e_prompt_selection"] = part_e(sp, pts, bpp)
    if "c" in a.part or "d" in a.part:
        summ["c_volumetric_agreement"], u, n = part_c()
        summ["d_size_strata"] = part_d(u, n)
    if "g" in a.part:
        summ["g_box_dice_between_recons"] = part_g()
    f.write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1)[:6000])


if __name__ == "__main__":
    main()
