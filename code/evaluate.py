"""
FROZEN EVALUATION (STUDY_PROTOCOL.md §3–4, D4–D6). Committed before any section is run; the test set and
the Aim-1 patients are scored once with this file. Any later change is logged as a deviation.

Models (all checkpoints = best validation Dice): conventional v2_conv_s{0,1,2}; MAML v2_maml_s{0,1,2};
leave-one-vendor-out v2_lovo_<vendor> (conv + MAML, seed 0); box-guided v2_box_s{0,1,2}; nnU-Net 2D.
Metric everywhere: per-patient slab 3D Dice / volume on the 1.25 mm grid (nlst_seg.slab_metrics).
Unit = patient. Seeds: per-patient values are averaged over the 3 seeds for tests; per-seed results are
reported as mean ± SD. Tests: paired Wilcoxon signed-rank (Mann–Whitney U for unpaired groups);
patient-level bootstrap 95% CIs (10,000 resamples, seed 0); Holm across the secondary endpoints listed in
HOLM_FAMILY.

Sections
  aim1    PRIMARY. 128 Aim-1 patients (118672 excluded, D5), one verified pair each. Conventional models
          on the annotated reconstruction (A) and its paired reconstruction (B), both scored against the
          same expert mask. Primary: per-patient Dice difference B − A. Key secondary: volume change
          % = 100 (V_B − V_A) / V_A (patients with V_A > 0), its absolute value, and the fraction of
          patients with |change| > 25% (the volume-growth threshold used in lung screening, e.g. NELSON).
          Context: model seed-to-seed volume variability on A, symmetric % = 100 |x − y| / mean(x, y),
          compared with the same symmetric % between reconstructions. Caveat recorded: the expert mask
          was drawn on A. Subgroups by reconstruction pair type (descriptive).
  aim2    Test patients only. For each vendor V: LOVO model (trained without V) Dice on V's test patients
          (unseen) vs other vendors' test patients (seen): gap = seen − unseen, two-group bootstrap CI,
          Mann–Whitney U. Secondary: on V's test patients, all-vendor model (v2_conv_s0) − LOVO model.
  aim3    Few-shot adaptation, k = 1, 2, 4, 8 tumour slices of the scan being scored, 10 steps, lr 1e-3
          (settings fixed during MAML validation), support slices drawn with random.Random(f"{pid}-{k}")
          and excluded from scoring; patients need >= k+1 tumour slices. Every comparison at a given k
          uses the same excluded slices.
          3a (Aim-1 gap): paired reconstruction B; conventional model fine-tuned vs MAML model adapted,
             seeds 0–2. Reference = conventional, unadapted, on A. Fraction of gap recovered =
             (mean Dice_k − mean conv Dice_0 on B) / (mean conv Dice_0 on A − mean conv Dice_0 on B),
             reported only if the Aim-1 Dice gap 95% CI excludes 0.
          3b (Aim-2 gap): unseen-vendor test patients; LOVO conv fine-tuned vs LOVO MAML adapted.
             Reference = all-vendor conv s0 unadapted; recovered fraction defined analogously.
          Main comparison at each k: MAML vs conventional fine-tuning (Wilcoxon on per-patient Dice_k).
  box     D6 secondary. Patients with a real radiologist prompt in the test set (endpoint) and in the
          held-out pool test ∪ Aim-1 (annotated reconstruction; never used in training or selection).
          Box-guided models vs automatic models vs automatic output cropped to the same prompt region.
          Exploratory: box-guided volume change between reconstructions on Aim-1 patients with prompts.
  readers Radiologists (LIDC-annot-NLST501, 4 reads each) scored with the same slab metric and grid
          against NLSTSeg, on held-out patients (test ∪ Aim-1); a missed lesion scores 0. Compared per
          patient (mean of the 4 reads) with the box-guided and automatic models.
  nnunet  nnU-Net 2D predictions (nnunet/pred_ts) mapped to the 1.25 mm grid: test Dice, per-vendor test
          Dice, and the Aim-1 analysis (single seed).
Usage: python code/evaluate.py --sections aim1,aim2,box,readers [--cpu] [--threads 6]
       --dry: code-path check only — validation patients stand in for the test set, 2 Aim-1 patients
       (the pair already used by the protocol's smoke stage), 1 seed, k in (1, 2), 200 bootstrap
       resamples, output to results/eval_dryrun. Its numbers are not results and are not read.
Outputs: results/eval/<section>_*.csv, results/eval/summary.json, results/eval/EVAL_REPORT.md
"""
from __future__ import annotations
import argparse, glob, json, os, random, time
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
import nlst_seg as S
import box_mode as B

ROOT = S.ROOT
OUT = ROOT / "results" / "eval"
SEEDS = (0, 1, 2)
VENDORS = ["GE MEDICAL SYSTEMS", "SIEMENS", "TOSHIBA", "Philips"]
KS, STEPS, LR = (1, 2, 4, 8), 10, 1e-3
NBOOT = 10_000
DRY = False
HOLM_FAMILY = ["aim1_volume_signed", "aim2_GE MEDICAL SYSTEMS", "aim2_SIEMENS", "aim2_TOSHIBA", "aim2_Philips",
               "box_vs_auto_test"]


# ─────────────────────────── helpers ───────────────────────────
def boot(x, f=np.mean, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float("nan")] * 2
    idx = np.random.default_rng(seed).integers(0, len(x), (NBOOT, len(x)))
    v = f(x[idx], axis=1)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def boot2(a, b, seed=0):
    """CI of mean(a) - mean(b), independent groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.random.default_rng(seed)
    v = a[r.integers(0, len(a), (NBOOT, len(a)))].mean(1) - b[r.integers(0, len(b), (NBOOT, len(b)))].mean(1)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def wilcox(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 2 or np.allclose(a[ok], b[ok]):
        return float("nan")
    return float(stats.wilcoxon(a[ok], b[ok]).pvalue)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def load(path, cin):
    m = S.UNet(cin=cin, base=32).to(S.DEV)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def params_of(m):
    return {n: t.detach() for n, t in m.named_parameters()}


def box_view(st, prompts):
    """BoxStore that shares st's arrays (no second copy in RAM)."""
    b = B.BoxStore.__new__(B.BoxStore)
    b.d, b.windows, b.cin, b.prompt = st.d, st.windows, st.cin + 1, dict(prompts)
    return b


def split_lists():
    sp = S.load_split()
    excl = set(pd.read_csv(ROOT / "results" / "qc" / "exclusions_D5.csv", dtype={"patient_id": str}).patient_id)
    pts = pd.read_csv(S.CACHE / "patients.csv", dtype={"patient_id": str}).set_index("patient_id")
    pairs = pd.read_csv(ROOT / "results" / "qc" / "aim1_pairs_final.csv", dtype={"patient_id": str})
    ptype = pairs.drop_duplicates("patient_id").set_index("patient_id").pair.to_dict()
    aim1 = [p for p in sp["aim1"] if p not in excl]
    if DRY:
        sp = {**sp, "test": sp["val"]}
        aim1 = sp["aim1"][:2]
    return sp, aim1, pts, ptype


def save(section, summary):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "summary.json"
    allsum = json.loads(f.read_text()) if f.exists() else {}
    allsum[section] = summary
    allsum[section]["_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    f.write_text(json.dumps(allsum, indent=2))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "eval.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ─────────────────────────── Aim 1 ───────────────────────────
def volume_summary(df, a="vol_a", b="vol_b", key="patient_id"):
    """df: per seed x patient. Returns volume-change summary (per-patient seed means)."""
    d = df[df[a] > 0].copy()
    d["pct"] = 100 * (d[b] - d[a]) / d[a]
    d["abs_pct"] = d.pct.abs()
    d["sym_pct"] = 100 * (d[b] - d[a]).abs() / ((d[a] + d[b]) / 2)
    pp = d.groupby(key).agg(pct=("pct", "mean"), abs_pct=("abs_pct", "mean"), sym_pct=("sym_pct", "mean"),
                            va=(a, "mean"), vb=(b, "mean"))
    over = (pp.abs_pct > 25).astype(float)
    return pp, {"n_patients_with_V_A>0": len(pp),
                "median_pct": r4(pp.pct.median()), "median_pct_CI": boot(pp.pct, np.median),
                "median_abs_pct": r4(pp.abs_pct.median()), "median_abs_pct_CI": boot(pp.abs_pct, np.median),
                "mean_abs_pct": r4(pp.abs_pct.mean()),
                "frac_abs_pct_gt25": r4(over.mean()), "frac_abs_pct_gt25_CI": boot(over),
                "wilcoxon_p_VA_vs_VB": wilcox(pp.va, pp.vb)}


def aim1(a):
    sp, pids, pts, ptype = split_lists()
    st = S.Store(pids)
    rows = []
    for s in SEEDS:
        m = load(ROOT / "results" / f"v2_conv_s{s}" / "conv_best.pt", st.cin)
        for p in pids:
            ra, rb = S.eval_patient(m, st, p), S.eval_patient(m, st, p, pair=True)
            rows.append({"seed": s, "patient_id": p, "pair": ptype[p], "vendor": pts.vendor[p],
                         "dice_a": ra["dice"], "dice_b": rb["dice"], "vol_a": ra["vol_pred_ml"],
                         "vol_b": rb["vol_pred_ml"], "vol_true": ra["vol_true_ml"]})
        log(f"aim1 seed {s} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aim1_per_seed.csv", index=False)
    summ = aim1_summary(df)
    save("aim1", summ)
    return summ


def aim1_summary(df):
    pp = df.groupby("patient_id").agg(dice_a=("dice_a", "mean"), dice_b=("dice_b", "mean"), pair=("pair", "first"),
                                      vol_true=("vol_true", "first"))
    pp["d"] = pp.dice_b - pp.dice_a
    per_seed = df.assign(d=df.dice_b - df.dice_a).groupby("seed").d.mean()
    vpp, vol = volume_summary(df)
    # seed-to-seed context on A (symmetric %), only when >1 seed
    noise = {}
    if df.seed.nunique() > 1:
        w = df.pivot(index="patient_id", columns="seed", values="vol_a")
        sym = []
        for p, r in w.iterrows():
            v = [100 * abs(r[i] - r[j]) / ((r[i] + r[j]) / 2) for i, j in combinations(w.columns, 2) if r[i] + r[j] > 0]
            sym.append(np.mean(v) if v else np.nan)
        w["seed_sym_pct"] = sym
        j = vpp.join(w[["seed_sym_pct"]], how="inner")
        noise = {"median_seed_to_seed_sym_pct_on_A": r4(j.seed_sym_pct.median()),
                 "median_recon_sym_pct": r4(j.sym_pct.median()),
                 "wilcoxon_p_recon_vs_seed": wilcox(j.sym_pct, j.seed_sym_pct), "n": len(j)}
    vm = df.groupby("patient_id")[["vol_a", "vol_b"]].mean()
    err_a = 100 * (vm.vol_a - pp.vol_true).abs() / pp.vol_true
    err_b = 100 * (vm.vol_b - pp.vol_true).abs() / pp.vol_true
    sub = {}
    for t, g in pp.groupby("pair"):
        vg = vpp.loc[vpp.index.intersection(g.index)]
        sub[t] = {"n": len(g), "mean_dice_a": r4(g.dice_a.mean()), "mean_dice_b": r4(g.dice_b.mean()),
                  "mean_d": r4(g.d.mean()), "median_pct": r4(vg.pct.median()), "median_abs_pct": r4(vg.abs_pct.median()),
                  "frac_abs_pct_gt25": r4((vg.abs_pct > 25).mean())}
    return {"n_patients": len(pp), "n_seeds": int(df.seed.nunique()),
            "mean_dice_A": r4(pp.dice_a.mean()), "mean_dice_B": r4(pp.dice_b.mean()),
            "primary_mean_dice_diff_B_minus_A": r4(pp.d.mean()), "primary_CI": boot(pp.d),
            "primary_wilcoxon_p": wilcox(pp.dice_a, pp.dice_b),
            "per_seed_mean_diff_mean_sd": [r4(per_seed.mean()), r4(per_seed.std())],
            "median_abs_dice_diff": r4(pp.d.abs().median()),
            "volume": vol, "noise_reference": noise,
            "volume_error_vs_expert_median_pct": {"A": r4(err_a.median()), "B": r4(err_b.median()),
                                                  "wilcoxon_p": wilcox(err_a, err_b)},
            "by_pair_type": sub,
            "caveat": "expert mask was drawn on reconstruction A"}


# ─────────────────────────── Aim 2 ───────────────────────────
def lovo_tag(v):
    return "v2_lovo_" + v.replace(" ", "_")


def aim2(a):
    sp, _, pts, _ = split_lists()
    test = sp["test"]
    st = S.Store(test)
    full = load(ROOT / "results" / "v2_conv_s0" / "conv_best.pt", st.cin)
    rows, summ = [], {}
    for v in VENDORS:
        m = load(ROOT / "results" / lovo_tag(v) / "conv_best.pt", st.cin)
        for p in test:
            r = {"heldout_vendor": v, "patient_id": p, "vendor": pts.vendor[p], "unseen": pts.vendor[p] == v,
                 "dice_lovo": S.eval_patient(m, st, p)["dice"]}
            if r["unseen"]:
                r["dice_full"] = S.eval_patient(full, st, p)["dice"]
            rows.append(r)
        d = pd.DataFrame([r for r in rows if r["heldout_vendor"] == v])
        un, se = d[d.unseen], d[~d.unseen]
        summ[v] = {"n_unseen": len(un), "n_seen": len(se),
                   "dice_unseen": r4(un.dice_lovo.mean()), "dice_seen": r4(se.dice_lovo.mean()),
                   "gap_seen_minus_unseen": r4(se.dice_lovo.mean() - un.dice_lovo.mean()),
                   "gap_CI": boot2(se.dice_lovo, un.dice_lovo),
                   "mannwhitney_p": float(stats.mannwhitneyu(se.dice_lovo, un.dice_lovo).pvalue),
                   "dice_unseen_allvendor_model": r4(un.dice_full.mean()),
                   "allvendor_minus_lovo_on_unseen": r4((un.dice_full - un.dice_lovo).mean()),
                   "allvendor_minus_lovo_CI": boot(un.dice_full - un.dice_lovo),
                   "allvendor_vs_lovo_wilcoxon_p": wilcox(un.dice_full, un.dice_lovo)}
        log(f"aim2 {v}: {summ[v]}")
    pd.DataFrame(rows).to_csv(OUT / "aim2_rows.csv", index=False)
    save("aim2", summ)
    return summ


# ─────────────────────────── Aim 3 ───────────────────────────
def few_shot(model, st, p, k, pair, extra=()):
    """Returns d0, dk (Dice excluding support) + Dice_0 of `extra` models/inputs on the same slices."""
    tum = st.tumor_idx(p)
    if len(tum) < k + 1:
        return None
    sup = random.Random(f"{p}-{k}").sample(tum, k)
    msk, dz = st.d[p]["msk"], st.d[p]["dz"]
    base = params_of(model)
    d0 = S.slab_metrics(S.predict(model, base, st, p, pair), msk, dz, exclude=sup)["dice"]
    prm = {n: t.clone().requires_grad_(True) for n, t in base.items()}
    prm = S.adapt(model, prm, st.x(p, sup, pair), st.y(p, sup), STEPS, LR)
    dk = S.slab_metrics(S.predict(model, prm, st, p, pair), msk, dz, exclude=sup)["dice"]
    out = {"d0": d0, "dk": dk}
    for name, (m2, pair2) in dict(extra).items():
        out[name] = S.slab_metrics(S.predict(m2, params_of(m2), st, p, pair2), msk, dz, exclude=sup)["dice"]
    return out


def aim3(a):
    sp, aim1p, pts, _ = split_lists()
    rows = []
    st = S.Store(aim1p)
    for s in SEEDS:
        conv = load(ROOT / "results" / f"v2_conv_s{s}" / "conv_best.pt", st.cin)
        maml = load(ROOT / "results" / f"v2_maml_s{s}" / "maml_best.pt", st.cin)
        for p in aim1p:
            for k in KS:
                rc = few_shot(conv, st, p, k, True, extra={"conv_d0_on_A": (conv, False)})
                if rc is None:
                    continue
                rm = few_shot(maml, st, p, k, True)
                rows.append({"part": "3a", "seed": s, "patient_id": p, "k": k, "conv_d0": rc["d0"], "conv_ft_dk": rc["dk"],
                             "maml_d0": rm["d0"], "maml_dk": rm["dk"], "ref_d0": rc["conv_d0_on_A"]})
        log(f"aim3a seed {s} done")
    test = sp["test"]
    st2 = S.Store(test)
    full = load(ROOT / "results" / "v2_conv_s0" / "conv_best.pt", st2.cin)
    for v in VENDORS:
        conv = load(ROOT / "results" / lovo_tag(v) / "conv_best.pt", st2.cin)
        maml = load(ROOT / "results" / lovo_tag(v) / "maml_best.pt", st2.cin)
        for p in [p for p in test if pts.vendor[p] == v]:
            for k in KS:
                rc = few_shot(conv, st2, p, k, False, extra={"full_d0": (full, False)})
                if rc is None:
                    continue
                rm = few_shot(maml, st2, p, k, False)
                rows.append({"part": "3b", "seed": 0, "vendor": v, "patient_id": p, "k": k, "conv_d0": rc["d0"],
                             "conv_ft_dk": rc["dk"], "maml_d0": rm["d0"], "maml_dk": rm["dk"], "ref_d0": rc["full_d0"]})
        log(f"aim3b {v} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aim3_rows.csv", index=False)
    summ = aim3_summary(df)
    save("aim3", summ)
    return summ


def aim3_summary(df):
    allsum = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    gap_ok = None
    if "aim1" in allsum:
        ci = allsum["aim1"]["primary_CI"]
        gap_ok = not (ci[0] <= 0 <= ci[1])
    summ = {"aim1_gap_CI_excludes_0": gap_ok}
    for part, g0 in df.groupby("part"):
        out = {}
        for k, g in g0.groupby("k"):
            pp = g.groupby("patient_id")[["conv_d0", "conv_ft_dk", "maml_d0", "maml_dk", "ref_d0"]].mean()
            gap = pp.ref_d0.mean() - pp.conv_d0.mean()
            rec = {}
            if part == "3b" or gap_ok:
                rec = {"conv_ft": r4((pp.conv_ft_dk.mean() - pp.conv_d0.mean()) / gap) if gap else None,
                       "maml": r4((pp.maml_dk.mean() - pp.conv_d0.mean()) / gap) if gap else None}
            out[f"k={k}"] = {"n": len(pp), "ref_unadapted": r4(pp.ref_d0.mean()), "conv_unadapted": r4(pp.conv_d0.mean()),
                             "conv_finetuned": r4(pp.conv_ft_dk.mean()), "maml_unadapted": r4(pp.maml_d0.mean()),
                             "maml_adapted": r4(pp.maml_dk.mean()),
                             "maml_minus_convft": r4((pp.maml_dk - pp.conv_ft_dk).mean()),
                             "maml_minus_convft_CI": boot(pp.maml_dk - pp.conv_ft_dk),
                             "maml_vs_convft_wilcoxon_p": wilcox(pp.maml_dk, pp.conv_ft_dk),
                             "gap": r4(gap), "fraction_recovered": rec}
        summ[part] = out
    return summ


# ─────────────────────────── box-guided (D6) ───────────────────────────
def box(a):
    sp, aim1p, pts, ptype = split_lists()
    ok, real = B.real_prompts()
    sets = {"test": [p for p in sp["test"] if p in real], "aim1": [p for p in aim1p if p in real]}
    pids = sets["test"] + sets["aim1"]
    st = S.Store(pids)
    bst = box_view(st, {p: real[p] for p in pids})
    rows = []
    for s in SEEDS:
        conv = load(ROOT / "results" / f"v2_conv_s{s}" / "conv_best.pt", st.cin)
        bm = load(ROOT / "results" / f"v2_box_s{s}" / "box_best.pt", bst.cin)
        for grp, ids in sets.items():
            for p in ids:
                msk, dz = st.d[p]["msk"], st.d[p]["dz"]
                prob = S.predict(conv, params_of(conv), st, p)
                r = {"set": grp, "seed": s, "patient_id": p, "vendor": pts.vendor[p],
                     "dice_auto": S.slab_metrics(prob, msk, dz)["dice"],
                     "dice_crop": S.slab_metrics(prob * B.region(bst, p), msk, dz)["dice"]}
                mb = B.eval_box(bm, bst, p)
                r.update(dice_box=mb["dice"], vol_box_a=mb["vol_pred_ml"], vol_true=mb["vol_true_ml"])
                if grp == "aim1":
                    pb = B.eval_box(bm, bst, p, pair=True)
                    pa_auto = S.slab_metrics(S.predict(conv, params_of(conv), st, p, True), msk, dz)
                    r.update(pair=ptype[p], dice_box_b=pb["dice"], vol_box_b=pb["vol_pred_ml"],
                             vol_auto_a=S.slab_metrics(prob, msk, dz)["vol_pred_ml"], vol_auto_b=pa_auto["vol_pred_ml"])
                rows.append(r)
        log(f"box seed {s} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "box_rows.csv", index=False)
    summ = {"n_prompts": {k: len(v) for k, v in sets.items()},
            "n_without_prompt": {"test": len(sp["test"]) - len(sets["test"]), "aim1": len(aim1p) - len(sets["aim1"])}}
    for name, g0 in [("test", df[df.set == "test"]), ("heldout_test_plus_aim1", df)]:
        per_seed = g0.groupby("seed")[["dice_box", "dice_auto", "dice_crop"]].mean()
        pp = g0.groupby("patient_id")[["dice_box", "dice_auto", "dice_crop"]].mean()
        summ[name] = {"n": len(pp),
                      **{f"mean_{c}": r4(pp[c].mean()) for c in pp}, **{f"median_{c}": r4(pp[c].median()) for c in pp},
                      **{f"per_seed_{c}_mean_sd": [r4(per_seed[c].mean()), r4(per_seed[c].std())] for c in pp},
                      "box_minus_auto": r4((pp.dice_box - pp.dice_auto).mean()), "box_minus_auto_CI": boot(pp.dice_box - pp.dice_auto),
                      "box_vs_auto_wilcoxon_p": wilcox(pp.dice_box, pp.dice_auto),
                      "box_minus_crop": r4((pp.dice_box - pp.dice_crop).mean()), "box_minus_crop_CI": boot(pp.dice_box - pp.dice_crop),
                      "box_vs_crop_wilcoxon_p": wilcox(pp.dice_box, pp.dice_crop)}
    a1 = df[df.set == "aim1"]
    _, vb = volume_summary(a1, "vol_box_a", "vol_box_b")
    _, va = volume_summary(a1, "vol_auto_a", "vol_auto_b")
    summ["aim1_exploratory_volume_change"] = {"box_guided": vb, "automatic_same_patients": va}
    save("box", summ)
    return summ


# ─────────────────────────── radiologists ───────────────────────────
def reader_slab(pid, ct_series, seg_series, files, zslab, shape_grid):
    """Per reader: union of manual findings overlapping NLSTSeg (native), mapped to the cache slab grid."""
    import qc_gate1 as Q
    import reader_agreement as RA
    from prepare_nlst import to_grid
    heads = Q.ct_headers(Q.find(pid, "CT", ct_series)[0])
    masks, _ = Q.seg_masks(glob.glob(os.path.join(Q.find(pid, "SEG", seg_series)[0], "*.dcm"))[0], heads)
    zs = sorted({round(v["pos"][2], 2) for v in heads.values()})
    zidx = {z: i for i, z in enumerate(zs)}
    any_s = next(iter(heads.values()))
    shape = (len(zs),) + tuple(any_s["rc"])
    ref = np.zeros(shape, bool)
    for s_, m in masks.items():
        if s_ in heads and m.sum() >= Q.MIN_PX:
            ref[zidx[round(heads[s_]["pos"][2], 2)]] |= m
    zarr = np.array(zs)
    out = {}
    for f in sorted(files):
        import pydicom
        reader = str(pydicom.dcmread(f, stop_before_pixels=True).get("ContentCreatorName"))
        segs, _ = RA.reader_masks(f, zidx, shape, any_s["pos"][:2])
        if segs is None:
            continue
        hit = [m for m in segs.values() if (m & ref).any()]
        u = np.logical_or.reduce(hit) if hit else np.zeros(shape, bool)
        g = np.zeros(shape_grid, np.float32)
        for i, z in enumerate(zslab):
            j = int(np.argmin(np.abs(zarr - z)))
            if abs(zarr[j] - z) <= RA.Z_TOL and u[j].any():
                g[i] = to_grid(u[j].astype(np.float32), any_s["sp"], True)
        out[reader] = g
    return out


def readers(a):
    sp, aim1p, pts, _ = split_lists()
    man = pd.read_csv(ROOT / "nlstseg_manifest.csv", dtype={"PatientID": str}).set_index("PatientID")
    sub = pd.read_csv(ROOT / "raw" / "lidc_annot_nlst501" / "our_seg_subset.csv", dtype={"patient_id": str})
    held = set(sp["test"]) | set(aim1p)
    pids = sorted(set(sub.patient_id) & held)
    ok, real = B.real_prompts()
    st = S.Store(pids)
    bst = box_view(st, {p: real[p] for p in pids if p in real})
    convs = [load(ROOT / "results" / f"v2_conv_s{s}" / "conv_best.pt", st.cin) for s in SEEDS]
    boxes = [load(ROOT / "results" / f"v2_box_s{s}" / "box_best.pt", st.cin + 1) for s in SEEDS]
    rows = []
    for p in pids:
        c = np.load(S.CACHE / f"{p}.npz")
        msk, dz = st.d[p]["msk"], st.d[p]["dz"]
        files = glob.glob(str(ROOT / "raw" / "lidc_annot_nlst501" / "seg" / p / "*" / "*.dcm"))
        rd = reader_slab(p, man.loc[p, "ct_series"], man.loc[p, "seg_series"], files, np.round(c["z"].astype(float), 2), msk.shape)
        rdice = {r: S.slab_metrics(g, msk, dz)["dice"] for r, g in rd.items()}
        pair_d = [S.slab_metrics(rd[x], (rd[y] > 0).astype(np.uint8), dz)["dice"] for x, y in combinations(sorted(rd), 2)
                  if rd[x].any() or rd[y].any()]
        row = {"patient_id": p, "set": "test" if p in sp["test"] else "aim1", "n_reads": len(rd),
               "reader_dice_mean": np.mean(list(rdice.values())) if rdice else np.nan,
               "reader_dice_min": min(rdice.values()) if rdice else np.nan,
               "reader_dice_max": max(rdice.values()) if rdice else np.nan,
               "inter_reader_dice_mean": np.mean(pair_d) if pair_d else np.nan,
               "auto_dice": np.mean([S.eval_patient(m, st, p)["dice"] for m in convs])}
        if p in real:
            row["box_dice"] = np.mean([B.eval_box(m, bst, p)["dice"] for m in boxes])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "readers_rows.csv", index=False)
    wb = df.dropna(subset=["box_dice"]) if "box_dice" in df else df.iloc[:0]
    summ = {"n_patients": len(df), "by_set": {k: int(v) for k, v in df.set.value_counts().items()},
            "reader_vs_nlstseg_mean": r4(df.reader_dice_mean.mean()), "reader_vs_nlstseg_CI": boot(df.reader_dice_mean),
            "reader_vs_nlstseg_median": r4(df.reader_dice_mean.median()),
            "inter_reader_mean": r4(df.inter_reader_dice_mean.mean()),
            "auto_mean": r4(df.auto_dice.mean()), "auto_minus_reader": r4((df.auto_dice - df.reader_dice_mean).mean()),
            "auto_minus_reader_CI": boot(df.auto_dice - df.reader_dice_mean),
            "auto_vs_reader_wilcoxon_p": wilcox(df.auto_dice, df.reader_dice_mean),
            "n_with_prompt": len(wb), "box_mean": r4(wb.box_dice.mean()) if len(wb) else None,
            "reader_mean_same_patients": r4(wb.reader_dice_mean.mean()) if len(wb) else None,
            "box_minus_reader": r4((wb.box_dice - wb.reader_dice_mean).mean()) if len(wb) else None,
            "box_minus_reader_CI": boot(wb.box_dice - wb.reader_dice_mean) if len(wb) else None,
            "box_vs_reader_wilcoxon_p": wilcox(wb.box_dice, wb.reader_dice_mean) if len(wb) else None}
    save("readers", summ)
    return summ


# ─────────────────────────── nnU-Net ───────────────────────────
def nnunet(a):
    import SimpleITK as sitk
    from prepare_nlst import to_grid
    sp, aim1p, pts, ptype = split_lists()
    pred = ROOT / "nnunet" / "pred_ts"
    rows = []

    def grid_pred(case, p):
        f = pred / f"{case}.nii.gz"
        if not f.exists():
            return None
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)
        spx = img.GetSpacing()
        c = np.load(S.CACHE / f"{p}.npz")
        if len(arr) != len(c["msk"]):
            log(f"nnunet {case}: {len(arr)} slices vs cache {len(c['msk'])} - skipped")
            return None
        g = np.stack([to_grid(s.astype(np.float32), (spx[1], spx[0]), True) for s in arr])
        return g, c["msk"], float(c["dz_mm"])

    for p in sp["test"]:
        r = grid_pred(f"NLST_{p}", p)
        if r:
            rows.append({"set": "test", "patient_id": p, "vendor": pts.vendor[p], "dice": S.slab_metrics(*r)["dice"]})
    a1 = []
    for p in aim1p:
        ra, rb = grid_pred(f"NLST_{p}", p), grid_pred(f"NLST_{p}p", p)
        if ra and rb:
            ma, mb = S.slab_metrics(*ra), S.slab_metrics(*rb)
            a1.append({"seed": 0, "patient_id": p, "pair": ptype[p], "vendor": pts.vendor[p], "dice_a": ma["dice"],
                       "dice_b": mb["dice"], "vol_a": ma["vol_pred_ml"], "vol_b": mb["vol_pred_ml"], "vol_true": ma["vol_true_ml"]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "nnunet_test_rows.csv", index=False)
    d1 = pd.DataFrame(a1)
    d1.to_csv(OUT / "nnunet_aim1_rows.csv", index=False)
    summ = {"test_n": len(t), "test_mean_dice": r4(t.dice.mean()) if len(t) else None,
            "test_CI": boot(t.dice) if len(t) else None,
            "test_by_vendor": {k: r4(v) for k, v in t.groupby("vendor").dice.mean().items()} if len(t) else {},
            "aim1": aim1_summary(d1) if len(d1) else None}
    save("nnunet", summ)
    return summ


# ─────────────────────────── report ───────────────────────────
def holm_and_report():
    f = OUT / "summary.json"
    s = json.loads(f.read_text())
    ps = {}
    if "aim1" in s:
        ps["aim1_volume_signed"] = s["aim1"]["volume"]["wilcoxon_p_VA_vs_VB"]
    if "aim2" in s:
        for v in VENDORS:
            ps[f"aim2_{v}"] = s["aim2"][v]["mannwhitney_p"]
    if "box" in s:
        ps["box_vs_auto_test"] = s["box"]["test"]["box_vs_auto_wilcoxon_p"]
    fam = [(k, ps[k]) for k in HOLM_FAMILY if k in ps and ps[k] == ps[k]]
    fam.sort(key=lambda t: t[1])
    m, adj, run = len(fam), {}, 0.0
    for i, (k, p) in enumerate(fam):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = round(run, 6)
    s["holm_adjusted_p"] = {"family_size_available": m, "family_declared": HOLM_FAMILY, "adjusted": adj}
    f.write_text(json.dumps(s, indent=2))
    md = ["# Evaluation report (generated by code/evaluate.py — do not edit by hand)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Values are copied from `summary.json`.", ""]
    for sec, v in s.items():
        md += [f"## {sec}", "", "```json", json.dumps(v, indent=2), "```", ""]
    (OUT / "EVAL_REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="aim1,aim2,box,readers")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    global OUT, SEEDS, KS, NBOOT, DRY
    if a.dry:
        DRY, OUT, SEEDS, KS, NBOOT = True, ROOT / "results" / "eval_dryrun", (0,), (1, 2), 200
    S.CACHE = ROOT / "cache" / "NLST_v2"
    if a.cpu:
        S.DEV = torch.device("cpu")
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    fns = {"aim1": aim1, "aim2": aim2, "aim3": aim3, "box": box, "readers": readers, "nnunet": nnunet}
    for sec in a.sections.split(","):
        log(f"START {sec} on {S.DEV}")
        r = fns[sec](a)
        log(f"DONE  {sec}: {json.dumps(r)[:600]}")
    holm_and_report()


if __name__ == "__main__":
    main()
