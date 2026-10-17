"""size matched all vendor unet for the GE control (D8f)"""
import argparse, json, random
import pandas as pd
import nlst_seg as S

S.CACHE = S.ROOT / "cache" / "NLST_v2"
RUN = "v2_sizematched_s0"


def main():
    sp = S.load_split()
    pts = pd.read_csv(S.CACHE / "patients.csv", dtype={"patient_id": str}).set_index("patient_id")
    train = sp["train"]
    n_target = sum(pts.vendor[p] != "GE MEDICAL SYSTEMS" for p in train)
    rng = random.Random(0)
    by_v = {}
    for p in train:
        by_v.setdefault(pts.vendor[p], []).append(p)
    quota = {v: round(len(ps) * n_target / len(train)) for v, ps in by_v.items()}
    while sum(quota.values()) != n_target:  # fix rounding so it adds up exactly
        big = max(quota, key=lambda v: len(by_v[v]))
        quota[big] += 1 if sum(quota.values()) < n_target else -1
    sub = sorted(p for v, ps in by_v.items() for p in rng.sample(sorted(ps), quota[v]))
    (S.ROOT / "results" / RUN).mkdir(parents=True, exist_ok=True)
    (S.ROOT / "results" / RUN / "subset.json").write_text(json.dumps({"n": len(sub), "quota": quota, "patients": sub}, indent=1))
    a = argparse.Namespace(seed=0, base=32, epochs=100, batch=24, crop=160, fg_prob=0.67, lr=1e-3, aug=True)
    store = S.Store(sorted(set(sub + sp["val"])))
    S.log(RUN, f"D8(f) size-matched all-vendor model: train {len(sub)} {quota} | val {len(sp['val'])}")
    S.train_conv(RUN, store, sub, sp["val"], a)


if __name__ == "__main__":
    main()
