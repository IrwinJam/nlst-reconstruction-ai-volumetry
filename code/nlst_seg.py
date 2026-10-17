"""main code: data store, unet, maml, training and eval"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call

ROOT = Path(r"C:\MetaSeg_data")
CACHE = ROOT / "cache" / "NLST"
SPLIT = ROOT / "results" / "split.json"
PX_MM = 1.25
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KS = [1, 2, 4, 8]


# data
class Store:
    """all patient slabs in ram. v2 cache has a lung window and a soft tissue window per slice, input is 3 neighbor slices x windows"""

    def __init__(self, pids):
        self.d = {}
        for p in pids:
            z = np.load(CACHE / f"{p}.npz")
            rec = {k: z[k] for k in ("img", "msk", "tumor")}
            rec["dz"] = float(z["dz_mm"])
            for k in ("img_pair", "img_st", "img_pair_st"):
                if k in z.files:
                    rec[k] = z[k]
            self.d[p] = rec
        self.windows = ["img", "img_st"] if all("img_st" in r for r in self.d.values()) else ["img"]
        self.cin = 3 * len(self.windows)

    def n(self, p):
        return len(self.d[p]["img"])

    def tumor_idx(self, p):
        return np.nonzero(self.d[p]["tumor"])[0].tolist()

    def _stack(self, p, i, pair=False):
        chans = []
        for w in self.windows:
            img = self.d[p][w.replace("img", "img_pair") if pair else w]
            n = len(img)
            chans += [img[max(i - 1, 0)], img[i], img[min(i + 1, n - 1)]]
        return np.stack(chans)

    def x(self, p, idx, pair=False):
        out = np.stack([self._stack(p, i, pair) for i in idx])
        return torch.from_numpy(out.astype(np.float32)).to(DEV)

    def y(self, p, idx):
        return torch.from_numpy(self.d[p]["msk"][idx].astype(np.float32)).unsqueeze(1).to(DEV)

    def crops(self, items, size, rng, fg_prob):
        """crops w/ foreground oversampling. tumor is ~0.14% of pixels so without this it just learns to predict nothing"""
        xs, ys = [], []
        for p, i in items:
            m = self.d[p]["msk"][i]
            H = m.shape[0]
            stack = self._stack(p, i)
            if m.any() and rng.random() < fg_prob:
                yy, xx = np.nonzero(m)
                j = rng.randrange(len(yy))
                y0 = int(np.clip(yy[j] - rng.randrange(size), 0, H - size))
                x0 = int(np.clip(xx[j] - rng.randrange(size), 0, H - size))
            else:
                y0, x0 = rng.randrange(H - size + 1), rng.randrange(H - size + 1)
            xs.append(stack[:, y0:y0 + size, x0:x0 + size]); ys.append(m[None, y0:y0 + size, x0:x0 + size])
        x = torch.from_numpy(np.stack(xs).astype(np.float32)).to(DEV)
        y = torch.from_numpy(np.stack(ys).astype(np.float32)).to(DEV)
        return x, y


def augment(x, y, rng):
    """v2 augmentation on gpu (D3): rotate +-15, scale 0.85-1.15, flips, contrast/brightness, a bit of noise. masks follow the geometric stuff"""
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device)
    for k in range(b):
        if rng.random() < 0.7:
            a = np.deg2rad(rng.uniform(-15, 15)); s = rng.uniform(0.85, 1.15)
        else:
            a, s = 0.0, 1.0
        c, si = np.cos(a) / s, np.sin(a) / s
        theta[k] = torch.tensor([[c, -si, 0.0], [si, c, 0.0]])
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y = (F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False) > 0.5).float()
    if rng.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if rng.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if rng.random() < 0.5:
        con = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
        bri = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.08, 0.08)
        mu = x.mean(dim=(2, 3), keepdim=True)
        x = ((x - mu) * con + mu + bri).clamp(0, 1)
    if rng.random() < 0.3:
        x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    return x, y


# split gets locked once and never redone, test set stays untouched til gate 2
def lock_split(seed=42, force=False):
    if SPLIT.exists() and not force:
        raise SystemExit(f"{SPLIT} already exists, the split is locked. not overwriting.")
    pts = pd.read_csv(CACHE / "patients.csv", dtype={"patient_id": str})
    aim1 = sorted(pts[pts.paired_series.fillna("") != ""].patient_id)
    pool = pts[~pts.patient_id.isin(aim1)]
    rng = random.Random(seed)
    sp = {"aim1": aim1, "train": [], "val": [], "test": []}
    for vendor, g in pool.groupby("vendor"):
        ids = sorted(g.patient_id)
        rng.shuffle(ids)
        n_test, n_val = round(0.25 * len(ids)), max(1, round(0.10 * len(ids)))
        sp["test"] += ids[:n_test]; sp["val"] += ids[n_test:n_test + n_val]; sp["train"] += ids[n_test + n_val:]
    sp = {k: sorted(v) for k, v in sp.items()}
    sp["_meta"] = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
                   "rule": "Aim-1 paired patients held out entirely; remainder 65/10/25 stratified by vendor",
                   "counts": {k: len(v) for k, v in sp.items() if not k.startswith("_")},
                   "by_vendor": {k: pts[pts.patient_id.isin(v)].vendor.value_counts().to_dict()
                                 for k, v in sp.items() if not k.startswith("_")}}
    SPLIT.write_text(json.dumps(sp, indent=1))
    return sp


def load_split():
    return json.loads(SPLIT.read_text())


# model
def gn(c):
    return nn.GroupNorm(min(8, c), c)


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), gn(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), gn(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4 level unet w/ groupnorm (no running stats so functional_call works right for maml)"""

    def __init__(self, cin=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([Block(cin, c[0]), Block(c[0], c[1]), Block(c[1], c[2]), Block(c[2], c[3])])
        self.mid = Block(c[3], c[4])
        self.dec = nn.ModuleList([Block(c[4] + c[3], c[3]), Block(c[3] + c[2], c[2]),
                                  Block(c[2] + c[1], c[1]), Block(c[1] + c[0], c[0])])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for d, s in zip(self.dec, reversed(skips)):
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.out(x)


def seg_loss(logits, y):
    """bce + soft dice pooled over the batch so empty slices dont break it"""
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, y) + dice


def adapt(model, params, x, y, steps, lr, keep_graph=False):
    for _ in range(steps):
        loss = seg_loss(functional_call(model, params, (x,)), y)
        g = torch.autograd.grad(loss, list(params.values()), create_graph=keep_graph)
        params = ({n: p - lr * gi for (n, p), gi in zip(params.items(), g)} if keep_graph else
                  {n: (p - lr * gi).detach().requires_grad_(True) for (n, p), gi in zip(params.items(), g)})
    return params


# evaluation
@torch.no_grad()
def predict(model, params, store, p, pair=False, bs=12):
    probs = []
    for i in range(0, store.n(p), bs):
        idx = list(range(i, min(i + bs, store.n(p))))
        probs.append(torch.sigmoid(functional_call(model, params, (store.x(p, idx, pair),))).squeeze(1).cpu().numpy())
    return np.concatenate(probs)


# scored over the whole slab so false positives off the tumor count too
def slab_metrics(prob, gt, dz, exclude=(), thr=0.5):
    keep = np.ones(len(gt), bool)
    keep[list(exclude)] = False
    pb, g = (prob[keep] >= thr), gt[keep].astype(bool)
    inter = (pb & g).sum()
    denom = pb.sum() + g.sum()
    vox_ml = PX_MM * PX_MM * dz / 1000
    return {"dice": float(2 * inter / denom) if denom else 1.0,
            "vol_pred_ml": float(pb.sum() * vox_ml), "vol_true_ml": float(g.sum() * vox_ml)}


def eval_patient(model, store, p, k=0, lr=0.0, steps=0, seed=0, pair=False):
    """k=0 is plain inference. k>0 adapts on k tumor slices of the same scan then scores the slab without those slices"""
    base = {n: t.detach() for n, t in model.named_parameters()}
    params = {n: t.clone().requires_grad_(k > 0) for n, t in base.items()}
    sup = []
    if k > 0:
        tum = store.tumor_idx(p)
        sup = random.Random(seed).sample(tum, min(k, len(tum)))
        params = adapt(model, params, store.x(p, sup, pair), store.y(p, sup), steps, lr)
    prob = predict(model, params, store, p, pair)
    return slab_metrics(prob, store.d[p]["msk"], store.d[p]["dz"], exclude=sup)


def eval_set(model, store, pids, **kw):
    model.eval()
    return {p: eval_patient(model, store, p, **kw) for p in pids}


def mean_dice(res):
    return float(np.mean([v["dice"] for v in res.values()])) if res else float("nan")


# training
def log(run, msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "results" / run / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def train_conv(run, store, train, val, a):
    rd = ROOT / "results" / run
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    model = UNet(cin=store.cin, base=a.base).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ep0, best = 0, -1.0
    if (rd / "conv_last.pt").exists():
        ck = torch.load(rd / "conv_last.pt", map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best = ck["ep"], ck["best"]
        log(run, f"CONV resume from epoch {ep0} (best val Dice {best:.4f})")
    tum = [(p, i) for p in train for i in store.tumor_idx(p)]
    t0 = time.time()
    for ep in range(ep0 + 1, a.epochs + 1):
        rng = random.Random(a.seed * 1000 + ep)
        # each epoch = every tumor slice once + the same number of random slab slices (mostly empty)
        items = tum + [(p, rng.randrange(store.n(p))) for p, _ in (rng.choice(tum) for _ in range(len(tum)))]
        rng.shuffle(items)
        model.train(); losses = []
        for b in range(0, len(items), a.batch):
            x, y = store.crops(items[b:b + a.batch], a.crop, rng, a.fg_prob)
            if a.aug:
                x, y = augment(x, y, rng)
            elif rng.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            loss = seg_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        vd = mean_dice(eval_set(model, store, val))
        if vd > best:
            best = vd; torch.save(model.state_dict(), rd / "conv_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ep": ep, "best": best}, rd / "conv_last.pt")
        log(run, f"CONV epoch {ep}/{a.epochs} loss {np.mean(losses):.4f} val Dice {vd:.4f} (best {best:.4f}) "
                 f"[{(time.time()-t0)/60:.1f} min]")


# first order maml by default, second order eats too much memory on an 8gb card
def train_maml(run, store, train, val, a):
    rd = ROOT / "results" / run
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    model = UNet(cin=store.cin, base=a.base).to(DEV)
    if a.init and Path(a.init).exists():
        model.load_state_dict(torch.load(a.init, map_location=DEV))
        log(run, f"MAML initialized from {a.init}")
    opt = torch.optim.Adam(model.parameters(), lr=a.meta_lr)
    it0, best = 0, -1.0
    if (rd / "maml_last.pt").exists():
        ck = torch.load(rd / "maml_last.pt", map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); it0, best = ck["it"], ck["best"]
        log(run, f"MAML resume from iteration {it0} (best val Dice {best:.4f})")
    rng = random.Random(a.seed * 7 + it0)
    tasks = [p for p in train if len(store.tumor_idx(p)) >= 2]
    t0 = time.time()
    for it in range(it0 + 1, a.meta_iters + 1):
        opt.zero_grad(); tot = 0.0
        for _ in range(a.meta_batch):
            p = rng.choice(tasks)
            tum = store.tumor_idx(p)
            k = min(rng.choice(KS), len(tum) - 1)
            sup = rng.sample(tum, k)
            rest = [i for i in range(store.n(p)) if i not in sup]
            qry = rng.sample(rest, min(a.query, len(rest)))
            params = dict(model.named_parameters())
            xs, ys = store.crops([(p, i) for i in sup], a.crop, rng, 1.0)
            xq, yq = store.crops([(p, i) for i in qry], a.crop, rng, a.fg_prob)
            if a.aug:
                xs, ys = augment(xs, ys, rng); xq, yq = augment(xq, yq, rng)
            if a.second_order:
                fast = adapt(model, params, xs, ys, a.inner_steps, a.inner_lr, keep_graph=True)
                loss = seg_loss(functional_call(model, fast, (xq,)), yq) / a.meta_batch
                loss.backward()
            else:
                start = {n: t.detach().clone().requires_grad_(True) for n, t in params.items()}
                fast = adapt(model, start, xs, ys, a.inner_steps, a.inner_lr)
                loss = seg_loss(functional_call(model, fast, (xq,)), yq) / a.meta_batch
                for (n, t), g in zip(params.items(), torch.autograd.grad(loss, list(fast.values()))):
                    t.grad = g if t.grad is None else t.grad + g
            tot += loss.item()
        opt.step()
        if it % 50 == 0:
            log(run, f"MAML it {it}/{a.meta_iters} meta-loss {tot:.4f} [{(time.time()-t0)/60:.1f} min]")
        if it % a.val_every == 0 or it == a.meta_iters:
            vd = mean_dice(eval_set(model, store, val, k=4, lr=a.inner_lr, steps=a.eval_steps, seed=11))
            if vd > best:
                best = vd; torch.save(model.state_dict(), rd / "maml_best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "best": best}, rd / "maml_last.pt")
            log(run, f"MAML it {it} val Dice@k4 {vd:.4f} (best {best:.4f})")


# cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["split", "conv", "maml", "smoke"])
    ap.add_argument("--run", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--fg-prob", type=float, default=0.67)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--meta-iters", type=int, default=4000)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--meta-lr", type=float, default=1e-4)
    ap.add_argument("--inner-steps", type=int, default=3)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--query", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--second-order", action="store_true")
    ap.add_argument("--init", default="")
    ap.add_argument("--exclude-vendor", default="", help="Aim 2: drop this vendor from train/val")
    ap.add_argument("--aug", action="store_true", help="recipe-v2 augmentation (D3)")
    ap.add_argument("--cache", default="", help="cache dir (default cache/NLST; recipe v2: cache/NLST_v2)")
    a = ap.parse_args()

    if a.stage == "split":
        sp = lock_split()
        print(json.dumps(sp["_meta"], indent=1)); return

    global CACHE
    if a.cache:
        CACHE = Path(a.cache)
    sp = load_split()
    (ROOT / "results" / a.run).mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(CACHE / "patients.csv", dtype={"patient_id": str}).set_index("patient_id")
    train, val = sp["train"], sp["val"]
    if a.exclude_vendor:
        train = [p for p in train if pts.vendor[p] != a.exclude_vendor]
        val = [p for p in val if pts.vendor[p] != a.exclude_vendor]
    if a.stage == "smoke":
        train, val = train[:10], val[:3]
        a.epochs, a.meta_iters, a.val_every = 1, 20, 20
    store = Store(sorted(set(train + val)))
    log(a.run, f"stage {a.stage} seed {a.seed} | train {len(train)} val {len(val)} patients | device {DEV}")
    if a.stage in ("conv", "smoke"):
        train_conv(a.run, store, train, val, a)
    if a.stage in ("maml", "smoke"):
        train_maml(a.run, store, train, val, a)
    if a.stage == "smoke":
        # quick aim 1 style check on 2 pairs
        s1 = Store(sp["aim1"][:2])
        m = UNet(cin=s1.cin, base=a.base).to(DEV); m.load_state_dict(torch.load(ROOT / "results" / a.run / "conv_best.pt", map_location=DEV))
        for p in sp["aim1"][:2]:
            ra, rb = eval_patient(m, s1, p), eval_patient(m, s1, p, pair=True)
            log(a.run, f"SMOKE aim1 {p}: annotated recon {ra} | paired recon {rb}")
        log(a.run, f"SMOKE peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if DEV.type == "cuda" else "cpu")


if __name__ == "__main__":
    main()
