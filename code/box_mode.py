"""box guided unet. trains with simulated boxes, evals with the real radiologist boxes (D6)"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import nlst_seg as S

ROOT = S.ROOT
BOXDIR = ROOT / "cache" / "NLST_v2_box"
DIL_PX, DIL_Z = 8, 1  # 8px at 1.25mm = 10mm margin around the box


def real_prompts():
    d = pd.read_csv(BOXDIR / "prompts.csv", dtype={"patient_id": str})
    ok = d[d.status == "ok"]
    return ok, {r.patient_id: [int(v) for v in r.prompt.split()] for r in ok.itertuples()}


class BoxStore(S.Store):
    def __init__(self, pids):
        super().__init__(pids)
        self.cin += 1
        self.prompt = {}

    def _stack(self, p, i, pair=False):
        base = super()._stack(p, i, pair)
        b = np.zeros(base.shape[1:], np.float32)
        z0, z1, y0, y1, x0, x1 = self.prompt[p]
        if z0 <= i <= z1:
            b[y0:y1 + 1, x0:x1 + 1] = 1
        return np.concatenate([base, b[None]])

    def gt_bbox(self, p):
        z, y, x = np.nonzero(self.d[p]["msk"])
        return [z.min(), z.max(), y.min(), y.max(), x.min(), x.max()]

    def simulate(self, p, offsets, rng):
        """fake training box = expert bbox +/- a real offset row from the training boxes, redraw if it comes out invalid"""
        g = self.gt_bbox(p)
        n, H = self.n(p), self.d[p]["msk"].shape[1]
        for _ in range(20):
            o = offsets[rng.randrange(len(offsets))]
            pr = [g[0] - o[0], g[1] + o[1], g[2] - o[2], g[3] + o[3], g[4] - o[4], g[5] + o[5]]
            pr = [int(np.clip(v, 0, (n if k < 2 else H) - 1)) for k, v in enumerate(pr)]
            if pr[0] <= pr[1] and pr[2] <= pr[3] and pr[4] <= pr[5]:
                return pr
        return [int(v) for v in g]


def region(store, p):
    """box + 10mm in plane and 1 slice up/down, everything outside gets zeroed"""
    n, H = store.n(p), store.d[p]["msk"].shape[1]
    z0, z1, y0, y1, x0, x1 = store.prompt[p]
    r = np.zeros((n, H, H), bool)
    r[max(z0 - DIL_Z, 0):z1 + DIL_Z + 1, max(y0 - DIL_PX, 0):y1 + DIL_PX + 1, max(x0 - DIL_PX, 0):x1 + DIL_PX + 1] = True
    return r


@torch.no_grad()
def eval_box(model, store, p, pair=False, return_prob=False):
    params = {n: t.detach() for n, t in model.named_parameters()}
    prob = S.predict(model, params, store, p, pair) * region(store, p)
    m = S.slab_metrics(prob, store.d[p]["msk"], store.d[p]["dz"])
    return (m, prob) if return_prob else m


def load_init(model, path):
    sd = torch.load(path, map_location="cpu")
    k = "enc.0.net.0.weight"
    w = sd[k]
    sd[k] = torch.cat([w, torch.zeros(w.shape[0], 1, *w.shape[2:], device=w.device)], 1)
    model.load_state_dict(sd)


# starts from the matching seed's unet, the new box channel starts at zero
def train(a):
    rd = ROOT / "results" / a.run
    rd.mkdir(parents=True, exist_ok=True)
    sp = S.load_split()
    ok, real = real_prompts()
    offsets = ok[ok.group == "train"][["off_z0", "off_z1", "off_y0", "off_y1", "off_x0", "off_x1"]].values.tolist()
    train, val = sp["train"], [p for p in sp["val"] if p in real]
    if a.smoke:
        train, val, a.epochs = train[:10], val[:3], 1
    store = BoxStore(sorted(set(train + val)))
    for p in val:
        store.prompt[p] = real[p]
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    model = S.UNet(cin=store.cin, base=32).to(S.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ep0, best = 0, -1.0
    if (rd / "box_last.pt").exists():
        ck = torch.load(rd / "box_last.pt", map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best = ck["ep"], ck["best"]
        S.log(a.run, f"BOX resume from epoch {ep0} (best val Dice {best:.4f})")
    else:
        load_init(model, a.init)
        S.log(a.run, f"BOX init from {a.init} | train {len(train)} (simulated prompts from {len(offsets)} real "
                     f"training-box offsets) | val {len(val)} (real radiologist prompts) | device {S.DEV}")
    tum = [(p, i) for p in train for i in store.tumor_idx(p)]
    t0 = time.time()
    for ep in range(ep0 + 1, a.epochs + 1):
        rng = random.Random(a.seed * 1000 + ep)
        for p in train:
            store.prompt[p] = store.simulate(p, offsets, rng)
        items = tum + [(p, rng.randrange(store.n(p))) for p, _ in (rng.choice(tum) for _ in range(len(tum)))]
        rng.shuffle(items)
        model.train(); losses = []
        for b in range(0, len(items), a.batch):
            x, y = store.crops(items[b:b + a.batch], a.crop, rng, a.fg_prob)
            x, y = S.augment(x, y, rng)
            x[:, -1] = (x[:, -1] > 0.5).float()  # jitter messes up the box channel so snap it back to 0/1
            loss = S.seg_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        model.eval()
        vd = float(np.mean([eval_box(model, store, p)["dice"] for p in val]))
        if vd > best:
            best = vd; torch.save(model.state_dict(), rd / "box_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ep": ep, "best": best}, rd / "box_last.pt")
        S.log(a.run, f"BOX epoch {ep}/{a.epochs} loss {np.mean(losses):.4f} val Dice {vd:.4f} (best {best:.4f}) "
                     f"[{(time.time()-t0)/60:.1f} min]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--fg-prob", type=float, default=0.67)
    ap.add_argument("--init", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU (e.g. smoke test while the GPU is busy)")
    a = ap.parse_args()
    S.CACHE = ROOT / "cache" / "NLST_v2"
    if a.cpu:
        S.DEV = torch.device("cpu")
    a.init = a.init or str(ROOT / "results" / f"v2_conv_s{a.seed}" / "conv_best.pt")
    train(a)


if __name__ == "__main__":
    main()
