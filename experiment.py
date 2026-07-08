"""
greedy-lgn: Backprop-free, layer-by-layer training of Logic Gate Networks
with immediate discretization, adaptive depth, and incremental logic simplification.

Runs on CPU in a few minutes. Requirements: torch, scikit-learn.

Usage:
    python experiment.py                     # default config (a few minutes on CPU)
    python experiment.py --gates 200 --epochs 30 --max-layers 3   # quick smoke test
    python experiment.py --device cuda       # same experiment on GPU
    python experiment.py --skip-input        # re-expose input bits to every layer
    python experiment.py --skip-all          # DenseNet-style: all previous layers
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ----------------------------- 16 two-input gates -----------------------------
# Real-valued relaxations (exact on {0,1}). Order follows difflogic convention.
GATE_NAMES = ["FALSE", "AND", "A&!B", "A", "!A&B", "B", "XOR", "OR",
              "NOR", "XNOR", "!B", "A|!B", "!A", "!A|B", "NAND", "TRUE"]
SWAP = {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3, 6: 6, 7: 7, 8: 8, 9: 9,
        10: 12, 11: 13, 12: 10, 13: 11, 14: 14, 15: 15}  # fn under input swap

def all16(a, b):
    ab = a * b
    return torch.stack([
        torch.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab,
        1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a,
        1 - a + ab, 1 - ab, torch.ones_like(a)], dim=-1)

def f16(fn, a, b):
    ab = a * b
    return [a * 0, ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab,
            1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a,
            1 - a + ab, 1 - ab, a * 0 + 1][fn]

class LogicLayer(nn.Module):
    """One layer of 2-input logic gates with fixed random wiring."""
    def __init__(self, in_dim, n_gates, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("ia", torch.randint(0, in_dim, (n_gates,), generator=g))
        self.register_buffer("ib", torch.randint(0, in_dim, (n_gates,), generator=g))
        self.logits = nn.Parameter(torch.randn(n_gates, 16, generator=g))
    def forward(self, x):  # soft (training) mode
        return (all16(x[:, self.ia], x[:, self.ib])
                * F.softmax(self.logits, -1)).sum(-1)
    @torch.no_grad()
    def hard(self, x):  # discretized (inference) mode
        sel = self.logits.argmax(-1)
        return all16(x[:, self.ia], x[:, self.ib]).gather(
            -1, sel.view(1, -1, 1).expand(x.shape[0], -1, 1)).squeeze(-1)

# ----------------------------- data -----------------------------
def load_data(seed=0):
    X, y = load_digits(return_X_y=True)          # 8x8 digits, values 0..16
    Xb = np.concatenate([(X > t).astype(np.float32) for t in (3, 7, 11)], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(
        Xb, y, test_size=0.25, stratify=y, random_state=seed)
    return (torch.tensor(Xtr), torch.tensor(Xte),
            torch.tensor(ytr), torch.tensor(yte))

# ----------------------------- training utils -----------------------------
def group_sum(h, n_class, tau):
    return h.view(h.shape[0], n_class, -1).sum(-1) / tau

def accuracy(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()

def train_layer(layer, Xin, yin, epochs, n_class, tau, lr):
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(group_sum(layer(Xin), n_class, tau), yin).backward()
        opt.step()

# ----------------------------- (A) greedy layer-wise -----------------------------
def run_greedy(Xtr, Xte, ytr, yte, cfg):
    print("=== (A) Greedy layer-wise: local loss -> discretize -> freeze ==="
          + (" [skip-all wiring]" if cfg.skip_all else
             " [skip-input wiring]" if cfg.skip_input else ""))
    tau = float(np.sqrt(cfg.gates / cfg.n_class))
    layers, pool_tr, pool_te = [], Xtr, Xte
    best_acc, best_depth, since_best = -1.0, 0, 0
    t0 = time.time()
    for d in range(1, cfg.max_layers + 1):
        L = LogicLayer(pool_tr.shape[1], cfg.gates, seed=cfg.seed * 100 + d).to(cfg.device)
        train_layer(L, pool_tr, ytr, cfg.epochs, cfg.n_class, tau, cfg.lr)
        h_tr, h_te = L.hard(pool_tr), L.hard(pool_te)   # freeze on HARD bits
        a_te = accuracy(group_sum(h_te, cfg.n_class, tau), yte)
        a_tr = accuracy(group_sum(h_tr, cfg.n_class, tau), ytr)
        print(f"  layer {d}: hard probe  train={a_tr:.4f}  test={a_te:.4f}")
        layers.append(L)
        # next layer's wiring pool: gate outputs, optionally with earlier bits re-exposed
        if cfg.skip_all:      # DenseNet-style: input + every previous layer
            pool_tr, pool_te = torch.cat([pool_tr, h_tr], 1), torch.cat([pool_te, h_te], 1)
        elif cfg.skip_input:  # input + previous layer only
            pool_tr, pool_te = torch.cat([Xtr, h_tr], 1), torch.cat([Xte, h_te], 1)
        else:
            pool_tr, pool_te = h_tr, h_te
        if a_te > best_acc + 1e-4:
            best_acc, best_depth, since_best = a_te, d, 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                print(f"  -> stop: no improvement for {cfg.patience} layers")
                break
    print(f"  greedy: best hard test acc = {best_acc:.4f} at depth {best_depth}"
          f"  ({time.time() - t0:.0f}s)\n")
    return layers[:best_depth], best_acc, best_depth

# ----------------------------- (B) end-to-end baseline -----------------------------
def run_e2e(Xtr, Xte, ytr, yte, depth, cfg):
    print(f"=== (B) End-to-end backprop baseline, {depth} layers ===")
    tau = float(np.sqrt(cfg.gates / cfg.n_class))
    net = nn.ModuleList([LogicLayer(Xtr.shape[1] if i == 0 else cfg.gates,
                                    cfg.gates, seed=cfg.seed * 200 + i)
                         for i in range(depth)]).to(cfg.device)
    opt = torch.optim.Adam([p for L in net for p in L.parameters()], lr=cfg.lr)
    epochs = min(cfg.epochs * depth, cfg.e2e_max_epochs)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        h = Xtr
        for L in net:
            h = L(h)
        F.cross_entropy(group_sum(h, cfg.n_class, tau), ytr).backward()
        opt.step()
    with torch.no_grad():
        hs, hh = Xte, Xte
        for L in net:
            hs = L(hs)
        for L in net:
            hh = L.hard(hh)
    soft = accuracy(group_sum(hs, cfg.n_class, tau), yte)
    hard = accuracy(group_sum(hh, cfg.n_class, tau), yte)
    print(f"  soft={soft:.4f}  discretized={hard:.4f}  gap={soft - hard:+.4f}"
          f"  ({epochs} epochs, {time.time() - t0:.0f}s)")
    print(f"  float logits during training: greedy={cfg.gates * 16:,} (1 layer)"
          f"  vs  e2e={cfg.gates * 16 * depth:,} (x{depth})\n")
    return soft, hard

# ----------------------------- (C) logic simplification -----------------------------
def simplify(layers, Xte, yte, cfg):
    """Constant folding, pass-through/NOT reduction, duplicate merge,
    dead-gate elimination. Verifies the simplified circuit is bit-identical."""
    print("=== (C) Logic simplification of the greedy hard network ===")
    in_bits, G, D = Xte.shape[1], cfg.gates, len(layers)
    tau = float(np.sqrt(G / cfg.n_class))
    net, base = [], in_bits
    skip = getattr(cfg, "skip_input", False)
    dense = getattr(cfg, "skip_all", False)
    for li, L in enumerate(layers):
        prev = base - G  # global id of the previous layer's first gate (li > 0)
        def src(j, li=li, prev=prev):
            if li == 0:
                return j                                  # wired to input bits
            if dense:                                     # pool = [input || all layers]:
                return j                                  #   pool order == global id order
            if skip:                                      # pool = [input || prev layer]
                return j if j < in_bits else prev + (j - in_bits)
            return prev + j                               # pool = prev layer only
        fn = L.logits.argmax(-1).tolist()
        net.append([(base + i, src(int(L.ia[i])), src(int(L.ib[i])), fn[i])
                    for i in range(G)])
        base += G
    total_before = D * G
    final_ids = [g[0] for g in net[-1]]

    # reference evaluation
    vals = {i: Xte[:, i] for i in range(in_bits)}
    for layer in net:
        for gid, ia, ib, fn in layer:
            vals[gid] = f16(fn, vals[ia], vals[ib])
    ref = torch.stack([vals[i] for i in final_ids], 1)
    acc0 = accuracy(group_sum(ref, cfg.n_class, tau), yte)

    resolve = {}
    def res(n):
        while True:
            r = resolve.get(n)
            if r is None:
                return ('node', n)
            if r[0] == 'const':
                return r
            n = r[1]

    c = dict(const=0, passthru=0, dup=0)
    kept = []
    for layer in net:
        seen, out = {}, []
        for gid, ia0, ib0, fn in layer:
            ra, rb = res(ia0), res(ib0)
            if ra[0] == 'node' and rb[0] == 'node' and ra[1] > rb[1]:
                ra, rb, fn = rb, ra, SWAP[fn]
            if ra[0] == 'const' and rb[0] == 'const':
                v = int(f16(fn, torch.tensor(float(ra[1])),
                            torch.tensor(float(rb[1]))).item())
                resolve[gid] = ('const', v); c['const'] += 1; continue
            if ra[0] == 'const' or rb[0] == 'const':
                cv = float((ra if ra[0] == 'const' else rb)[1])
                other = (rb if ra[0] == 'const' else ra)[1]
                z, o = torch.tensor(0.), torch.tensor(1.)
                if ra[0] == 'const':
                    v0, v1 = int(f16(fn, torch.tensor(cv), z).item()), \
                             int(f16(fn, torch.tensor(cv), o).item())
                else:
                    v0, v1 = int(f16(fn, z, torch.tensor(cv)).item()), \
                             int(f16(fn, o, torch.tensor(cv)).item())
                if v0 == v1:
                    resolve[gid] = ('const', v0); c['const'] += 1; continue
                if (v0, v1) == (0, 1):
                    resolve[gid] = ('node', other); c['passthru'] += 1; continue
                ra = rb = ('node', other); fn = 12       # NOT(other)
            ia, ib = ra[1], rb[1]
            if fn == 3:
                resolve[gid] = ('node', ia); c['passthru'] += 1; continue
            if fn == 5:
                resolve[gid] = ('node', ib); c['passthru'] += 1; continue
            if fn == 0:
                resolve[gid] = ('const', 0); c['const'] += 1; continue
            if fn == 15:
                resolve[gid] = ('const', 1); c['const'] += 1; continue
            key = (ia, ib, fn)
            if key in seen:
                resolve[gid] = ('node', seen[key]); c['dup'] += 1; continue
            seen[key] = gid
            out.append((gid, ia, ib, fn))
        kept.append(out)

    live = set()
    for gid in final_ids:
        r = res(gid)
        if r[0] == 'node' and r[1] >= in_bits:
            live.add(r[1])
    for layer in reversed(kept):
        for gid, ia, ib, fn in layer:
            if gid in live:
                if ia >= in_bits: live.add(ia)
                if ib >= in_bits: live.add(ib)
    kept = [[g for g in l if g[0] in live] for l in kept]
    total_after = sum(len(l) for l in kept)
    dead = total_before - c['const'] - c['passthru'] - c['dup'] - total_after

    # verify simplified circuit
    vals2 = {i: Xte[:, i] for i in range(in_bits)}
    for layer in kept:
        for gid, ia, ib, fn in layer:
            vals2[gid] = f16(fn, vals2[ia], vals2[ib])
    cols = []
    for gid in final_ids:
        r = res(gid)
        cols.append(torch.full((len(Xte),), float(r[1]))
                    if r[0] == 'const' else vals2[r[1]])
    out = torch.stack(cols, 1)
    acc1 = accuracy(group_sum(out, cfg.n_class, tau), yte)
    identical = torch.equal(ref, out)

    print(f"  gates: {total_before:,} -> {total_after:,}"
          f" ({100 * total_after / total_before:.1f}%)")
    print(f"  const-folded={c['const']}, pass-through={c['passthru']},"
          f" duplicates={c['dup']}, dead={dead}")
    print(f"  test acc {acc0:.4f} -> {acc1:.4f}, outputs identical = {identical}\n")
    assert identical, "simplification changed the function!"
    return total_before, total_after

# ----------------------------- main -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gates", type=int, default=500, help="gates per layer (multiple of 10)")
    p.add_argument("--epochs", type=int, default=120, help="epochs per greedy layer")
    p.add_argument("--max-layers", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--e2e-max-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-e2e", action="store_true", help="skip the backprop baseline")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--e2e-depth", type=int, default=None,
                   help="override e2e baseline depth (default: greedy's chosen depth)")
    p.add_argument("--skip-input", action="store_true",
                   help="concatenate the input bits into every greedy layer's wiring"
                        " pool (skip connections; e2e baseline is unaffected)")
    p.add_argument("--skip-all", action="store_true",
                   help="DenseNet-style: wiring pool = input bits + ALL previous"
                        " layers' outputs (overrides --skip-input)")
    cfg = p.parse_args()
    cfg.n_class = 10
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    Xtr, Xte, ytr, yte = [t.to(cfg.device) for t in load_data()]
    print(f"data: {Xtr.shape[0]} train / {Xte.shape[0]} test, {Xtr.shape[1]} input bits"
          f"  (device={cfg.device})\n")

    layers, greedy_acc, depth = run_greedy(Xtr, Xte, ytr, yte, cfg)
    e2e_soft = e2e_hard = None
    if not cfg.skip_e2e:
        e2e_soft, e2e_hard = run_e2e(Xtr, Xte, ytr, yte, cfg.e2e_depth or depth, cfg)
    # simplification is pure-Python graph rewriting -> always run on CPU
    before, after = simplify([L.cpu() for L in layers], Xte.cpu(), yte.cpu(), cfg)

    summary = {"greedy_hard_test_acc": round(greedy_acc, 4),
               "greedy_depth": depth,
               "e2e_soft_test_acc": e2e_soft and round(e2e_soft, 4),
               "e2e_hard_test_acc": e2e_hard and round(e2e_hard, 4),
               "gates_before": before, "gates_after_simplify": after}
    print("=== summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
