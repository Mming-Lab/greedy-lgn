"""論理簡略化+ビット等価検証: 定数畳み込み・パススルー/NOT除去・重複マージ・
デッドゲート削除。簡略化後の回路が元とビット単位で一致することをassertする。"""
import numpy as np
import torch
from core import SWAP, f16, group_sum, accuracy

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

