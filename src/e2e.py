"""逆伝播ベースライン: 同構成のLGNをend-to-endで学習(比較対象)。"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import LogicLayer, group_sum, accuracy, hard_batched

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
    g = torch.Generator().manual_seed(cfg.seed * 777)
    for _ in range(epochs):
        if cfg.batch and cfg.batch < len(Xtr):
            for idx in torch.randperm(len(Xtr), generator=g).split(cfg.batch):
                idx = idx.to(Xtr.device)
                opt.zero_grad()
                h = Xtr[idx]
                for L in net:
                    h = L(h)
                F.cross_entropy(group_sum(h, cfg.n_class, tau), ytr[idx]).backward()
                opt.step()
        else:
            opt.zero_grad()
            h = Xtr
            for L in net:
                h = L(h)
            F.cross_entropy(group_sum(h, cfg.n_class, tau), ytr).backward()
            opt.step()
    with torch.no_grad():
        chunks = []
        for c in Xte.split(4096):   # chunked soft eval bounds the [B, G, 16] temporary
            for L in net:
                c = L(c)
            chunks.append(c)
        hs = torch.cat(chunks)
        hh = Xte
        for L in net:
            hh = hard_batched(L, hh)
    soft = accuracy(group_sum(hs, cfg.n_class, tau), yte)
    hard = accuracy(group_sum(hh, cfg.n_class, tau), yte)
    print(f"  soft={soft:.4f}  discretized={hard:.4f}  gap={soft - hard:+.4f}"
          f"  ({epochs} epochs, {time.time() - t0:.0f}s)")
    print(f"  float logits during training: greedy={cfg.gates * 16:,} (1 layer)"
          f"  vs  e2e={cfg.gates * 16 * depth:,} (x{depth})\n")
    return soft, hard

