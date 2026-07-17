"""チェックポイント: 凍結済み層の保存・復元(--checkpoint)。
run_greedy(src/greedy.py)から呼ばれる。学習途中の状態(optimizer等)は保存しない
— 保存単位は「離散化済み・凍結済みの層」のみ(このアーキテクチャは層が確定した
時点で再学習不要になるので、これで十分な粒度になる)。
対応: groupsum(dense)=LogicLayer(ia/ib/logits) と ConvGroupSum=ConvLogicLayer
(leaf/logits/幾何)。FF/seq/ensemble/carryはexperiment.py側のp.errorで
--checkpointと排他にしている。"""
import os
import torch
from core import LogicLayer
from conv import ConvLogicLayer

# フィンガープリント対象: 層の中身(RNGドロー・学習結果)を左右する設定のみ。
# max_layers/patience/device/checkpoint/stop_file/skip_e2eは再開のたびに
# 変えてよい設定なので意図的に除外(例: --max-layersを80→120に伸ばして
# 続きから再開する使い方を許すため)
FIELDS = ["dataset", "gates", "seed", "thresholds", "window", "commit", "carry",
          "skip_input", "skip_all", "group_residual", "group_boost", "group_loss",
          "warm_start", "local", "recur", "objective", "win_loss", "lr", "batch",
          "epoch_stop", "epoch_peak", "epoch_peak_decay", "epoch_chain",
          "epoch_min", "epoch_check", "epoch_patience", "epochs", "n_class",
          "conv", "conv_k", "conv_tree", "conv_pool", "conv_sched"]

# FIELDSへ後から追加したキーの既定値(後方互換)。conv対応(2026-07-17)以前に
# 保存された.pt(task29/vol2のRelease資産を含む)はこれらのキーを持たないので、
# 読み込み時に既定値で埋めてfingerprint照合を通す。既定値=「そのレバーはオフ」
# なので、旧チェックポイントは全てdense=conv無しであり意味も正しい
COMPAT_DEFAULTS = {"conv": 0, "conv_k": 3, "conv_tree": 2, "conv_pool": 2,
                   "conv_sched": None}

def fingerprint(cfg):
    return {f: getattr(cfg, f) for f in FIELDS}

def _pack(L):
    """層1枚を保存用dictへ(型で分岐)"""
    if isinstance(L, ConvLogicLayer):
        return {"leaf": L.leaf.cpu(), "logits": L.logits.detach().cpu(),
                "geom": [L.Cin, L.H, L.W, L.C, L.k,
                         int(L.L).bit_length() - 1, L.pool]}   # tree = log2(葉数)
    return {"ia": L.ia.cpu(), "ib": L.ib.cpu(),
            "logits": L.logits.detach().cpu()}

def save(path, layers, cfg):
    """凍結済みlayersを丸ごと書き直す(層1つ確定するたびに呼ぶ想定、数MB程度)。
    アトミック書き込み: 一時ファイルに書いてからos.replaceで差し替えるので、
    保存中に電源が落ちても既存のpathは無傷のまま(古い状態がそのまま残る)"""
    payload = {"fingerprint": fingerprint(cfg),
               "layers": [_pack(L) for L in layers]}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # Windows NTFSでもatomic rename

def load(path):
    # weights_only=True: pickleの任意コード実行を許さない安全な読み込み。
    # payloadはテンソルとプリミティブ(fingerprintはargparse由来のstr/int/float/
    # bool/None/リスト)だけなので制限付きunpicklerで完全に復元できる。配布された
    # チェックポイント(Release資産)を読む運用があるので既定を安全側に
    payload = torch.load(path, weights_only=True)
    fp = payload["fingerprint"]
    for k, v in COMPAT_DEFAULTS.items():
        fp.setdefault(k, v)
    return fp, payload["layers"]

def rebuild(d, device):
    """保存dictから層を再構築(型はキーで判別)。RNGドローはwires=/上書きで
    完全に置き換えるので、コンストラクタに渡すseedはダミー"""
    if "leaf" in d:
        Cin, H, W, C, k, tree, pool = d["geom"]
        L = ConvLogicLayer(Cin, H, W, C, k, tree, pool, seed=0)
        L.leaf = d["leaf"].clone()
        L.logits = torch.nn.Parameter(d["logits"].clone())
        return L.to(device)
    L = LogicLayer(1, len(d["ia"]), seed=0, wires=(d["ia"], d["ib"]))
    L.logits = torch.nn.Parameter(d["logits"].clone())
    return L.to(device)

def rebuild_layer(ia, ib, logits, device):
    """旧API(dense専用)。tools等の既存呼び出しのために残す"""
    return rebuild({"ia": ia, "ib": ib, "logits": logits}, device)
