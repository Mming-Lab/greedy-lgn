"""チェックポイント: 凍結済み層(ia/ib/logits)の保存・復元(--checkpoint)。
run_greedy(src/greedy.py)から呼ばれる。学習途中の状態(optimizer等)は保存しない
— 保存単位は「離散化済み・凍結済みの層」のみ(このアーキテクチャは層が確定した
時点で再学習不要になるので、これで十分な粒度になる)。
現状のスコープ: groupsum(dense)オブジェクトのみ。ConvGroupSumは別クラス
ConvLogicLayer(src/conv.py)を使うため対象外。FF/seq/ensemble/carryは
experiment.py側のp.errorで--checkpointと排他にしている。"""
import os
import torch
from core import LogicLayer

# フィンガープリント対象: 層の中身(RNGドロー・学習結果)を左右する設定のみ。
# max_layers/patience/device/checkpoint/stop_file/skip_e2eは再開のたびに
# 変えてよい設定なので意図的に除外(例: --max-layersを80→120に伸ばして
# 続きから再開する使い方を許すため)
FIELDS = ["dataset", "gates", "seed", "thresholds", "window", "commit", "carry",
          "skip_input", "skip_all", "group_residual", "group_boost", "group_loss",
          "warm_start", "local", "recur", "objective", "win_loss", "lr", "batch",
          "epoch_stop", "epoch_peak", "epoch_peak_decay", "epoch_chain",
          "epoch_min", "epoch_check", "epoch_patience", "epochs", "n_class"]

def fingerprint(cfg):
    return {f: getattr(cfg, f) for f in FIELDS}

def save(path, layers, cfg):
    """凍結済みlayersを丸ごと書き直す(層1つ確定するたびに呼ぶ想定、数MB程度)。
    アトミック書き込み: 一時ファイルに書いてからos.replaceで差し替えるので、
    保存中に電源が落ちても既存のpathは無傷のまま(古い状態がそのまま残る)"""
    payload = {"fingerprint": fingerprint(cfg),
               "layers": [{"ia": L.ia.cpu(), "ib": L.ib.cpu(),
                           "logits": L.logits.detach().cpu()} for L in layers]}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # Windows NTFSでもatomic rename

def load(path):
    payload = torch.load(path, weights_only=False)
    return (payload["fingerprint"],
            [(d["ia"], d["ib"], d["logits"]) for d in payload["layers"]])

def rebuild_layer(ia, ib, logits, device):
    """保存済み(ia, ib, logits)からLogicLayerを再構築。wires=で配線を上書きする
    既存の仕組み(--local用)を流用するので、core.pyの変更は不要。
    in_dim/seedはwires=で上書きされるダミー値(値そのものは意味を持たない)"""
    L = LogicLayer(1, len(ia), seed=0, wires=(ia, ib))
    L.logits = torch.nn.Parameter(logits.clone())
    return L.to(device)
