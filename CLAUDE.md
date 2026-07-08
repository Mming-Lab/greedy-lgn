# greedy-lgn — プロジェクトコンテキスト

## これは何か
論理ゲートネットワーク(DLGN)を逆伝播なしで1層ずつ学習する研究プロトタイプ。
各層をローカル損失(GroupSum+CE)で学習→即離散化→凍結→次層はハードな0/1ビット上で学習。
検証精度が頭打ちで層追加を停止(深さ自動決定)。学習後に論理簡略化(定数畳み込み・
パススルー除去・重複マージ・デッドゲート削除)を行い、ビット単位の等価性を検証する。

di fflogic(Petersen et al.)系の先行研究に対する位置づけ・関連論文はREADME.mdを参照。
この組み合わせ(LGN × backprop-free × 適応深さ × 逐次簡略化)は2026年中頃時点で
公開された前例が見当たらない。早期公開でタイムスタンプを残すのが方針。

## 検証済みの結果(このリポジトリのexperiment.pyで再現可能)
- sklearn digits, 500ゲート/層, CPU数分:
  - greedy: 深さ4自動選択, ハード回路テスト精度 88.2%, 離散化ギャップ0(構造的)
  - end-to-end逆伝播(同構成): 93.6%(greedyは精度で約5pt負ける — 隠さないこと)
  - 簡略化: 2,000→1,316ゲート(65.8%), 出力完全一致を検証済み, 重複マージは0件
- 小規模設定(--gates 200 --epochs 30)ではe2e側に離散化ギャップ+8.2ptが出現、greedyはゼロ

## ファイル
- experiment.py : 全実験を1ファイルに統合(greedy / e2eベースライン / 簡略化+検証)
- README.md    : 英語本体+日本語概要。結果表・限界・ロードマップ記載済み
- requirements.txt, LICENSE(MIT, 著作権者名は要確認)

## 直近のタスク(優先順)
1. GitHub公開: `gh repo create greedy-lgn --public --source . --push`
   - Description: "Backprop-free, layer-by-layer training of Differentiable Logic
     Gate Networks with zero discretization gap, adaptive depth, and incremental
     logic simplification"
   - Topics: differentiable-logic-gate-networks, forward-forward,
     backpropagation-free, fpga, binary-neural-networks
   - 公開前にLICENSEの著作権者名をユーザーに確認
2. 実行ログをissueまたはREADME末尾に貼る(再現性の証拠)
3. 深さストレステスト: 30〜40層まで層を積んでgreedyが学習できるか
   (逆伝播が勾配消失で壊れる領域。これが主結果候補。RTX 3060 / difflogic本家の
   CUDAカーネル利用を検討)
4. メモリ等価比較: greedy側の層幅を4倍にしてe2eと精度比較
5. MNIST移植(data読み込みとGATES定数の変更で可能)

## 方針・制約
- 結果は正直に報告する(負けている数字を隠さない)。ユーザーはデータ駆動で
  誠実な分析を好む。楽観的なフレーミングは不要。
- 実験の変更時はexperiment.py内の等価性検証(assert identical)を必ず維持する。
- ユーザー環境: Windows, RTX 3060 6GB(HP OMEN), 音声入力のため固有名詞の
  誤変換があり得る。
