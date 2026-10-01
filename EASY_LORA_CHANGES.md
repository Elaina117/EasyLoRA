# Easy LoRA 改造内容

## 目的

TrainTrainの既存学習エンジンを維持しながら、初心者が

`画像投入 → 学習対象を選択 → 自動タグ生成・選別 → Dataset完成`

まで少ない操作で済むようにする。

## 追加・変更

### Easy LoRA UI

`Easy LoRA` を最初のタブとして追加。

- 画像複数選択
- ZIP入力
- Colab / Google Drive のフォルダパス入力
- データセット名
- トリガーワード
- プリセット選択
  - キャラクター
  - 衣装
  - 画風
  - オブジェクト
- WD14しきい値
- 手動で必ず残す / 消すタグ
- 自動準備
- 自動チェック
- そのまま学習開始

従来の複雑なUIは `Advanced` 内にまとめて残している。

### ローカルWD14

`SmilingWolf/wd-swinv2-tagger-v3` のONNX版を使用。

- 初回利用時にHugging Faceからダウンロード
- CUDA Execution Providerがあれば優先
- CPU fallbackあり
- 448x448 / mean=0.5 / std=0.5
- バッチ推論対応

### タグ分類

WD14のカテゴリ番号を利用しつつ、generalタグを名前ベースで追加分類する。

- character
- copyright
- artist
- clothing
- hair
- appearance
- pose
- expression
- background
- style
- object
- quality
- other

### 自動選別

未知のタグは削除しない。

各プリセットは「削除カテゴリ」を定義しており、それ以外をcaptionに残す。

つまり、LLMを使わずに決定的な処理を行う。

### 元データ保護

出力は

```text
easy_datasets/
  <dataset>/
    original/
    prepared/
    captions_original/
    meta/
```

に分ける。

`prepared/` が学習用Dataset。

`captions_original/` にWD14の生タグとスコアを保存する。

`meta/review.csv` と `meta/summary.json` に自動処理の結果を保存する。

## 学習設定

既存TrainTrainの設定をベースに、

- LoRA / lierla
- rank 16
- alpha 8
- batch size 1
- learning rate 1e-4
- AdamW
- cosine
- fp16
- gradient checkpointing ON
- tag shuffle ON

などの初心者向け初期値だけを上書きする。

学習stepは画像枚数とプリセットから自動計算する。

これは「最初に試すための初期値」であり、すべてのDatasetに対して最適とは限らない。

## 依存関係

`install.py` に以下を追加。

- `huggingface_hub>=0.24`
- `onnxruntime-gpu>=1.18`

GPU版ONNX Runtimeが使えない環境では、`onnxruntime` に変更してCPU推論へ切り替えられる。

## 使い方

1. このリポジトリを自分のGitHubへpush
2. A1111 / Forge / reForgeからTrainTrainを通常通り導入
3. 初回起動時にEasy LoRA用依存関係をインストール
4. `Easy LoRA` タブを開く
5. 画像を入れる
6. LoRA種類を選ぶ
7. `自動で学習準備`
8. 内容を確認
9. `学習開始`

### 完全自動寄りにする

`準備後、そのまま学習を開始` をONにすると、準備後にそのままTrainTrainのLoRA学習を開始する。

初回はOFFを推奨。
