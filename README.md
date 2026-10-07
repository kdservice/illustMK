# illustMK

AIイラスト制作アプリの Phase1 実装です。

## ディレクトリ

- `/Users/kds/Projects/illustMK`: Electron + React アプリ本体、ドキュメント
- `/Volumes/ARC-8050T2/illustMK`: Docker 構築先、モデル保存先、生成画像出力先

## Phase1

- macOS ネイティブ ComfyUI ランタイムの初期化
- ComfyUI clone / venv 作成 / requirements install
- ComfyUI の開始 / 停止 / ログ表示
- Checkpoint モデルの URL ダウンロード
- ComfyUI API 接続確認
- 固定 workflow によるテスト生成
- 出力画像のアプリ内表示

## 起動

```bash
TMPDIR=/tmp electron_config_cache=/Users/kds/Projects/illustMK/.electron-cache npm install
npm run dev
```

## ランタイム構成

Phase1 では ComfyUI を macOS 上で直接起動し、MPS/Metal 利用を狙います。

```text
/Volumes/ARC-8050T2/illustMK
  native/
    ComfyUI/
    venv/
    logs/
  models/
    checkpoints/
    loras/
    vae/
  outputs/
  workflows/
  logs/
  downloads/
```

Docker 版の試作資産は後始末済みです。Docker Desktop 本体は削除していません。

## Phase2 方針

Atelier Mira 風の制作UIは Phase2 として追加します。

- プロンプト編集
- タグビルダー
- LoRA / モデル管理
- LLM補助
- 生成履歴 / ライブラリ
- ローカル Docker LLM / 別PC Linux LLM / クラウド LLM の統合
