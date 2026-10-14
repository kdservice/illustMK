# illustMK

AIイラスト制作アプリの Phase1 実装です。

## ディレクトリ

- リポジトリ: Electron + React アプリ本体、ドキュメント
- DataPath: ComfyUI構築先、モデル保存先、生成画像出力先（アプリ内で切替可能）

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

## Ubuntu 22.04.5 LTS

Ubuntu用セットアップは、Node.js 22、Python 3、Electron実行ライブラリ、ビルド依存関係を導入し、型検査とビルドまで実行します。

```bash
chmod +x setup-linux.sh
./setup-linux.sh
npm run dev:linux
```

起動後は `環境 > ComfyUI > Linux` を選び、`初期化`、`インストール`、`開始` の順で実行します。既定DataPathは `~/illustMK-data`、アプリ設定は `~/.local/share/illustmk` です。NVIDIA GPUを使用する場合、Ubuntu側のドライバと `nvidia-smi` は事前に利用可能な状態にしてください。

## ランタイム構成

ComfyUIはmacOS、Linux、Dockerの実行方式を環境タブで切り替えられます。macOSではMPS/Metal、LinuxではCUDAを利用し、利用可能なGPUがない場合はCPUで起動します。

```text
DataPath
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
