# Phase1 Design

## 目的

Phase1 は、アプリ単独起動からローカル生成環境の準備、ComfyUI への接続、テスト生成、画像表示までを確認できる状態にする。

## 実装範囲

```text
Electron App
  Main Process
    Runtime Manager
      Docker確認
      runtime directory作成
      Dockerfile/compose生成
      build/start/stop
      model download
      ComfyUI prompt submit

  Renderer
    Setup UI
    Model Download UI
    Test Generation UI
    Preview
```

## 外部委託設計

LLMや補助機能は「補助サービス」として扱う。

```text
Assistant Service
  provider: local-docker | remote-linux | cloud
  baseUrl: string
  capabilities:
    - prompt_refine
    - tag_suggest
    - caption
    - safety_check
```

Phase1ではUIに思想だけ表示し、接続実装はPhase2以降に回す。

## セキュリティ方針

- Docker socket はマウントしない
- privileged は使わない
- no-new-privileges を有効化
- cap_drop ALL
- モデルは read-only mount
- outputs のみ read-write mount
- LLM と ComfyUI は将来的に別コンテナ化する
