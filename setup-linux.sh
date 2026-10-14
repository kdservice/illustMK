#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ! -r /etc/os-release ]]; then
  echo "エラー: /etc/os-releaseを読み込めません。Ubuntu 22.04.5 LTS上で実行してください。" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "エラー: Ubuntu 22.04.x LTS専用です。検出: ${PRETTY_NAME:-不明}" >&2
  exit 1
fi

if [[ ${EUID} -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "エラー: システムパッケージの導入にsudoが必要です。" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ILLUSTMK_RUNTIME_ROOT:-${HOME}/illustMK-data}"

echo "[1/5] Ubuntuパッケージを更新します"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg git build-essential \
  python3 python3-dev python3-pip python3-venv \
  ffmpeg lsof xdg-utils \
  libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 \
  libdrm2 libgbm1 libglib2.0-0 libgtk-3-0 libnss3 libnotify4 \
  libx11-xcb1 libxcomposite1 libxdamage1 libxfixes3 libxkbcommon0 \
  libxrandr2 libxshmfence1 libxss1 libgl1

echo "[2/5] Node.js 22を確認します"
NODE_MAJOR=0
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
fi
if (( NODE_MAJOR < 22 )); then
  "${SUDO[@]}" install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | "${SUDO[@]}" gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/nodesource.list >/dev/null
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y nodejs
fi

echo "[3/5] DataPathを準備します: ${DATA_ROOT}"
mkdir -p "${DATA_ROOT}" "${HOME}/.local/share/illustmk"

echo "[4/5] Electronアプリの依存関係を導入します"
cd "${SCRIPT_DIR}"
npm ci

echo "[5/5] TypeScriptとプロダクションビルドを確認します"
npm run typecheck
npm run build

cat <<EOF

セットアップが完了しました。
DataPath: ${DATA_ROOT}

開発起動:
  npm run dev:linux

ビルド済みアプリ起動:
  npm run start:linux

起動後は「環境 > ComfyUI > Linux」を選び、初期化、インストール、開始の順で実行してください。
NVIDIA GPUを使う場合は、事前にUbuntu側で対応ドライバを導入し、nvidia-smiが成功する状態にしてください。
EOF
