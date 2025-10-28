#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(realpath -m "$SCRIPT_DIR/..")"
MISAKI_DIR="$(realpath -m "$SCRIPT_DIR/../misaki")"
KOKORO_DIR="$SCRIPT_DIR"

cd "$WORKSPACE_DIR"

echo "[INFO] Kokoro workspace: $WORKSPACE_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HF_TOKEN environment variable is required." >&2
  exit 1
fi

VENV_PATH="${VENV_PATH:-$WORKSPACE_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[INFO] Creating virtual environment at $VENV_PATH"
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if command -v apt-get >/dev/null 2>&1; then
  echo "[INFO] Installing system libraries via apt-get"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends libsndfile1 espeak-ng
fi

# Ensure the Misaki fork is present and up to date
MISAKI_REPO="https://github.com/neiom-systems/misaki.git"
if [[ ! -d "$MISAKI_DIR/.git" ]]; then
  echo "[INFO] Cloning Misaki fork from $MISAKI_REPO"
  git clone "$MISAKI_REPO" "$MISAKI_DIR"
else
  echo "[INFO] Using existing Misaki repository at $MISAKI_DIR"
  git -C "$MISAKI_DIR" remote set-url origin "$MISAKI_REPO"
  git -C "$MISAKI_DIR" fetch --tags --prune origin
  git -C "$MISAKI_DIR" checkout main
  git -C "$MISAKI_DIR" reset --hard origin/main
fi

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.3.1}"

echo "[INFO] Installing PyTorch (index: $TORCH_INDEX_URL)"
pip install --upgrade "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url "$TORCH_INDEX_URL"

echo "[INFO] Installing training dependencies"
pip install --upgrade python-dotenv librosa soundfile pyworld textgrid tensorboard tqdm accelerate hf_transfer

echo "[INFO] Installing local packages"
pip install --upgrade -e "$MISAKI_DIR"
pip install --upgrade -e "$KOKORO_DIR"

echo "[INFO] Downloading Kokoro base model"
python kokoro/download_base_model.py

echo "[INFO] Downloading Luxembourgish dataset"
python kokoro/data/scripts/prepare_luxembourgish_male_only.py --work-dir kokoro/data/luxembourgish_male_corpus

echo "[INFO] Generating Luxembourgish voice table"
python kokoro/scripts/generate_voice_table.py --config kokoro/train_luxembourgish.toml --num-clips "${VOICE_CLIPS:-64}"

FEATURE_FORCE_FLAG=()
if [[ "${FEATURE_FORCE:-}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  FEATURE_FORCE_FLAG=(--force)
fi

echo "[INFO] Extracting acoustic features"
python kokoro/scripts/generate_features.py \
  --config kokoro/train_luxembourgish.toml \
  --splits train test \
  --log-level "${FEATURE_LOG_LEVEL:-INFO}" \
  "${FEATURE_FORCE_FLAG[@]}"

TRAIN_ARGS=(--config kokoro/train_luxembourgish.toml)
if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${TRAIN_EXTRA_ARGS})
  TRAIN_ARGS+=("${EXTRA_ARR[@]}")
fi

echo "[INFO] Starting fine-tuning"
python -m kokoro.training.train "${TRAIN_ARGS[@]}"

echo "[INFO] Fine-tuning completed"
