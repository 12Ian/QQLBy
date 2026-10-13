#!/usr/bin/env bash
set -euo pipefail
# Server bootstrap for the military intercept-engagement reproduction.
# AutoDL-tested: works on a stock AutoDL PyTorch/CUDA image or a bare conda.
#
#   bash bash/setup_server.sh [VENV_DIR] [cpu|cu121|cu124|cu126|cu128|auto]
#
# "auto" (the default) inspects the preinstalled torch if any and picks the
# matching flavour; on a bare image it probes the GPU compute capability and
# picks cu128 for Blackwell (RTX 50xx, sm_120 -- a cu124/cu126 wheel installs
# fine but dies with "no kernel image" on the first kernel launch) or cu124
# otherwise. Creates the virtual environment, installs the pinned dependencies
# and runs a short end-to-end self-check so a broken environment is caught
# before a multi-hour training run starts.
if [[ $# -gt 2 ]]; then
  echo "Usage: bash bash/setup_server.sh [VENV_DIR] [cpu|cu121|cu124|cu126|cu128|auto]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xuance_root="$(cd -- "$script_dir/.." && pwd)"
module="$xuance_root/Trajectory Prediction"
venv="${1:-$xuance_root/.venv}"
flavour="${2:-auto}"
python_bin="${PYTHON:-python3}"

command -v "$python_bin" >/dev/null 2>&1 || { echo "python3 not found; set PYTHON=..." >&2; exit 1; }
echo "[1/6] python: $($python_bin --version 2>&1)"

if [[ "$flavour" == "auto" ]]; then
  # AutoDL images ship torch in the base conda env; reuse its CUDA flavour so
  # the venv wheel matches the driver, instead of guessing.
  detected="$("$python_bin" - <<'PY' 2>/dev/null || echo cpu
try:
    import torch
    print("cu121" if "cu121" in torch.__version__ else
          "cu124" if "cu124" in torch.__version__ else
          "cu126" if "cu126" in torch.__version__ else
          "cu128" if "cu128" in torch.__version__ else "cpu")
except Exception:
    print("cpu")
PY
)"
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$detected" == "cpu" ]]; then
    # No preinstalled torch but a GPU host: pick by compute capability.
    # Blackwell consumer cards (RTX 50xx, sm_120) need the cu128 wheel
    # (torch >= 2.7.0); older cards work on cu124. cu128 covers sm_50..sm_120
    # so it is safe universally, but cu124 is the smaller download on
    # pre-Blackwell hosts.
    capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    major="${capability%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]] && [[ "$major" -ge 10 ]]; then
      detected="cu128"
    else
      detected="cu124"
    fi
    echo "[1/6] GPU compute capability: ${capability:-unknown} -> torch $detected"
  fi
  flavour="$detected"
  echo "[1/6] torch flavour auto-detected: $flavour"
fi

if [[ ! -x "$venv/bin/python" ]]; then
  echo "[2/6] creating venv at $venv"
  "$python_bin" -m venv "$venv"
else
  echo "[2/6] reusing venv at $venv"
fi
py="$venv/bin/python"

echo "[3/6] upgrading pip"
"$py" -m pip install --upgrade pip setuptools wheel

echo "[4/6] installing torch ($flavour) and project requirements"
case "$flavour" in
  cpu)   "$py" -m pip install "torch==2.7.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu ;;
  cu121) "$py" -m pip install "torch==2.7.1+cu121" --extra-index-url https://download.pytorch.org/whl/cu121 ;;
  cu124) "$py" -m pip install "torch==2.7.1+cu124" --extra-index-url https://download.pytorch.org/whl/cu124 ;;
  cu126) "$py" -m pip install "torch==2.7.1+cu126" --extra-index-url https://download.pytorch.org/whl/cu126 ;;
  cu128) "$py" -m pip install "torch==2.7.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128 ;;
  *) echo "Unknown torch flavour: $flavour (cpu|cu121|cu124|cu126|cu128|auto)" >&2; exit 2 ;;
esac
# numpy is pinned <2 by requirements.txt, which is what the CPU validation used.
"$py" -m pip install -r "$module/requirements.txt"
"$py" -m pip install pytest

echo "[5/6] self-check"
"$py" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
export PYTHONPATH="$module/src:$xuance_root"
"$py" "$module/run.py" check --device cpu --config "$module/configs/cpu_engagement.json" --backward

echo "[6/6] disk-space sanity (AutoDL: keep big artefacts on the data disk)"
df -h "$xuance_root" | tail -1
if [[ -d /root/autodl-tmp ]]; then
  echo "AutoDL data disk detected: /root/autodl-tmp"
  echo "  put OUTPUT_DIR there, e.g. bash bash/run_engagement.sh /root/autodl-tmp/military cuda:0 forecast"
fi
echo
echo "Environment ready."
echo "  source $venv/bin/activate"
echo "  bash bash/run_engagement.sh outputs/military cuda:0 forecast"
