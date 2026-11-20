#!/bin/bash
# Provision a fresh GPU box for the plugin, its kernel tests and the benchmarks (tested: vast.ai RTX 5090, image
# vastai/pytorch:cuda-13.0.3-auto). Run it from a copy of this repository on the box:
#
#   bash benchmarks/setup_box.sh > setup.log 2>&1                # Nenya only
#   WITH_QWEN=1 bash benchmarks/setup_box.sh > setup.log 2>&1    # + Qwen3-0.6B and Qwen3.5-2B, for compare_models.sh
#
# Environment (all optional): VENV (default /root/vllm_env), MODELS_DIR (default /root/models), TORCH_BACKEND
# (default cu130: CUDA 13 wheels, which Blackwell / sm120 needs; "auto" picks for your driver), HF_TOKEN (higher Hub
# rate limits; every repository used here is public).
set -euxo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/root/vllm_env}
MODELS_DIR=${MODELS_DIR:-/root/models}
export UV_LINK_MODE=copy
command -v uv || pip install uv
uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"
uv pip install "vllm==0.33.0" --torch-backend="${TORCH_BACKEND:-cu130}"
# the plugin (pulls flash-linear-attention 0.5.2), the parallel downloader and pytest
uv pip install -e "$REPO"
uv pip install hf_transfer requests pytest
python -c "import vllm, torch, fla; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda, 'fla', fla.__version__)"
# FlashInfer JIT-compiles its fused-MoE kernels on the first engine start; unbounded, it runs one compiler job per
# core (several GB each) and can exhaust host RAM on many-core boxes. Build it once here with a bounded job count;
# later starts use ~/.cache/flashinfer.
export MAX_JOBS=${MAX_JOBS:-6}
mkdir -p "$MODELS_DIR"
dl() { python "$REPO/benchmarks/hf_fast_download.py" "$@"; }
# Nenya at its release revision (tag v1.0.0)
dl ArsalanZ/Nenya-2.5B-A0.5B-Base 8b7bb96599d01d99027227e08f4b646509881611 "$MODELS_DIR/nenya-base"
# Once per checkpoint: names ids 100270/100271 <raw_code>/</raw_code>, sets the stop ids and writes the chat template.
# Only the local copy's tokenizer files, generation_config.json and chat_template.jinja change; serve/serve.sh's
# preflight refuses a checkpoint that was not patched. Running it again gives the same files.
python -m nenya_vllm.olmo3_chat export-hf "$MODELS_DIR/nenya-base"
if [ "${WITH_QWEN:-0}" = 1 ]; then
    # the revisions the published comparison ran
    dl Qwen/Qwen3-0.6B-Base da87bfb608c14b7cf20ba1ce41287e8de496c0cd "$MODELS_DIR/qwen3-0.6b-base"
    dl Qwen/Qwen3.5-2B-Base b1485b2fa6dfa1287294f269f5fb618e03d52d7c "$MODELS_DIR/qwen3.5-2b-base"
fi
python "$REPO/benchmarks/smoke_generate.py" --model "$MODELS_DIR/nenya-base"   # builds the JIT modules
echo SETUP_DONE
