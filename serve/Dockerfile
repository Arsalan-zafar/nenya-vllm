# Nenya on vLLM's OpenAI-compatible server. Build from the repository root:
#   docker build -f serve/Dockerfile -t nenya-vllm:0.1.0 .
#   docker run --gpus all --ipc=host -p 8000:8000 -v /models:/models nenya-vllm:0.1.0 /models/nenya-base
# Arguments after the image name go to serve/serve.sh (model first, then any extra `vllm serve` flags); the
# environment variables documented in serve/serve.sh apply (docker run -e MAX_MODEL_LEN=8192 ...).
# Base image: vllm/vllm-openai:v0.33.0 = vLLM commit 4ba787ec, CUDA 13.0.2,
# Python 3.12, user root, entrypoint `vllm serve` (replaced below). This Dockerfile has not been built yet.
FROM vllm/vllm-openai:v0.33.0

COPY pyproject.toml README.md /opt/nenya-vllm/
COPY nenya_vllm /opt/nenya-vllm/nenya_vllm
COPY serve /opt/nenya-vllm/serve
# The image (root, system Python 3.12, CUDA 13.0) installs its packages with `uv pip install --system` and pins
# them through UV_OVERRIDE; do the same. vllm 0.33.0 and torch are already there: the plugin goes in without
# re-resolving them, fla 0.5.2 adds only fla-core.
RUN uv pip install --system --no-cache "flash-linear-attention==0.5.2" \
 && uv pip install --system --no-cache --no-deps /opt/nenya-vllm \
 && python3 -c "import fla, nenya_vllm; print('fla', fla.__version__, 'nenya_vllm', nenya_vllm.__version__)"

ENTRYPOINT ["/opt/nenya-vllm/serve/serve.sh"]
