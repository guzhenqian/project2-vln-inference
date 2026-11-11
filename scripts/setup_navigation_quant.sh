#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
quant_env="${HOME}/.venvs/project2-quant312"
if [[ ! -x "${quant_env}/bin/python" ]]; then
  "${HOME}/.local/bin/uv" venv --python "${HOME}/.venvs/project2/bin/python" "${quant_env}"
fi
"${HOME}/.local/bin/uv" pip install --python "${quant_env}/bin/python" -r setup/navigation-quant-requirements.txt
"${HOME}/.local/bin/uv" pip freeze --python "${quant_env}/bin/python" > setup/navigation-quant-lock.txt
"${quant_env}/bin/python" -c 'import torch, transformers, llmcompressor; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
