# Development Environment Reference

This file captures the user's current WSL development/runtime environment as a reference point for repo work.

Date captured:
- 2026-03-18

## GPU

From `nvidia-smi`:

- GPU: `NVIDIA GeForce RTX 4080`
- Driver Version: `581.95`
- CUDA Version: `13.0`
- Persistence Mode: `On`

Observed at capture time:
- Temperature: `55C`
- Power: `27W / 167W`
- Memory Usage: `3753 MiB / 12282 MiB`
- GPU Utilization: `39%`

## Python Environment

Project commands should use:

```bash
./.venv/bin/python
```

Important installed packages reported in the environment:

### MRZ / OCR stack

- `paddleocr==3.4.0`
- `paddlepaddle-gpu==3.2.0`
- `paddlex==3.4.2`
- `pytesseract==0.3.13`
- `opencv-python==4.13.0.92`
- `opencv-contrib-python==4.10.0.84`
- `Pillow==12.1.1`
- `PyMuPDF==1.27.2`
- `pypdfium2==5.6.0`

### API stack

- `fastapi==0.135.1`
- `starlette==0.52.1`
- `uvicorn==0.41.0`
- `httpx==0.28.1`
- `python-multipart==0.0.22`
- `pydantic==2.12.5`

### ML / CUDA stack

- `torch==2.9.1+cu130`
- `torchvision==0.24.1+cu130`
- `torchaudio==2.9.1+cu130`
- `triton==3.5.1`
- CUDA 12 / 13 support libraries are both present in the environment

## Full `pip list` Snapshot

```text
Package                  Version      Editable project location
------------------------ ------------ -------------------------
aiohappyeyeballs         2.6.1
aiohttp                  3.13.3
aiosignal                1.4.0
aistudio-sdk             0.3.8
annotated-doc            0.0.4
annotated-types          0.7.0
anyio                    4.12.1
attrs                    25.4.0
bce-python-sdk           0.9.63
certifi                  2026.2.25
chardet                  7.1.0
charset-normalizer       3.4.5
click                    8.3.1
colorlog                 6.10.1
fastapi                  0.135.1
filelock                 3.25.2
frozenlist               1.8.0
fsspec                   2026.2.0
future                   1.0.0
h11                      0.16.0
hf-xet                   1.4.2
httpcore                 1.0.9
httpx                    0.28.1
huggingface_hub          1.7.1
idna                     3.11
imagesize                2.0.0
Jinja2                   3.1.6
joblib                   1.5.3
lightning                2.6.1
lightning-utilities      0.15.3
lmdb                     1.8.1
markdown-it-py           4.0.0
MarkupSafe               3.0.2
mdurl                    0.1.2
modelscope               1.34.0
mpmath                   1.3.0
multidict                6.7.1
networkx                 3.6.1
nltk                     3.9.3
numpy                    2.4.3
nvidia-cublas            13.0.0.19
nvidia-cublas-cu12       12.9.0.13
nvidia-cuda-cccl-cu12    12.9.27
nvidia-cuda-cupti        13.0.48
nvidia-cuda-cupti-cu12   12.9.19
nvidia-cuda-nvrtc        13.0.48
nvidia-cuda-nvrtc-cu12   12.9.41
nvidia-cuda-runtime      13.0.48
nvidia-cuda-runtime-cu12 12.9.37
nvidia-cudnn-cu12        9.9.0.52
nvidia-cudnn-cu13        9.13.0.50
nvidia-cufft             12.0.0.15
nvidia-cufft-cu12        11.4.0.6
nvidia-cufile            1.15.0.42
nvidia-cufile-cu12       1.14.0.30
nvidia-curand            10.4.0.35
nvidia-curand-cu12       10.3.10.19
nvidia-cusolver          12.0.3.29
nvidia-cusolver-cu12     11.7.4.40
nvidia-cusparse          12.6.2.49
nvidia-cusparse-cu12     12.5.9.5
nvidia-cusparselt-cu12   0.7.1
nvidia-cusparselt-cu13   0.8.0
nvidia-nccl-cu12         2.27.3
nvidia-nccl-cu13         2.27.7
nvidia-nvjitlink         13.0.39
nvidia-nvjitlink-cu12    12.9.41
nvidia-nvshmem-cu13      3.3.24
nvidia-nvtx              13.0.39
nvidia-nvtx-cu12         12.9.19
opencv-contrib-python    4.10.0.84
opencv-python            4.13.0.92
opt-einsum               3.3.0
packaging                26.0
paddleocr                3.4.0
paddlepaddle-gpu         3.2.0
paddlex                  3.4.2
pandas                   3.0.1
pillow                   12.1.1
pip                      26.0.1
prettytable              3.17.0
propcache                0.4.1
protobuf                 7.34.0
psutil                   7.2.2
py-cpuinfo               9.0.0
pyclipper                1.4.0
pycryptodome             3.23.0
pydantic                 2.12.5
pydantic_core            2.41.5
Pygments                 2.19.2
PyMuPDF                  1.27.2
pypdfium2                5.6.0
pytesseract              0.3.13
python-bidi              0.6.7
python-dateutil          2.9.0.post0
python-multipart         0.0.22
pytorch-lightning        2.6.1
PyYAML                   6.0.2
regex                    2026.2.28
requests                 2.32.5
rich                     14.3.3
ruamel.yaml              0.19.1
safetensors              0.7.0
sentencepiece            0.2.1
setuptools               82.0.1
shapely                  2.1.2
shellingham              1.5.4
six                      1.17.0
starlette                0.52.1
strhub                   1.2.0        /tmp/parseq
sympy                    1.14.0
timm                     1.0.25
tokenizers               0.22.2
torch                    2.9.1+cu130
torchaudio               2.9.1+cu130
torchmetrics             1.9.0
torchvision              0.24.1+cu130
tqdm                     4.67.3
transformers             5.3.0
triton                   3.5.1
typer                    0.24.1
typing_extensions        4.15.0
typing-inspection        0.4.2
ujson                    5.12.0
urllib3                  2.6.3
uvicorn                  0.41.0
wcwidth                  0.6.0
yarl                     1.23.0
```

## Usage Note

This file is a reference snapshot, not a lockfile. The user's WSL terminal remains the source of truth for:

- GPU visibility
- Paddle runtime behavior
- warm-process API performance
- full OCR benchmark verification
