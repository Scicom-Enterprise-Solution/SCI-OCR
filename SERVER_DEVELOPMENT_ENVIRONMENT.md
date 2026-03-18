# Server Development Environment Reference

This file captures the server runtime environment for this repository as built at:

- Date captured: 2026-03-18
- Repo path: `/home/inam/reaper/SCI-OCR`

This is separate from `DEVELOPMENT_ENVIRONMENT.md`, which remains the laptop snapshot.

## GPU

From `nvidia-smi`:

- GPU: `Tesla V100-PCIE-32GB`
- Driver Version: `570.86.15`
- CUDA Version: `12.8`
- Persistence Mode: `Off`

Observed at capture time:

- Temperature: `31C`
- Power: `34W / 250W`
- Memory Usage: `1112 MiB / 32768 MiB`
- GPU Utilization: `0%`

## Python Environment

Project commands should use:

```bash
./.venv/bin/python
```

Environment runtime:

- Python: `3.12.12`
- pip: `26.0.1`

## Important Installed Packages

### MRZ / OCR stack

- `paddleocr==3.4.0`
- `paddlepaddle-gpu==3.2.0`
- `paddlex==3.4.2`
- `pytesseract==0.3.13`
- `opencv-python==4.13.0.92`
- `opencv-contrib-python==4.10.0.84`
- `opencv-python-headless==4.13.0.92`
- runtime `cv2.__version__ == 4.13.0`
- `pillow==12.1.1`
- `PyMuPDF==1.27.2`
- `pypdfium2==5.6.0`

### API stack

- `fastapi==0.135.1`
- `starlette==0.52.1`
- `uvicorn==0.42.0`
- `httpx==0.28.1`
- `python-multipart==0.0.22`
- `pydantic==2.12.5`

### CUDA / Paddle support libraries

- `nvidia-cublas-cu12==12.6.4.1`
- `nvidia-cuda-cccl-cu12==12.6.77`
- `nvidia-cuda-cupti-cu12==12.6.80`
- `nvidia-cuda-nvrtc-cu12==12.6.77`
- `nvidia-cuda-runtime-cu12==12.6.77`
- `nvidia-cudnn-cu12==9.5.1.17`
- `nvidia-cufft-cu12==11.3.0.4`
- `nvidia-cufile-cu12==1.11.1.6`
- `nvidia-curand-cu12==10.3.7.77`
- `nvidia-cusolver-cu12==11.7.1.2`
- `nvidia-cusparse-cu12==12.5.4.2`
- `nvidia-cusparselt-cu12==0.6.3`
- `nvidia-nccl-cu12==2.25.1`
- `nvidia-nvjitlink-cu12==12.6.85`
- `nvidia-nvtx-cu12==12.6.77`

## Full `pip freeze` Snapshot

```text
aistudio-sdk==0.3.8
albucore==0.0.24
albumentations==2.0.8
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.12.1
astor==0.8.1
bce-python-sdk==0.9.63
beautifulsoup4==4.14.3
certifi==2026.2.25
chardet==7.1.0
charset-normalizer==3.4.6
click==8.3.1
colorlog==6.10.1
cuda-bindings==12.9.5
cuda-pathfinder==1.3.3
cuda-python==12.9.4
Cython==3.2.4
decorator==5.2.1
fastapi==0.135.1
filelock==3.25.2
fire==0.7.1
fonttools==4.62.1
fsspec==2026.2.0
future==1.0.0
h11==0.16.0
hf-xet==1.4.2
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.7.1
idna==3.11
ImageIO==2.37.3
imagesize==2.0.0
lazy-loader==0.5
lmdb==1.8.1
lxml==6.0.2
markdown-it-py==4.0.0
mdurl==0.1.2
modelscope==1.35.0
networkx==3.6.1
numpy==2.4.3
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cccl-cu12==12.6.77
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.25.1
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
opencv-contrib-python==4.10.0.84
opencv-python==4.13.0.92
opencv-python-headless==4.13.0.92
opt-einsum==3.3.0
packaging==26.0
paddleocr==3.4.0
paddlepaddle-gpu==3.2.0
paddlex==3.4.2
pandas==3.0.1
pillow==12.1.1
prettytable==3.17.0
protobuf==7.34.0
psutil==7.2.2
pyclipper==1.4.0
py-cpuinfo==9.0.0
pycryptodome==3.23.0
pydantic==2.12.5
pydantic_core==2.41.5
Pygments==2.19.2
PyMuPDF==1.27.2
pypdfium2==5.6.0
pytesseract==0.3.13
python-bidi==0.6.7
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-multipart==0.0.22
PyYAML==6.0.2
RapidFuzz==3.14.3
requests==2.32.5
rich==14.3.3
ruamel.yaml==0.19.1
safetensors==0.7.0
scikit-image==0.26.0
scipy==1.17.1
setuptools==82.0.1
shapely==2.1.2
shellingham==1.5.4
simsimd==6.5.16
six==1.17.0
soupsieve==2.8.3
starlette==0.52.1
stringzilla==4.6.0
termcolor==3.3.0
tifffile==2026.3.3
tqdm==4.67.3
typer==0.24.1
typing_extensions==4.15.0
typing-inspection==0.4.2
ujson==5.12.0
urllib3==2.6.3
uvicorn==0.42.0
wcwidth==0.6.0
wheel==0.46.3
```

## Notes

- This document describes the server environment created in this repo at `/home/inam/reaper/SCI-OCR/.venv`.
- The package set was matched against the active server environment in another checkout and verified with `pip freeze`.
- `paddlepaddle-gpu==3.2.0` required the official Paddle CUDA wheel index rather than the default package source.
- `cv2` was verified to resolve to `4.13.0` in this environment.