import contextlib
import inspect
import io
import os
import re

import cv2
import numpy as np


_PADDLE_OCR_INSTANCE = None


def resolve_paddle_cache_home() -> str:
    env_cache = os.getenv("PADDLE_PDX_CACHE_HOME", "").strip()
    if env_cache:
        return env_cache

    return os.path.abspath(os.path.join(os.getcwd(), ".paddlex"))


def ensure_stub_ccache_on_path() -> None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        if path and os.path.exists(os.path.join(path, "ccache")):
            return

    stub_dir = os.path.join(resolve_paddle_cache_home(), "bin")
    os.makedirs(stub_dir, exist_ok=True)
    stub_path = os.path.join(stub_dir, "ccache")

    if not os.path.exists(stub_path):
        with open(stub_path, "w", encoding="ascii") as f:
            f.write("#!/bin/sh\n")
            f.write('exec "$@"\n')
        os.chmod(stub_path, 0o755)

    os.environ["PATH"] = stub_dir + os.pathsep + os.environ.get("PATH", "")


def should_disable_paddle_model_source_check(paddle_cache_home: str) -> bool:
    env_value = os.getenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "").strip()
    if env_value:
        return env_value.lower() in {"1", "true", "yes", "on"}

    official_models_dir = os.path.join(paddle_cache_home, "official_models")
    return os.path.isdir(official_models_dir)


def resolve_paddle_use_gpu(use_gpu_requested: bool) -> bool:
    if not use_gpu_requested:
        return False

    ensure_stub_ccache_on_path()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            import paddle
        except ImportError:
            print("[WARN] Paddle GPU requested but 'paddle' is not installed")
            return False

    if not paddle.device.is_compiled_with_cuda():
        print("[WARN] Paddle GPU requested but the installed Paddle build has no CUDA support")
        return False

    try:
        device_count = paddle.device.cuda.device_count()
    except Exception as exc:
        print(f"[WARN] Paddle GPU requested but CUDA device detection failed: {exc}")
        return False

    if device_count < 1:
        print("[WARN] Paddle GPU requested but no CUDA devices are visible; falling back to CPU")
        return False

    return True


def get_paddle_ocr(lang: str, use_gpu_requested: bool):
    global _PADDLE_OCR_INSTANCE

    if _PADDLE_OCR_INSTANCE is not None:
        return _PADDLE_OCR_INSTANCE

    paddle_cache_home = resolve_paddle_cache_home()
    os.environ["PADDLE_PDX_CACHE_HOME"] = paddle_cache_home
    os.makedirs(paddle_cache_home, exist_ok=True)
    ensure_stub_ccache_on_path()
    if should_disable_paddle_model_source_check(paddle_cache_home):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR backend requested but 'paddleocr' is not installed in the active environment"
            ) from exc

    use_gpu = resolve_paddle_use_gpu(use_gpu_requested)
    kwargs = {"lang": lang}
    signature = inspect.signature(PaddleOCR.__init__)

    if "use_gpu" in signature.parameters:
        kwargs["use_gpu"] = use_gpu
    if "show_log" in signature.parameters:
        kwargs["show_log"] = False
    if "use_doc_orientation_classify" in signature.parameters:
        kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in signature.parameters:
        kwargs["use_doc_unwarping"] = False
    if "use_textline_orientation" in signature.parameters:
        kwargs["use_textline_orientation"] = False
    if "ocr_version" in signature.parameters:
        kwargs["ocr_version"] = "PP-OCRv5"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _PADDLE_OCR_INSTANCE = PaddleOCR(**kwargs)
    return _PADDLE_OCR_INSTANCE


def extract_paddle_lines(result, normalize_mrz) -> list[str]:
    texts = []

    def walk(node):
        if node is None:
            return

        if isinstance(node, str):
            text = normalize_mrz(node)
            if text:
                texts.append(text)
            return

        if isinstance(node, dict):
            for key in ("rec_texts", "texts"):
                values = node.get(key)
                if isinstance(values, list):
                    for value in values:
                        walk(value)
                    return

            text = node.get("rec_text") or node.get("text")
            if isinstance(text, str):
                walk(text)
                return

            for value in node.values():
                walk(value)
            return

        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)):
                if node[1] and isinstance(node[1][0], str):
                    walk(node[1][0])
                    return
            for value in node:
                walk(value)

    walk(result)
    return texts


def _prepare_paddle_image(img):
    paddle_img = img
    if isinstance(paddle_img, np.ndarray) and paddle_img.ndim == 2:
        paddle_img = cv2.cvtColor(paddle_img, cv2.COLOR_GRAY2BGR)
    return paddle_img


def _run_paddle_inference(ocr, paddle_input):
    if hasattr(ocr, "predict"):
        return ocr.predict(paddle_input)
    if hasattr(ocr, "ocr"):
        return ocr.ocr(paddle_input, cls=False)
    raise RuntimeError("Unsupported PaddleOCR instance: missing predict/ocr method")


def trim_line1_spill(text: str, normalize_td3_line1) -> str:
    text = normalize_td3_line1(text)
    text = re.sub(r"<{4,}[A-Z0-9]{1,6}$", lambda m: "<" * len(m.group(0)), text)

    if "<<" not in text:
        return text

    name_zone = text[5:]
    if any(ch.isdigit() for ch in name_zone):
        digits = sum(ch.isdigit() for ch in name_zone)
        text = normalize_td3_line1(re.sub(r"[0-9]", "<", text))
        if digits >= 2:
            return text

    return text


def best_paddle_text_candidate(
    texts: list[str],
    line_kind: str,
    *,
    normalize_mrz,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
    trim_line1_spill_func,
) -> str:
    cleaned = [normalize_mrz(text) for text in texts if normalize_mrz(text)]
    if not cleaned:
        return ""

    candidates = []
    seen = set()

    def add_candidate(text: str) -> None:
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(text)

    for text in cleaned:
        add_candidate(text)
        for fragment in re.findall(r"[A-Z0-9<]{6,}", text):
            add_candidate(fragment)

    joined = normalize_mrz("".join(cleaned))
    add_candidate(joined)

    if line_kind == "line1":
        scored = []
        for cand in candidates:
            trimmed = trim_line1_spill_func(cand)
            score = score_td3_line1(trimmed)
            if any(ch.isdigit() for ch in trimmed[5:]):
                score -= 80.0
            if re.search(r"<{4,}[A-Z0-9]{1,6}$", cand):
                score -= 40.0
            scored.append((score, trimmed))
        return max(scored, key=lambda item: item[0])[1]

    scored = []
    for cand in candidates:
        normalized = normalize_td3_line2(cand)
        score, checks = score_td3_line2(normalized)
        if not re.match(r"^[A-Z0-9<]{1,2}[A-Z0-9<]{7,}", normalized):
            score -= 20.0
        score += checks.get("passed_count", 0) * 5.0
        scored.append((score, normalized))

    return max(scored, key=lambda item: item[0])[1]


def paddle_ocr_image(
    img,
    *,
    line_kind: str | None,
    paddle_lang: str,
    paddle_use_gpu: bool,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
) -> str:
    ocr = get_paddle_ocr(paddle_lang, paddle_use_gpu)
    paddle_img = _prepare_paddle_image(img)

    try:
        result = _run_paddle_inference(ocr, paddle_img)
    except NotImplementedError as exc:
        raise RuntimeError(
            "Installed Paddle runtime failed during CPU inference. "
            "This project expects the official PaddleOCR-supported runtime; "
            "reinstall the pinned versions from requirements-paddle.txt in .venv."
        ) from exc

    texts = extract_paddle_lines(result, normalize_mrz)
    if not texts:
        return ""

    if line_kind in {"line1", "line2"}:
        return best_paddle_text_candidate(
            texts,
            line_kind,
            normalize_mrz=normalize_mrz,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
            trim_line1_spill_func=lambda text: trim_line1_spill(text, normalize_td3_line1),
        )

    return normalize_mrz("".join(texts))


def paddle_ocr_images(
    images,
    *,
    line_kind: str | None,
    paddle_lang: str,
    paddle_use_gpu: bool,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
):
    ocr = get_paddle_ocr(paddle_lang, paddle_use_gpu)
    paddle_imgs = [_prepare_paddle_image(img) for img in images]

    batch_results = None
    if paddle_imgs:
        try:
            candidate_results = _run_paddle_inference(ocr, paddle_imgs)
            if isinstance(candidate_results, list) and len(candidate_results) == len(paddle_imgs):
                batch_results = candidate_results
        except Exception:
            batch_results = None

    if batch_results is None:
        texts = []
        for paddle_img in paddle_imgs:
            try:
                result = _run_paddle_inference(ocr, paddle_img)
            except NotImplementedError as exc:
                raise RuntimeError(
                    "Installed Paddle runtime failed during CPU inference. "
                    "This project expects the official PaddleOCR-supported runtime; "
                    "reinstall the pinned versions from requirements-paddle.txt in .venv."
                ) from exc
            texts.append(
                _extract_text_from_paddle_result(
                    result,
                    line_kind=line_kind,
                    normalize_mrz=normalize_mrz,
                    normalize_td3_line1=normalize_td3_line1,
                    normalize_td3_line2=normalize_td3_line2,
                    score_td3_line1=score_td3_line1,
                    score_td3_line2=score_td3_line2,
                )
            )
        return texts

    return [
        _extract_text_from_paddle_result(
            result,
            line_kind=line_kind,
            normalize_mrz=normalize_mrz,
            normalize_td3_line1=normalize_td3_line1,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
        )
        for result in batch_results
    ]


def _extract_text_from_paddle_result(
    result,
    *,
    line_kind: str | None,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
) -> str:
    texts = extract_paddle_lines(result, normalize_mrz)
    if not texts:
        return ""

    if line_kind in {"line1", "line2"}:
        return best_paddle_text_candidate(
            texts,
            line_kind,
            normalize_mrz=normalize_mrz,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
            trim_line1_spill_func=lambda text: trim_line1_spill(text, normalize_td3_line1),
        )

    return normalize_mrz("".join(texts))
