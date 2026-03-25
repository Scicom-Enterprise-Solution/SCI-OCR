import contextlib
import inspect
import io
import os
import re
from itertools import zip_longest

import cv2
import numpy as np


_PADDLE_OCR_INSTANCES = {}
_PADDLE_OCR_STATS = {}


def reset_paddle_ocr_stats() -> None:
    global _PADDLE_OCR_STATS
    _PADDLE_OCR_STATS = {
        "inference_calls": 0,
        "batch_requests": 0,
        "batched_calls": 0,
        "batch_fallbacks": 0,
        "serial_calls": 0,
        "images_submitted": 0,
        "batch_sizes": [],
        "predict_calls": 0,
        "ocr_calls": 0,
    }


def get_paddle_ocr_stats() -> dict:
    return {
        **_PADDLE_OCR_STATS,
        "batch_sizes": list(_PADDLE_OCR_STATS.get("batch_sizes", [])),
    }


reset_paddle_ocr_stats()


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


def reset_paddle_ocr_cache() -> None:
    global _PADDLE_OCR_INSTANCES
    _PADDLE_OCR_INSTANCES = {}


def _parse_optional_float_env(name: str):
    raw = os.getenv(name, "").strip()
    if not raw:
        return None

    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Ignoring invalid float in {name}: {raw}")
        return None


def _parse_choice_env(name: str, allowed: set[str]) -> str:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return ""
    if raw in allowed:
        return raw
    print(f"[WARN] Ignoring invalid value in {name}: {raw}")
    return ""


def _resolve_paddle_det_model_name(variant: str) -> str | None:
    if not variant:
        return None
    return f"PP-OCRv5_{variant}_det"


def _resolve_paddle_rec_model_name(lang: str, variant: str) -> str | None:
    if not variant:
        return None
    if variant == "server":
        return "PP-OCRv5_server_rec"
    if lang == "en":
        return f"en_PP-OCRv5_mobile_rec"
    return None


def build_paddle_ocr_kwargs(PaddleOCR, lang: str, use_gpu_requested: bool) -> dict:
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

    det_variant = _parse_choice_env("PADDLEOCR_DET_MODEL_VARIANT", {"mobile", "server"})
    rec_variant = _parse_choice_env("PADDLEOCR_REC_MODEL_VARIANT", {"mobile", "server"})
    det_model_name = os.getenv("PADDLEOCR_TEXT_DETECTION_MODEL_NAME", "").strip()
    rec_model_name = os.getenv("PADDLEOCR_TEXT_RECOGNITION_MODEL_NAME", "").strip()
    det_model_dir = os.getenv("PADDLEOCR_TEXT_DETECTION_MODEL_DIR", "").strip()
    rec_model_dir = os.getenv("PADDLEOCR_TEXT_RECOGNITION_MODEL_DIR", "").strip()
    rec_score_thresh = _parse_optional_float_env("PADDLEOCR_TEXT_REC_SCORE_THRESH")

    if not det_model_name:
        det_model_name = _resolve_paddle_det_model_name(det_variant) or ""
    if not rec_model_name:
        rec_model_name = _resolve_paddle_rec_model_name(lang, rec_variant) or ""

    if det_model_name and "text_detection_model_name" in signature.parameters:
        kwargs["text_detection_model_name"] = det_model_name
    if rec_model_name and "text_recognition_model_name" in signature.parameters:
        kwargs["text_recognition_model_name"] = rec_model_name
    if det_model_dir and "text_detection_model_dir" in signature.parameters:
        kwargs["text_detection_model_dir"] = det_model_dir
    if rec_model_dir and "text_recognition_model_dir" in signature.parameters:
        kwargs["text_recognition_model_dir"] = rec_model_dir
    if rec_score_thresh is not None and "text_rec_score_thresh" in signature.parameters:
        kwargs["text_rec_score_thresh"] = rec_score_thresh

    return kwargs


def get_paddle_ocr(lang: str, use_gpu_requested: bool):
    global _PADDLE_OCR_INSTANCES

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

    kwargs = build_paddle_ocr_kwargs(PaddleOCR, lang, use_gpu_requested)
    cache_key = tuple(sorted(kwargs.items()))
    cached = _PADDLE_OCR_INSTANCES.get(cache_key)
    if cached is not None:
        return cached

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _PADDLE_OCR_INSTANCES[cache_key] = PaddleOCR(**kwargs)
    return _PADDLE_OCR_INSTANCES[cache_key]


def _normalize_paddle_confidence(value):
    if value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(confidence):
        return None

    return max(0.0, min(confidence, 1.0))


def extract_paddle_entries(result, normalize_mrz) -> list[dict]:
    entries = []

    def add_entry(text: str, score=None) -> None:
        normalized = normalize_mrz(text)
        if not normalized:
            return

        entry = {"text": normalized}
        confidence = _normalize_paddle_confidence(score)
        if confidence is not None:
            entry["line_confidence"] = confidence
        entries.append(entry)

    def walk(node):
        if node is None:
            return

        if isinstance(node, str):
            add_entry(node)
            return

        if isinstance(node, dict):
            for text_key, score_key in (("rec_texts", "rec_scores"), ("texts", "scores")):
                values = node.get(text_key)
                if isinstance(values, list):
                    scores = node.get(score_key)
                    if not isinstance(scores, list):
                        scores = []
                    for value, score in zip_longest(values, scores, fillvalue=None):
                        if isinstance(value, str):
                            add_entry(value, score)
                        else:
                            walk(value)
                    return

            text = node.get("rec_text") or node.get("text")
            if isinstance(text, str):
                add_entry(text, node.get("rec_score") or node.get("score"))
                return

            for value in node.values():
                walk(value)
            return

        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)):
                if node[1] and isinstance(node[1][0], str):
                    score = node[1][1] if len(node[1]) > 1 else None
                    add_entry(node[1][0], score)
                    return
            for value in node:
                walk(value)

    walk(result)
    return entries


def extract_paddle_lines(result, normalize_mrz) -> list[str]:
    return [entry["text"] for entry in extract_paddle_entries(result, normalize_mrz)]


def _prepare_paddle_image(img):
    paddle_img = img
    if isinstance(paddle_img, np.ndarray) and paddle_img.ndim == 2:
        paddle_img = cv2.cvtColor(paddle_img, cv2.COLOR_GRAY2BGR)
    return paddle_img


def _run_paddle_inference(ocr, paddle_input):
    is_batch = isinstance(paddle_input, list)
    batch_size = len(paddle_input) if is_batch else 1
    _PADDLE_OCR_STATS["inference_calls"] += 1
    _PADDLE_OCR_STATS["images_submitted"] += batch_size
    if is_batch:
        _PADDLE_OCR_STATS["batch_requests"] += 1
        _PADDLE_OCR_STATS["batch_sizes"].append(batch_size)
    else:
        _PADDLE_OCR_STATS["serial_calls"] += 1

    if hasattr(ocr, "predict"):
        _PADDLE_OCR_STATS["predict_calls"] += 1
        return ocr.predict(paddle_input)
    if hasattr(ocr, "ocr"):
        _PADDLE_OCR_STATS["ocr_calls"] += 1
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


def best_paddle_entry_candidate(
    entries: list[dict],
    line_kind: str,
    *,
    normalize_mrz,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
    trim_line1_spill_func,
) -> dict:
    cleaned = []
    for entry in entries:
        text = normalize_mrz(entry.get("text", ""))
        if not text:
            continue
        cleaned.append({
            "text": text,
            "line_confidence": _normalize_paddle_confidence(entry.get("line_confidence")),
        })

    if not cleaned:
        return {"text": ""}

    candidates = []
    seen = {}

    def add_candidate(text: str, source_confidence) -> None:
        if not text:
            return

        existing = seen.get(text)
        if existing is None:
            candidate = {"text": text, "line_confidence": source_confidence}
            seen[text] = candidate
            candidates.append(candidate)
            return

        if source_confidence is not None:
            previous = existing.get("line_confidence")
            if previous is None or source_confidence > previous:
                existing["line_confidence"] = source_confidence

    for entry in cleaned:
        text = entry["text"]
        confidence = entry.get("line_confidence")
        add_candidate(text, confidence)
        for fragment in re.findall(r"[A-Z0-9<]{6,}", text):
            add_candidate(fragment, confidence)

    joined_confidence = max(
        (
            entry.get("line_confidence")
            for entry in cleaned
            if entry.get("line_confidence") is not None
        ),
        default=None,
    )
    add_candidate(normalize_mrz("".join(entry["text"] for entry in cleaned)), joined_confidence)

    if line_kind == "line1":
        scored = []
        for cand in candidates:
            trimmed = trim_line1_spill_func(cand["text"])
            score = score_td3_line1(trimmed)
            if any(ch.isdigit() for ch in trimmed[5:]):
                score -= 80.0
            if re.search(r"<{4,}[A-Z0-9]{1,6}$", cand["text"]):
                score -= 40.0
            scored.append((score, trimmed, cand.get("line_confidence")))
        _, text, confidence = max(scored, key=lambda item: item[0])
    else:
        scored = []
        for cand in candidates:
            normalized = normalize_td3_line2(cand["text"])
            score, checks = score_td3_line2(normalized)
            if not re.match(r"^[A-Z0-9<]{1,2}[A-Z0-9<]{7,}", normalized):
                score -= 20.0
            score += checks.get("passed_count", 0) * 5.0
            scored.append((score, normalized, cand.get("line_confidence")))
        _, text, confidence = max(scored, key=lambda item: item[0])

    payload = {"text": text}
    if confidence is not None:
        payload["line_confidence"] = confidence
        payload["ocr_confidence_source"] = "line"
    return payload


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

    return _extract_candidate_from_paddle_result(
        result,
        line_kind=line_kind,
        normalize_mrz=normalize_mrz,
        normalize_td3_line1=normalize_td3_line1,
        normalize_td3_line2=normalize_td3_line2,
        score_td3_line1=score_td3_line1,
        score_td3_line2=score_td3_line2,
    )["text"]


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
        _PADDLE_OCR_STATS["batch_fallbacks"] += 1
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
                _extract_candidate_from_paddle_result(
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

    _PADDLE_OCR_STATS["batched_calls"] += 1
    return [
        _extract_candidate_from_paddle_result(
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


def _extract_candidate_from_paddle_result(
    result,
    *,
    line_kind: str | None,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
) -> dict:
    entries = extract_paddle_entries(result, normalize_mrz)
    if not entries:
        return {"text": ""}

    if line_kind in {"line1", "line2"}:
        return best_paddle_entry_candidate(
            entries,
            line_kind,
            normalize_mrz=normalize_mrz,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
            trim_line1_spill_func=lambda text: trim_line1_spill(text, normalize_td3_line1),
        )

    payload = {"text": normalize_mrz("".join(entry["text"] for entry in entries))}
    confidence = max(
        (
            entry.get("line_confidence")
            for entry in entries
            if entry.get("line_confidence") is not None
        ),
        default=None,
    )
    if confidence is not None:
        payload["line_confidence"] = confidence
        payload["ocr_confidence_source"] = "line"
    return payload
