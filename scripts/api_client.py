import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _join_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _guess_content_type(path: str) -> str:
    content_type, _ = mimetypes.guess_type(path)
    return content_type or "application/octet-stream"


def _build_multipart_body(field_name: str, filename: str, payload: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----mrzclient{uuid.uuid4().hex}"
    lines = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{os.path.basename(filename)}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        payload,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(lines), boundary


def _http_json(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {url} failed with HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"{method} {url} failed: {exc.reason}") from exc


def _http_bytes(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {url} failed with HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"{method} {url} failed: {exc.reason}") from exc


def upload_document(base_url: str, file_path: str) -> dict:
    with open(file_path, "rb") as f:
        payload = f.read()
    content_type = _guess_content_type(file_path)
    body, boundary = _build_multipart_body("file", os.path.basename(file_path), payload, content_type)
    return _http_json(
        _join_url(base_url, "/api/uploads"),
        method="POST",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def build_extraction_payload(
    *,
    document_id: str,
    input_mode: str,
    enable_correction: bool | None,
    use_face_hint: bool,
) -> dict:
    payload = {
        "document_id": document_id,
        "input_mode": input_mode,
        "use_face_hint": use_face_hint,
    }
    if enable_correction is not None:
        payload["enable_correction"] = enable_correction
    return payload


def extract_document(
    base_url: str,
    *,
    document_id: str,
    input_mode: str,
    enable_correction: bool | None,
    use_face_hint: bool,
) -> dict:
    payload = build_extraction_payload(
        document_id=document_id,
        input_mode=input_mode,
        enable_correction=enable_correction,
        use_face_hint=use_face_hint,
    )
    return _http_json(
        _join_url(base_url, "/api/extractions"),
        method="POST",
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def _print_summary(upload_result: dict, extraction_result: dict, base_url: str) -> None:
    report_path = extraction_result.get("report_path")
    extraction_id = extraction_result.get("extraction_id", "")
    report_url = _join_url(base_url, f"/api/extractions/{extraction_id}/report") if extraction_id else None
    summary = {
        "document_id": upload_result.get("document_id"),
        "upload": {
            "filename": upload_result.get("filename"),
            "deduplicated": upload_result.get("deduplicated"),
            "source_type": upload_result.get("source_type"),
            "preview_width": upload_result.get("preview_width"),
            "preview_height": upload_result.get("preview_height"),
        },
        "extraction": {
            "extraction_id": extraction_result.get("extraction_id"),
            "status": extraction_result.get("status"),
            "filename": extraction_result.get("filename"),
            "duration_ms": extraction_result.get("duration_ms"),
            "line1": extraction_result.get("line1"),
            "line2": extraction_result.get("line2"),
            "report_path": report_path,
            "report_url": report_url,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a document and run MRZ extraction against the local API.",
    )
    parser.add_argument("file", help="Path to the image or PDF to upload.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:3000",
        help="API base URL. Default: http://127.0.0.1:3000",
    )
    parser.add_argument(
        "--input-mode",
        choices=("raw", "frontend"),
        default="raw",
        help="Extraction input mode. raw runs correction by default; frontend skips it by default.",
    )
    correction_group = parser.add_mutually_exclusive_group()
    correction_group.add_argument(
        "--enable-correction",
        action="store_true",
        help="Force backend correction on, regardless of input mode.",
    )
    correction_group.add_argument(
        "--disable-correction",
        action="store_true",
        help="Force backend correction off, regardless of input mode.",
    )
    parser.add_argument(
        "--use-face-hint",
        action="store_true",
        help="Request face-hint orientation when correction is enabled.",
    )
    parser.add_argument(
        "--save",
        help="Write the full upload and extraction responses to this JSON file.",
    )
    parser.add_argument(
        "--report-save",
        help="Download /api/extractions/{id}/report to this local file after extraction.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    file_path = os.path.abspath(args.file)
    if not os.path.isfile(file_path):
        raise SystemExit(f"file not found: {file_path}")

    enable_correction = None
    if args.enable_correction:
        enable_correction = True
    elif args.disable_correction:
        enable_correction = False

    upload_result = upload_document(args.base_url, file_path)
    extraction_result = extract_document(
        args.base_url,
        document_id=upload_result["document_id"],
        input_mode=args.input_mode,
        enable_correction=enable_correction,
        use_face_hint=args.use_face_hint,
    )
    if args.report_save:
        extraction_id = extraction_result.get("extraction_id")
        if not extraction_id:
            raise SystemExit("extraction response did not include extraction_id; cannot download report")
        report_bytes = _http_bytes(
            _join_url(args.base_url, f"/api/extractions/{extraction_id}/report"),
            method="GET",
        )
        _write_bytes(os.path.abspath(args.report_save), report_bytes)
    if args.save:
        _write_json(
            os.path.abspath(args.save),
            {
                "request": {
                    "file": file_path,
                    "base_url": args.base_url,
                    "input_mode": args.input_mode,
                    "enable_correction": enable_correction,
                    "use_face_hint": args.use_face_hint,
                    "report_save": os.path.abspath(args.report_save) if args.report_save else None,
                },
                "upload": upload_result,
                "extraction": extraction_result,
            },
        )
    _print_summary(upload_result, extraction_result, args.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
