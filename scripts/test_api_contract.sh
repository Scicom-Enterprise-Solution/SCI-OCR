#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
API_BASE="${API_BASE:-http://127.0.0.1:3000}"
SAMPLE_PATH="${SAMPLE_PATH:-$ROOT/samples/11.png}"
TMP_DIR="${TMP_DIR:-$ROOT/tmp/api_contract}"
SMALL_IMAGE_PATH="${SMALL_IMAGE_PATH:-}"

mkdir -p "$TMP_DIR"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

if [[ ! -f "$SAMPLE_PATH" ]]; then
  echo "sample not found: $SAMPLE_PATH" >&2
  exit 1
fi

json_get() {
  local expr="$1"
  "$PYTHON_BIN" -c '
import json
import sys

expr = sys.argv[1]
payload = json.loads(sys.stdin.read())
value = payload
for part in expr.split("."):
    if part:
        value = value.get(part)
if value is None:
    sys.exit(1)
if isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
' "$expr"
}

run_python_client() {
  local name="$1"
  shift
  echo
  echo "== $name =="
  "$PYTHON_BIN" "$ROOT/scripts/api_client.py" "$SAMPLE_PATH" "$@"
}

assert_error_detail() {
  local payload="$1"
  local expected="$2"
  local body
  body="$(curl -sS -X POST "$API_BASE/api/extractions" \
    -H 'Content-Type: application/json' \
    -d "$payload")"
  echo "$body"
  local detail=""
  if ! detail="$(printf '%s' "$body" | json_get "detail" 2>/dev/null)"; then
    detail=""
  fi
  if [[ "$detail" != "$expected" ]]; then
    echo "unexpected error detail" >&2
    echo "expected: $expected" >&2
    echo "actual:   $detail" >&2
    exit 1
  fi
}

assert_success_status() {
  local payload="$1"
  local body
  body="$(curl -sS -X POST "$API_BASE/api/extractions" \
    -H 'Content-Type: application/json' \
    -d "$payload")"
  echo "$body"
  local status=""
  if ! status="$(printf '%s' "$body" | json_get "status" 2>/dev/null)"; then
    status=""
  fi
  if [[ "$status" != "success" ]]; then
    echo "expected success status but got: $status" >&2
    exit 1
  fi
}

echo "Using API base: $API_BASE"
echo "Using sample:   $SAMPLE_PATH"

run_python_client \
  "1. raw mode default correction" \
  --input-mode raw \
  --save "$TMP_DIR/api_raw.json" \
  --report-save "$TMP_DIR/api_raw_report.json"

run_python_client \
  "2. frontend mode no correction" \
  --input-mode frontend \
  --disable-correction \
  --save "$TMP_DIR/api_frontend.json" \
  --report-save "$TMP_DIR/api_frontend_report.json"

run_python_client \
  "3. frontend mode explicit correction override" \
  --input-mode frontend \
  --enable-correction \
  --save "$TMP_DIR/api_frontend_forcecorr.json" \
  --report-save "$TMP_DIR/api_frontend_forcecorr_report.json"

echo
echo "== Upload sample for direct boundary tests =="
DOC_ID="$(
  curl -sS -X POST "$API_BASE/api/uploads" \
    -F "file=@$SAMPLE_PATH" | json_get "document_id"
)"
echo "document_id: $DOC_ID"

BOUNDARY_MSG="frontend mode does not accept crop/transform/rotation; send final prepared image instead"

echo
echo "== 4. frontend rejects crop =="
assert_error_detail \
  "{
    \"document_id\":\"$DOC_ID\",
    \"input_mode\":\"frontend\",
    \"enable_correction\":false,
    \"crop\":{\"x\":0.1,\"y\":0.1,\"width\":0.5,\"height\":0.5}
  }" \
  "$BOUNDARY_MSG"

echo
echo "== 5. frontend rejects transform =="
assert_error_detail \
  "{
    \"document_id\":\"$DOC_ID\",
    \"input_mode\":\"frontend\",
    \"enable_correction\":false,
    \"transform\":{\"micro_rotation\":1.0}
  }" \
  "$BOUNDARY_MSG"

echo
echo "== 6. frontend rejects non-zero rotation =="
assert_error_detail \
  "{
    \"document_id\":\"$DOC_ID\",
    \"input_mode\":\"frontend\",
    \"enable_correction\":false,
    \"rotation\":90
  }" \
  "$BOUNDARY_MSG"

echo
echo "== 7. raw mode still accepts correction inputs =="
assert_success_status \
  "{
    \"document_id\":\"$DOC_ID\",
    \"input_mode\":\"raw\",
    \"enable_correction\":true,
    \"rotation\":90,
    \"transform\":{\"micro_rotation\":1.0},
    \"crop\":{\"x\":0.1,\"y\":0.1,\"width\":0.5,\"height\":0.5}
  }"

if [[ -n "$SMALL_IMAGE_PATH" ]]; then
  if [[ ! -f "$SMALL_IMAGE_PATH" ]]; then
    echo "small image not found: $SMALL_IMAGE_PATH" >&2
    exit 1
  fi

  echo
  echo "== 8. frontend rejects undersized image =="
  SMALL_ID="$(
    curl -sS -X POST "$API_BASE/api/uploads" \
      -F "file=@$SMALL_IMAGE_PATH" | json_get "document_id"
  )"
  echo "small document_id: $SMALL_ID"
  assert_error_detail \
    "{
      \"document_id\":\"$SMALL_ID\",
      \"input_mode\":\"frontend\",
      \"enable_correction\":false
    }" \
    "frontend input image is too small; minimum size is 600x400"
else
  echo
  echo "== 8. skipping undersized frontend test (set SMALL_IMAGE_PATH to enable) =="
fi

echo
echo "== 9. warm regression check =="
"$PYTHON_BIN" "$ROOT/scripts/check_reference_set_paddle.py"

echo
echo "All contract checks completed."
