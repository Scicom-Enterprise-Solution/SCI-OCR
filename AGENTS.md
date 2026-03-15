# AGENTS.md

## Purpose

This repository is a machine-readable zone (MRZ) extraction project focused on **passport MRZ only**.

Current project scope is intentionally narrow:
- **TD3 passport MRZ only**
- **2 lines**
- **44 characters per line**

Do not expand scope to TD1, TD2, visas, residence permits, or other MRTD formats unless explicitly requested.

---

## Repo working notes

- Use the project virtual environment for Python commands: `./.venv/bin/python`.
- Treat the user's WSL terminal as the source of truth for PaddleOCR and GPU-enabled runs.
- The Codex sandbox may not have visibility into the host GPU or the same Paddle runtime state as WSL, so sandbox-side Paddle/GPU results are not authoritative.
- Sandbox verification should focus on unit tests, static code checks, and non-GPU local validation unless the user explicitly asks otherwise.
- The repo now includes a FastAPI + SQLite API layer under `api/` and `db/`. Treat this as real repo architecture, not temporary scaffolding.
- The API is intended to behave like a long-lived production process. Warm-process behavior is more representative than one-process-per-file CLI timing.
- Runtime storage belongs under `storage/` and must stay git-ignored.
- API report files belong under `storage/reports/`.
- CLI reference/regression combined reports belong under `samples/reports/`.
- Per-sample pipeline reports still belong under `output/<sample>/`.
- After OCR failures, inspect the generated per-sample report files under `output/<sample>/`. These contain deeper details than the terminal summary and should drive mismatch analysis.
- Prefer repo-relative serialized paths over absolute paths in reports, DB records, and API responses. Internal absolute resolution is fine only at the point of file access.
- Keep cross-platform portability in mind. Stored path strings should remain portable across Linux, WSL, and Windows.

---

## Core project philosophy

This project is **truth-first**, not beauty-first.

Primary goals:
1. Extract the MRZ as it is actually printed.
2. Preserve benchmark/reference truth.
3. Avoid “smart” fixes that improve pass rate by corrupting source truth.
4. Keep parsing, scoring, checksum logic, OCR, and repair conceptually separate.
5. Make minimal, high-confidence changes.

If there is tension between pretty normalization and visible document truth, **visible truth wins** unless an explicit normalization policy is requested.

---

## Non-negotiable rules

### 1. Preserve visible source truth
- Do not silently rewrite extracted MRZ text into a prettier or more standard-looking version.
- If the image visibly supports a value, preserve it.
- Do not normalize printed prefixes or name separators unless explicitly instructed.

### 2. TD3 passport only
Assume all parsing/validation logic is for TD3 passport MRZ unless the user says otherwise.

Expected structure:
- line count: 2
- line length: 44 each

### 3. Line 2 is structurally stronger than line 1
- Treat checksum-backed **line 2** as more trustworthy than heuristic line 1 recovery.
- Passport number, DOB, expiry, nationality, and related check-digit segments have strong structural value.
- Line 1 is fragile and must be handled conservatively.

### 4. Be extremely conservative with line 1 repair
Especially avoid aggressive rewrites of:
- names
- separator counts (`<` vs `<<`)
- token boundaries
- document subtype slot (character 2)

Line 1 has no checksum safety net. A “smart” fix can easily turn truth into fiction.

### 5. No benchmark gaming
Do not implement hacks that merely force benchmark matches while reducing real-world correctness.
If a change is dataset-shaped, overfit, or risky, say so explicitly.

---

## Passport document code policy

For TD3 passport line 1:
- character 1 should strongly prefer `P`
- character 2 is a **real document-type slot**, not automatically filler

Accepted as structurally plausible:
- `P<`
- `P[A-Z]`

Do **not** blindly normalize the second character to `<`.

Examples of policy:
- If the passport visibly prints `P<`, preserve `P<`
- If the passport visibly prints `PP`, preserve `PP`
- If the passport visibly prints `PB`, preserve `PB`
- If a second-letter code is unusual or non-standard, preserve it if visibly printed and flag it rather than silently correcting it

The parser/validator may classify a second-letter code as:
- standardized / expected
- plausible but uncommon
- non-standard / legacy / unknown

But extraction should still preserve visible truth.

---

## Project architecture expectations

The code should remain logically separable across these concerns:

1. **OCR / backend routing**
2. **Preprocessing**
3. **Candidate generation**
4. **Line splitting / segmentation**
5. **TD3 structure detection**
6. **Checksum logic**
7. **Scoring / ranking**
8. **Parsing**
9. **Repair**
10. **Benchmark / reference evaluation**

Do not collapse these into one giant “god file” if avoidable.

When touching code, prefer changes in the correct layer instead of downstream hacks.

Examples:
- OCR misread -> fix OCR/preprocessing/candidate scoring
- bad line split -> fix segmentation/splitting
- checksum mismatch -> fix parsing/validation/checksum logic
- unsafe name rewrite -> fix repair conservatism, not parser structure
- benchmark mismatch due to forced normalization -> remove or gate normalization

---

## Known project realities

This repo already shows a clear pattern:

- Line 2 is generally strong
- Line 1 is the weak side
- Common failure modes are clustered, not random

Typical failure classes seen so far:
1. **Document-code slot confusion** at line 1 char 2 (`<` vs `C` / `O` / other letters)
2. **Line 1 separator over-insertion** (`<` vs `<<`)
3. **Single-character OCR confusions** inside names (`N/M`, etc.)
4. **Tail truncation** or over-pruning in long names
5. **Dataset-shaped heuristics** that may overfit current samples

Use these patterns when diagnosing bugs.

---

## Repair policy

Repair is allowed, but must be careful.

### Safe or relatively safer repair areas
- obvious OCR cleanup in checksum-backed line 2 fields
- constrained recovery when check digits strongly support one interpretation
- trimming obvious garbage outside the valid TD3 structure
- cautious candidate ranking using structural evidence

### Dangerous repair areas
- beautifying names
- inserting/removing `<` in line 1 aggressively
- identity text rewrites driven by “looks better” heuristics
- forced normalization of passport prefix
- any fix that makes benchmark output look better at the cost of printed truth

When in doubt:
- preserve rawer truth
- add a warning/meta flag
- avoid rewriting the final extracted MRZ

---

## How to reason about problems

When analyzing a bug or failure, classify it first.

Use one or more of these categories:
- OCR confusion
- preprocessing issue
- segmentation/crop issue
- candidate ranking issue
- TD3 structural parsing bug
- checksum bug
- repair-layer corruption
- benchmark/reference mismatch
- normalization policy mistake

Fix the issue at the correct layer.

Do not patch a root-cause OCR problem by hacking parser output unless explicitly asked for a temporary benchmark-specific workaround.

---

## Expected working style

Before proposing changes:
1. Read the relevant files
2. Identify the runtime path
3. Explain what the code is doing now
4. State what is fragile, wrong, or overfit
5. Propose the smallest safe change
6. State residual risks

Ground all claims in the actual codebase.

Do not:
- invent behavior
- assume architecture without reading
- suggest broad rewrites without clear justification
- present guesses as facts

Always distinguish:
- **observed fact from code**
- **inference**
- **recommended change**

---

## Change philosophy

Prefer:
- surgical patches
- high-confidence diffs
- local fixes at the correct abstraction layer
- preserving public behavior unless requested otherwise

Avoid:
- broad refactors unless requested
- merging responsibilities into one file
- hidden normalization
- speculative cleanup unrelated to the current bug

If a refactor is suggested, explain:
- what coupling/problem it solves
- why current design is fragile
- which boundaries should exist
- how to migrate safely

---

## Testing and evaluation expectations

This project is benchmark-sensitive.

When changing code:
- think about exact-match MRZ output, not just parsed fields
- consider both reference truth and structural validity
- validate against current benchmark/reference set when practical
- explicitly mention which failure class the change targets

Important:
- a line 2 improvement backed by checksums is usually more trustworthy than a line 1 “beautification”
- an exact-match gain caused by destructive normalization is not a real win

Preferred evaluation mindset:
- exact line1 / line2 match
- checksum consistency
- structural TD3 correctness
- no silent corruption of visible truth

Additional repo-specific evaluation rules:
- For benchmark/reference runs, prefer the in-process warm runners in `scripts/check_reference_set_*.py` over subprocess-per-file timing.
- When reviewing performance, separate cold-start cost from steady-state cost.
- `TOTAL_RAW` includes warm-up; `TOTAL` and averages in the combined CLI report are the steady-state numbers.
- If a failure occurs in a reference run, inspect the relevant per-sample `output/<sample>/..._report.json` before proposing fixes.

---

## Recent performance findings

The project now has enough timing instrumentation that performance work should be evidence-driven.

Observed findings from recent Paddle investigation:
- For single-file TD3 runs, the dominant cost is **Stage 3 candidate OCR**, not Stage 1 or Stage 2.
- `document_preparation`, MRZ detection, variant-preparation, and final ranking are all comparatively small.
- The main wall-clock cost sits in `mrz.ocr.timing_ms.candidate_ocr_ms`.
- Paddle GPU is available and resolves to `gpu:0`, but utilization is bursty because the runtime is doing multiple small batched inference calls rather than one dense sustained job.
- Current Paddle telemetry shows batching is working and does not normally fall back to serial.
- Reducing image-save/debug I/O in normal mode did **not** materially change runtime.
- CPU-side preprocessing parallelism helped only marginally; it is not the main remaining bottleneck.
- The earlier apparent 25s-40s per-sample cost from regression runs was heavily distorted by cold-start behavior from spawning one Python process per sample.
- After switching the regression runner to a single warm in-process run, steady-state Paddle timing dropped to roughly single-digit seconds per sample, which is much closer to API behavior.
- Warm-process benchmarking is therefore the correct baseline for operational performance discussions.

Current proven-safe speed baseline:
- `PADDLE_PROFILE=exhaustive`
- `PADDLE_FAST=False`
- `MRZ_VARIANT_WORKERS=4`

What was learned from experiments:
- A very aggressive Paddle fast profile reduced runtime but caused benchmark regressions and should not be the default without explicit tradeoff acceptance.
- Pruning line-1 split evaluation based only on line-2 split quality caused a real regression (`inam_new.png`) and should be treated as unsafe unless redesigned more carefully.
- Narrow line-1 scoring penalties for obvious garbage tails can be safe when backed by report evidence and targeted tests.
- Warm API extraction latency can be far lower than the old subprocess benchmark suggested, and API timing should not be confused with reference-backed output reuse.

Guidance for future speed work:
- For single-file latency, the main lever is reducing **redundant OCR candidate evaluation** without losing the candidate that carries the correct line 1.
- Prefer conservative, report-backed changes over broad search-space cuts.
- When a new optimization changes candidate flow, rerun the full Paddle reference set and inspect failed sample reports before keeping it.
- Always inspect `mrz.ocr.backend_stats.paddle` and timing fields in the report before drawing conclusions about GPU behavior.
- Do not use cold-start subprocess benchmark timing as the main argument for changing search profiles.

---

## API And Storage Notes

Current API shape:
- `POST /api/uploads`
- `POST /api/extractions`
- `GET /api/references`
- `POST /api/references`
- `GET /api/health`

Current API behavior:
- Uploads are deduplicated by SHA-256 hash.
- `document_id` is the primary handle for later extraction.
- Extraction only requires `document_id`; `crop`, `rotation`, and `use_face_hint` are optional.
- `use_face_hint` should default to `False` unless explicitly requested.
- API extraction should suppress internal Stage 1/2/3 debug spam when `DEBUG=False`.

Path/report conventions:
- API reports: `storage/reports/<document_id>_<ddMMyyHHmmss>.json`
- CLI combined run reports: `samples/reports/<ddMMyyHHmmss>_<backend>.json`
- Avoid absolute paths in serialized output.

Architecture guidance:
- Keep API concerns in `api/`.
- Keep persistence concerns in `db/`.
- Do not entangle API/controller code with MRZ core logic more than necessary.
- Prefer service-layer calls into the existing pipeline rather than duplicating OCR logic inside routes.

---

## Output expectations for repository work

Depending on task, respond in one of these modes:

### REPO MAP
Explain:
- project purpose
- runtime flow
- major modules
- hotspots
- technical debt
- safest extension points

### BUG HUNT
Explain:
- what failed
- failure class
- root cause path
- smallest safe fix

### PATCH
Implement:
- minimal diff
- correct-layer fix
- no unnecessary cleanup

### REVIEW
Audit:
- fragility
- coupling
- overfitting
- repair risks
- architectural debt

### REFACTOR PLAN
Propose:
- cleaner boundaries
- phased migration
- low-risk sequencing

### TEST PLAN
Define:
- smallest useful regression set
- benchmark targets
- failure classes covered

---

## Repository-specific cautions

Be especially skeptical of code that mixes too many responsibilities in one place, especially if one file is doing all of:
- env parsing
- OCR routing
- preprocessing
- candidate generation
- line splitting
- scoring
- checksum logic
- field repair
- country repair
- parsing
- image saving

That pattern is fragile and should be called out clearly.

Also treat the following as high risk:
- support-bonus tuning that is dataset-shaped
- token repair thresholds tuned to current samples
- name-token scoring that “beautifies” identity text
- vowel/consonant or appearance-based name rewriting
- spill trimming that may truncate valid long names

These may help current benchmarks while silently corrupting truth.

---

## Final principle

Act like the original engineer responsible for long-term correctness.

That means:
- understand first
- change as little as possible
- protect source truth
- respect TD3 structure
- trust checksum-backed evidence
- stay conservative with names
- never fake correctness
