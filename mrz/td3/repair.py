import itertools
import re

from mrz.td3.normalize import (
    MRZ_LINE_LEN,
    _generate_country_code_variants,
    _sanitize_name_token,
    _sanitize_name_zone,
    _split_tail_name_tokens,
    normalize_td3_line1,
)
from mrz.td3.score import _name_token_score, score_td3_line1


TOKEN_AMBIGUOUS_SUBS = {
    "T": ("I",),
    "I": ("L",),
    "L": ("I",),
    "M": ("N",),
    "N": ("M",),
}
VOWELS = set("AEIOUY")

MIN_TOKEN_REPAIR_SCORE_GAIN = 6.0
MIN_LINE1_ZONE_REPAIR_GAIN = 4.0


HIGH_RISK_REPAIR_REASONS = {
    "name_noise_collapse",
    "token_ambiguity_repair",
    "surname_ambiguity_repair",
    "paddle_surname_ambiguity_repair",
    "trim_long_given_token_noise",
    "trim_long_given_token_noise_with_token_repair",
}

MEDIUM_RISK_REPAIR_REASONS = {
    "country_code_ambiguity_repair",
    "drop_noise_before_country_code",
    "prefer_passport_filler_document_code",
}

LINE1_NAME_REWRITE_POSITIONS = {
    "surname",
    "given_name_zone",
    "given_name_token",
    "name",
}


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    prev = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        cur = [i]
        for j, rch in enumerate(right, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if lch == rch else 1),
            ))
        prev = cur
    return prev[-1]


def _substitution_count(left: str, right: str) -> int:
    if len(left) != len(right):
        return 0
    return sum(1 for src, dst in zip(left, right) if src != dst)


def _repair_risk_weight(repair: dict) -> tuple[float, str]:
    reason = (repair.get("reason") or "").strip()
    position = (repair.get("position") or "").strip()
    field = (repair.get("field") or "").strip()

    if position in LINE1_NAME_REWRITE_POSITIONS or reason in HIGH_RISK_REPAIR_REASONS:
        return 0.18, "high"
    if reason in MEDIUM_RISK_REPAIR_REASONS:
        return 0.10, "medium"
    if field == "line2":
        return 0.05, "low"
    return 0.07, "low"


def build_repair_confidence(repairs: list[dict]) -> dict:
    if not repairs:
        return {
            "penalty": 0.0,
            "score": 1.0,
            "risk": "none",
            "repair_count": 0,
            "edit_estimate": 0,
            "substitution_count": 0,
            "name_rewrite_count": 0,
            "checksum_backed_repair_count": 0,
            "warnings": [],
            "details": [],
        }

    total_penalty = 0.0
    total_edits = 0
    total_substitutions = 0
    name_rewrite_count = 0
    checksum_backed_repair_count = 0
    details = []
    risk_rank = "none"
    warnings = []
    rank_order = {"none": 0, "low": 1, "medium": 2, "high": 3}

    for repair in repairs:
        from_text = str(repair.get("from", "") or "")
        to_text = str(repair.get("to", "") or "")
        edit_distance = _edit_distance(from_text, to_text)
        substitutions = _substitution_count(from_text, to_text)
        base_penalty, risk = _repair_risk_weight(repair)
        repair_penalty = base_penalty + (edit_distance * 0.015) + (substitutions * 0.01)
        total_penalty += repair_penalty
        total_edits += edit_distance
        total_substitutions += substitutions
        if rank_order[risk] > rank_order[risk_rank]:
            risk_rank = risk

        if repair.get("position") in LINE1_NAME_REWRITE_POSITIONS:
            name_rewrite_count += 1
        if repair.get("field") == "line2":
            checksum_backed_repair_count += 1

        details.append({
            "field": repair.get("field"),
            "position": repair.get("position"),
            "reason": repair.get("reason"),
            "risk": risk,
            "edit_distance": edit_distance,
            "substitution_count": substitutions,
            "penalty": round(repair_penalty, 4),
        })

    if len(repairs) >= 2:
        warnings.append("multiple_repairs_applied")
    if name_rewrite_count > 0:
        warnings.append("line1_name_rewritten")

    total_penalty = max(0.0, min(total_penalty, 1.0))
    return {
        "penalty": round(total_penalty, 4),
        "score": round(max(0.0, 1.0 - total_penalty), 4),
        "risk": risk_rank,
        "repair_count": len(repairs),
        "edit_estimate": total_edits,
        "substitution_count": total_substitutions,
        "name_rewrite_count": name_rewrite_count,
        "checksum_backed_repair_count": checksum_backed_repair_count,
        "warnings": warnings,
        "details": details,
    }


def _generate_token_variants(token: str, max_edits: int = 2) -> set[str]:
    token = _sanitize_name_token(token)
    if not token:
        return {token}

    variants = {token}
    positions = [i for i, ch in enumerate(token) if ch in TOKEN_AMBIGUOUS_SUBS]

    for i in positions:
        ch = token[i]
        for repl in TOKEN_AMBIGUOUS_SUBS[ch]:
            variants.add(token[:i] + repl + token[i + 1:])

    if max_edits >= 2 and len(positions) >= 2:
        for i, j in itertools.combinations(positions, 2):
            for repl_i in TOKEN_AMBIGUOUS_SUBS[token[i]]:
                for repl_j in TOKEN_AMBIGUOUS_SUBS[token[j]]:
                    chars = list(token)
                    chars[i] = repl_i
                    chars[j] = repl_j
                    variants.add("".join(chars))

    # Allow one conservative insertion repair for short tokens where OCR likely
    # collapsed a vertical stroke inside a repeated-vowel cluster, e.g.
    # ADEEA -> ADEELA. This remains gated by downstream token-score gain.
    if 4 <= len(token) <= 6 and "L" not in token and "I" not in token:
        for i in range(1, len(token)):
            if token[i - 1] not in VOWELS or token[i] not in VOWELS:
                continue
            repeated_vowel_cluster = token[i - 1] == token[i] or (
                i >= 2 and token[i - 2] == token[i - 1]
            )
            if repeated_vowel_cluster:
                variants.add(token[:i] + "L" + token[i:])

    return {v for v in variants if v}


def _split_given_name_zone(given_raw: str) -> tuple[str, str]:
    raw_first, sep, raw_tail = given_raw.partition("<")
    given_token = _sanitize_name_token(raw_first)
    given_tail = _sanitize_name_zone(sep + raw_tail) if sep else ""
    return given_token, given_tail


def _preserved_double_letter_count(source: str, candidate: str) -> int:
    source = _sanitize_name_token(source)
    candidate = _sanitize_name_token(candidate)
    if not source or not candidate:
        return 0

    source_pairs = {
        source[i:i + 2]
        for i in range(len(source) - 1)
        if source[i] == source[i + 1]
    }
    if not source_pairs:
        return 0

    return sum(1 for pair in source_pairs if pair in candidate)


def _normalize_given_tail(given_tail: str) -> str:
    # Preserve explicit double separators inside the tail; collapsing them
    # can rewrite visible truth in multi-token given-name zones.
    if "<<" in (given_tail or "").strip("<"):
        return given_tail
    tail_tokens, tail_remainder = _split_tail_name_tokens(given_tail)
    if not tail_tokens:
        return given_tail
    return "<" + "<".join(tail_tokens) + tail_remainder


def _build_td3_line1(
    issuing_country: str,
    surname: str,
    given_token: str,
    given_tail: str = "",
    document_code: str = "P<",
) -> str:
    document_code = re.sub(r"[^A-Z<]", "", (document_code or "P<")).upper()
    if document_code.startswith("P") and len(document_code) >= 2:
        document_code = document_code[:2]
    elif document_code.startswith("P"):
        document_code = "P<"
    else:
        document_code = "P<"

    issuing_country = re.sub(r"[^A-Z<]", "", (issuing_country or ""))
    issuing_country = issuing_country[:3].ljust(3, "<")
    surname = _sanitize_name_token(surname)
    given_token = _sanitize_name_token(given_token)
    given_tail = _sanitize_name_zone(given_tail)

    if given_tail and not given_tail.startswith("<"):
        given_tail = "<" + given_tail

    rebuilt = f"{document_code}{issuing_country}{surname}<<{given_token}{given_tail}"
    return rebuilt[:MRZ_LINE_LEN].ljust(MRZ_LINE_LEN, "<")


def repair_given_name_token(token: str) -> tuple[str, dict]:
    raw = _sanitize_name_token(token)
    if not raw:
        return raw, {"changed": False, "from": token, "to": raw, "reason": "empty"}

    raw_score = _name_token_score(raw)

    # Preserve short, already-plausible clean tokens. These are high-risk to
    # "repair" heuristically and have caused visible-truth regressions such as
    # LIM -> IIN in line 1 surname handling.
    if len(raw) <= 3 and raw_score >= 12:
        return raw, {
            "changed": False,
            "from": raw,
            "to": raw,
            "reason": "short_token_preserved",
            "candidates_considered": 1,
            "best_score": raw_score,
        }

    candidates = {
        c for c in _generate_token_variants(raw, max_edits=2)
        if re.fullmatch(r"[A-Z]+", c)
    }
    if not candidates:
        candidates = {raw}
    ranked = sorted(
        ((cand, _name_token_score(cand)) for cand in candidates),
        key=lambda x: (
            x[1],
            _preserved_double_letter_count(raw, x[0]),
            -abs(len(x[0]) - 5),
            x[0][-1] in "AEIOUY",
            x[0].endswith("A"),
            x[0],
        ),
        reverse=True,
    )

    best, best_score = ranked[0]
    if best != raw and best_score < raw_score + MIN_TOKEN_REPAIR_SCORE_GAIN:
        best = raw
        best_score = raw_score

    changed = best != raw

    return best, {
        "changed": changed,
        "from": raw,
        "to": best,
        "reason": "token_ambiguity_repair",
        "candidates_considered": len(candidates),
        "best_score": best_score,
    }


def _build_filler_tail(length: int) -> str:
    if length <= 0:
        return ""
    return "<" * length


def repair_document_code(line1: str) -> tuple[str, dict | None]:
    line = normalize_td3_line1(line1)
    document_code = line[:2]

    if document_code != "PO":
        return line, None

    repaired = normalize_td3_line1("P<" + line[2:])
    if score_td3_line1(repaired) < score_td3_line1(line):
        return line, None

    return repaired, {
        "field": "line1",
        "position": "document_code",
        "from": document_code,
        "to": "P<",
        "reason": "prefer_passport_filler_document_code",
    }


def repair_surname_ambiguity(
    issuing_country: str,
    surname: str,
    given_token: str,
    given_tail: str,
    *,
    document_code: str,
) -> tuple[str, dict | None]:
    if not surname or not any(ch in {"M", "N"} for ch in surname):
        return surname, None

    baseline_line = _build_td3_line1(
        issuing_country,
        surname,
        given_token,
        given_tail,
        document_code=document_code,
    )
    baseline_score = score_td3_line1(baseline_line)

    candidates = {surname}
    for i, ch in enumerate(surname):
        if ch == "M":
            candidates.add(surname[:i] + "N" + surname[i + 1:])
        elif ch == "N":
            candidates.add(surname[:i] + "M" + surname[i + 1:])

    ranked = []
    for candidate in candidates:
        rebuilt = _build_td3_line1(
            issuing_country,
            candidate,
            given_token,
            given_tail,
            document_code=document_code,
        )
        ranked.append((
            score_td3_line1(rebuilt),
            _name_token_score(candidate),
            candidate.endswith("N"),
            -candidate.count("M"),
            candidate,
        ))

    _, _, _, _, best = max(ranked)
    if best == surname:
        return surname, None

    repaired_line = _build_td3_line1(
        issuing_country,
        best,
        given_token,
        given_tail,
        document_code=document_code,
    )
    repaired_score = score_td3_line1(repaired_line)
    if repaired_score < baseline_score:
        return surname, None

    if not (
        best.endswith("MAN")
        and surname.endswith("M")
        and not surname.endswith("MAN")
    ):
        return surname, None

    return best, {
        "field": "line1",
        "position": "surname",
        "from": surname,
        "to": best,
        "reason": "surname_ambiguity_repair",
        "score_gain": round(repaired_score - baseline_score, 2),
    }


def _candidate_given_zone_repairs(given_raw: str) -> list[dict]:
    raw_first, sep, raw_tail = given_raw.partition("<")
    raw_token = _sanitize_name_token(raw_first)
    raw_tail_zone = _sanitize_name_zone(sep + raw_tail) if sep else ""

    candidates: list[dict] = [{
        "given_token": raw_token,
        "given_tail": raw_tail_zone,
        "reason": "baseline",
    }]

    if 3 <= len(raw_token) <= 8:
        repaired_token, meta = repair_given_name_token(raw_token)
        if meta["changed"]:
            candidates.append({
                "given_token": repaired_token,
                "given_tail": raw_tail_zone,
                "reason": "token_ambiguity_repair",
            })

    if len(raw_token) > 8:
        max_prefix_len = min(12, len(raw_token))
        for prefix_len in range(3, max_prefix_len + 1):
            prefix = raw_token[:prefix_len]
            suffix_len = len(raw_token) - prefix_len
            filler_tail = _build_filler_tail(suffix_len + len(raw_tail_zone))
            candidates.append({
                "given_token": prefix,
                "given_tail": filler_tail,
                "reason": "trim_long_given_token_noise",
            })

            repaired_prefix, meta = repair_given_name_token(prefix)
            if meta["changed"]:
                candidates.append({
                    "given_token": repaired_prefix,
                    "given_tail": filler_tail,
                    "reason": "trim_long_given_token_noise_with_token_repair",
                })

    deduped = {}
    for candidate in candidates:
        key = (candidate["given_token"], candidate["given_tail"])
        deduped.setdefault(key, candidate)
    return list(deduped.values())


def repair_given_name_zone(
    issuing_country: str,
    surname: str,
    given_raw: str,
    document_code: str = "P<",
) -> tuple[str, str, dict | None]:
    baseline_token, baseline_tail = _split_given_name_zone(given_raw)
    baseline_line = _build_td3_line1(
        issuing_country,
        surname,
        baseline_token,
        baseline_tail,
        document_code=document_code,
    )
    baseline_score = score_td3_line1(baseline_line)

    ranked = []
    for candidate in _candidate_given_zone_repairs(given_raw):
        given_token = candidate["given_token"]
        given_tail = candidate["given_tail"]
        if not given_token:
            continue

        line = _build_td3_line1(
            issuing_country,
            surname,
            given_token,
            given_tail,
            document_code=document_code,
        )
        score = score_td3_line1(line)
        ranked.append((score, given_token, given_tail, candidate["reason"]))

    if not ranked:
        return baseline_token, baseline_tail, None

    best_score, best_token, best_tail, best_reason = max(
        ranked,
        key=lambda item: (
            item[0],
            _name_token_score(item[1]),
            _preserved_double_letter_count(baseline_token, item[1]),
            -len(item[2]),
            item[1].endswith("A"),
            item[1],
        ),
    )

    if (
        (best_token, best_tail) == (baseline_token, baseline_tail)
        or best_score < baseline_score + MIN_LINE1_ZONE_REPAIR_GAIN
    ):
        return baseline_token, baseline_tail, None

    return best_token, best_tail, {
        "field": "line1",
        "position": "given_name_zone",
        "from": f"{baseline_token}{baseline_tail}",
        "to": f"{best_token}{best_tail}",
        "reason": best_reason,
        "score_gain": round(best_score - baseline_score, 2),
    }


def repair_issuing_country_code(line1: str) -> tuple[str, dict | None]:
    line = normalize_td3_line1(line1)

    if not line.startswith("P"):
        return line, None

    document_code = line[:2]
    issuing_country = line[2:5]
    from mrz.td3.country_codes import is_valid_mrz_country_code

    if is_valid_mrz_country_code(issuing_country):
        return line, None

    shifted_country = line[3:6]
    if is_valid_mrz_country_code(shifted_country):
        repaired = normalize_td3_line1(document_code + line[3:])
        return repaired, {
            "field": "line1",
            "position": "issuing_country",
            "from": issuing_country,
            "to": shifted_country,
            "reason": "drop_noise_before_country_code",
        }

    valid_variants = [
        candidate
        for candidate in _generate_country_code_variants(issuing_country)
        if is_valid_mrz_country_code(candidate)
    ]
    if not valid_variants:
        return line, None

    best_country = max(
        valid_variants,
        key=lambda candidate: (
            score_td3_line1(document_code + candidate + line[5:]),
            candidate,
        ),
    )
    repaired = normalize_td3_line1(document_code + best_country + line[5:])
    return repaired, {
        "field": "line1",
        "position": "issuing_country",
        "from": issuing_country,
        "to": best_country,
        "reason": "country_code_ambiguity_repair",
    }


def repair_td3_line1(line1: str, *, trim_line1_spill_func) -> tuple[str, list[dict]]:
    repairs = []
    line = trim_line1_spill_func(line1)

    if not line.startswith("P"):
        return line, repairs

    line, document_code_repair = repair_document_code(line)
    if document_code_repair is not None:
        repairs.append(document_code_repair)

    document_code = line[:2]
    line, country_repair = repair_issuing_country_code(line)
    if country_repair is not None:
        repairs.append(country_repair)

    issuing_country = line[2:5]
    name_zone = line[5:]

    name_core = name_zone.rstrip("<")
    if "<<" not in name_core:
        return line, repairs

    surname_raw, given_core = name_core.split("<<", 1)
    given_raw = given_core + name_zone[len(name_core):]
    surname = _sanitize_name_token(surname_raw)
    given_token, given_tail = _split_given_name_zone(given_raw)

    if not surname:
        return line, repairs

    given_tail = _normalize_given_tail(given_tail)

    if 3 <= len(surname) <= 10:
        repaired_surname, meta = repair_given_name_token(surname)
        if meta["changed"]:
            repairs.append({
                "field": "line1",
                "position": "surname",
                "from": surname,
                "to": repaired_surname,
                "reason": meta["reason"],
                "candidates_considered": meta["candidates_considered"],
                "best_score": meta["best_score"],
            })
            surname = repaired_surname

    surname, surname_repair = repair_surname_ambiguity(
        issuing_country,
        surname,
        given_token,
        given_tail,
        document_code=document_code,
    )
    if surname_repair is not None:
        repairs.append(surname_repair)

    repaired_given_token, repaired_given_tail, given_zone_repair = repair_given_name_zone(
        issuing_country,
        surname,
        f"{given_token}{given_tail}",
        document_code=document_code,
    )
    if given_zone_repair is not None:
        given_token = repaired_given_token
        given_tail = repaired_given_tail
        repairs.append(given_zone_repair)

    rebuilt = _build_td3_line1(
        issuing_country,
        surname,
        given_token,
        given_tail,
        document_code=document_code,
    )

    original_line_score = score_td3_line1(line)
    rebuilt_line_score = score_td3_line1(rebuilt)
    late_tail_fragment = re.search(r"<{8,}[A-Z]{1,4}<{2,}$", given_raw) is not None
    if rebuilt != line:
        if not late_tail_fragment and (repairs or rebuilt_line_score > original_line_score):
            repairs.append({
                "field": "line1",
                "position": "name",
                "from": name_zone,
                "to": f"{surname}<<{given_token}{given_tail}",
                "reason": "name_noise_collapse",
            })
            line = rebuilt

    if 3 <= len(given_token) <= 8:
        repaired_token, meta = repair_given_name_token(given_token)
        if meta["changed"]:
            rebuilt2 = _build_td3_line1(
                issuing_country,
                surname,
                repaired_token,
                given_tail,
                document_code=document_code,
            )
            repairs.append({
                "field": "line1",
                "position": "given_name_token",
                "from": given_token,
                "to": repaired_token,
                "reason": meta["reason"],
                "candidates_considered": meta["candidates_considered"],
                "best_score": meta["best_score"],
            })
            line = rebuilt2

    return line, repairs


def repair_paddle_line1_candidate(line1: str, *, trim_line1_spill_func) -> tuple[str, dict | None]:
    line = trim_line1_spill_func(line1)

    if not line.startswith("P"):
        return line, None

    document_code = line[:2]
    name_zone = line[5:]
    name_core = name_zone.rstrip("<")
    if "<<" not in name_core:
        return line, None

    issuing_country = line[2:5]
    surname_raw, given_core = name_core.split("<<", 1)
    given_raw = given_core + name_zone[len(name_core):]
    surname = _sanitize_name_token(surname_raw)

    if not surname:
        return line, None

    given_token, given_tail = _split_given_name_zone(given_raw)
    given_tail = _normalize_given_tail(given_tail)

    baseline_score = score_td3_line1(line)
    ranked = []
    surname_variants = {surname}
    for i, ch in enumerate(surname):
        if ch == "N":
            surname_variants.add(surname[:i] + "M" + surname[i + 1:])
        elif ch == "M":
            surname_variants.add(surname[:i] + "N" + surname[i + 1:])

    for variant in surname_variants:
        rebuilt = _build_td3_line1(
            issuing_country,
            variant,
            given_token,
            given_tail,
            document_code=document_code,
        )
        ranked.append((
            score_td3_line1(rebuilt),
            variant.endswith("MAN"),
            variant.count("M") - variant.count("N"),
            variant,
            rebuilt,
        ))

    best_score, _, _, best_surname, best_line = max(ranked)
    if best_surname == surname or best_score < baseline_score:
        return line, None
    if best_score == baseline_score:
        changed_positions = [i for i, (a, b) in enumerate(zip(surname, best_surname)) if a != b]
        if (
            len(changed_positions) != 1
            or changed_positions[0] < len(surname) - 3
            or best_surname[-3:] not in {"MAN", "MAD"}
        ):
            return line, None

    return best_line, {
        "field": "line1",
        "position": "surname",
        "from": surname,
        "to": best_surname,
        "reason": "paddle_surname_ambiguity_repair",
        "score_gain": round(best_score - baseline_score, 2),
    }
