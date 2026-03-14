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

MIN_TOKEN_REPAIR_SCORE_GAIN = 6.0
MIN_LINE1_ZONE_REPAIR_GAIN = 4.0


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

    return {v for v in variants if v}


def _split_given_name_zone(given_raw: str) -> tuple[str, str]:
    raw_first, sep, raw_tail = given_raw.partition("<")
    given_token = _sanitize_name_token(raw_first)
    given_tail = _sanitize_name_zone(sep + raw_tail) if sep else ""
    return given_token, given_tail


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

    candidates = {
        c for c in _generate_token_variants(raw, max_edits=2)
        if re.fullmatch(r"[A-Z]+", c)
    }
    if not candidates:
        candidates = {raw}
    ranked = sorted(
        ((cand, _name_token_score(cand)) for cand in candidates),
        key=lambda x: (x[1], -abs(len(x[0]) - 5), x[0][-1] in "AEIOUY"),
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
    if score_td3_line1(repaired) <= score_td3_line1(line):
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

    if "<<" not in name_zone:
        return line, repairs

    surname_raw, given_raw = name_zone.split("<<", 1)
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

    if rebuilt != line:
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
    if "<<" not in name_zone:
        return line, None

    issuing_country = line[2:5]
    surname_raw, given_raw = name_zone.split("<<", 1)
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

    return best_line, {
        "field": "line1",
        "position": "surname",
        "from": surname,
        "to": best_surname,
        "reason": "paddle_surname_ambiguity_repair",
        "score_gain": round(best_score - baseline_score, 2),
    }
