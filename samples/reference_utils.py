from __future__ import annotations


def normalize_reference_samples(payload: dict) -> dict[str, dict]:
    samples = payload.get("samples", {}) if isinstance(payload, dict) else {}

    if isinstance(samples, dict):
        return samples

    if isinstance(samples, list):
        normalized: dict[str, dict] = {}
        for item in samples:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename") or item.get("image")
            line1 = item.get("line1")
            line2 = item.get("line2")
            if not isinstance(filename, str) or not isinstance(line1, str) or not isinstance(line2, str):
                continue
            normalized[filename] = {
                "line1": line1,
                "line2": line2,
            }
        return normalized

    return {}
