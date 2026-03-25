import importlib
import os
import sys
import unittest
import numpy as np
from unittest import mock

from document_preparation.passport import resize_aligned_image
from mrz.td3.ocr_pipeline import (
    _auto_requires_tesseract,
    _apply_candidate_support_bonus,
    _best_paddle_text_candidate,
    _rank_candidates,
    _line1_selection_penalty,
    _paddle_ocr_image,
    _pair_selection_key,
    _prepare_variants,
    _repair_paddle_line1_candidate,
    _resolve_ocr_backends,
    _resolve_paddle_use_gpu,
    _run_auto_paddle_batches,
    _trim_line1_spill,
    build_split_candidates,
    estimate_ocr_search_space,
    is_valid_mrz_country_code,
    normalize_td3_line1,
    pair_consistency_bonus,
    repair_given_name_zone,
    repair_given_name_token,
    repair_issuing_country_code,
    repair_td3_line1,
    score_td3_line1,
    score_td3_line2,
    validate_and_correct_mrz,
    validate_td3_checks,
)
from mrz.td3.checksums import build_checksum_confidence
from mrz.td3.checksums import DOC_NUMBER_AMBIGUOUS_SUBS, correct_field
from mrz.td3.ocr_runner import extract_ocr_confidence
from mrz.td3.repair import build_repair_confidence
from mrz.td3.score import (
    assemble_td3_confidence,
    build_candidate_margin_confidence,
    build_structure_confidence,
)
from mrz.td3.detect import merge_bboxes, prepare_detection_roi, scale_bboxes_back
from ocr_backends.paddle_backend import (
    build_paddle_ocr_kwargs,
    get_paddle_ocr_stats,
    paddle_ocr_images,
    reset_paddle_ocr_cache,
    reset_paddle_ocr_stats,
)


class TestLine1Repair(unittest.TestCase):
    def test_resize_aligned_image_caps_oversized_input(self) -> None:
        img = np.zeros((6752, 12000, 3), dtype=np.uint8)

        resized, meta = resize_aligned_image(img, max_dim=2400)

        self.assertTrue(meta["resized"])
        self.assertEqual(meta["original_width"], 12000)
        self.assertEqual(meta["original_height"], 6752)
        self.assertEqual(max(resized.shape[:2]), 2400)
        self.assertEqual(meta["width"], resized.shape[1])
        self.assertEqual(meta["height"], resized.shape[0])

    def test_merge_bboxes_keeps_single_detected_line_bbox(self) -> None:
        merged = merge_bboxes([(100, 900, 1000, 40)])

        self.assertEqual(merged, (100, 900, 1000, 40))

    def test_normalize_td3_line1_preserves_two_character_document_code(self) -> None:
        line = normalize_td3_line1("POJORALMOUSA<<AWS<WAJDI<FAROUR<<<<<<<<<<<<<")

        self.assertEqual(line[:5], "POJOR")

    def test_preserves_valid_name_ending_in_t(self) -> None:
        line = normalize_td3_line1("P<PAKALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<")

        repaired, repairs = repair_td3_line1(line)

        self.assertEqual(repaired, line)
        self.assertEqual(repairs, [])

    def test_preserves_multiple_given_names(self) -> None:
        line = normalize_td3_line1("P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<")

        repaired, repairs = repair_td3_line1(line)

        self.assertEqual(repaired, line)
        self.assertEqual(repairs, [])

    def test_single_separator_name_zone_is_not_rewritten_from_filler_tail(self) -> None:
        line = "PRMDVYOOSUF<AZYAN<FARHATH<<<<<<<<<<<<<<<<<<<"

        repaired, repairs = repair_td3_line1(line)

        self.assertEqual(repaired, line)
        self.assertEqual(repairs, [])

    def test_keeps_plausible_name_when_score_gain_is_small(self) -> None:
        repaired, meta = repair_given_name_token("SHAHADAT")

        self.assertEqual(repaired, "SHAHADAT")
        self.assertFalse(meta["changed"])

    def test_repair_given_name_token_prefers_l_over_i_for_adeela(self) -> None:
        repaired, meta = repair_given_name_token("ADEEIA")

        self.assertEqual(repaired, "ADEELA")
        self.assertTrue(meta["changed"])

    def test_repair_given_name_token_restores_missing_l_for_adeela(self) -> None:
        repaired, meta = repair_given_name_token("ADEEA")

        self.assertEqual(repaired, "ADEELA")
        self.assertTrue(meta["changed"])

    def test_repair_given_name_token_preserves_short_clean_surname(self) -> None:
        repaired, meta = repair_given_name_token("LIM")

        self.assertEqual(repaired, "LIM")
        self.assertFalse(meta["changed"])

    def test_repair_given_name_zone_trims_ocr_noise_and_restores_sadia(self) -> None:
        token, tail, meta = repair_given_name_zone(
            "PAK",
            "WAQAR",
            "SADTAKKKKKEEEKKKRRRRRECEK<<<<<<<",
        )

        self.assertEqual(token, "SADIA")
        self.assertTrue(tail.startswith("<"))
        self.assertIsNotNone(meta)
        self.assertIn("trim_long_given_token_noise", meta["reason"])

    def test_repair_td3_line1_collapses_noisy_tail_after_given_name(self) -> None:
        repaired, repairs = repair_td3_line1(
            "P<PAKWAQAR<<SADTAKKKKKEEEKKKRRRRRECEK<<<<<<<"
        )

        self.assertEqual(
            repaired,
            normalize_td3_line1("P<PAKWAQAR<<SADIA"),
        )
        self.assertTrue(any(r["position"] == "given_name_zone" for r in repairs))

    def test_preserves_reference_given_names_for_ww_sample(self) -> None:
        line = "PPMLIDIALLO<<FATOUMATA<ZAHARA<<<<<<<<<<<<<<<"

        repaired, _ = repair_td3_line1(line)

        self.assertEqual(repaired, normalize_td3_line1(line))

    def test_preserves_visual_pb_prefix_for_sri_lankan_sample(self) -> None:
        line = "PBLKACOLAMBAGE<<KANISHKA<NETHMI<COLAMBAGE<<<"

        repaired, repairs = repair_td3_line1(line)

        self.assertEqual(repaired, normalize_td3_line1(line))
        self.assertEqual(repairs, [])

    def test_preserves_visual_pp_prefix_for_sri_lankan_sample(self) -> None:
        line = "PPLKAPERERA<<THANTHREEGE<SHINE<INNOCENT<<<<<"

        repaired, repairs = repair_td3_line1(line)

        self.assertEqual(repaired, normalize_td3_line1(line))
        self.assertEqual(repairs, [])

    def test_preserves_reference_given_names_for_xcxc_sample(self) -> None:
        line = "P<GINCAMARA<<FATOUMATA<BOBO<<<<<<<<<<<<<<<<<"

        repaired, repairs = repair_td3_line1(line)

        self.assertEqual(repaired, normalize_td3_line1(line))
        self.assertEqual(repairs, [])

    def test_repairs_po_prefix_to_passport_filler_for_xcxc_sample(self) -> None:
        repaired, repairs = repair_td3_line1(
            "POGINCAMARA<<FATOUMATA<BOBO<<<<<<<<<<<<<<<<<"
        )

        self.assertEqual(
            repaired,
            normalize_td3_line1("P<GINCAMARA<<FATOUMATA<BOBO<<<<<<<<<<<<<<<<<"),
        )
        self.assertTrue(any(r["position"] == "document_code" for r in repairs))

    def test_repairs_accidental_double_separator_inside_given_names(self) -> None:
        repaired, repairs = repair_td3_line1(
            "P<JORTAWALBEH<<NAKA<AHMAD<ABDEL<<RAHMAN<<<<<"
        )

        self.assertEqual(
            repaired,
            normalize_td3_line1("P<JORTAWALBEH<<NAKA<AHMAD<ABDEL<<RAHMAN<<<<<"),
        )
        self.assertEqual(repairs, [])

    def test_repairs_surname_mn_ambiguity_for_rahman(self) -> None:
        repaired, repairs = repair_td3_line1(
            "P<BGDRAHMAM<<MD<MAHBUBUR<<<<<<<<<<<<<<<<<<<<"
        )

        self.assertEqual(
            repaired,
            normalize_td3_line1("P<BGDRAHMAN<<MD<MAHBUBUR<<<<<<<<<<<<<<<<<<<<"),
        )
        self.assertTrue(any(r["reason"] == "surname_ambiguity_repair" for r in repairs))

    def test_score_prefers_full_dzhakhongir_over_trimmed_dzhakho(self) -> None:
        full = score_td3_line1("P<RUSAMIRKULOV<<DZHAKHONGIR<<<<<<<<<<<<<<<<<")
        trimmed = score_td3_line1("P<RUSAMIRKULOV<<DZHAKHO<<<<<<<<<<<<<<<<<<<<<")

        self.assertGreater(full, trimmed)

    def test_repairs_noise_before_country_code(self) -> None:
        repaired, meta = repair_issuing_country_code(
            "P<SJORALMOUSA<<AWS<WAJDI<FAROUR<<<<<<<<<<<<<"
        )

        self.assertEqual(repaired[:5], "P<JOR")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["reason"], "drop_noise_before_country_code")

    def test_score_prefers_valid_issuing_country_code(self) -> None:
        valid = score_td3_line1("P<JORALMOUSA<<AWS<WAJDI<FAROUR<<<<<<<<<<<<<")
        invalid = score_td3_line1("P<SJORALMOUSA<<AWS<WAJDI<FAROUR<<<<<<<<<<<<<")

        self.assertGreater(valid, invalid)

    def test_score_prefers_passport_filler_prefix_over_unknown_pc_prefix(self) -> None:
        expected = score_td3_line1("P<JORALMOUSA<<AWS<WAJDI<FAROUQ<<<<<<<<<<<<<<")
        weak = score_td3_line1("PCJORALMOUSA<<AWS<WAJDI<FAROUQ<<<<<<<<<<<<<<")

        self.assertGreater(expected, weak)

    def test_score_prefers_passport_filler_prefix_over_po_for_standard_passport_line(self) -> None:
        expected = score_td3_line1("P<GINCAMARA<<FATOUMATA<BOBO<<<<<<<<<<<<<<<<<")
        weak = score_td3_line1("POGINCAMARA<<FATOUMATA<BOBO<<<<<<<<<<<<<<<<<")

        self.assertGreater(expected, weak)

    def test_score_prefers_adeela_over_adeea_for_first_given_name_token(self) -> None:
        expected = score_td3_line1("P<BGDAHMAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<")
        weak = score_td3_line1("P<BGDAHMAD<<ADEEA<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        self.assertGreater(expected, weak)

    def test_score_accepts_visual_pb_and_pp_prefixes_as_plausible(self) -> None:
        pb = score_td3_line1("PBLKACOLAMBAGE<<KANISHKA<NETHMI<COLAMBAGE<<<")
        pp = score_td3_line1("PPLKAPERERA<<THANTHREEGE<SHINE<INNOCENT<<<<<")
        unknown = score_td3_line1("PCLKAPERERA<<THANTHREEGE<SHINE<INNOCENT<<<<<")

        self.assertGreater(pb, unknown)
        self.assertGreater(pp, unknown)

    def test_score_prefers_valid_multi_token_given_names_over_trimmed_variant(self) -> None:
        full = score_td3_line1("P<MLIDIALLO<<FATOUMATA<ZAHARA<<<<<<<<<<<<<<<")
        trimmed = score_td3_line1("P<MLIDIALLO<<FATOUMA<<<<<<<<<<<<<<<<<<<<<<<<")

        self.assertGreater(full, trimmed)

    def test_score_penalizes_short_garbage_tail_fragments_after_valid_given_name(self) -> None:
        clean = score_td3_line1("P<BGDALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<")
        noisy = score_td3_line1("P<BGDALAM<<SHAHADAT<<<<CCC<<<CCCCC<KCAK<<<<<")

        self.assertGreater(clean, noisy)

    def test_auto_skips_tesseract_when_paddle_split_is_strong(self) -> None:
        with mock.patch("mrz.td3.ocr_pipeline.OCR_BACKEND", "auto"):
            line1_candidates = [{
                "text": "P<BGDRAHMAN<<MD<MAHBUBUR<<<<<<<<<<<<<<<<<<<<",
                "score": 108.0,
            }]
            line2_candidates = [{
                "text": "A132435974BGD0207278M33112041030761520<<<<18",
                "score": 151.0,
                "checks": {
                    "passed_count": 5,
                    "composite_valid": True,
                },
            }]

            self.assertFalse(_auto_requires_tesseract(line1_candidates, line2_candidates))

    def test_auto_uses_tesseract_when_paddle_line2_checks_are_weak(self) -> None:
        with mock.patch("mrz.td3.ocr_pipeline.OCR_BACKEND", "auto"):
            line1_candidates = [{
                "text": "P<BGDRAHMAN<<MD<MAHBUBUR<<<<<<<<<<<<<<<<<<<<",
                "score": 108.0,
            }]
            line2_candidates = [{
                "text": "A132435974BGD0207278M33112041030761520<<<<18",
                "score": 145.0,
                "checks": {
                    "passed_count": 4,
                    "composite_valid": False,
                },
            }]

            self.assertTrue(_auto_requires_tesseract(line1_candidates, line2_candidates))

    def test_auto_batches_all_paddle_line_variants_by_line_kind(self) -> None:
        prepared_splits = [
            {
                "line1_variants": [{"variant_id": "l1a", "meta": {}, "image": np.zeros((4, 8), dtype=np.uint8)}],
                "line2_variants": [{"variant_id": "l2a", "meta": {}, "image": np.zeros((4, 8), dtype=np.uint8)}],
            },
            {
                "line1_variants": [{"variant_id": "l1b", "meta": {}, "image": np.zeros((4, 8), dtype=np.uint8)}],
                "line2_variants": [{"variant_id": "l2b", "meta": {}, "image": np.zeros((4, 8), dtype=np.uint8)}],
            },
        ]

        def fake_paddle_ocr_images(images, *, line_kind, **kwargs):
            if line_kind == "line1":
                return ["LINE1_A", "LINE1_B"]
            return ["LINE2_A", "LINE2_B"]

        with mock.patch("mrz.td3.ocr_pipeline._paddle_ocr_images_impl", side_effect=fake_paddle_ocr_images):
            grouped = _run_auto_paddle_batches(prepared_splits)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["line1"][0]["text_raw"], "LINE1_A")
        self.assertEqual(grouped[1]["line1"][0]["text_raw"], "LINE1_B")
        self.assertEqual(grouped[0]["line2"][0]["text_raw"], "LINE2_A")
        self.assertEqual(grouped[1]["line2"][0]["text_raw"], "LINE2_B")

    def test_accepts_uto_mrz_country_code(self) -> None:
        self.assertTrue(is_valid_mrz_country_code("UTO"))

    def test_support_bonus_prefers_consensus_line1_candidate(self) -> None:
        expected = "P<JORALMOUSA<<AWS<WAJDI<FAROUQ<<<<<<<<<<<<<<"
        wrong = "P<JORALMOUSA<<AUS<WAJDI<FAROUR<<<<<<<<<<<<<<"
        candidates = [
            {"text": expected, "score": 82.5}
            for _ in range(18)
        ] + [
            {"text": wrong, "score": 86.5}
            for _ in range(5)
        ]

        _apply_candidate_support_bonus(candidates)

        best = max(candidates, key=lambda cand: cand["score"])

        self.assertEqual(best["text"], expected)
        self.assertGreater(best["score"], max(c["score"] for c in candidates if c["text"] == wrong))


class TestLine2Checks(unittest.TestCase):
    def test_valid_td3_line2_reports_composite_check(self) -> None:
        checks = validate_td3_checks("A098919372BGD0601038M36020815119012168<<<<26")

        self.assertEqual(checks["passed_count"], 5)
        self.assertTrue(checks["composite_valid"])
        self.assertEqual(checks["composite"]["expected"], "6")

    def test_pair_consistency_bonus_rewards_matching_country(self) -> None:
        bonus = pair_consistency_bonus(
            "P<JORALMOUSA<<AWS<WAJDI<FAROUR<<<<<<<<<<<<<<",
            "T1493973<9JOR0706265M31030839000186766<<<<56",
        )

        self.assertGreater(bonus, 0.0)

    def test_pair_consistency_bonus_ignores_mismatched_country(self) -> None:
        bonus = pair_consistency_bonus(
            "P<SJORALMOUSA<<AUS<WAJDI<FAROUR<<<<<<<<<<<<<",
            "T1493973<9JOR0706265M31030839000186766<<<<56",
        )

        self.assertEqual(bonus, 0.0)

    def test_validate_and_correct_mrz_normalizes_numeric_date_fields(self) -> None:
        _, repaired, checks = validate_and_correct_mrz(
            "",
            "0Q07341557GIN00L2083F27051142001208017021578",
        )

        self.assertEqual(repaired[13:19], "001208")
        self.assertEqual(checks["passed_count"], 5)

class TestPostSelectionConfidence(unittest.TestCase):
    def test_build_checksum_confidence_warns_on_composite_failure(self) -> None:
        checks = validate_td3_checks("A098919372BGD0601038M36020815119012168<<<<20")

        confidence = build_checksum_confidence(checks)

        self.assertFalse(confidence["composite_valid"])
        self.assertIn("composite_checksum_failed", confidence["warnings"])
        self.assertLess(confidence["score"], 1.0)

    def test_build_repair_confidence_flags_multiple_name_repairs(self) -> None:
        repairs = [
            {
                "field": "line1",
                "position": "given_name_zone",
                "from": "SADTAKKKKK",
                "to": "SADIA<<<<",
                "reason": "trim_long_given_token_noise",
            },
            {
                "field": "line1",
                "position": "name",
                "from": "WAQAR<<SADTAKKKKK",
                "to": "WAQAR<<SADIA<<<<",
                "reason": "name_noise_collapse",
            },
        ]

        confidence = build_repair_confidence(repairs)

        self.assertEqual(confidence["risk"], "high")
        self.assertIn("multiple_repairs_applied", confidence["warnings"])
        self.assertIn("line1_name_rewritten", confidence["warnings"])
        self.assertGreater(confidence["penalty"], 0.0)

    def test_candidate_margin_confidence_handles_single_candidate_without_warning(self) -> None:
        best_pair = {
            "line1": {"text": "L1", "variant_id": "v1", "psm": "p1", "split_label": "projection", "score": 120.0},
            "line2": {
                "text": "A098919372BGD0601038M36020815119012168<<<<26",
                "variant_id": "v2",
                "psm": "p2",
                "split_label": "projection",
                "score": 150.0,
                "checks": validate_td3_checks("A098919372BGD0601038M36020815119012168<<<<26"),
            },
            "pair_score": 290.0,
        }

        confidence = build_candidate_margin_confidence(
            best_pair,
            [best_pair["line1"]],
            [best_pair["line2"]],
        )

        self.assertGreaterEqual(confidence["score"], 0.7)
        self.assertFalse(confidence["line1_runner_up_available"])
        self.assertFalse(confidence["line2_runner_up_available"])
        self.assertFalse(confidence["pair_runner_up_available"])
        self.assertEqual(confidence["warnings"], [])

    def test_assemble_td3_confidence_adds_tension_and_missing_ocr_warnings(self) -> None:
        line1 = "P<PAKWAQAR<<SADIA<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "A098919372BGD0601038M36020815119012168<<<<20"
        checks = validate_td3_checks(line2)
        checksum_confidence = build_checksum_confidence(checks)
        structure_confidence = build_structure_confidence(line1, line2)
        repair_confidence = build_repair_confidence([
            {
                "field": "line1",
                "position": "name",
                "from": "WAQAR<<SADTAKKKKK",
                "to": "WAQAR<<SADIA<<<<",
                "reason": "name_noise_collapse",
            }
        ])
        margin_confidence = {
            "score": 0.2,
            "line1_margin": 1.0,
            "line2_margin": 0.5,
            "pair_margin": 0.7,
            "line1_margin_score": 0.0,
            "line2_margin_score": 0.0,
            "pair_margin_score": 0.0,
            "line1_runner_up_available": True,
            "line2_runner_up_available": True,
            "pair_runner_up_available": True,
            "warnings": ["selected_candidate_margin_small"],
        }

        confidence = assemble_td3_confidence(
            checksum_confidence=checksum_confidence,
            structure_confidence=structure_confidence,
            repair_confidence=repair_confidence,
            margin_confidence=margin_confidence,
            ocr_confidence=None,
            selected_meta={"line1_score": 80.0, "line2_score": 150.0, "pair_score": 230.0},
        )

        self.assertIn("composite_checksum_failed", confidence["warnings"])
        self.assertIn("repair_checksum_tension", confidence["warnings"])
        self.assertIn("ocr_confidence_unavailable", confidence["warnings"])
        self.assertIn("selected_candidate_margin_small", confidence["warnings"])
        self.assertTrue(confidence["suspicious"])
        self.assertGreaterEqual(confidence["final_score"], 0.0)
        self.assertLessEqual(confidence["final_score"], 0.65)

    def test_extract_ocr_confidence_marks_unavailable_without_real_backend_data(self) -> None:
        confidence = extract_ocr_confidence({"backend": "paddle", "text": "ABC"})

        self.assertFalse(confidence["available"])
        self.assertIn("ocr_confidence_unavailable", confidence["warnings"])

    def test_extract_ocr_confidence_uses_real_paddle_line_confidence(self) -> None:
        confidence = extract_ocr_confidence({
            "backend": "paddle",
            "text": "ABC",
            "line_confidence": 0.8732,
            "ocr_confidence_source": "line",
        })

        self.assertTrue(confidence["available"])
        self.assertEqual(confidence["source"], "line")
        self.assertEqual(confidence["score"], 0.8732)
        self.assertEqual(confidence["warnings"], [])

    def test_clean_duplicate_same_text_support_does_not_trigger_small_margin_warning(self) -> None:
        line1 = "P<BGDALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "A098919372BGD0601038M36020815119012168<<<<26"
        checks = validate_td3_checks(line2)
        best_pair = {
            "line1": {"text": line1, "variant_id": "l1a", "psm": "p1", "split_label": "projection", "score": 112.0, "support_count": 3},
            "line2": {"text": line2, "variant_id": "l2a", "psm": "p1", "split_label": "projection", "score": 158.0, "checks": checks},
            "pair_score": 290.0,
        }
        line1_ranked = [
            best_pair["line1"],
            {"text": line1, "variant_id": "l1b", "psm": "p2", "split_label": "projection", "score": 111.5, "support_count": 3},
            {"text": "P<BGDALAM<<SHAHADAT<<<<CCC<<<CCCCC<KCAK<<<<<", "variant_id": "l1c", "psm": "p3", "split_label": "projection", "score": 101.0},
        ]
        line2_ranked = [
            best_pair["line2"],
            {"text": line2, "variant_id": "l2b", "psm": "p2", "split_label": "projection", "score": 157.8, "checks": checks},
            {"text": "A098919372BGD0601038M36020815119012168<<<<20", "variant_id": "l2c", "psm": "p3", "split_label": "projection", "score": 141.0, "checks": validate_td3_checks("A098919372BGD0601038M36020815119012168<<<<20")},
        ]

        confidence = build_candidate_margin_confidence(best_pair, line1_ranked, line2_ranked)

        self.assertEqual(confidence["warnings"], [])
        self.assertGreaterEqual(confidence["line1_same_text_support"], 2)
        self.assertGreaterEqual(confidence["line2_same_text_support"], 2)

    def test_missing_ocr_confidence_does_not_inflate_score(self) -> None:
        line1 = "P<BGDALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "A098919372BGD0601038M36020815119012168<<<<26"
        checks = validate_td3_checks(line2)
        payload = assemble_td3_confidence(
            checksum_confidence=build_checksum_confidence(checks),
            structure_confidence=build_structure_confidence(line1, line2),
            repair_confidence=build_repair_confidence([]),
            margin_confidence={"score": 1.0, "warnings": []},
            ocr_confidence=None,
            selected_meta={"line1_score": 100.0, "line2_score": 150.0, "pair_score": 270.0},
        )

        self.assertEqual(payload["effective_weights"]["ocr_quality"], 0.0)
        self.assertIsNone(payload["details"]["ocr_confidence_effective_score"])
        self.assertIn("ocr_confidence_unavailable", payload["warnings"])

    def test_exact_match_clean_case_lands_in_high_confidence_band(self) -> None:
        line1 = "P<BGDALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "A098919372BGD0601038M36020815119012168<<<<26"
        checks = validate_td3_checks(line2)
        best_pair = {
            "line1": {"text": line1, "variant_id": "l1a", "psm": "p1", "split_label": "projection", "score": 112.0, "support_count": 3},
            "line2": {"text": line2, "variant_id": "l2a", "psm": "p1", "split_label": "projection", "score": 158.0, "checks": checks},
            "pair_score": 290.0,
        }
        line1_ranked = [
            best_pair["line1"],
            {"text": line1, "variant_id": "l1b", "psm": "p2", "split_label": "projection", "score": 111.5, "support_count": 3},
            {"text": "P<BGDALAM<<SHAHADAT<<<<CCC<<<CCCCC<KCAK<<<<<", "variant_id": "l1c", "psm": "p3", "split_label": "projection", "score": 101.0},
        ]
        line2_ranked = [
            best_pair["line2"],
            {"text": line2, "variant_id": "l2b", "psm": "p2", "split_label": "projection", "score": 157.8, "checks": checks},
            {"text": "A098919372BGD0601038M36020815119012168<<<<20", "variant_id": "l2c", "psm": "p3", "split_label": "projection", "score": 141.0, "checks": validate_td3_checks("A098919372BGD0601038M36020815119012168<<<<20")},
        ]

        payload = assemble_td3_confidence(
            checksum_confidence=build_checksum_confidence(checks),
            structure_confidence=build_structure_confidence(line1, line2),
            repair_confidence=build_repair_confidence([]),
            margin_confidence=build_candidate_margin_confidence(best_pair, line1_ranked, line2_ranked),
            ocr_confidence=None,
            selected_meta={"line1_score": 112.0, "line2_score": 158.0, "pair_score": 290.0},
        )

        self.assertGreaterEqual(payload["final_score"], 0.9)
        self.assertFalse(payload["suspicious"])


class TestStage2DetectionScaling(unittest.TestCase):
    def test_prepare_detection_roi_upscales_small_inputs(self) -> None:
        gray = np.zeros((350, 512), dtype=np.uint8)

        roi, roi_top, upscale = prepare_detection_roi(gray)

        self.assertEqual(roi_top, int(350 * 0.75))
        self.assertGreater(upscale, 1.0)
        self.assertAlmostEqual(upscale, 216 / (350 - int(350 * 0.75)), places=2)
        self.assertEqual(roi.shape[1], int(round(512 * upscale)))

    def test_prepare_detection_roi_does_not_upscale_medium_image_with_tall_roi(self) -> None:
        gray = np.zeros((921, 687), dtype=np.uint8)

        roi, roi_top, upscale = prepare_detection_roi(gray)

        self.assertEqual(roi_top, int(921 * 0.75))
        self.assertEqual(upscale, 1.0)
        self.assertEqual(roi.shape[1], 687)

    def test_prepare_detection_roi_does_not_upscale_medium_width_even_if_roi_is_short(self) -> None:
        gray = np.zeros((500, 687), dtype=np.uint8)

        roi, _, upscale = prepare_detection_roi(gray)

        self.assertEqual(upscale, 1.0)
        self.assertEqual(roi.shape[1], 687)

    def test_scale_bboxes_back_maps_detection_boxes_to_original_size(self) -> None:
        bboxes = [(77, 480, 479, 101)]

        scaled = scale_bboxes_back(bboxes, 1.76)

        self.assertEqual(scaled[0][0], 44)
        self.assertEqual(scaled[0][1], 273)

    def test_validate_and_correct_mrz_repairs_document_number_q_to_o_when_checksum_matches(self) -> None:
        _, repaired, checks = validate_and_correct_mrz(
            "",
            "Q0Q7341557GIN0012083F27051142001208017021578",
        )

        self.assertEqual(repaired[:10], "O007341557")
        self.assertEqual(checks["passed_count"], 5)

    def test_validate_and_correct_mrz_repairs_nationality_zero_to_letter_o(self) -> None:
        _, repaired, _ = validate_and_correct_mrz(
            "",
            "T1493973<9J0R0706265M31030839000186766<<<<56",
        )

        self.assertEqual(repaired[10:13], "JOR")

    def test_score_td3_line2_prefers_fewer_ambiguous_doc_chars(self) -> None:
        cleaner_score, _ = score_td3_line2(
            "0Q07341557GIN0012083F27051142001208017021578"
        )
        noisier_score, _ = score_td3_line2(
            "O00734L557GIN0012083F27051142001208017021578"
        )

        self.assertGreater(cleaner_score, noisier_score)


class TestPairSelection(unittest.TestCase):
    def test_rank_candidates_prefers_cleaner_same_text_instance(self) -> None:
        candidates = [
            {
                "text": "P<BGDAHMAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<",
                "score": 161.9,
                "repairs": [
                    {"reason": "country_code_ambiguity_repair"},
                    {"reason": "paddle_surname_ambiguity_repair"},
                ],
            },
            {
                "text": "P<BGDAHMAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<",
                "score": 161.9,
                "repairs": [],
            },
            {
                "text": "P<BGDAHMAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<",
                "score": 161.9,
                "repairs": [{"reason": "token_ambiguity_repair"}],
            },
        ]

        ranked = _rank_candidates(candidates, "score")

        self.assertEqual(ranked[0]["repairs"], [])
        self.assertEqual(ranked[1]["repairs"], [{"reason": "token_ambiguity_repair"}])

    def test_pair_selection_prefers_lower_candidate_rank_over_lexicographic_text(self) -> None:
        better_pair = {
            "pair_score": 294.0,
            "pair_bonus": 10.0,
            "line1": {
                "score": 114.0,
                "candidate_rank": 0,
                "text": "P<BGDALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<",
            },
            "line2": {
                "score": 150.0,
                "candidate_rank": 0,
                "text": "A098919372BGD0601038M36020815119012168<<<<26",
            },
        }
        worse_pair = {
            "pair_score": 294.0,
            "pair_bonus": 10.0,
            "line1": {
                "score": 114.0,
                "candidate_rank": 5,
                "text": "P<BGDALAN<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<",
            },
            "line2": {
                "score": 150.0,
                "candidate_rank": 0,
                "text": "A098919372BGD0601038M36020815119012168<<<<26",
            },
        }

        self.assertGreater(_pair_selection_key(better_pair), _pair_selection_key(worse_pair))

    def test_pair_selection_uses_line2_ocr_confidence_as_tiebreak(self) -> None:
        lower_conf_pair = {
            "pair_score": 294.0,
            "pair_bonus": 10.0,
            "line1": {
                "score": 114.0,
                "candidate_rank": 0,
                "text": "P<BGDALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<",
            },
            "line2": {
                "score": 150.0,
                "candidate_rank": 0,
                "text": "A098919372BGD0601038M36020815119012168<<<<26",
                "line_confidence": 0.82,
            },
        }
        higher_conf_pair = {
            "pair_score": 294.0,
            "pair_bonus": 10.0,
            "line1": {
                "score": 114.0,
                "candidate_rank": 0,
                "text": "P<BGDALAM<<SHAHADAT<<<<<<<<<<<<<<<<<<<<<<<<<",
            },
            "line2": {
                "score": 150.0,
                "candidate_rank": 0,
                "text": "A098919372BGD0601038M36020815119012168<<<<26",
                "line_confidence": 0.97,
            },
        }

        self.assertGreater(_pair_selection_key(higher_conf_pair), _pair_selection_key(lower_conf_pair))

    def test_paddle_spill_penalty_demotes_spill_trimmed_line1_candidates(self) -> None:
        paddle_candidate = {
            "backend": "paddle",
            "spill_trimmed": True,
            "repairs": [{"reason": "name_noise_collapse"}],
        }
        tesseract_candidate = {
            "backend": "tesseract",
            "spill_trimmed": True,
            "repairs": [{"reason": "name_noise_collapse"}],
        }

        self.assertGreater(_line1_selection_penalty(paddle_candidate), 0.0)
        self.assertEqual(_line1_selection_penalty(tesseract_candidate), 0.0)


class TestOcrSearchProfiles(unittest.TestCase):
    def test_fast_profile_limits_variant_generation(self) -> None:
        profile = {
            "variant_sources": ("gray", "clahe"),
            "variant_scales": (2,),
            "variant_thresholds": ("otsu",),
            "split_labels": ("projection", "half"),
            "variant_workers": 2,
        }
        image = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        variants = _prepare_variants(image, "line1_test", profile)

        self.assertEqual(len(variants), 2)
        self.assertEqual(
            {variant["variant_id"] for variant in variants},
            {
                "line1_test_gray_s2_otsu",
                "line1_test_clahe_s2_otsu",
            },
        )

    def test_parallel_variant_generation_preserves_order(self) -> None:
        profile = {
            "variant_sources": ("gray", "clahe"),
            "variant_scales": (2,),
            "variant_thresholds": ("otsu", "adaptive"),
            "split_labels": ("projection", "half"),
            "variant_workers": 2,
        }
        image = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        variants = _prepare_variants(image, "line1_test", profile)

        self.assertEqual(
            [variant["variant_id"] for variant in variants],
            [
                "line1_test_gray_s2_otsu",
                "line1_test_gray_s2_adaptive",
                "line1_test_clahe_s2_otsu",
                "line1_test_clahe_s2_adaptive",
            ],
        )

    def test_fast_profile_limits_split_candidates(self) -> None:
        profile = {
            "variant_sources": ("gray", "clahe"),
            "variant_scales": (2,),
            "variant_thresholds": ("otsu",),
            "split_labels": ("projection", "half"),
        }
        gray = np.zeros((20, 10), dtype=np.uint8)

        candidates = build_split_candidates(gray, {"split_y": 9}, profile)

        self.assertEqual(
            [candidate["label"] for candidate in candidates],
            ["projection", "half"],
        )

    def test_default_search_space_estimate_matches_exhaustive_profile(self) -> None:
        search_space = estimate_ocr_search_space()

        if search_space["fast_ocr"]:
            self.assertEqual(search_space["psms"], [7])
            self.assertEqual(search_space["split_count"], 2)
            self.assertEqual(search_space["per_line_variants"], 2)
            self.assertEqual(search_space["total_tesseract_calls"], 8)
        elif search_space.get("paddle_fast"):
            self.assertEqual(search_space["psms"], [7, 6, 13])
            self.assertEqual(search_space["split_count"], 3)
            self.assertEqual(search_space["per_line_variants"], 4)
            self.assertEqual(search_space["total_tesseract_calls"], 72)
        elif search_space.get("profile_name") == "balanced":
            self.assertEqual(search_space["psms"], [7, 6, 13])
            self.assertEqual(search_space["split_count"], 4)
            self.assertEqual(search_space["per_line_variants"], 12)
            self.assertEqual(search_space["total_tesseract_calls"], 288)
        else:
            self.assertEqual(search_space["psms"], [7, 6, 13])
            self.assertEqual(search_space["split_count"], 6)
            self.assertEqual(search_space["per_line_variants"], 18)
            self.assertEqual(search_space["total_tesseract_calls"], 648)


class TestOcrBackendSelection(unittest.TestCase):
    def test_backend_resolution_honors_environment(self) -> None:
        previous = os.environ.get("OCR_BACKEND")
        os.environ["OCR_BACKEND"] = "tesseract"
        try:
            from mrz.td3 import ocr_pipeline
            importlib.reload(ocr_pipeline)
            self.assertEqual(ocr_pipeline._resolve_ocr_backends(), ["tesseract"])
        finally:
            if previous is None:
                os.environ.pop("OCR_BACKEND", None)
            else:
                os.environ["OCR_BACKEND"] = previous


class TestChecksumFieldRepair(unittest.TestCase):
    def test_document_number_prefers_alpha_prefix_over_ambiguous_digit(self) -> None:
        repaired = correct_field(
            "1A36E7777",
            "8",
            substitutions=DOC_NUMBER_AMBIGUOUS_SUBS,
            field_kind="document_number",
        )

        self.assertEqual(repaired, "LA36E7777")

    def test_document_number_keeps_valid_numeric_prefix(self) -> None:
        repaired = correct_field(
            "551356455",
            "1",
            substitutions=DOC_NUMBER_AMBIGUOUS_SUBS,
            field_kind="document_number",
        )

        self.assertEqual(repaired, "551356455")




class TestPaddleLine1Repair(unittest.TestCase):
    def test_paddle_repair_skips_single_separator_name_zone(self) -> None:
        line = "PRMDVYOOSUFAZYANFARHATH<<<<<<<<<<<<<<<<<<<<<"

        repaired, meta = _repair_paddle_line1_candidate(line)

        self.assertEqual(repaired, line)
        self.assertIsNone(meta)

class TestPaddleOcrAdapter(unittest.TestCase):
    def test_resolve_paddle_use_gpu_returns_true_when_cuda_device_is_visible(self) -> None:
        fake_paddle = mock.Mock()
        fake_paddle.device.is_compiled_with_cuda.return_value = True
        fake_paddle.device.cuda.device_count.return_value = 1

        with mock.patch.dict(sys.modules, {"paddle": fake_paddle}):
            self.assertTrue(_resolve_paddle_use_gpu({"paddle_use_gpu": True}))

    def test_build_paddle_ocr_kwargs_uses_server_rec_variant_for_en(self) -> None:
        class FakePaddleOCR:
            def __init__(
                self,
                lang=None,
                use_gpu=None,
                show_log=None,
                use_doc_orientation_classify=None,
                use_doc_unwarping=None,
                use_textline_orientation=None,
                ocr_version=None,
                text_detection_model_name=None,
                text_recognition_model_name=None,
                text_rec_score_thresh=None,
                **kwargs,
            ):
                pass

        with mock.patch.dict(
            os.environ,
            {
                "PADDLEOCR_DET_MODEL_VARIANT": "server",
                "PADDLEOCR_REC_MODEL_VARIANT": "server",
                "PADDLEOCR_TEXT_REC_SCORE_THRESH": "0.83",
            },
            clear=False,
        ):
            kwargs = build_paddle_ocr_kwargs(FakePaddleOCR, "en", False)

        self.assertEqual(kwargs["ocr_version"], "PP-OCRv5")
        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_server_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "PP-OCRv5_server_rec")
        self.assertEqual(kwargs["text_rec_score_thresh"], 0.83)

    def test_build_paddle_ocr_kwargs_respects_explicit_model_names(self) -> None:
        class FakePaddleOCR:
            def __init__(
                self,
                lang=None,
                use_gpu=None,
                show_log=None,
                use_doc_orientation_classify=None,
                use_doc_unwarping=None,
                use_textline_orientation=None,
                ocr_version=None,
                text_detection_model_name=None,
                text_recognition_model_name=None,
                **kwargs,
            ):
                pass

        with mock.patch.dict(
            os.environ,
            {
                "PADDLEOCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLEOCR_TEXT_RECOGNITION_MODEL_NAME": "PP-OCRv5_server_rec",
                "PADDLEOCR_DET_MODEL_VARIANT": "server",
                "PADDLEOCR_REC_MODEL_VARIANT": "mobile",
            },
            clear=False,
        ):
            kwargs = build_paddle_ocr_kwargs(FakePaddleOCR, "en", False)

        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "PP-OCRv5_server_rec")

    def test_paddle_ocr_image_converts_grayscale_to_bgr(self) -> None:
        seen = {}

        class FakePaddleOCR:
            def predict(self, img):
                seen["shape"] = img.shape
                return [{"rec_texts": ["P<PAKWAQAR<<SADIA"]}]

        with mock.patch("mrz.td3.ocr_pipeline._get_paddle_ocr", return_value=FakePaddleOCR()):
            text = _paddle_ocr_image(np.full((12, 24), 255, dtype=np.uint8))

        self.assertEqual(seen["shape"], (12, 24, 3))
        self.assertEqual(text, "P<PAKWAQAR<<SADIA")

    def test_paddle_ocr_images_uses_batch_predict_when_available(self) -> None:
        seen = {}
        reset_paddle_ocr_stats()
        reset_paddle_ocr_cache()

        class FakePaddleOCR:
            def predict(self, imgs):
                seen["count"] = len(imgs)
                seen["shapes"] = [img.shape for img in imgs]
                return [
                    {"rec_texts": ["P<PAKWAQAR<<SADIA"], "rec_scores": [0.93]},
                    {"rec_texts": ["P<BGDAHAMED<<INAM"], "rec_scores": [0.81]},
                ]

        with mock.patch("ocr_backends.paddle_backend.get_paddle_ocr", return_value=FakePaddleOCR()):
            texts = paddle_ocr_images(
                [
                    np.full((12, 24), 255, dtype=np.uint8),
                    np.full((10, 18), 255, dtype=np.uint8),
                ],
                line_kind="line1",
                paddle_lang="en",
                paddle_use_gpu=False,
                normalize_mrz=normalize_td3_line1,
                normalize_td3_line1=normalize_td3_line1,
                normalize_td3_line2=lambda value: value,
                score_td3_line1=score_td3_line1,
                score_td3_line2=score_td3_line2,
            )

        self.assertEqual(seen["count"], 2)
        self.assertEqual(seen["shapes"], [(12, 24, 3), (10, 18, 3)])
        self.assertEqual(
            texts,
            [
                {
                    "text": normalize_td3_line1("P<PAKWAQAR<<SADIA"),
                    "line_confidence": 0.93,
                    "ocr_confidence_source": "line",
                },
                {
                    "text": normalize_td3_line1("P<BGDAHAMED<<INAM"),
                    "line_confidence": 0.81,
                    "ocr_confidence_source": "line",
                },
            ],
        )
        stats = get_paddle_ocr_stats()
        self.assertEqual(stats["batch_requests"], 1)
        self.assertEqual(stats["batched_calls"], 1)
        self.assertEqual(stats["batch_fallbacks"], 0)
        self.assertEqual(stats["serial_calls"], 0)
        self.assertEqual(stats["images_submitted"], 2)

    def test_trim_line1_spill_removes_line2_prefix_leak(self) -> None:
        repaired = _trim_line1_spill(
            "P<BGDAHMAD<<ADEE<<<<<<<<<<<<<<<<<<<<<<<<<<EK"
        )

        self.assertEqual(
            repaired,
            "P<BGDAHMAD<<ADEE<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        )

    def test_best_paddle_line1_candidate_prefers_clean_fragment_over_joined_spill(self) -> None:
        texts = [
            "P<BGDAHMAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "EK",
        ]

        best = _best_paddle_text_candidate(texts, "line1")

        self.assertEqual(
            best,
            "P<BGDAHMAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<",
        )

    def test_paddle_specific_line1_repair_prefers_ahmad_over_ahnad(self) -> None:
        repaired, meta = _repair_paddle_line1_candidate(
            "P<BGDAHNAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )

        self.assertEqual(
            repaired,
            "P<BGDAHMAD<<ADEELA<<<<<<<<<<<<<<<<<<<<<<<<<<",
        )
        self.assertIsNotNone(meta)



if __name__ == "__main__":
    unittest.main()
