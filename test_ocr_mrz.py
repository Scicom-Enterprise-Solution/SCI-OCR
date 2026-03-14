import importlib
import os
import sys
import unittest
import numpy as np
from unittest import mock

from mrz.td3.ocr_pipeline import (
    _apply_candidate_support_bonus,
    _best_paddle_text_candidate,
    _line1_selection_penalty,
    _paddle_ocr_image,
    _pair_selection_key,
    _prepare_variants,
    _repair_paddle_line1_candidate,
    _resolve_ocr_backends,
    _resolve_paddle_use_gpu,
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
from ocr_backends.paddle_backend import paddle_ocr_images


class TestLine1Repair(unittest.TestCase):
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

    def test_keeps_plausible_name_when_score_gain_is_small(self) -> None:
        repaired, meta = repair_given_name_token("SHAHADAT")

        self.assertEqual(repaired, "SHAHADAT")
        self.assertFalse(meta["changed"])

    def test_repair_given_name_token_prefers_l_over_i_for_adeela(self) -> None:
        repaired, meta = repair_given_name_token("ADEEIA")

        self.assertEqual(repaired, "ADEELA")
        self.assertTrue(meta["changed"])

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
            from mrz.td3 import ocr_pipeline
            importlib.reload(ocr_pipeline)

class TestPaddleOcrAdapter(unittest.TestCase):
    def test_resolve_paddle_use_gpu_returns_true_when_cuda_device_is_visible(self) -> None:
        fake_paddle = mock.Mock()
        fake_paddle.device.is_compiled_with_cuda.return_value = True
        fake_paddle.device.cuda.device_count.return_value = 1

        with mock.patch("mrz.td3.ocr_pipeline.PADDLEOCR_USE_GPU", True):
            with mock.patch.dict(sys.modules, {"paddle": fake_paddle}):
                self.assertTrue(_resolve_paddle_use_gpu())

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

        class FakePaddleOCR:
            def predict(self, imgs):
                seen["count"] = len(imgs)
                seen["shapes"] = [img.shape for img in imgs]
                return [
                    {"rec_texts": ["P<PAKWAQAR<<SADIA"]},
                    {"rec_texts": ["P<BGDAHAMED<<INAM"]},
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
                normalize_td3_line1("P<PAKWAQAR<<SADIA"),
                normalize_td3_line1("P<BGDAHAMED<<INAM"),
            ],
        )

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
