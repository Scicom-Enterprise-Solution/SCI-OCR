import unittest

from ocr_mrz import (
    _pair_selection_key,
    is_valid_mrz_country_code,
    normalize_td3_line1,
    pair_consistency_bonus,
    repair_given_name_token,
    repair_issuing_country_code,
    repair_td3_line1,
    score_td3_line1,
    score_td3_line2,
    validate_and_correct_mrz,
    validate_td3_checks,
)


class TestLine1Repair(unittest.TestCase):
    def test_normalize_td3_line1_collapses_po_prefix_to_passport_prefix(self) -> None:
        line = normalize_td3_line1("POJORALMOUSA<<AWS<WAJDI<FAROUR<<<<<<<<<<<<<")

        self.assertEqual(line[:5], "P<JOR")

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

    def test_accepts_uto_mrz_country_code(self) -> None:
        self.assertTrue(is_valid_mrz_country_code("UTO"))


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



if __name__ == "__main__":
    unittest.main()