import unittest

from ocr_mrz import (
    normalize_td3_line1,
    pair_consistency_bonus,
    repair_given_name_token,
    repair_td3_line1,
    validate_td3_checks,
)


class TestLine1Repair(unittest.TestCase):
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



if __name__ == "__main__":
    unittest.main()