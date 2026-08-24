import copy
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import submission as model


class SubmissionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.underlyings = [
            model.Underlying("FED", 1, 3.0),
            model.Underlying("AJR", 2, 500.0),
            model.Underlying("THR", 3, 600.0),
        ]
        self.options = [
            model.BinaryOption((model.OptionLeg(1, 1.0),), 1, 1, 3.0),
            model.BinaryOption((model.OptionLeg(1, 1.0),), 2, 5, 3.5),
            model.BinaryOption((model.OptionLeg(2, 1.0),), 3, 1, 500.0),
            model.BinaryOption((model.OptionLeg(3, 1.0),), 4, 10, 650.0),
            model.BinaryOption(
                (model.OptionLeg(3, 1.0), model.OptionLeg(2, -1.0)),
                5,
                1,
                0.0,
            ),
            model.BinaryOption(
                (model.OptionLeg(3, 1.0), model.OptionLeg(2, -1.0)),
                6,
                10,
                0.0,
            ),
        ]
        self.parameters = model.MarketParameters(
            ajarai_drift=0.001,
            ajarai_idio_std_dev=0.01,
            ajarai_rate_beta=-0.02,
            ajarai_sector_beta=1.0,
            rate_down_probability=0.2,
            rate_reversion_strength=0.1,
            rate_up_probability=0.25,
            sector_std_dev=0.02,
            theriodic_drift=0.0015,
            theriodic_idio_std_dev=0.012,
            theriodic_rate_beta=-0.015,
            theriodic_sector_beta=1.0,
            rate_step=0.25,
            rate_target=2.0,
        )

    def maker(self):
        return model.MarketMaker(self.underlyings, self.options, 20.0)

    def test_source_is_below_platform_limit(self):
        self.assertLess((ROOT / "submission.py").stat().st_size, 65_536)

    def test_published_theoretical_values(self):
        expected = (0.7000, 0.0471, 0.5309, 0.2068, 1.0000, 0.9999)
        maker = self.maker()
        actual = tuple(
            maker.price_option_from_parameters(self.parameters, option)
            for option in self.options
        )
        for value, target in zip(actual, expected):
            self.assertAlmostEqual(value, target, delta=0.00006)

    def test_pricing_is_side_effect_free(self):
        maker = self.maker()
        before = copy.deepcopy(maker.__dict__)
        before_position = dict(
            before.pop("position").option_quantity_by_option_id
        )
        for option in self.options:
            maker.price_option(option)
            maker.price_option_from_parameters(self.parameters, option)
        after = copy.deepcopy(maker.__dict__)
        after_position = dict(after.pop("position").option_quantity_by_option_id)
        self.assertEqual(after, before)
        self.assertEqual(after_position, before_position)

    def test_quotes_are_legal_and_boundary_capacity_is_selective(self):
        maker = self.maker()
        maker.models_ready = True
        option = self.options[2]
        for fair, uncertainty in (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.03, 0.0),
            (0.97, 0.0),
            (0.50, 0.05),
        ):
            maker._estimated_price_and_uncertainty = (
                lambda _option, value=fair, error=uncertainty: (value, error)
            )
            quote = maker.quote(option, 123)
            self.assertTrue(
                math.isfinite(quote.bid_price)
                and math.isfinite(quote.offer_price)
            )
            self.assertTrue(0.0 <= quote.bid_price < quote.offer_price <= 1.0)
            self.assertAlmostEqual(
                quote.bid_price * 100,
                round(quote.bid_price * 100),
            )
            self.assertAlmostEqual(
                quote.offer_price * 100,
                round(quote.offer_price * 100),
            )
            if quote.bid_price == 0.0:
                self.assertGreaterEqual(quote.bid_quantity, 50)
            if quote.offer_price == 1.0:
                self.assertGreaterEqual(quote.offer_quantity, 50)

    def test_zero_loss_fok_only_bypasses_at_exact_boundaries(self):
        maker = self.maker()
        maker.models_ready = True
        option = self.options[2]
        sell_zero = model.FokOrder(
            1, option.option_id, model.OrderType.SELL, 0.0, 500
        )
        buy_one = model.FokOrder(
            2, option.option_id, model.OrderType.BUY, 1.0, 500
        )
        wrong_option = model.FokOrder(
            3, 999_999, model.OrderType.SELL, 0.0, 500
        )
        self.assertTrue(maker.respond_to_fok(option, sell_zero))
        self.assertTrue(maker.respond_to_fok(option, buy_one))
        self.assertFalse(maker.respond_to_fok(option, wrong_option))


if __name__ == "__main__":
    unittest.main()
