# Akuna Binary Options Market Maker

Final submission for the Akuna Capital binary-options market-making challenge.

> Competition-safety note: this repository is private by default. Do not make it
> public while the challenge is active unless the competition rules explicitly
> permit sharing complete solutions.

## Game summary

The market maker trades event contracts on three daily underlyings:

- `FED`: the fed funds rate, which moves up or down by 0.25 or remains unchanged,
  with state-dependent mean reversion.
- `AJR`: AjarAI's valuation.
- `THR`: Theriodic's valuation.

Every binary option pays `1.0` if its event is true at expiry and `0.0`
otherwise. The available contracts are single-underlying threshold options or
AJR/THR relative-value spreads.

The submission must:

- estimate the hidden market parameters from the warm-up history;
- return a side-effect-free theoretical probability for each live option;
- quote both a bid and an offer in response to RFQs;
- decide whether to accept all-or-none FOK orders, whose fills may be shared
  among accepting market makers;
- manage maximum-loss cash usage so that it never becomes bankrupt;
- remain below the platform's 64 KB source limit.

RFQ execution is competitive: the best eligible quote receives the trade, with
the platform's tie rule determining allocation among equally good quotes.

## Final strategy

`submission.py` contains the standalone final strategy, named
`Posterior Ridge Zero-Loss Boundary Dynamic-FOK`.

Its main components are:

1. A clipped, mean-reverting state model for `FED`, estimated from warm-up
   transitions using likelihood-based parameter search and posterior weighting.
2. A joint factor model for `AJR` and `THR`, separating shared sector risk,
   rate sensitivity, drift, and idiosyncratic volatility.
3. Exact FED-state mixing when pricing valuation and relative-value binary
   options.
4. Uncertainty-aware RFQ spreads and dynamic FOK edge requirements.
5. Cash, per-option position, and aggregate factor-exposure limits.
6. Extra capacity only at true zero-maximum-loss boundaries: buying at `0.00`
   or selling at `1.00`.

The zero-loss overlay does not alter the theoretical-pricing or warm-up methods;
away from the exact `0.00`/`1.00` boundaries, execution remains identical to the
underlying Dynamic-FOK strategy.

## Validation

The final source was checked for:

- Python syntax and standalone module loading;
- no duplicate imports;
- source size below 64 KB;
- side-effect-free theoretical-pricing methods;
- finite prices in `[0, 1]`;
- legal penny-rounded two-sided quotes;
- full requested-quantity FOK risk checks;
- expiry-time position and risk-budget cleanup;
- exact zero-loss boundary behavior.

The challenge's six published theoretical-value examples passed with:

| Contract | Model value |
| --- | ---: |
| 1d FED ≥ 3.00 | 0.7000 |
| 5d FED ≥ 3.50 | 0.0471 |
| 1d AJR ≥ 500 | 0.5309 |
| 10d THR ≥ 650 | 0.2068 |
| 1d THR − AJR ≥ 0 | 1.0000 |
| 10d THR − AJR ≥ 0 | 0.9999 |

Official result: `PASS (max_error=0.0000)`.

Run the local regression checks with:

```bash
python -m unittest discover -s tests -v
```

## Repository contents

- `submission.py`: final standalone HackerRank submission.
- `tests/test_submission.py`: compact regression and safety checks.

This repository documents a personal competition solution and is not affiliated
with or endorsed by Akuna Capital.
