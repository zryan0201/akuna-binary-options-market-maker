import argparse
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
from typing import Any, Final, Iterable, Mapping, Sequence

UP = 'up'
DOWN = 'down'
STAY = 'stay'

@dataclass(frozen=True)
class FedParameters:
    rate_up_probability: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self):
        if self.rate_step <= 0:
            raise ValueError('rate_step must be positive')
        if not 0.0 < self.rate_up_probability <= 1.0:
            raise ValueError('rate_up_probability must be in (0, 1]')
        if not 0.0 < self.rate_down_probability <= 1.0:
            raise ValueError('rate_down_probability must be in (0, 1]')
        if self.rate_up_probability + self.rate_down_probability > 1.0 + 1e-12:
            raise ValueError('base up/down probabilities must sum to at most one')
        if not 0.0 <= self.rate_reversion_strength <= 1.0:
            raise ValueError('rate_reversion_strength must be in [0, 1]')

@dataclass(frozen=True)
class GridStage:
    step: float
    radius: float | None

@dataclass(frozen=True)
class PosteriorPoint:
    parameters: FedParameters
    log_likelihood: float
    log_posterior: float
    weight: float = 0.0

@dataclass(frozen=True)
class FedFit:
    mle: FedParameters
    map_estimate: FedParameters
    posterior_mean: FedParameters
    posterior: tuple[PosteriorPoint, ...]
    posterior_mass_retained: float
    log_likelihood_at_mle: float

def transition_probabilities(parameters, rate):
    tilt = parameters.rate_reversion_strength * (parameters.rate_target - rate)
    up = min(max(parameters.rate_up_probability + tilt, 0.0), 1.0)
    down = min(max(parameters.rate_down_probability - tilt, 0.0), 1.0 - up)
    flat = max(0.0, 1.0 - up - down)
    return (up, down, flat)

def _rate_to_tick(rate, step):
    tick = round(rate / step)
    reconstructed = tick * step
    if abs(reconstructed - rate) > 1e-06:
        raise ValueError(f'rate {rate!r} is not on the {step} grid')
    if tick < 0:
        raise ValueError('FED rate cannot be negative')
    return tick

def aggregate_transitions(history, step=0.25):
    if len(history) < 2:
        raise ValueError('at least two historical observations are required')
    counts = defaultdict(Counter)
    ticks = [_rate_to_tick(rate, step) for rate in history]
    for current_tick, next_tick in zip(ticks, ticks[1:]):
        delta = next_tick - current_tick
        if delta == 1:
            outcome = UP
        elif delta == -1:
            outcome = DOWN
        elif delta == 0:
            outcome = STAY
        else:
            raise ValueError(f'invalid FED transition: {current_tick * step:.2f} -> {next_tick * step:.2f}')
        counts[current_tick][outcome] += 1
    return dict(counts)

def observed_transition_probabilities(parameters, current_tick):
    rate = current_tick * parameters.rate_step
    up, down, flat = transition_probabilities(parameters, rate)
    if current_tick == 0:
        return (up, 0.0, down + flat)
    return (up, down, flat)

def fed_log_likelihood(counts, parameters):
    score = 0.0
    for tick, state_counts in counts.items():
        p_up, p_down, p_stay = observed_transition_probabilities(parameters, tick)
        for outcome, probability in ((UP, p_up), (DOWN, p_down), (STAY, p_stay)):
            count = state_counts.get(outcome, 0)
            if count == 0:
                continue
            if probability <= 0.0:
                return -math.inf
            score += count * math.log(probability)
    return score

def log_prior(parameters, *, base_dirichlet_alpha=(1.0, 1.0, 1.0), reversion_beta_alpha=(1.0, 1.0)):
    if any((alpha < 1.0 for alpha in (*base_dirichlet_alpha, *reversion_beta_alpha))):
        raise ValueError('prior alpha values must be at least one')
    up = parameters.rate_up_probability
    down = parameters.rate_down_probability
    flat = max(0.0, 1.0 - up - down)
    k = parameters.rate_reversion_strength
    score = 0.0
    for value, alpha in zip((up, down, flat), base_dirichlet_alpha):
        coefficient = alpha - 1.0
        if coefficient == 0.0:
            continue
        if value <= 0.0:
            return -math.inf
        score += coefficient * math.log(value)
    for value, alpha in zip((k, 1.0 - k), reversion_beta_alpha):
        coefficient = alpha - 1.0
        if coefficient == 0.0:
            continue
        if value <= 0.0:
            return -math.inf
        score += coefficient * math.log(value)
    return score

def _aligned_values(low, high, step):
    first = math.ceil((low - 1e-12) / step)
    last = math.floor((high + 1e-12) / step)
    return [round(index * step, 12) for index in range(first, last + 1)]

def _candidate_parameters(*, step, rate_step, rate_target, centers, radius, minimum_base_probability):
    if centers is None:
        boxes = [(minimum_base_probability, 1.0 - minimum_base_probability, minimum_base_probability, 1.0 - minimum_base_probability, 0.0, 1.0)]
    else:
        if radius is None:
            raise ValueError('a refinement stage requires a radius')
        boxes = [(max(minimum_base_probability, center.rate_up_probability - radius), min(1.0 - minimum_base_probability, center.rate_up_probability + radius), max(minimum_base_probability, center.rate_down_probability - radius), min(1.0 - minimum_base_probability, center.rate_down_probability + radius), max(0.0, center.rate_reversion_strength - radius), min(1.0, center.rate_reversion_strength + radius)) for center in centers]
    seen = set()
    for up_low, up_high, down_low, down_high, k_low, k_high in boxes:
        for up in _aligned_values(up_low, up_high, step):
            for down in _aligned_values(down_low, down_high, step):
                if up + down > 1.0 + 1e-12:
                    continue
                for k in _aligned_values(k_low, k_high, step):
                    key = (up, down, k)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield FedParameters(rate_up_probability=up, rate_down_probability=down, rate_reversion_strength=k, rate_step=rate_step, rate_target=rate_target)

def _select_separated_centers(points, *, count, minimum_separation):
    centers = []
    for point in sorted(points, key=lambda item: item.log_posterior, reverse=True):
        candidate = point.parameters
        if all((max(abs(candidate.rate_up_probability - center.rate_up_probability), abs(candidate.rate_down_probability - center.rate_down_probability), abs(candidate.rate_reversion_strength - center.rate_reversion_strength)) >= minimum_separation for center in centers)):
            centers.append(candidate)
            if len(centers) == count:
                break
    return centers

def _normalise_and_compress_posterior(points, *, posterior_limit):
    finite = [point for point in points if math.isfinite(point.log_posterior)]
    if not finite:
        raise ValueError('no grid point assigned positive probability to the history')
    if posterior_limit <= 0:
        raise ValueError('posterior_limit must be positive')
    maximum = max((point.log_posterior for point in finite))
    raw_weights = [math.exp(point.log_posterior - maximum) for point in finite]
    total = sum(raw_weights)
    weighted = sorted(((point, weight / total) for point, weight in zip(finite, raw_weights)), key=lambda item: (item[0].parameters.rate_up_probability, item[0].parameters.rate_down_probability, item[0].parameters.rate_reversion_strength))
    if len(weighted) <= posterior_limit:
        return (tuple((replace(point, weight=weight) for point, weight in weighted)), 1.0)
    selected_counts = Counter()
    weighted_index = 0
    cumulative = weighted[0][1]
    for sample_index in range(posterior_limit):
        target = (sample_index + 0.5) / posterior_limit
        while target > cumulative and weighted_index + 1 < len(weighted):
            weighted_index += 1
            cumulative += weighted[weighted_index][1]
        selected_counts[weighted_index] += 1
    posterior = tuple((replace(weighted[index][0], weight=count / posterior_limit) for index, count in sorted(selected_counts.items())))
    return (posterior, 1.0)

def fit_fed_model(history, *, rate_step=0.25, rate_target=2.0, stages=(GridStage(step=0.025, radius=None), GridStage(step=0.005, radius=0.05), GridStage(step=0.001, radius=0.01)), max_refinement_centers=3, posterior_limit=512, minimum_base_probability=0.001, base_dirichlet_alpha=(1.0, 1.0, 1.0), reversion_beta_alpha=(1.0, 1.0)):
    if not stages or stages[0].radius is not None:
        raise ValueError('the first grid stage must be global (radius=None)')
    if any((stage.step <= 0 for stage in stages)):
        raise ValueError('grid steps must be positive')
    counts = aggregate_transitions(history, rate_step)
    previous_centers = None
    evaluated = []
    for stage_index, stage in enumerate(stages):
        evaluated = []
        for parameters in _candidate_parameters(step=stage.step, rate_step=rate_step, rate_target=rate_target, centers=previous_centers, radius=stage.radius, minimum_base_probability=minimum_base_probability):
            likelihood = fed_log_likelihood(counts, parameters)
            prior = log_prior(parameters, base_dirichlet_alpha=base_dirichlet_alpha, reversion_beta_alpha=reversion_beta_alpha)
            evaluated.append(PosteriorPoint(parameters=parameters, log_likelihood=likelihood, log_posterior=likelihood + prior))
        if not evaluated:
            raise ValueError(f'grid stage {stage_index} produced no legal candidates')
        if stage_index + 1 < len(stages):
            next_radius = stages[stage_index + 1].radius
            assert next_radius is not None
            previous_centers = _select_separated_centers(evaluated, count=max_refinement_centers, minimum_separation=max(next_radius, stage.step))
    posterior, retained_mass = _normalise_and_compress_posterior(evaluated, posterior_limit=posterior_limit)
    mle_point = max(evaluated, key=lambda point: point.log_likelihood)
    map_point = max(evaluated, key=lambda point: point.log_posterior)

    def weighted(attribute):
        return sum((point.weight * getattr(point.parameters, attribute) for point in posterior))
    posterior_mean = FedParameters(rate_up_probability=weighted('rate_up_probability'), rate_down_probability=weighted('rate_down_probability'), rate_reversion_strength=weighted('rate_reversion_strength'), rate_step=rate_step, rate_target=rate_target)
    return FedFit(mle=mle_point.parameters, map_estimate=map_point.parameters, posterior_mean=posterior_mean, posterior=posterior, posterior_mass_retained=retained_mass, log_likelihood_at_mle=mle_point.log_likelihood)

@lru_cache(maxsize=16384)
def terminal_distribution(parameters, *, current_rate, steps):
    if steps < 0:
        raise ValueError('steps must be non-negative')
    start_tick = _rate_to_tick(current_rate, parameters.rate_step)
    distribution = {start_tick: 1.0}
    for _ in range(steps):
        next_distribution = defaultdict(float)
        for tick, state_probability in distribution.items():
            rate = tick * parameters.rate_step
            p_up, p_down, p_flat = transition_probabilities(parameters, rate)
            next_distribution[tick + 1] += state_probability * p_up
            next_distribution[max(tick - 1, 0)] += state_probability * p_down
            next_distribution[tick] += state_probability * p_flat
        distribution = dict(next_distribution)
    return {round(tick * parameters.rate_step, 12): probability for tick, probability in distribution.items()}

@lru_cache(maxsize=65536)
def price_fed_binary(parameters, *, current_rate, steps, strike, weight=1.0):
    if weight == 0.0:
        raise ValueError('option weight cannot be zero')
    distribution = terminal_distribution(parameters, current_rate=current_rate, steps=steps)
    return sum((probability for rate, probability in distribution.items() if weight * rate >= strike))

def posterior_predictive_price(fit, *, current_rate, steps, strike, weight=1.0):
    priced = [(point.weight, price_fed_binary(point.parameters, current_rate=current_rate, steps=steps, strike=strike, weight=weight)) for point in fit.posterior]
    mean = sum((weight_value * price for weight_value, price in priced))
    variance = sum((weight_value * (price - mean) ** 2 for weight_value, price in priced))
    return (mean, math.sqrt(max(variance, 0.0)))

def posterior_terminal_distribution(fit, *, current_rate, steps):
    mixture = defaultdict(float)
    for point in fit.posterior:
        distribution = terminal_distribution(point.parameters, current_rate=current_rate, steps=steps)
        for rate, probability in distribution.items():
            mixture[rate] += point.weight * probability
    total = sum(mixture.values())
    if total <= 0.0:
        raise ValueError('posterior terminal distribution has no probability mass')
    return {rate: probability / total for rate, probability in mixture.items()}

@dataclass(frozen=True)
class RegressionDiagnostics:
    num_observations: int
    predictor_mean: float
    predictor_sum_squares: float
    ridge_penalty: float
    degrees_of_freedom: int
    finite_sample_variance_multiplier: float

@dataclass(frozen=True)
class ConditionalLogReturnModel:
    drift: float
    rate_beta: float
    residual_variance: float
    diagnostics: RegressionDiagnostics | None = None

    def __post_init__(self):
        if self.residual_variance < 0.0:
            raise ValueError('residual_variance cannot be negative')

    def cumulative_moments(self, *, steps, total_rate_change, include_estimation_uncertainty=True):
        if steps < 0:
            raise ValueError('steps must be non-negative')
        mean = steps * self.drift + self.rate_beta * total_rate_change
        variance = steps * self.residual_variance
        diagnostics = self.diagnostics
        if include_estimation_uncertainty and diagnostics is not None:
            n = diagnostics.num_observations
            denominator = diagnostics.predictor_sum_squares + diagnostics.ridge_penalty
            intercept_leverage = steps * steps / n
            centered_future_predictor = total_rate_change - steps * diagnostics.predictor_mean
            slope_leverage = centered_future_predictor * centered_future_predictor / denominator if denominator > 0.0 else 0.0
            variance += self.residual_variance * (intercept_leverage + slope_leverage)
            variance *= diagnostics.finite_sample_variance_multiplier
        return (mean, max(variance, 0.0))

@dataclass(frozen=True)
class CompanyModels:
    ajarai: ConditionalLogReturnModel
    theriodic: ConditionalLogReturnModel
    relative_ajarai_to_theriodic: ConditionalLogReturnModel

@dataclass(frozen=True)
class CompanySimulationParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    def __post_init__(self):
        if min(self.ajarai_idio_std_dev, self.sector_std_dev, self.theriodic_idio_std_dev) < 0.0:
            raise ValueError('standard deviations cannot be negative')

@dataclass(frozen=True)
class SimulatedMarketHistory:
    fed: tuple[float, ...]
    ajarai: tuple[float, ...]
    theriodic: tuple[float, ...]

def _log_returns(values):
    if len(values) < 2:
        raise ValueError('at least two company values are required')
    if any((value <= 0.0 for value in values)):
        raise ValueError('company values must be positive to compute log returns')
    return [math.log(next_value / value) for value, next_value in zip(values, values[1:])]

def fit_conditional_log_return_model(rate_changes, log_returns, *, rate_step=0.25, ridge_move_equivalents=2.0, apply_finite_sample_inflation=True, variance_floor=1e-12):
    if len(rate_changes) != len(log_returns):
        raise ValueError('rate changes and log returns must have equal length')
    if not rate_changes:
        raise ValueError('at least one return observation is required')
    if ridge_move_equivalents < 0.0:
        raise ValueError('ridge_move_equivalents cannot be negative')
    n = len(rate_changes)
    x_mean = statistics.fmean(rate_changes)
    y_mean = statistics.fmean(log_returns)
    x_ss = sum(((value - x_mean) ** 2 for value in rate_changes))
    xy_ss = sum(((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(rate_changes, log_returns)))
    ridge_penalty = ridge_move_equivalents * rate_step * rate_step
    denominator = x_ss + ridge_penalty
    rate_beta = xy_ss / denominator if denominator > 0.0 else 0.0
    drift = y_mean - rate_beta * x_mean
    residuals = [y_value - drift - rate_beta * x_value for x_value, y_value in zip(rate_changes, log_returns)]
    degrees_of_freedom = max(n - 2, 1)
    residual_variance = max(sum((residual * residual for residual in residuals)) / degrees_of_freedom, variance_floor)
    if apply_finite_sample_inflation:
        variance_multiplier = degrees_of_freedom / (degrees_of_freedom - 2) if degrees_of_freedom > 2 else 3.0
    else:
        variance_multiplier = 1.0
    return ConditionalLogReturnModel(drift=drift, rate_beta=rate_beta, residual_variance=residual_variance, diagnostics=RegressionDiagnostics(num_observations=n, predictor_mean=x_mean, predictor_sum_squares=x_ss, ridge_penalty=ridge_penalty, degrees_of_freedom=degrees_of_freedom, finite_sample_variance_multiplier=variance_multiplier))

def fit_company_models(fed_history, ajarai_history, theriodic_history, *, rate_step=0.25, ridge_move_equivalents=2.0, apply_finite_sample_inflation=True):
    if not len(fed_history) == len(ajarai_history) == len(theriodic_history):
        raise ValueError('FED, AJR, and THR histories must have equal length')
    if len(fed_history) < 3:
        raise ValueError('at least three historical days are required')
    rate_changes = [next_rate - rate for rate, next_rate in zip(fed_history, fed_history[1:])]
    ajarai_returns = _log_returns(ajarai_history)
    theriodic_returns = _log_returns(theriodic_history)
    relative_returns = [ajarai_return - theriodic_return for ajarai_return, theriodic_return in zip(ajarai_returns, theriodic_returns)]
    fit_kwargs = {'rate_step': rate_step, 'ridge_move_equivalents': ridge_move_equivalents, 'apply_finite_sample_inflation': apply_finite_sample_inflation}
    return CompanyModels(ajarai=fit_conditional_log_return_model(rate_changes, ajarai_returns, **fit_kwargs), theriodic=fit_conditional_log_return_model(rate_changes, theriodic_returns, **fit_kwargs), relative_ajarai_to_theriodic=fit_conditional_log_return_model(rate_changes, relative_returns, **fit_kwargs))

def known_company_models(parameters):
    ajarai_variance = (parameters.ajarai_sector_beta * parameters.sector_std_dev) ** 2 + parameters.ajarai_idio_std_dev ** 2
    theriodic_variance = (parameters.theriodic_sector_beta * parameters.sector_std_dev) ** 2 + parameters.theriodic_idio_std_dev ** 2
    relative_variance = ((parameters.ajarai_sector_beta - parameters.theriodic_sector_beta) * parameters.sector_std_dev) ** 2 + parameters.ajarai_idio_std_dev ** 2 + parameters.theriodic_idio_std_dev ** 2
    return CompanyModels(ajarai=ConditionalLogReturnModel(drift=parameters.ajarai_drift, rate_beta=parameters.ajarai_rate_beta, residual_variance=ajarai_variance), theriodic=ConditionalLogReturnModel(drift=parameters.theriodic_drift, rate_beta=parameters.theriodic_rate_beta, residual_variance=theriodic_variance), relative_ajarai_to_theriodic=ConditionalLogReturnModel(drift=parameters.ajarai_drift - parameters.theriodic_drift, rate_beta=parameters.ajarai_rate_beta - parameters.theriodic_rate_beta, residual_variance=relative_variance))

def _probability_at_least(threshold, mean, variance):
    if variance <= 1e-18:
        return 1.0 if mean >= threshold else 0.0
    z = (threshold - mean) / math.sqrt(2.0 * variance)
    return min(max(0.5 * math.erfc(z), 0.0), 1.0)

def _probability_at_most(threshold, mean, variance):
    if variance <= 1e-18:
        return 1.0 if mean <= threshold else 0.0
    z = (mean - threshold) / math.sqrt(2.0 * variance)
    return min(max(0.5 * math.erfc(z), 0.0), 1.0)

def _validate_terminal_distribution(distribution):
    if not distribution:
        raise ValueError('FED terminal distribution cannot be empty')
    if any((probability < 0.0 for probability in distribution.values())):
        raise ValueError('FED terminal probabilities cannot be negative')
    if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-08):
        raise ValueError('FED terminal probabilities must sum to one')

def price_single_name_binary(model, *, current_value, current_fed, fed_terminal_distribution, steps, strike, weight=1.0, include_estimation_uncertainty=True):
    if current_value <= 0.0:
        raise ValueError('current company value must be positive')
    if weight == 0.0:
        raise ValueError('option weight cannot be zero')
    _validate_terminal_distribution(fed_terminal_distribution)
    level_threshold = strike / weight
    if weight > 0.0 and level_threshold <= 0.0:
        return 1.0
    if weight < 0.0 and level_threshold <= 0.0:
        return 0.0
    log_threshold = math.log(level_threshold)
    current_log_value = math.log(current_value)
    price = 0.0
    for terminal_fed, fed_probability in fed_terminal_distribution.items():
        mean_return, variance = model.cumulative_moments(steps=steps, total_rate_change=terminal_fed - current_fed, include_estimation_uncertainty=include_estimation_uncertainty)
        terminal_log_mean = current_log_value + mean_return
        conditional_probability = _probability_at_least(log_threshold, terminal_log_mean, variance) if weight > 0.0 else _probability_at_most(log_threshold, terminal_log_mean, variance)
        price += fed_probability * conditional_probability
    return min(max(price, 0.0), 1.0)

def price_company_comparison_binary(relative_model, *, current_ajarai, current_theriodic, current_fed, fed_terminal_distribution, steps, ajarai_weight=1.0, theriodic_weight=-1.0, strike=0.0, include_estimation_uncertainty=True):
    if current_ajarai <= 0.0 or current_theriodic <= 0.0:
        raise ValueError('current company values must be positive')
    if abs(strike) > 1e-12:
        raise ValueError('analytic comparison pricing requires strike zero')
    if ajarai_weight * theriodic_weight >= 0.0:
        raise ValueError('comparison legs must have opposite non-zero signs')
    _validate_terminal_distribution(fed_terminal_distribution)
    ratio_threshold = abs(theriodic_weight / ajarai_weight)
    log_ratio_threshold = math.log(ratio_threshold)
    current_log_ratio = math.log(current_ajarai / current_theriodic)
    ajarai_is_positive_leg = ajarai_weight > 0.0
    price = 0.0
    for terminal_fed, fed_probability in fed_terminal_distribution.items():
        mean_relative_return, variance = relative_model.cumulative_moments(steps=steps, total_rate_change=terminal_fed - current_fed, include_estimation_uncertainty=include_estimation_uncertainty)
        terminal_log_ratio_mean = current_log_ratio + mean_relative_return
        conditional_probability = _probability_at_least(log_ratio_threshold, terminal_log_ratio_mean, variance) if ajarai_is_positive_leg else _probability_at_most(log_ratio_threshold, terminal_log_ratio_mean, variance)
        price += fed_probability * conditional_probability
    return min(max(price, 0.0), 1.0)
    
AJARAI_NAME: Final[str] = 'AJR'
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = 'FED'
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = 'THR'
THERIODIC_UNDERLYING_ID: Final[int] = 3
UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {AJARAI_UNDERLYING_ID: AJARAI_NAME, FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME, THERIODIC_UNDERLYING_ID: THERIODIC_NAME}

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: 'tuple[OptionLeg, ...]'
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self):
        if self.steps_until_expiry < 0:
            raise ValueError('Steps until expiry must be non-negative')
        if not self.legs:
            raise ValueError('Binary option must have at least one leg')
        underlying_ids = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError('Binary option legs must reference distinct underlyings')
        if any((leg.weight == 0 for leg in self.legs)):
            raise ValueError('Binary option leg weights must be non-zero')

    def __str__(self):
        terms = []
        for index, leg in enumerate(self.legs):
            name = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude = abs(leg.weight)
            magnitude_str = '' if magnitude == 1 else f'{magnitude:.2f}*'
            if index == 0:
                sign = '-' if leg.weight < 0 else ''
            else:
                sign = ' - ' if leg.weight < 0 else ' + '
            terms.append(f'{sign}{magnitude_str}{name}')
        observable_expression = ''.join(terms)
        return f'{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})'

    def advance_step(self):
        if self.steps_until_expiry == 0:
            return self
        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other):
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id):
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id):
        return sum((leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs))

@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: 'OrderType'
    price: float
    quantity: int

    def __post_init__(self):
        if self.price < 0:
            raise ValueError('FOK order price must be non-negative')
        if self.quantity <= 0:
            raise ValueError('FOK order quantity must be positive')

@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self):
        lengths = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError('All underlyings must have the same number of historical days')
        if lengths and next(iter(lengths)) <= 0:
            raise ValueError('Market history must contain at least one day')

    @property
    def num_days(self):
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))

@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float
    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self):
        if self.rate_step <= 0:
            raise ValueError('Rate step must be positive')
        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError('Rate up/down probabilities must both be positive')
        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError('Rate up/down probabilities must not sum to more than 1')
        if self.rate_target < 0:
            raise ValueError('Rate target must be non-negative')
        if not 0 <= self.rate_reversion_strength <= 1:
            raise ValueError('Rate reversion strength must be between 0 and 1')
        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError('Standard deviations must be non-negative')

    def advance_company_value(self, current_value, rate_change, sector_shock, *, drift, rate_beta, sector_beta, idio_std_dev):
        idiosyncratic_shock = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return = drift + rate_beta * rate_change + sector_beta * sector_shock + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value):
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)
        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)
        return rate_value

    def advance_step(self, value_by_underlying_id):
        current_rate_value = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value = self.advance_rate(current_rate_value)
        rate_change = round(rate_value - current_rate_value, 2)
        sector_shock = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {FED_FUNDS_RATE_UNDERLYING_ID: rate_value, AJARAI_UNDERLYING_ID: self.advance_company_value(value_by_underlying_id[AJARAI_UNDERLYING_ID], rate_change, sector_shock, drift=self.ajarai_drift, rate_beta=self.ajarai_rate_beta, sector_beta=self.ajarai_sector_beta, idio_std_dev=self.ajarai_idio_std_dev), THERIODIC_UNDERLYING_ID: self.advance_company_value(value_by_underlying_id[THERIODIC_UNDERLYING_ID], rate_change, sector_shock, drift=self.theriodic_drift, rate_beta=self.theriodic_rate_beta, sector_beta=self.theriodic_sector_beta, idio_std_dev=self.theriodic_idio_std_dev)}

    def next_rate_value(self, rate_value, num_grid_steps):
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value):
        tilt = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return (up_probability, down_probability)

@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float

class OrderType(StrEnum):
    BUY = 'buy'
    SELL = 'sell'

class Position:

    def __init__(self):
        self.option_quantity_by_option_id = defaultdict(int)

    def add_option_quantity(self, option_id, quantity):
        self.option_quantity_by_option_id[option_id] += quantity

@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self):
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError('Quote quantities must be positive')
        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError('Quote prices must be between 0 and 1')
        if self.bid_price >= self.offer_price:
            raise ValueError('Quote bid price must be less than offer price')
        if any((abs(round(price * 100) - price * 100) > 1e-06 for price in (self.bid_price, self.offer_price))):
            raise ValueError('Quote prices must be in whole pennies (multiples of 0.01)')

@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other):
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id

class MarketMaker:
    RFQ_BASE_HALF_SPREAD: Final[float] = 0.03
    FOK_MINIMUM_EDGE: Final[float] = 0.03
    CASH_USAGE_LIMIT: Final[float] = 0.75
    RFQ_TRADE_RISK_LIMIT: Final[float] = 0.1
    MAX_RFQ_QUANTITY: Final[int] = 10
    MAX_ABS_POSITION_PER_OPTION: Final[int] = 10
    INVENTORY_SKEW_PER_CONTRACT: Final[float] = 0.002
    MAX_INVENTORY_SKEW: Final[float] = 0.03
    MAX_UNCERTAINTY_BUFFER: Final[float] = 0.05

    def __init__(self, underlying_initial_state, option_initial_state, cash_balance):
        self.underlying_state = underlying_initial_state
        self.active_option_state = option_initial_state
        self.cash_balance = cash_balance
        self.position = Position()
        self.initial_cash_balance = cash_balance
        self.risk_budget_remaining = max(cash_balance * self.CASH_USAGE_LIMIT, 0.0)
        self.gross_long_quantity_by_option_id = defaultdict(int)
        self.gross_short_quantity_by_option_id = defaultdict(int)
        self.fed_fit = None
        self.company_models = None
        self.models_ready = False

    def on_step_advance(self, new_underlying_state, new_option_state):
        new_values = {underlying.underlying_id: underlying.value for underlying in new_underlying_state}
        for option in self.active_option_state:
            if option.steps_until_expiry != 1:
                continue
            option_id = option.option_id
            long_quantity = self.gross_long_quantity_by_option_id.pop(option_id, 0)
            short_quantity = self.gross_short_quantity_by_option_id.pop(option_id, 0)
            if long_quantity == 0 and short_quantity == 0:
                continue
            expiry_value = option.expiry_valuation(new_values)
            settlement_credit = long_quantity * expiry_value + short_quantity * (1.0 - expiry_value)
            self.risk_budget_remaining += settlement_credit
            self.position.option_quantity_by_option_id[option_id] = 0
        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state

    def on_trade(self, option, price, quantity, counterparty_id):
        self.position.add_option_quantity(option.option_id, quantity)
        if quantity > 0:
            self.gross_long_quantity_by_option_id[option.option_id] += quantity
            maximum_loss = quantity * max(price, 0.0)
        elif quantity < 0:
            self.gross_short_quantity_by_option_id[option.option_id] += -quantity
            maximum_loss = -quantity * max(1.0 - price, 0.0)
        else:
            maximum_loss = 0.0
        self.risk_budget_remaining = max(self.risk_budget_remaining - maximum_loss, 0.0)

    @property
    def name(self):
        return 'Posterior Ridge Submission'

    def _value_by_underlying_id(self):
        return {underlying.underlying_id: underlying.value for underlying in self.underlying_state}

    @staticmethod
    def _safe_probability(value, fallback=0.5):
        if not math.isfinite(value):
            return fallback
        return min(max(value, 0.0), 1.0)

    def _price_with_company_models(self, option, *, models, fed_distribution, include_estimation_uncertainty):
        values = self._value_by_underlying_id()
        current_fed = values[FED_FUNDS_RATE_UNDERLYING_ID]
        if len(option.legs) == 1:
            leg = option.legs[0]
            if leg.underlying_id == AJARAI_UNDERLYING_ID:
                return price_single_name_binary(models.ajarai, current_value=values[AJARAI_UNDERLYING_ID], current_fed=current_fed, fed_terminal_distribution=fed_distribution, steps=option.steps_until_expiry, strike=option.strike, weight=leg.weight, include_estimation_uncertainty=include_estimation_uncertainty)
            if leg.underlying_id == THERIODIC_UNDERLYING_ID:
                return price_single_name_binary(models.theriodic, current_value=values[THERIODIC_UNDERLYING_ID], current_fed=current_fed, fed_terminal_distribution=fed_distribution, steps=option.steps_until_expiry, strike=option.strike, weight=leg.weight, include_estimation_uncertainty=include_estimation_uncertainty)
        if len(option.legs) == 2:
            weight_by_id = {leg.underlying_id: leg.weight for leg in option.legs}
            if set(weight_by_id) == {AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID}:
                return price_company_comparison_binary(models.relative_ajarai_to_theriodic, current_ajarai=values[AJARAI_UNDERLYING_ID], current_theriodic=values[THERIODIC_UNDERLYING_ID], current_fed=current_fed, fed_terminal_distribution=fed_distribution, steps=option.steps_until_expiry, ajarai_weight=weight_by_id[AJARAI_UNDERLYING_ID], theriodic_weight=weight_by_id[THERIODIC_UNDERLYING_ID], strike=option.strike, include_estimation_uncertainty=include_estimation_uncertainty)
        raise ValueError('unsupported company option structure')

    def _estimated_price_and_uncertainty(self, option):
        values = self._value_by_underlying_id()
        if option.steps_until_expiry == 0:
            return (option.expiry_valuation(values), 0.0)
        if not self.models_ready or self.fed_fit is None or self.company_models is None:
            return (0.5, self.MAX_UNCERTAINTY_BUFFER)
        try:
            if len(option.legs) == 1 and option.legs[0].underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                leg = option.legs[0]
                fair_value, price_std = posterior_predictive_price(self.fed_fit, current_rate=values[FED_FUNDS_RATE_UNDERLYING_ID], steps=option.steps_until_expiry, strike=option.strike, weight=leg.weight)
                return (self._safe_probability(fair_value), min(max(price_std, 0.0), self.MAX_UNCERTAINTY_BUFFER))
            posterior_fed_distribution = posterior_terminal_distribution(self.fed_fit, current_rate=values[FED_FUNDS_RATE_UNDERLYING_ID], steps=option.steps_until_expiry)
            fair_value = self._price_with_company_models(option, models=self.company_models, fed_distribution=posterior_fed_distribution, include_estimation_uncertainty=True)
            mle_fed_distribution = terminal_distribution(self.fed_fit.mle, current_rate=values[FED_FUNDS_RATE_UNDERLYING_ID], steps=option.steps_until_expiry)
            plug_in_value = self._price_with_company_models(option, models=self.company_models, fed_distribution=mle_fed_distribution, include_estimation_uncertainty=False)
            uncertainty = min(abs(fair_value - plug_in_value), self.MAX_UNCERTAINTY_BUFFER)
            return (self._safe_probability(fair_value), uncertainty)
        except (KeyError, ValueError, ZeroDivisionError, OverflowError):
            return (0.5, self.MAX_UNCERTAINTY_BUFFER)

    def price_option(self, option):
        fair_value, _uncertainty = self._estimated_price_and_uncertainty(option)
        return fair_value

    def price_option_from_parameters(self, market_parameters, option):
        values = self._value_by_underlying_id()
        if option.steps_until_expiry == 0:
            return option.expiry_valuation(values)
        try:
            fed_parameters = FedParameters(rate_up_probability=market_parameters.rate_up_probability, rate_down_probability=market_parameters.rate_down_probability, rate_reversion_strength=market_parameters.rate_reversion_strength, rate_step=market_parameters.rate_step, rate_target=market_parameters.rate_target)
            current_fed = values[FED_FUNDS_RATE_UNDERLYING_ID]
            if len(option.legs) == 1 and option.legs[0].underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                leg = option.legs[0]
                return self._safe_probability(price_fed_binary(fed_parameters, current_rate=current_fed, steps=option.steps_until_expiry, strike=option.strike, weight=leg.weight))
            company_parameters = CompanySimulationParameters(ajarai_drift=market_parameters.ajarai_drift, ajarai_idio_std_dev=market_parameters.ajarai_idio_std_dev, ajarai_rate_beta=market_parameters.ajarai_rate_beta, ajarai_sector_beta=market_parameters.ajarai_sector_beta, sector_std_dev=market_parameters.sector_std_dev, theriodic_drift=market_parameters.theriodic_drift, theriodic_idio_std_dev=market_parameters.theriodic_idio_std_dev, theriodic_rate_beta=market_parameters.theriodic_rate_beta, theriodic_sector_beta=market_parameters.theriodic_sector_beta)
            known_models = known_company_models(company_parameters)
            fed_distribution = terminal_distribution(fed_parameters, current_rate=current_fed, steps=option.steps_until_expiry)
            return self._safe_probability(self._price_with_company_models(option, models=known_models, fed_distribution=fed_distribution, include_estimation_uncertainty=False))
        except (KeyError, ValueError, ZeroDivisionError, OverflowError):
            return 0.5

    @staticmethod
    def _floor_to_penny(price):
        return math.floor((price + 1e-12) * 100.0) / 100.0

    @staticmethod
    def _ceil_to_penny(price):
        return math.ceil((price - 1e-12) * 100.0) / 100.0

    def _rfq_quantity(self, *, per_contract_loss, position_room):
        if position_room <= 0:
            return 0
        risk_limit = min(self.initial_cash_balance * self.RFQ_TRADE_RISK_LIMIT, self.risk_budget_remaining)
        if per_contract_loss <= 1e-12:
            risk_quantity = self.MAX_RFQ_QUANTITY
        else:
            risk_quantity = math.floor((risk_limit + 1e-12) / per_contract_loss)
        return max(min(risk_quantity, position_room, self.MAX_RFQ_QUANTITY), 0)

    def quote(self, option, counterparty_id):
        fair_value, uncertainty = self._estimated_price_and_uncertainty(option)
        if not self.models_ready and option.steps_until_expiry > 0:
            return Quote(0.0, 1, 1.0, 1)
        current_position = self.position.option_quantity_by_option_id[option.option_id]
        inventory_skew = min(max(self.INVENTORY_SKEW_PER_CONTRACT * current_position, -self.MAX_INVENTORY_SKEW), self.MAX_INVENTORY_SKEW)
        reservation_price = fair_value - inventory_skew
        half_spread = self.RFQ_BASE_HALF_SPREAD + uncertainty
        raw_bid = reservation_price - half_spread
        raw_offer = reservation_price + half_spread
        bid = min(max(self._floor_to_penny(raw_bid), 0.0), 0.99)
        offer = min(max(self._ceil_to_penny(raw_offer), 0.01), 1.0)
        if bid >= offer:
            if fair_value <= 0.5:
                bid = max(0.0, offer - 0.01)
            else:
                offer = min(1.0, bid + 0.01)
        if bid >= offer:
            bid, offer = (0.0, 1.0)
        bid_quantity = self._rfq_quantity(per_contract_loss=bid, position_room=self.MAX_ABS_POSITION_PER_OPTION - current_position)
        offer_quantity = self._rfq_quantity(per_contract_loss=1.0 - offer, position_room=self.MAX_ABS_POSITION_PER_OPTION + current_position)
        if bid_quantity == 0:
            bid = 0.0
            bid_quantity = 1
        if offer_quantity == 0:
            offer = 1.0
            offer_quantity = 1
        if bid >= offer:
            bid, offer = (0.0, 1.0)
        return Quote(bid_price=round(bid, 2), bid_quantity=bid_quantity, offer_price=round(offer, 2), offer_quantity=offer_quantity)

    def respond_to_fok(self, option, fok_order):
        if fok_order.option_id != option.option_id:
            return False
        if not self.models_ready and option.steps_until_expiry > 0:
            return False
        fair_value, uncertainty = self._estimated_price_and_uncertainty(option)
        current_position = self.position.option_quantity_by_option_id[option.option_id]
        if fok_order.order_type == OrderType.BUY:
            maker_quantity = -fok_order.quantity
            edge = fok_order.price - fair_value
            maximum_loss = fok_order.quantity * max(1.0 - fok_order.price, 0.0)
        elif fok_order.order_type == OrderType.SELL:
            maker_quantity = fok_order.quantity
            edge = fair_value - fok_order.price
            maximum_loss = fok_order.quantity * max(fok_order.price, 0.0)
        else:
            return False
        resulting_position = current_position + maker_quantity
        if abs(resulting_position) > self.MAX_ABS_POSITION_PER_OPTION:
            return False
        if maximum_loss > self.risk_budget_remaining + 1e-12:
            return False
        required_edge = self.FOK_MINIMUM_EDGE + min(uncertainty, self.MAX_UNCERTAINTY_BUFFER)
        return edge >= required_edge

    def warm_up(self, market_history):
        self.models_ready = False
        try:
            fed_history = market_history.values_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
            ajarai_history = market_history.values_by_underlying_id[AJARAI_UNDERLYING_ID]
            theriodic_history = market_history.values_by_underlying_id[THERIODIC_UNDERLYING_ID]
            self.fed_fit = fit_fed_model(fed_history, rate_step=RATE_STRIKE_GRID, rate_target=2.0)
            self.company_models = fit_company_models(fed_history, ajarai_history, theriodic_history, rate_step=RATE_STRIKE_GRID, ridge_move_equivalents=2.0, apply_finite_sample_inflation=True)
            self.models_ready = True
        except (KeyError, ValueError, ZeroDivisionError, OverflowError):
            self.fed_fit = None
            self.company_models = None
_StableMarketMaker = MarketMaker

class MarketMaker(_StableMarketMaker):
    FACTOR_RISK_LIMIT_FRACTION: Final[float] = 0.3
    MAX_FACTOR_SKEW: Final[float] = 0.03
    FACTOR_SKEW_STRENGTH: Final[float] = 2.0
    FULL_SIZE_ROBUST_EDGE: Final[float] = 0.05
    MEDIUM_UTILIZATION: Final[float] = 0.4
    HIGH_UTILIZATION: Final[float] = 0.6
    MEDIUM_SIZE_MULTIPLIER: Final[float] = 0.75
    HIGH_SIZE_MULTIPLIER: Final[float] = 0.5

    @property
    def name(self):
        return 'Posterior Ridge Dynamic-FOK Submission'

    def _factor_limit(self):
        return max(self.initial_cash_balance * self.FACTOR_RISK_LIMIT_FRACTION, 1e-12)

    def _factor_loadings(self, option):
        values = self._value_by_underlying_id()
        raw = {}
        company_scale = 0.0
        sector = 0.0
        indirect_fed = 0.0
        ajr_rate_beta = self.company_models.ajarai.rate_beta if self.company_models is not None else 0.0
        thr_rate_beta = self.company_models.theriodic.rate_beta if self.company_models is not None else 0.0
        for leg in option.legs:
            if leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                raw['FED'] = raw.get('FED', 0.0) + leg.weight
                continue
            if leg.underlying_id not in (AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID):
                continue
            loading = leg.weight * values[leg.underlying_id]
            company_scale = max(company_scale, abs(loading))
            sector += loading
            if leg.underlying_id == AJARAI_UNDERLYING_ID:
                raw['AJR'] = raw.get('AJR', 0.0) + loading
                indirect_fed += loading * ajr_rate_beta
            else:
                raw['THR'] = raw.get('THR', 0.0) + loading
                indirect_fed += loading * thr_rate_beta
        if company_scale > 0.0:
            for factor in ('AJR', 'THR'):
                if factor in raw:
                    raw[factor] /= company_scale
            raw['SECTOR'] = sector / company_scale
            raw['FED'] = raw.get('FED', 0.0) + indirect_fed / company_scale
        weights = {leg.underlying_id: leg.weight for leg in option.legs}
        if len(option.legs) == 2 and set(weights) == {AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID} and (option.strike == 0.0) and (weights[AJARAI_UNDERLYING_ID] * weights[THERIODIC_UNDERLYING_ID] < 0.0):
            raw['SECTOR'] = 0.0
            if self.company_models is not None:
                ajr_direction = 1.0 if weights[AJARAI_UNDERLYING_ID] > 0.0 else -1.0
                raw['FED'] = ajr_direction * self.company_models.relative_ajarai_to_theriodic.rate_beta
        return {factor: max(min(loading, 1.0), -1.0) for factor, loading in raw.items() if abs(loading) > 1e-12}

    @staticmethod
    def _signed_position_risk(position, fair_value):
        if position > 0:
            return position * fair_value
        if position < 0:
            return position * (1.0 - fair_value)
        return 0.0

    def _factor_contribution(self, option, position, fair_value=None):
        if position == 0:
            return {}
        probability = self.price_option(option) if fair_value is None else self._safe_probability(fair_value)
        signed_risk = self._signed_position_risk(position, probability)
        return {factor: loading * signed_risk for factor, loading in self._factor_loadings(option).items()}

    def _factor_exposure(self):
        exposure = {}
        seen = set()
        for option in self.active_option_state:
            if option.option_id in seen:
                continue
            seen.add(option.option_id)
            position = self.position.option_quantity_by_option_id[option.option_id]
            for factor, value in self._factor_contribution(option, position).items():
                exposure[factor] = exposure.get(factor, 0.0) + value
        return exposure

    def _project_exposure(self, option, exposure, current_position, projected_position, fair_value):
        projected = dict(exposure)
        current = self._factor_contribution(option, current_position, fair_value)
        replacement = self._factor_contribution(option, projected_position, fair_value)
        for factor in set(current) | set(replacement):
            projected[factor] = projected.get(factor, 0.0) - current.get(factor, 0.0) + replacement.get(factor, 0.0)
        return projected

    def _factor_score(self, exposure):
        limit = self._factor_limit()
        return sum(((value / limit) ** 2 for value in exposure.values()))

    def _reduces_portfolio(self, before, after):
        before_max = max((abs(value) for value in before.values()), default=0.0)
        after_max = max((abs(value) for value in after.values()), default=0.0)
        return self._factor_score(after) <= self._factor_score(before) + 1e-12 and after_max <= before_max + 1e-12

    def _factor_inventory_skew(self, option, exposure):
        loadings = self._factor_loadings(option)
        denominator = sum((value * value for value in loadings.values()))
        if denominator <= 1e-12:
            return 0.0
        aligned = sum((loading * exposure.get(factor, 0.0) for factor, loading in loadings.items())) / denominator
        return self.MAX_FACTOR_SKEW * math.tanh(self.FACTOR_SKEW_STRENGTH * aligned / self._factor_limit())

    def _factor_allowed_quantity(self, option, hard_quantity, direction, starting_position, fair_value, starting_exposure):
        limit = self._factor_limit()
        for quantity in range(hard_quantity, 0, -1):
            projected = self._project_exposure(option, starting_exposure, starting_position, starting_position + direction * quantity, fair_value)
            newly_breached = any((abs(projected.get(factor, 0.0)) > limit + 1e-12 and abs(projected.get(factor, 0.0)) > abs(starting_exposure.get(factor, 0.0)) + 1e-12 for factor in set(projected) | set(starting_exposure)))
            if self._reduces_portfolio(starting_exposure, projected) or not newly_breached:
                return quantity
        return 0

    def _portfolio_quantity(self, option, hard_quantity, direction, price, fair_value, uncertainty, exposure):
        current_position = self.position.option_quantity_by_option_id[option.option_id]
        directly_reducing = min(hard_quantity, max(-direction * current_position, 0))
        reduced_position = current_position + direction * directly_reducing
        reduced_exposure = self._project_exposure(option, exposure, current_position, reduced_position, fair_value)
        increasing_hard = hard_quantity - directly_reducing
        increasing_allowed = self._factor_allowed_quantity(option, increasing_hard, direction, reduced_position, fair_value, reduced_exposure)
        fully_projected = self._project_exposure(option, reduced_exposure, reduced_position, reduced_position + direction * increasing_allowed, fair_value)
        if direction > 0:
            robust_edge = max(fair_value - uncertainty, 0.0) - price
        else:
            robust_edge = price - min(fair_value + uncertainty, 1.0)
        if self._reduces_portfolio(reduced_exposure, fully_projected) or robust_edge >= self.FULL_SIZE_ROBUST_EDGE - 1e-12:
            multiplier = 1.0
        else:
            initial_budget = self.initial_cash_balance * self.CASH_USAGE_LIMIT
            global_utilization = 1.0 - self.risk_budget_remaining / initial_budget if initial_budget > 1e-12 else 1.0
            touched = self._factor_loadings(option)
            factor_utilization = max((abs(reduced_exposure.get(factor, 0.0)) / self._factor_limit() for factor in touched), default=0.0)
            utilization = max(min(max(global_utilization, 0.0), 1.0), factor_utilization)
            if utilization < self.MEDIUM_UTILIZATION:
                multiplier = 1.0
            elif utilization < self.HIGH_UTILIZATION:
                multiplier = self.MEDIUM_SIZE_MULTIPLIER
            else:
                multiplier = self.HIGH_SIZE_MULTIPLIER
        increasing = math.ceil(multiplier * increasing_allowed - 1e-12) if increasing_allowed > 0 else 0
        return min(hard_quantity, directly_reducing + increasing)

    def quote(self, option, counterparty_id):
        del counterparty_id
        fair_value, uncertainty = self._estimated_price_and_uncertainty(option)
        if not self.models_ready and option.steps_until_expiry > 0:
            return Quote(0.0, 1, 1.0, 1)
        current_position = self.position.option_quantity_by_option_id[option.option_id]
        option_skew = min(max(self.INVENTORY_SKEW_PER_CONTRACT * current_position, -self.MAX_INVENTORY_SKEW), self.MAX_INVENTORY_SKEW)
        exposure = self._factor_exposure()
        reservation_price = fair_value - option_skew - self._factor_inventory_skew(option, exposure)
        half_spread = self.RFQ_BASE_HALF_SPREAD + uncertainty
        bid = min(max(self._floor_to_penny(reservation_price - half_spread), 0.0), 0.99)
        offer = min(max(self._ceil_to_penny(reservation_price + half_spread), 0.01), 1.0)
        if bid >= offer:
            if fair_value <= 0.5:
                bid = max(0.0, offer - 0.01)
            else:
                offer = min(1.0, bid + 0.01)
        if bid >= offer:
            bid, offer = (0.0, 1.0)
        bid_hard = self._rfq_quantity(per_contract_loss=bid, position_room=self.MAX_ABS_POSITION_PER_OPTION - current_position)
        offer_hard = self._rfq_quantity(per_contract_loss=1.0 - offer, position_room=self.MAX_ABS_POSITION_PER_OPTION + current_position)
        bid_quantity = self._portfolio_quantity(option, bid_hard, 1, bid, fair_value, uncertainty, exposure)
        offer_quantity = self._portfolio_quantity(option, offer_hard, -1, offer, fair_value, uncertainty, exposure)
        if bid_quantity == 0:
            bid, bid_quantity = (0.0, 1)
        if offer_quantity == 0:
            offer, offer_quantity = (1.0, 1)
        if bid >= offer:
            bid, offer = (0.0, 1.0)
        return Quote(bid_price=round(bid, 2), bid_quantity=bid_quantity, offer_price=round(offer, 2), offer_quantity=offer_quantity)

    def _execution_family(self, option):
        if len(option.legs) != 1:
            return 'SPREAD'
        uid = option.legs[0].underlying_id
        if uid == FED_FUNDS_RATE_UNDERLYING_ID:
            return 'FED'
        if uid == AJARAI_UNDERLYING_ID:
            return 'AJR'
        if uid == THERIODIC_UNDERLYING_ID:
            return 'THR'
        return 'UNKNOWN'

    def _calibrated_uncertainty(self, option, fair_value, raw):
        family = self._execution_family(option)
        steps = max(option.steps_until_expiry, 1)
        if family == 'FED':
            estimate = max(2.0 * raw, 0.006 * math.sqrt(steps))
        elif family in ('AJR', 'THR', 'SPREAD'):
            if family == 'AJR':
                model = self.company_models.ajarai
            elif family == 'THR':
                model = self.company_models.theriodic
            else:
                model = self.company_models.relative_ajarai_to_theriodic
            n = max(model.diagnostics.num_observations, 1) if model.diagnostics is not None else 1
            scale = 0.3 if family == 'SPREAD' else 0.8
            estimate = scale * math.sqrt(max(fair_value * (1.0 - fair_value), 0.0)) * math.sqrt(steps / n)
        else:
            estimate = raw
        return min(max(raw, estimate, 0.0), 0.12)

    def respond_to_fok(self, option, fok_order):
        if fok_order.option_id != option.option_id:
            return False
        if not self.models_ready and option.steps_until_expiry > 0:
            return False
        fair_value, uncertainty = self._estimated_price_and_uncertainty(option)
        current_position = self.position.option_quantity_by_option_id[option.option_id]
        if fok_order.order_type == OrderType.BUY:
            maker_quantity = -fok_order.quantity
            edge = fok_order.price - fair_value
            maximum_loss = fok_order.quantity * max(1.0 - fok_order.price, 0.0)
        elif fok_order.order_type == OrderType.SELL:
            maker_quantity = fok_order.quantity
            edge = fair_value - fok_order.price
            maximum_loss = fok_order.quantity * max(fok_order.price, 0.0)
        else:
            return False
        projected_position = current_position + maker_quantity
        if abs(projected_position) > self.MAX_ABS_POSITION_PER_OPTION:
            return False
        if maximum_loss > self.risk_budget_remaining + 1e-12:
            return False
        direction = 1 if maker_quantity > 0 else -1
        exposure = self._factor_exposure()
        allowed = self._factor_allowed_quantity(option, fok_order.quantity, direction, current_position, fair_value, exposure)
        if allowed < fok_order.quantity:
            return False
        if abs(projected_position) < abs(current_position):
            base_edge = 0.015
        else:
            projected = self._project_exposure(option, exposure, current_position, projected_position, fair_value)
            calibrated = self._calibrated_uncertainty(option, fair_value, uncertainty)
            if self._reduces_portfolio(exposure, projected) and calibrated <= 0.04:
                base_edge = 0.02
            else:
                base_edge = 0.03
        return edge >= base_edge + uncertainty - 1e-12

_OriginalDynamicFokMarketMaker = MarketMaker


class MarketMaker(_OriginalDynamicFokMarketMaker):
    """Add capacity only where a binary contract has zero maximum loss."""

    ZERO_LOSS_QUANTITY = 50

    @property
    def name(self):
        return 'Posterior Ridge Zero-Loss Boundary Dynamic-FOK'

    def quote(self, option, counterparty_id):
        base = super().quote(option, counterparty_id)
        bid_quantity = base.bid_quantity
        offer_quantity = base.offer_quantity
        if base.bid_price <= 1e-12:
            bid_quantity = max(bid_quantity, self.ZERO_LOSS_QUANTITY)
        if base.offer_price >= 1.0 - 1e-12:
            offer_quantity = max(offer_quantity, self.ZERO_LOSS_QUANTITY)
        return Quote(
            base.bid_price,
            bid_quantity,
            base.offer_price,
            offer_quantity,
        )

    def respond_to_fok(self, option, fok_order):
        if fok_order.option_id != option.option_id:
            return False
        if fok_order.order_type == OrderType.SELL and fok_order.price <= 1e-12:
            return True
        if fok_order.order_type == OrderType.BUY and fok_order.price >= 1.0 - 1e-12:
            return True
        return super().respond_to_fok(option, fok_order)