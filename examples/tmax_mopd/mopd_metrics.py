"""Multi-teacher OPD's per-rollout W&B metrics: each domain's own, under ``rollout/<domain>/``.

Two hooks in the rollout process, as tmax_metrics' and search_metrics':

* ``missing_reward_filter`` (``--dynamic-sampling-filter-path``) records each finished group with its domain's metrics
  module, then applies Miles' missing-reward filter: pure OPD trains on every group whose task reward is known;
* ``log_rollout_data`` (``--custom-rollout-log-function-path``) logs each domain's metrics once per trained batch
  (tmax_metrics' under ``rollout/terminal/``, search_metrics' under ``rollout/search/``), the domain's share of the
  trained samples (``rollout/<domain>/trained/fraction``) and the samples trained so far
  (``rollout/trained/samples_total``). Miles' own rollout metrics (``rollout/raw_reward``, ...) cover both domains.
"""

import search_metrics
import tmax_metrics

from miles.rollout.filter_hub.base_types import FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import apply_missing_reward_filter
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

_METRICS = {"terminal": tmax_metrics, "search": search_metrics}
_SAMPLES_TOTAL = "rollout/trained/samples_total"


def missing_reward_filter(args, samples, **kwargs) -> FilterOutput:
    """Record the group with its domain's metrics, then Miles' missing-reward filter alone (pure OPD: no
    zero-variance filter, since a uniform group still carries the full OPD signal)."""
    domain = next(iter(iter_samples(samples))).metadata["domain"]
    _METRICS[domain].record_group(args, samples)
    return apply_missing_reward_filter(args, samples, **kwargs)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    metrics = rollout_metrics(args, rollout_id, samples)
    if metrics:
        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        tracking.log(args, metrics, step_key="rollout/step")
    return False


def rollout_metrics(args, rollout_id: int, samples) -> dict[str, float]:
    metrics = {}
    for domain, module in _METRICS.items():
        trained = [s for s in samples if s.metadata["domain"] == domain]
        domain_metrics = module.rollout_metrics(args, rollout_id, trained)
        # Counts every domain's samples, not this domain's: logged once, without the prefix.
        if _SAMPLES_TOTAL in domain_metrics:
            metrics[_SAMPLES_TOTAL] = domain_metrics.pop(_SAMPLES_TOTAL)
        metrics |= {key.replace("rollout/", f"rollout/{domain}/", 1): value for key, value in domain_metrics.items()}
        if samples:
            metrics[f"rollout/{domain}/trained/fraction"] = len(trained) / len(samples)
    return metrics
