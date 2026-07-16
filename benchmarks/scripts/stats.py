"""Statistical utilities for benchmark trials."""
import math


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


T_TABLE_95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    15: 2.145, 20: 2.093, 25: 2.064, 30: 2.042,
}


def _t_value(n):
    if n in T_TABLE_95:
        return T_TABLE_95[n]
    for k in sorted(T_TABLE_95.keys()):
        if k >= n:
            return T_TABLE_95[k]
    return 1.96


def ci95(values):
    n = len(values)
    if n < 2:
        return None
    s = stddev(values)
    t = _t_value(n)
    margin = t * s / math.sqrt(n)
    m = mean(values)
    return {
        "mean": round(m, 4),
        "ci_low": round(m - margin, 4),
        "ci_high": round(m + margin, 4),
        "margin": round(margin, 4),
        "stddev": round(s, 4),
        "n": n,
    }


def summarize_trials(trial_values, label="metric"):
    if not trial_values:
        return {}
    result = ci95(trial_values)
    if result is None:
        return {"mean": trial_values[0], "n": 1}
    return result
