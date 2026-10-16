from __future__ import annotations

import pandas as pd


def compare_success_control(
    success: pd.DataFrame,
    control: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in metrics:
        if metric not in success or metric not in control:
            continue
        success_values = pd.to_numeric(success[metric], errors="coerce").dropna()
        control_values = pd.to_numeric(control[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "success_mean": success_values.mean(),
                "success_median": success_values.median(),
                "control_mean": control_values.mean(),
                "control_median": control_values.median(),
                "mean_difference": success_values.mean() - control_values.mean(),
            }
        )
    return pd.DataFrame(rows)

