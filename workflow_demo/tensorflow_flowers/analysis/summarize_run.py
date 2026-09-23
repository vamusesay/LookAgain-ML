"""Recompute paired stack-minus-single uncertainty from a completed public run."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
def probabilities(frame: pd.DataFrame, method: str) -> np.ndarray:
    columns = [f"pred_{method}__class_{index}" for index in range(5)]
    return frame[columns].to_numpy(dtype=float)

def bootstrap_stack_difference(frame: pd.DataFrame) -> dict[str, object]:
    selected = probabilities(frame, "siglip2_b16")
    stack = probabilities(frame, "linear_stack")
    y = frame["outcome"].to_numpy(dtype=int)
    groups = frame["group_id"].astype(str).to_numpy()
    unique = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group) for group in unique]

    def metrics(probability: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
        local = np.clip(probability[rows], 1e-15, 1.0)
        accuracy = float(np.mean(np.argmax(local, axis=1) == y[rows]))
        log_loss = float(-np.mean(np.log(local[np.arange(len(rows)), y[rows]])))
        return accuracy, log_loss

    rows = np.arange(len(frame))
    selected_point = metrics(selected, rows)
    stack_point = metrics(stack, rows)
    rng = np.random.default_rng(20260818)
    draws = np.empty((2000, 2), dtype=float)
    for index in range(len(draws)):
        sample = rng.integers(0, len(group_rows), size=len(group_rows))
        take = np.concatenate([group_rows[value] for value in sample])
        selected_draw = metrics(selected, take)
        stack_draw = metrics(stack, take)
        draws[index] = stack_draw[0] - selected_draw[0], stack_draw[1] - selected_draw[1]
    return {
        "selected_accuracy": selected_point[0],
        "selected_log_loss": selected_point[1],
        "stack_accuracy": stack_point[0],
        "stack_log_loss": stack_point[1],
        "accuracy_difference": stack_point[0] - selected_point[0],
        "log_loss_difference": stack_point[1] - selected_point[1],
        "accuracy_interval": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
        "log_loss_interval": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
        "models_retrained_in_draw": False,
    }
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    result=bootstrap_stack_difference(pd.read_csv(a.results/'fixed_test_predictions.csv'))
    (a.results/'stack_vs_selected_paired_bootstrap.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
