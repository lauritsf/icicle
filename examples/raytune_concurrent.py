"""Tune a classifier with ICICLE using three Ray Tune workers.

This extends raytune_freeze_thaw.py to train configurations concurrently.
It requires six CPU slots, with two assigned to each worker. ICICLE chooses
what to train as workers become available.

    uv pip install '.[ray]' matplotlib
    python examples/raytune_concurrent.py

Results and plots are saved under ray_results/. The script also prints the
maximum number of training steps observed running at the same time.
"""

from raytune_freeze_thaw import main

if __name__ == "__main__":
    results = main(
        workers=3, candidates=48, total_steps=480, features=2048,
    )
    events = []
    for result in results:
        if "val_acc" in result.metrics:
            rows = result.metrics_dataframe
            events.extend((row.step_started, 1) for row in rows.itertuples())
            events.extend((row.step_finished, -1) for row in rows.itertuples())
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    print(f"Maximum observed simultaneous training steps: {peak}")
