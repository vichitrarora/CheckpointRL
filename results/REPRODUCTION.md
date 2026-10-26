# Verified submission workflow

Checked on 2026-10-26 in a fresh Python 3.13.1 virtual environment on Windows x64,
using CPU execution. The environment was created with `python -m venv .venv` and
installed from `requirements-lock.txt`, without access to the older repository's
Python packages or source folders. `pip check` reported no broken requirements.

After separating DQN and PPO into dedicated training modules, the complete
workflow was rechecked on **2026-10-27**, using the same locked Python
environment. The updated receipt records this run. Both optimizer loops and
the effective hyperparameters were also checked against their pre-refactor
versions and are unchanged.

The latest automated command was:

```bash
python -m scripts.check_readme --full-evaluation --output runs/balanced-readme-check
```

| Check | Observed outcome |
| --- | --- |
| Data and trained-model integrity | All 8,100 worlds and six model hashes verified. |
| Reference ledger integrity | All 21,870 runs verified, with expected world/seed coverage. |
| Unit tests | 30 passed: event timing, failures during writes/recovery, timeout scoring, causal observations, masks, save/load, replay targets, metric aggregation and report guards. |
| PPO training smoke | 512 decisions; 16 optimizer steps; validation on 81 worlds; selected model saved and reloaded. |
| DQN training smoke | 8,192 decisions; 516 optimizer steps after replay warmup; validation on 81 worlds; selected model saved and reloaded. |
| New smoke-agent evaluation | Both agents loaded from their new selection manifests and completed test/OOD evaluation on 81 worlds per split. |
| Shipped-agent smoke evaluation | All five strategies; 1,458 runs matched the reference. |
| Full saved-agent evaluation | **21,870 of 21,870 runs matched** the reference at rtol=1e-10, atol=1e-7 for floats, exact counters/metadata. |
| Reports and figures | Full results regenerated; five figures exported in PNG and PDF; submitted PNG layouts visually inspected. |
| Full retraining | **Not performed.** Only short optimizer smoke runs and the unit-test learning checks were run. |

The full evaluation/report/plot pipeline took approximately 71 seconds on the
checked machine. Runtime varies by hardware. The saved training experiment's
runtime information is separately retained in `training_runtime.json`.

[reproduction_checks.json](reproduction_checks.json) records commands, exit codes,
durations, log hashes, package versions, asset-manifest hashes and the complete
evaluation verification receipt. Logs and smoke model files are generated under
`runs/` and excluded from version control; rerun the command above to create new
ones. The shipped models in `models/` were not changed by these checks.

This validates the documented installation and execution workflow and reproduces
the saved agents' measured behavior. It does not assert bit-identical full
training on every platform.
