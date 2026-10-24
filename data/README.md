# Dataset schema and split contract

Each JSONL line defines one complete job scenario. Load with
`checkpoint_rl.runtime.corpus(split)`. All durations and timestamps are seconds.

| File | Worlds | Use |
| --- | ---: | --- |
| train.jsonl | 5,184 | Environment interaction for RL fitting |
| validation.jsonl | 486 | Model checkpoint selection |
| test.jsonl | 1,620 | In-distribution evaluation |
| ood_test.jsonl | 810 | Evaluation under shifted conditions |

Each split covers 81 combinations of process, work band, risk band and checkpoint
cost band. IDs are unique across splits. `manifest.json` provides row counts and
SHA-256 checksums. Keep files unchanged when reproducing the submitted results.

| Field | Meaning and access |
| --- | --- |
| id | Unique scenario identifier; reporting only. |
| split | Split membership; reporting only. |
| cell | `work_band`, `risk_band`, `cost_band` (0–2) and `process`; analysis strata, not actor inputs. |
| work | Required useful computation, known at job start. |
| horizon | Execution cutoff, equal to eight times `work`. |
| nominal_mtbf | Scenario-level reference mean time between failures; not exposed directly to actors. |
| public_mtbf | Available job-level MTBF estimate used by observations and Young/Daly. |
| recovery | Restart duration; another failure during recovery restarts this delay. |
| block | Duration of each time block for cost and sensor lookup. |
| costs | Checkpoint durations by time block. Only the current entry is observable. |
| current_mtbf_estimates | MTBF sensor estimates by block. Only the current entry is observable. |
| failures | Ordered failure timestamps, consumed privately by the environment; never actor inputs. |
| telemetry | Primitive-environment telemetry array; not included in the retained agents' 16-feature observation. |
| aging_telemetry | Primitive-environment telemetry mode flag; not an actor input. |

`process` values are `stationary`, `variable_io` and `switching`. The last two
allow changing checkpoint costs; `switching` also has changing failure conditions.
OOD has longer jobs, greater failure exposure, larger checkpoint costs and noisier
current estimates. The model sees causal measurements, not future array values.

The evaluator gives every strategy the same scenario, including the same failure
timeline. A strategy's elapsed time determines which failures occur while it is
active. Thus active failure counts can differ even on identical worlds.

The benchmark assumes known total work, one durable checkpoint level, blocking
writes, and an available current MTBF estimate. There is no model of distributed
storage corruption, multiple checkpoint tiers or an unknown job length. Claims
should remain within this evaluated scope. The test splits have been previously
observed and must not be described as an untouched final holdout.
