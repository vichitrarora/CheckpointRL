# Metric definitions

## Primary objective

For each job, W is required useful computation, T is actual elapsed time and P
is penalized time: P=T on completion, otherwise P=10W after stopping at T=8W.
Normalized time N=P/W. Excess E=100(N-1). The primary score is the arithmetic
mean of E across all worlds and retained seeds, with equal weight per run.
It is not the ratio of total elapsed time to total required work. A 1,000 s job
finishing in 1,300 s has 30% excess; a timeout has 900% excess.

Rank independently within test and OOD by mean excess, lower first. Completion
and tails remain visible; no composite score combines diagnostics into a rank.

## Outcomes and tails

| Output field | Definition |
| --- | --- |
| runs / worlds | Executions / distinct scenario IDs. RL has three runs per world; a deterministic baseline has one. |
| training_seed_count | Number of retained fitted seeds (baseline reporting uses seed 0 as a placeholder). |
| mean_normalized_time / per_job_excess_pct | Mean N / mean E. |
| completion_pct / timeouts | 100 times fraction completed / number unfinished at the cap. |
| mean_elapsed_seconds | Mean actual T, including capped runs. |
| mean_penalized_seconds | Mean P, including timeout scoring penalties. |
| median_seconds | Median P. |
| p95_seconds / p95_excess_pct | 95th percentile of P / E, using NumPy's linear quantile convention. |
| cvar95_seconds / cvar95_excess_pct | Mean of the largest ceil(0.05*runs) values of P / E. |
| efficiency_pct | 100*sum(retained progress)/sum(T). Capped jobs contribute partial retained progress. |

Lower overhead and tails and higher completion are desirable. Mean seconds and
mean relative overhead weight different job sizes differently, so their rankings
can differ. P95 is a threshold; CVaR95 measures severity in the worst 5%.

## Costs and checkpoint counters

| Output field | Definition |
| --- | --- |
| attempts_per_job | Mean initiated checkpoint writes. |
| successful_writes_per_job | Mean fully committed writes. |
| failed_writes_per_job | Mean writes interrupted by a failure. |
| unfinished_writes_per_job | Mean writes interrupted by the episode cap. |
| writing_seconds_per_job | Mean time spent writing, including interrupted writes. |
| failed_write_seconds_per_job | Mean time in failure-interrupted writes, a subset of writing time. |
| lost_work_seconds_per_job | Mean discarded computation due to rollback. |
| recovery_seconds_per_job | Mean recovery delay, including restarted recovery. |
| used_writes_per_job | Mean distinct committed checkpoints restored at least once. |
| observed_unused_writes_per_job | Successful writes minus distinct used writes. Non-use does not prove a write was unnecessary. |
| compute_actions_per_job / checkpoint_actions_per_job | Mean primitive computation / write operations. |
| writing_fraction_pct / lost_work_fraction_pct / recovery_fraction_pct | 100*sum(component time)/sum(T), not the average of per-job fractions. |

Attempts = successful + failed + unfinished writes. Actual time satisfies
`T = retained progress + lost work + writing + recovery`. Efficiency and the
three component fractions therefore sum to 100%. Do not add failed-write time
again to writing time. Counts are diagnostics, not independent optimization goals.

## Failure protection and checkpoint gaps

| Output field | Definition |
| --- | --- |
| active_failures | Total failures during computation or writing. |
| recovery_failures | Total failures during recovery, counted separately. |
| usable_failures | Active failures with a durable checkpoint available. |
| failures_without_usable_checkpoint | Active minus usable failures. |
| usable_pct | 100*usable_failures/active_failures. |
| protected_failures / protected_pct | Active failures with a usable checkpoint and lost computation <=300 s / their percentage of **all active failures**. |
| lost_seconds_per_active_failure | Sum(lost work)/sum(active failures). |
| mean_gap_seconds / median_gap_seconds / p95_gap_seconds | Failure wall-clock time minus latest successful checkpoint completion, conditional on a usable checkpoint. |

Failure gaps can include writing or recovery time and differ from lost useful
computation. Failures without checkpoints are not assigned zero gaps. Undefined
ratios and empty gap populations are null in JSON and N/A in Markdown. Event
counts pool repeated seed-world runs and are not independent hardware failures.

## Stability, uncertainty and checks

| Output field | Definition |
| --- | --- |
| seed_excess_std_pp | Sample standard deviation (ddof=1) of the three seed mean excess scores; null for one seed. |
| seed_excess_min_pct / seed_excess_max_pct | Minimum / maximum per-seed mean excess. |
| excess_ci95_low_pct / excess_ci95_high_pct | 95% bootstrap interval for mean excess after averaging seeds within each world. |
| difference_excess_pp | Paired policy excess minus reference excess. Negative favors the policy. |
| ci95_low_pp / ci95_high_pp | Pointwise 95% paired world-bootstrap interval; 2,000 resamples. |
| world_win_pct / world_tie_pct | Fraction of paired worlds with lower mean normalized time / ties within 1e-10 after conversion to percentage-point differences. |
| max_accounting_residual | Maximum absolute time conservation error in seconds. |
| max_reward_identity_error | Maximum absolute error between raw episode return and 1-P/W. |

Bootstrap intervals hold fitted models fixed. They exclude uncertainty from
future training seeds and are not adjusted for multiple comparisons. Seed SD
describes only three fitted models. Verification errors should be near zero;
they are correctness checks, not performance metrics.

## Training diagnostics

Learning-curve JSON records decisions, completed episodes, training timeouts,
recent raw return, action counts, cell exposure and wall-clock runtime including
validation. DQN additionally logs epsilon, optimizer steps, mean squared TD loss
and gradient norm. PPO logs optimizer epochs/steps and library statistics such as
policy loss, value loss, entropy, approximate KL and clipping fraction. These
diagnose learning; the submitted ranking uses evaluation excess, not training loss.

Every policy has individual-seed results, 12 marginal slices (three values each
for process, work, risk and cost) and all 81 joint cells. These show where behavior
changes and do not replace the primary aggregate score.
