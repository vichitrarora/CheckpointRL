# Methods and reproducibility contract

## Five retained strategies

The package retains one PPO family, one DQN family and three baselines. All three
seeds are preserved for each RL family; no favorable seed is selected for reporting.

Family selection used the mean of the selected per-seed validation excess scores:
PPO with fractional actions: 25.6687%; PPO grid: 27.2893%; DQN grid: 27.4765%;
DQN fractions: 27.5503%. We retain PPO fractions and DQN grid. The fixed interval
is 1,200 s: validation excess 45.3552%, compared with 95.7500% for 300 s and
86.4487% for 4,800 s. These figures document selection, not additional packaged
strategies. The original experiment identifier is `canonical_ppo_dqn_v1`.

For each seed, the saved checkpoint minimizes validation mean normalized time;
ties use the earliest checkpoint. The six byte-preserved models, validation
scores, training budgets and selected steps are listed in `models/manifest.json`.

## Environment and actions

At each decision the actor chooses a numeric computation interval. It computes
for that duration, then checkpoints if the job has neither finished nor failed.
A failure interrupts the interval, rolls progress back to the latest committed
checkpoint and initiates recovery. After recovery the actor chooses again.
Writes are blocking and their duration is fixed when the write starts. A failure
during a write leaves the previous checkpoint intact. No checkpoint is written
after final completion. A failure exactly at completion takes precedence.

The DQN network has 18 outputs: 17 absolute intervals (60–15,360 s, listed in
`configs/environment.json`) and an explicit finish action. Intervals at least as
large as the remaining work are masked because they duplicate finish. The
15,360-second output (index 16) was never legal in training. The submitted DQN
uses the existing **trained-actions deployment variant**, which disables that
one output, including on longer OOD jobs. This restriction is fixed from the
training action support; it performs no search or baseline fallback. It is
essential for reproducing the reported 52.70% DQN OOD excess.

PPO has 24 outputs: the same grid and finish action plus 1/32, 1/16, 1/8, 1/4,
1/2 and 3/4 of remaining work, with fractions below 30 s masked. PPO uses the
original action mask. The fraction action library changes both resolution and
job-size scaling; this comparison is not an isolated PPO-versus-DQN algorithm
ablation with identical action spaces.

## Observations and reward

Both agents see the same 16 features, in this exact order:

1. Remaining work fraction.
2. Log remaining work.
3. Log total work.
4. Log current write cost.
5. Log public job-level MTBF estimate.
6. Log current MTBF estimate.
7. Current write cost / total work.
8. Remaining work / public MTBF estimate.
9. Remaining work / current MTBF estimate.
10. Recovery duration / total work.
11. Elapsed time / execution cap.
12. Time since last observed failure / public MTBF estimate.
13. Observed failure rate feature.
14. Phase within the current time block.
15. Log block duration.
16. Whether a durable checkpoint exists.

Exact normalization and clipping are in `checkpoint_rl/environment.py`. The
feature's historical name `log_nominal_mtbf` refers to `public_mtbf`, not the
private `nominal_mtbf` field. Actors receive only observation vectors and legal
action masks. Scenario IDs, process labels, future costs and future failures are
not actor inputs. No supervised predictor is trained.

For W required work and P scored execution time, the unscaled total return is
`1 - P/W`. Per-step rewards combine negative elapsed time / W with the change in
remaining-work potential. A timeout occurs at 8W and receives scored P=10W.
Gamma is 1. Training multiplies rewards by 10 without changing the objective.
Costs are not double-counted through extra checkpoint or failure penalties.

## Training

Double DQN uses a 96/96 ReLU MLP, a target network, proportional prioritized
replay (alpha 0.6), importance beta annealed from 0.4 to 1, squared TD error,
Adam updates and epsilon-greedy exploration. Network arithmetic, replay and
optimizer implementation are in `agents/dqn.py`; gamma=1 is explicit in the
target formula. Full training begins updates after 4,096 replay transitions.

PPO uses sb3-contrib MaskablePPO with separate 96/96 tanh actor and critic heads,
masked categorical actions, clipped policy updates, gamma=1 and GAE lambda=0.95.
Shared settings are in `configs/training.json`; algorithm hyperparameters are in
`configs/dqn.json` and `configs/ppo.json`. The merged settings are copied to each run's
`effective_config.json`. CPU threads are limited to one for reproducibility.

The DQN deployment restriction does not alter the original training loop: the
unsupported action is already masked by every training world. PPO rollout
rounding and repeated `learn(..., reset_num_timesteps=False)` calls preserve the
saved experiment's training procedure. Smoke mode explicitly changes only the
budget, validation subset and PPO rollout settings, and labels its outputs.

## Baselines

All three use the same computation, write, failure and recovery semantics.
Young uses `sqrt(2*C*M)`, where C is current checkpoint cost and M is public MTBF.
Daly uses `sqrt(2*C*M)*(1 + sqrt(z)/3 + z/9) - C`, `z=C/(2*M)`, with interval M
when C >= 2M. This names the exact Daly approximation used, rather than implying
a different implementation. Fixed uses 1,200 seconds. Each reschedules after
failure recovery. Formula outputs never enter the learned actors or training.

## Evaluation

Each strategy sees identical worlds. Frozen actors use deterministic inference;
training seeds remain separate runs. Per-world averages across seeds are used
before paired bootstrap resampling (2,000 replicates, fixed seed). Intervals are
conditional on these saved agents and are not confidence intervals over future
training runs; they have no multiple-comparison adjustment.

Dataset hashes, selected model hashes, complete panel coverage, time conservation
and reward identities are checked. Saved-model evaluation can be reproduced
without full retraining. Small numerical platform differences are possible;
the verifier fails explicitly if measured outcomes differ beyond its tolerance.
