"""Create publication-ready figures from measured evaluation and validation logs."""
from checkpoint_rl.runtime import *
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'ppo': '#176B70', 'dqn': '#E58B45', 'young': '#7786A2', 'daly': '#AA769A', 'fixed1200': '#A2A898'}
SHORT = {'ppo': 'PPO', 'dqn': 'DQN', 'young': 'Young', 'daly': 'Daly', 'fixed1200': 'Fixed 1,200 s'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results')
    p.add_argument('--curves', type=Path, default=ROOT / 'results/learning_curves')
    p.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read(args.results / 'all_metrics.json')
    paired = read(args.results / 'paired_comparisons.json')
    lookup = {(r['split'], r['policy']): r for r in rows}
    if set(lookup) != {(s, p) for s in ('test', 'ood_test') for p in POLICIES}:
        raise ValueError('Plots require the complete five-strategy, two-split report')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.axisbelow': True})

    def save(fig, name, note):
        fig.text(.5, .025, note, ha='center', fontsize=9, color='#49515A')
        fig.tight_layout(rect=(0, .08, 1, .94))
        for suffix in ('png', 'pdf'):
            fig.savefig(args.output / f'{name}.{suffix}', dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Execution overhead on identical benchmark worlds', fontsize=15, weight='bold')
    for ax, split, title in zip(axes, ('test', 'ood_test'), ('Test · 1,620 worlds', 'OOD · 810 worlds')):
        ordered = sorted(POLICIES, key=lambda p: lookup[split, p]['per_job_excess_pct'])
        means = np.array([lookup[split, p]['per_job_excess_pct'] for p in ordered])
        lows = np.array([lookup[split, p]['excess_ci95_low_pct'] for p in ordered])
        highs = np.array([lookup[split, p]['excess_ci95_high_pct'] for p in ordered])
        ax.barh(range(5), means, color=[COLORS[p] for p in ordered], height=.62)
        ax.errorbar(means, range(5), xerr=np.array([means-lows, highs-means]), fmt='none', ecolor='#222A32', capsize=3)
        for i, value in enumerate(means):
            ax.text(highs[i] + 1.4, i, f'{value:.2f}%', va='center', fontsize=9)
        ax.set(yticks=range(5), yticklabels=[SHORT[p] for p in ordered], title=title,
               xlabel='Mean per-job excess time (%) ↓', xlim=(0, max(highs)*1.22))
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    save(fig, 'overhead', '95% world-bootstrap intervals; three RL seeds averaged within each world. Lower is better.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Tail risk and completion', fontsize=15, weight='bold')
    x = np.arange(5)
    for ax, split, title in zip(axes, ('test', 'ood_test'), ('Test', 'OOD')):
        ax.bar(x-.18, [lookup[split,p]['p95_excess_pct'] for p in POLICIES], .36, label='P95 excess', color='#7786A2')
        ax.bar(x+.18, [lookup[split,p]['cvar95_excess_pct'] for p in POLICIES], .36, label='CVaR95 excess', color='#176B70')
        for i,p in enumerate(POLICIES):
            y = lookup[split,p]['cvar95_excess_pct']
            ax.text(i, y+8, f"{lookup[split,p]['completion_pct']:.2f}%\ncomplete", ha='center', fontsize=8)
        ax.set(title=title, ylabel='Excess time (%) ↓', xticks=x,
               xticklabels=[SHORT[p] for p in POLICIES])
        ax.tick_params(axis='x', labelrotation=15); ax.margins(y=.23); ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='upper left', frameon=False)
    save(fig, 'tail_risk', 'CVaR95 is the mean of the worst 5% of runs. Timeout penalties are included.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Where execution overhead is spent', fontsize=15, weight='bold')
    for ax, split, title in zip(axes, ('test', 'ood_test'), ('Test', 'OOD')):
        bottom = np.zeros(5)
        for key, label, color in [('writing_seconds_per_job','Writing','#176B70'),
                                  ('lost_work_seconds_per_job','Lost computation','#E58B45'),
                                  ('recovery_seconds_per_job','Recovery','#7786A2')]:
            values = np.array([lookup[split,p][key] for p in POLICIES])
            ax.bar(x, values, .65, bottom=bottom, label=label, color=color); bottom += values
        ax.set(title=title, ylabel='Measured seconds per job', xticks=x, xticklabels=[SHORT[p] for p in POLICIES])
        ax.tick_params(axis='x', labelrotation=15); ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False)
    save(fig, 'cost_breakdown', 'Actual time components, including capped jobs; the additional timeout scoring penalty is excluded.')

    fig, ax = plt.subplots(figsize=(10, 5))
    for policy in ('ppo', 'dqn'):
        curves = [read(args.curves / f'{policy}_seed{s}.json') for s in config()['seeds']]
        steps = np.array([r['decisions'] for r in curves[0]]) / 1e6
        if any([r['decisions'] for r in c] != [r['decisions'] for r in curves[0]] for c in curves):
            raise ValueError('Training curves have inconsistent evaluation steps')
        values = np.array([[r['validation']['per_job_excess_pct'] for r in c] for c in curves])
        ax.plot(steps, values.mean(axis=0), label=SHORT[policy], color=COLORS[policy], linewidth=2.3)
        ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=COLORS[policy], alpha=.16)
    ax.set(title='Validation performance during training', xlabel='Environment decisions (millions)',
           ylabel='Validation mean excess time (%) ↓')
    ax.grid(alpha=.2); ax.legend(frameon=False)
    save(fig, 'learning_curves', 'Lines: mean across three seeds. Shading: seed minimum–maximum, not a confidence interval.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle('PPO improvement against each retained comparator', fontsize=15, weight='bold')
    for ax, split, title in zip(axes, ('test', 'ood_test'), ('Test', 'OOD')):
        pairs = [r for r in paired if r['split'] == split and r['policy'] == 'ppo']
        for i, row in enumerate(pairs):
            ax.plot([row['ci95_low_pp'], row['ci95_high_pp']], [i, i], color='#176B70', linewidth=2)
            ax.scatter([row['difference_excess_pp']], [i], color='#176B70', s=45)
        ax.axvline(0, color='#7786A2', linestyle='--')
        ax.set(yticks=range(len(pairs)), yticklabels=[SHORT[r['reference']] for r in pairs],
               xlabel='PPO minus comparator (percentage points)', title=title)
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    save(fig, 'paired_comparisons', 'Negative favors PPO. Pointwise 95% world-bootstrap intervals, conditional on the saved models.')
    print('Saved five figures in PNG and PDF to', args.output)


if __name__ == '__main__':
    main()
