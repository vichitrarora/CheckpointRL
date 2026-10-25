"""Reproduce shipped evaluation and figures; training is deliberately separate."""
from checkpoint_rl.runtime import new_output
from pathlib import Path
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/reproduction')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    output = new_output(a.output)
    commands = [
        ['-m', 'scripts.verify'],
        ['evaluate.py', '--output', str(output / 'evaluation')],
        ['-m', 'scripts.verify', '--evaluation', str(output / 'evaluation')],
        ['report.py', '--input', str(output / 'evaluation/raw'), '--output', str(output / 'results')],
        ['-m', 'plots.make_plots', '--results', str(output / 'results'), '--output', str(output / 'figures')],
    ]
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=root, check=True)
    print('Reproduction complete:', output)


if __name__ == '__main__':
    main()
