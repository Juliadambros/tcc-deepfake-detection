from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.common.paths import resolve_from_root
from modelos.common.utils import load_json
from modelos.mesonet.train import train_single_run


def iter_search_space(search_space: dict):
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def main() -> None:
    parser = argparse.ArgumentParser(description='Executa grid search da MesoNet.')
    parser.add_argument('--config', required=True, help='JSON com search_space.')
    args = parser.parse_args()

    config_path = resolve_from_root(args.config)
    base_config = load_json(config_path)
    search_space = base_config.pop('search_space')

    all_results = []
    combinations = list(iter_search_space(search_space))
    print(f'Total de combinações: {len(combinations)}')

    for i, combo in enumerate(combinations, start=1):
        run_config = dict(base_config)
        run_config.update(combo)
        print('\n' + '=' * 80)
        print(f'Experimento {i}/{len(combinations)}')
        print(json.dumps(combo, indent=2, ensure_ascii=False))
        print('=' * 80)
        result = train_single_run(run_config)
        all_results.append(result)

    if all_results:
        best = max(all_results, key=lambda row: row['test_f1_macro'])
        print('\nMelhor experimento encontrado:')
        print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
