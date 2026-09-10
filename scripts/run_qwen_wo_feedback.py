"""Register and resume a Qwen server evaluation using stock feedback-v2 settings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import resume_closed_source_wo_feedback as controller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=('eb-hab', 'eb-nav'), required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=0, help='0 = all subsets in full; positive = Base pilot')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.output.resolve()
    cmd = controller.command_prefix(args.env) + ['--provider', 'qwen', '--model', args.model,
        '--env', args.env, '--track', 'rgb_only', '--eval-sets', 'all' if args.episodes == 0 else 'base',
        '--episodes', str(args.episodes), '--seed', '0', '--api-retries', '3',
        '--base-url', args.base_url, '--output', str(root), '--dry-run']
    config = json.loads(subprocess.check_output(cmd, cwd=controller.BENCH, text=True))
    if (root / 'config.json').exists():
        saved = controller.read(root / 'config.json')
        controller.compatible(saved, config)
        if any(saved[k] != config[k] for k in ('eval_sets', 'episodes', 'start_index')):
            raise ValueError('Existing experiment task range differs')
    elif root.exists():
        raise ValueError('Output exists without a registered config')
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    # Validate served ID before constructing a simulator or creating run data.
    req = Request(args.base_url.rstrip('/') + '/models',
                  headers={'Authorization': 'Bearer ' + (os.environ.get('QWEN_API_KEY') or 'EMPTY')})
    with urlopen(req, timeout=10) as response:
        models = json.load(response)
    if args.model not in [item['id'] for item in models.get('data', [])]:
        raise ValueError('Requested model ID is not advertised by this server')
    if not (root / 'config.json').exists():
        config.update(output=str(root), requested_output=str(root), dry_run=False,
                      source_hashes=controller.native_hashes(args.env), resume_managed=True,
                      endpoint_models=models)
        root.mkdir(parents=True, exist_ok=False)
        controller.write(root / 'config.json', config)
    return controller.main(['--run-dir', str(root)])


if __name__ == '__main__':
    raise SystemExit(main())
