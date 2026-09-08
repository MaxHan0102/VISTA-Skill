"""Four-rollout diagnostic recheck of a frozen candidate after an evidence bug fix."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.phase5_recovery_pilot import REPO, analyze_stage, preregister, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base-urls', nargs=2, required=True)
    args = parser.parse_args()
    path = Path(args.protocol).resolve()
    protocol = read(path)
    output = REPO / protocol['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in protocol['source_artifacts'].items():
        source = REPO / protocol['source_run'] / name
        if sha(source) != expected:
            raise ValueError(f'Source artifact changed: {name}')
        target = output / name
        if target.exists() and sha(target) != expected:
            raise ValueError(f'Recheck input changed: {name}')
        if not target.exists():
            shutil.copyfile(source, target)
    if sha(__file__) != protocol['recheck_driver_sha256']:
        raise ValueError('Recheck driver changed')
    manifest = preregister(path, protocol, args.base_urls)
    processes, logs = [], []
    for index in (0, 1):
        log = (output / f'mechanism_worker{index}.log').open('a')
        logs.append(log)
        processes.append(subprocess.Popen([
            sys.executable, str(REPO / 'scripts/phase5_recovery_pilot.py'),
            '--protocol', str(path), '--base-urls', *args.base_urls,
            '--worker', str(index), '--stage', 'mechanism',
        ], cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    if any(codes):
        write(output / 'run_status.json', {'status': 'execution_error', 'exit_codes': codes})
        raise RuntimeError(codes)
    verdict = analyze_stage(protocol, manifest, 'mechanism')
    write(output / 'run_status.json', {'status': 'completed', 'mechanism': verdict,
        'promotion': False, 'claim_eligible': False, 'held_out_stages_opened': False})
    print(verdict)


if __name__ == '__main__':
    main()
