"""Add this completed experiment to the existing portable review package."""
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / 'reproducibility'


def main():
    report = json.loads((HERE / 'geometry-analysis/report.json').read_text())
    assert report['new_formal_executions'] == 240
    for source, target in (
        ('mechanism-analysis/recorded-inputs.jsonl.gz', 'data/mechanism-recorded-inputs.jsonl.gz'),
        ('mechanism-analysis/report.json', 'reference/arrival-mechanism-results.json'),
        ('geometry-analysis/geometry-position.jsonl.gz', 'data/geometry-position.jsonl.gz'),
        ('geometry-analysis/pairing-inputs.jsonl.gz', 'data/geometry-holding-pairing.jsonl.gz'),
        ('geometry-analysis/report.json', 'reference/geometry-holding-results.json'),
        ('geometry/protocol.json', 'data/geometry-position-protocol.json'),
        ('geometry/formal-freeze.json', 'data/geometry-position-formal-freeze.json'),
        ('geometry/prefix-reproduction-check.json', 'data/geometry-prefix-reproduction-check.json')):
        target = DEST / target
        assert not target.exists(), target
        shutil.copy2(HERE / source, target)
    native = DEST / 'data/geometry-position-native'
    native.mkdir(exist_ok=False)
    files = sorted((HERE / 'geometry/formal').glob('*-native.npz'))
    assert len(files) == 240
    for file in files:
        shutil.copy2(file, native / file.name)
    for name, source in (('arrival-mechanism', 'mechanism-analysis'),
                         ('geometry-holding', 'geometry-analysis'), ('geometry-holding-effects', 'geometry-analysis')):
        for folder in (DEST / 'reference/tables', ROOT / 'paper/tables'):
            shutil.copy2(HERE / source / (name + '.tex'), folder / (name + '.tex'))
    for name in ('arrival-mechanism', 'geometry-holding'):
        for suffix in ('pdf', 'png'):
            for folder in (ROOT / 'paper/figures', DEST / 'figures'):
                shutil.copy2(HERE / (name + '.' + suffix), folder / (name + '.' + suffix))
    example = DEST / 'policy_example/geometry-position'
    example.mkdir(exist_ok=False)
    for name in ('campaign.py', 'geometry.py', 'observation.py', 'native.py', 'closed_loop.py',
                 'config.json', 'warmup-port.json', 'protocol.json', 'formal-freeze.json'):
        shutil.copy2(HERE / 'geometry' / name, example / name)
    shutil.copytree(HERE / 'geometry/frozen-source', example / 'frozen-source',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    print(json.dumps(dict(new_native_paths=len(files), tables_added=3, new_formal_executions=240,
                          total_formal_records=9436, total_physical_policy_executions=9400)))


if __name__ == '__main__':
    main()
