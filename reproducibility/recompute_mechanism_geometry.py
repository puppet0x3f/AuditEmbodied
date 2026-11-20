"""Recompute joint/command diagnostics and the 240-path geometry extension.

Python and NumPy only. Existing base and bounded-follow-up evaluators are
unchanged; this entry reuses their frozen numerical scoring functions.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import mechanism_geometry_audit as audit
import wait_geometry_audit as waiting
import controller_audit as controller
import prospective_geometry as geometry
import recompute as base


def read(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    (out / 'tables').mkdir()
    cases = [audit.mechanism_case(row) for row in read(HERE / 'data/mechanism-recorded-inputs.jsonl.gz')]
    mechanism = audit.mechanism_summary(cases)
    assert mechanism == json.loads((HERE / 'reference/arrival-mechanism-results.json').read_text())
    assert len(cases) == 240 and mechanism['new_physical_executions'] == 0
    tables = {'arrival-mechanism': audit.mechanism_table(mechanism)}
    print('Recomputed 240 recorded first approaches and action/joint diagnostics.', flush=True)
    geometry.configure('tray')
    rows, fits, new_fits, checks = [], 0, 0, 0
    for dataset, native_folder in (('geometry', 'geometry-native'), ('geometry-position', 'geometry-position-native')):
        for row in read(HERE / 'data' / (dataset + '.jsonl.gz')):
            result, count = audit.geometry_score(row, HERE / 'data' / native_folder / row['native_file'],
                base.score_lift, controller, waiting, geometry.fit)
            rows.append(result)
            fits += count
            if dataset == 'geometry-position':
                new_fits += count
                checks += audit.holding_commands(row)
    states = list(read(HERE / 'data/geometry-holding-pairing.jsonl.gz'))
    report = audit.geometry_summary(rows, waiting)
    report.update(audit.geometry_pairing(states), pose_fits_recomputed=fits,
                  new_pose_fits_recomputed=new_fits, holding_command_checks=checks,
                  new_native_samples=sum(r['native_samples'] for r in rows if r['hold_mode'] == 'measured_position'))
    assert report == json.loads((HERE / 'reference/geometry-holding-results.json').read_text())
    tables.update(audit.geometry_tables(report))
    for name, content in tables.items():
        reference = HERE / 'reference/tables' / (name + '.tex')
        assert content.split() == reference.read_text().split(), name
        (out / 'tables' / (name + '.tex')).write_text(content)
    result = dict(new_formal_executions_recomputed=240, reused_registered_executions=240,
                  mechanism_reanalyzed_executions=240, new_independent_seeds=0,
                  action_target_checks=mechanism['action_target_checks'],
                  holding_command_checks=checks, all_geometry_pose_fits=fits,
                  new_geometry_pose_fits=new_fits, new_tables_match=list(tables),
                  geometry_pairing=audit.geometry_pairing(states),
                  arrival_mechanism=mechanism, geometry_holding=report)
    (out / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'scored-geometry.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows))
    print(json.dumps({k: v for k, v in result.items() if k not in ('arrival_mechanism', 'geometry_holding')}, indent=2))


if __name__ == '__main__':
    main()
