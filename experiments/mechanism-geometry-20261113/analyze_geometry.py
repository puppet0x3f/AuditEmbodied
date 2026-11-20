"""Score the fixed geometry extension and retain the old registered baseline."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'reproducibility'))
import mechanism_geometry_audit as audit
import wait_geometry_audit as waiting
import controller_audit as controller
import prospective_geometry as geometry
import recompute as base
from export_from_workspace import project


def main():
    campaign = HERE / 'geometry'
    out = HERE / 'geometry-analysis'
    out.mkdir(exist_ok=False)
    frozen = json.loads((campaign / 'formal-freeze.json').read_text())
    for name, value in frozen['source_sha256'].items():
        assert hashlib.sha256((campaign / name).read_bytes()).hexdigest() == value, name
    completion = json.loads((campaign / 'formal-completion.json').read_text())
    assert len(completion) == 240 and not any(r['error'] for r in completion)
    geometry.configure('tray')
    rows, states, fits, checks = [], [], 0, 0
    source = ROOT / 'outputs/wait-hold-geometry-20261111'
    with (source / 'geometry-trials.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            result, count = audit.geometry_score(row, source / 'geometry' / row['native_file'],
                base.score_lift, controller, waiting, geometry.fit)
            rows.append(result)
            states.append(audit.geometry_state(row))
            fits += count
    new_fits = 0
    with gzip.open(out / 'geometry-position.jsonl.gz', 'wt') as archive:
        for path in sorted((campaign / 'formal').glob('*.json')):
            row = json.loads(path.read_text())
            assert row['nominal_ready'] and row['fault_class_valid'] and not row.get('error'), path.name
            assert not row['blocked_calls'] and row['contact_anchor_count'] == row['state_edit_count_after_initialization'] == 0
            assert row['hold_mode'] == 'measured_position'
            result, count = audit.geometry_score(row, campaign / 'formal' / row['native_file'],
                base.score_lift, controller, waiting, geometry.fit)
            rows.append(result)
            states.append(audit.geometry_state(row))
            fits += count
            new_fits += count
            checks += audit.holding_commands(row)
            compact = project(row, 'geometry')
            originals = {round(f['time'], 8): f for f in row['trace']}
            for frame in compact['trace']:
                if frame['stage'] == 'inference_wait' or abs(frame['time'] - row['prefix_state']['time']) < 1e-8:
                    original = originals[round(frame['time'], 8)]
                    frame['robot'] = {k: original['robot'][k] for k in ('positions', 'action')}
            archive.write(json.dumps(compact, separators=(',', ':')) + '\n')
    assert len(rows) == len(states) == 480
    report = audit.geometry_summary(rows, waiting)
    report.update(audit.geometry_pairing(states), pose_fits_recomputed=fits,
                  new_pose_fits_recomputed=new_fits, holding_command_checks=checks,
                  new_native_samples=sum(r['native_samples'] for r in rows if r['hold_mode'] == 'measured_position'))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'scored.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows))
    with gzip.open(out / 'pairing-inputs.jsonl.gz', 'wt') as stream:
        for state in states:
            stream.write(json.dumps(state, separators=(',', ':')) + '\n')
    for name, table in audit.geometry_tables(report).items():
        (out / (name + '.tex')).write_text(table)
    for cell in report['cells']:
        print(json.dumps(cell))
    print(json.dumps({k: v for k, v in report.items() if k != 'cells'}, indent=2))


if __name__ == '__main__':
    main()
