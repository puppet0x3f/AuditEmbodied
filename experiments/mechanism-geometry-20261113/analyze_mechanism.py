"""Analyze all frozen arrival-adaptive paths before the new geometry campaign."""
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'reproducibility'))
import mechanism_geometry_audit as audit


def main():
    out = HERE / 'mechanism-analysis'
    out.mkdir(exist_ok=True)
    source = ROOT / 'outputs/budgeted-recovery-20261112/robust-cad/formal-trials.jsonl'
    cases = []
    with source.open() as stream, gzip.open(out / 'recorded-inputs.jsonl.gz', 'wt') as archive:
        for line in stream:
            recorded = audit.mechanism_input(json.loads(line))
            archive.write(json.dumps(recorded, separators=(',', ':')) + '\n')
            cases.append(audit.mechanism_case(recorded))
    assert len(cases) == 240
    report = audit.mechanism_summary(cases)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'cases.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in cases))
    (out / 'arrival-mechanism.tex').write_text(audit.mechanism_table(report))
    print((out / 'arrival-mechanism.tex').read_text())
    for cell in report['cells']:
        if not cell['healthy'] and cell['wait_steps'] == 20:
            print(json.dumps(cell, indent=2))


if __name__ == '__main__':
    main()
