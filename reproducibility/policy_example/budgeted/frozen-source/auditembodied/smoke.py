"""Console entry point for the focused L0 smoke test."""

from __future__ import annotations

import argparse
import json

from .l0 import smoke_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AuditEmbodied L0 smoke checks")
    parser.add_argument("--render", action="store_true", help="also validate four-camera RGB observations")
    parser.add_argument("--output-dir", default="outputs/l0-smoke")
    args = parser.parse_args()
    report = smoke_test(render=args.render, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
