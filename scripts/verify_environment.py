#!/usr/bin/env python
from __future__ import annotations

import argparse

from ica_lens.config import write_json
from ica_lens.paths import RESULTS_DIR, V6_ROOT
from ica_lens.provenance import environment_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an ICA Lens environment report.")
    parser.parse_args(argv)
    output = RESULTS_DIR / "verification" / "environment_report.json"
    write_json(output, environment_report(V6_ROOT))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
