#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from ica_lens.artifacts import fetch_artifact_set, selected_artifact_sets
from ica_lens.config import write_json
from ica_lens.paths import RESULTS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch released ICA Lens artifacts.")
    parser.add_argument("--models", action="store_true", help="Fetch fitted ICA model artifacts.")
    parser.add_argument("--databases", action="store_true", help="Fetch explorer database artifacts.")
    parser.add_argument(
        "--database-variant",
        choices=["all", "mini", "full"],
        default="mini",
        help="When fetching databases, choose all databases, only the mini DB, or only the full DB.",
    )
    parser.add_argument("--explorer", action="store_true", help="Fetch artifacts needed by the explorer.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned downloads without downloading.")
    args = parser.parse_args(argv)

    models = bool(args.models or args.explorer)
    databases = bool(args.databases or args.explorer)
    selected = [
        _with_database_variant(item, str(args.database_variant))
        for item in selected_artifact_sets(models=models, databases=databases)
    ]
    reports = [fetch_artifact_set(item, dry_run=args.dry_run) for item in selected]

    output = RESULTS_DIR / "verification" / "fetch_artifacts_report.json"
    write_json(output, {"reports": reports})
    for report in reports:
        print(f"{report['name']}: {report['status']} -> {report['local_dir']}")
        if report["status"] == "not_configured":
            print(f"  {report['message']}", file=sys.stderr)
    print(f"Wrote {output}")
    return 0 if all(report["status"] != "not_configured" for report in reports) else 2


def _with_database_variant(artifact_set, variant: str):
    if artifact_set.name != "databases" or variant == "all":
        return artifact_set
    filename = "ica_probe_mini.sqlite" if variant == "mini" else "ica_probe_full.sqlite"
    return replace(artifact_set, allow_patterns=(f"databases/{filename}",))


if __name__ == "__main__":
    raise SystemExit(main())
