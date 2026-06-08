#!/usr/bin/env python
from __future__ import annotations

import argparse

from ica_lens.artifacts import verify_artifact_layout, verify_checksums
from ica_lens.config import write_json
from ica_lens.paths import RESULTS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify released ICA Lens artifacts.")
    parser.add_argument(
        "--database-variant",
        choices=["all", "mini", "full"],
        default="mini",
        help="Limit database checksum verification to one database variant.",
    )
    args = parser.parse_args(argv)

    layout_reports = verify_artifact_layout()
    checksum_reports = _filter_database_checksums(verify_checksums(), str(args.database_variant))
    passed = all(report["status"] in {"present", "empty"} for report in layout_reports) and all(
        report["status"] == "passed" for report in checksum_reports
    )
    output = RESULTS_DIR / "verification" / "artifact_verification_report.json"
    write_json(
        output,
        {
            "passed": passed,
            "layout_reports": layout_reports,
            "checksum_reports": checksum_reports,
            "note": "Run scripts/fetch_artifacts.py before verifying checksums.",
        },
    )
    for report in layout_reports:
        print(f"{report['name']}: {report['status']} ({report['local_dir']})")
    for report in checksum_reports:
        print(f"checksum {report['status']}: {report['path']}")
    print(f"Wrote {output}")
    return 0 if passed else 1


def _filter_database_checksums(reports: list[dict[str, object]], variant: str) -> list[dict[str, object]]:
    if variant == "all":
        return reports
    keep = "ica_probe_mini.sqlite" if variant == "mini" else "ica_probe_full.sqlite"
    filtered = []
    for report in reports:
        path = str(report.get("path", ""))
        if "/databases/" in path and not path.endswith(keep):
            continue
        filtered.append(report)
    return filtered


if __name__ == "__main__":
    raise SystemExit(main())
