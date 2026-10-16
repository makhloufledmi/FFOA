"""Build the single raw-evidence package used by the SWEVO manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCES = {
    "cec": "cec_full_corrected_v1_results.json",
    "ablation": "ffoa_ablation_cec2022_corrected_v1_results.json",
    "focused_ceo": "ceo_focused_corrected_v1_raw.json",
    "snl": "snl_corrected_v1_raw.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="swevo_evidence_raw_v2.json")
    for name, default in DEFAULT_SOURCES.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default)
    args = parser.parse_args()

    source_paths = {
        name: Path(getattr(args, name))
        for name in DEFAULT_SOURCES
    }
    experiments = {}
    manifest = {}
    for name, path in source_paths.items():
        experiments[name] = json.loads(path.read_text(encoding="utf-8"))
        manifest[name] = {
            "path": path.name,
            "sha256": sha256(path),
            "records": len(experiments[name]),
        }

    package = {
        "schema": "swevo-evidence-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_policy": (
            "Objective observations are preserved verbatim. Derived final-error "
            "statistics tie target-reaching runs at zero while retaining raw "
            "residuals and FEs-to-target."
        ),
        "source_manifest": manifest,
        "experiments": experiments,
    }
    output = Path(args.output)
    output.write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "experiments": {
            name: details["records"] for name, details in manifest.items()
        },
        "sha256": sha256(output),
    }, indent=2))


if __name__ == "__main__":
    main()
