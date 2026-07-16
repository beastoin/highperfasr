#!/usr/bin/env python3
"""
Validate benchmark report JSON files against the highperfasr report schema.

Usage:
    python3 validate_report.py benchmarks/results/2026-l4-nemo-batch/result.json
    python3 validate_report.py benchmarks/results/*/result.json
    python3 validate_report.py --all
"""

import argparse
import json
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "report-schema.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_file(path, schema, verbose=False):
    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema required. Install: pip install jsonschema", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    try:
        checker = jsonschema.FormatChecker()
    except AttributeError:
        checker = None
    validator = jsonschema.Draft202012Validator(schema, format_checker=checker)
    errors = list(validator.iter_errors(data))
    if not errors:
        if verbose:
            print(f"  VALID: {path}")
        return True

    print(f"  INVALID: {path}")
    for e in sorted(errors, key=lambda x: list(x.path)):
        loc = "/".join(str(p) for p in e.path) or "(root)"
        print(f"    - {loc}: {e.message}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark reports against schema")
    parser.add_argument("files", nargs="*", help="JSON files to validate")
    parser.add_argument("--all", action="store_true", help="Validate all result.json files in benchmarks/results/")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print valid files too")
    args = parser.parse_args()

    schema = load_schema()

    files = []
    if args.all:
        files = sorted(RESULTS_DIR.glob("*/result.json"))
    elif args.files:
        files = [Path(f) for f in args.files]
    else:
        parser.print_help()
        return 1

    if not files:
        print("No files to validate")
        return 1

    print(f"Validating {len(files)} files against {SCHEMA_PATH.name} ({schema.get('version', '?')}):")
    valid = sum(1 for f in files if validate_file(f, schema, args.verbose))
    invalid = len(files) - valid
    print(f"\n{valid}/{len(files)} valid, {invalid} invalid")
    return 1 if invalid > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
