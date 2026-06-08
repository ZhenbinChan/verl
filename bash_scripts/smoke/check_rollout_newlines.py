#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distinguish real newlines from literal backslash-n sequences in rollout outputs."
    )
    parser.add_argument("rollout_dir", type=Path, help="Directory containing rollout JSONL files.")
    parser.add_argument("--examples", type=int, default=3, help="Maximum examples to print per category.")
    return parser.parse_args()


def compact_repr(text: str, limit: int = 500) -> str:
    rendered = repr(text)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    files = sorted(args.rollout_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found under {args.rollout_dir}")

    total = 0
    real_double_newline = []
    literal_double_backslash_n = []

    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                output = row.get("output", "")
                if not isinstance(output, str):
                    continue

                total += 1
                location = f"{path}:{line_number}"
                if "\n\n" in output:
                    real_double_newline.append((location, output))
                if r"\n\n" in output:
                    literal_double_backslash_n.append((location, output))

    print(f"rollout_dir: {args.rollout_dir}")
    print(f"jsonl_files: {len(files)}")
    print(f"outputs: {total}")
    print(f"real LF LF (actual two newline characters): {len(real_double_newline)}")
    print(f"literal \\\\n\\\\n (four visible characters): {len(literal_double_backslash_n)}")
    if literal_double_backslash_n:
        print("RESULT: found literal \\\\n\\\\n in decoded rollout output.")
    else:
        print("RESULT: no literal \\\\n\\\\n found in decoded rollout output.")

    categories = (
        ("real LF LF examples", real_double_newline),
        ("literal \\\\n\\\\n examples", literal_double_backslash_n),
    )
    for title, matches in categories:
        if not matches:
            continue
        print(f"\n{title}:")
        for location, output in matches[: args.examples]:
            print(f"- {location}")
            print(f"  {compact_repr(output)}")


if __name__ == "__main__":
    main()
