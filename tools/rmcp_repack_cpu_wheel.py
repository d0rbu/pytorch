"""Repackage one pinned CPU wheel with the reviewed Dynamo Python change.

This preserves the upstream native build and wheel metadata. The build tag
identifies a downstream Python-only revision; it is not a native rebuild.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

BASE_SHA256 = "ee40b8a4b4b2cf0670c6fd4f35a7ef23871af956fecb238fbf5da15a72650b1d"
BASE_SOURCE_SHA256 = "3e1725a8609e088e7e7a5bfc786a79bf6596f10efe06ae37b2c92d2dc6a1995a"
BUILD = "1rmcp20261103"
FILENAME = f"torch-2.10.0+cpu-{BUILD}-cp312-cp312-manylinux_2_28_x86_64.whl"
SOURCE_MEMBER = "torch/_dynamo/trace_rules.py"
WHEEL_MEMBER = "torch-2.10.0+cpu.dist-info/WHEEL"
RECORD_MEMBER = "torch-2.10.0+cpu.dist-info/RECORD"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
        b"="
    ).decode("ascii")


def repackage(original: Path, source: Path, destination: Path) -> dict[str, object]:
    if file_sha256(original) != BASE_SHA256:
        raise ValueError(
            "Input must be the pinned upstream Linux x86_64 CPython 3.12 CPU wheel"
        )
    if destination.name != FILENAME:
        raise ValueError(f"Destination filename must be {FILENAME}")
    with zipfile.ZipFile(original) as base:
        baseline = base.read(SOURCE_MEMBER)
        if hashlib.sha256(baseline).hexdigest() != BASE_SOURCE_SHA256:
            raise ValueError(
                "Pinned upstream trace_rules.py differs from the reviewed baseline"
            )
        changed = source.read_bytes()
        if changed == baseline:
            raise ValueError("The source change is missing")
        compile(changed, SOURCE_MEMBER, "exec")
        wheel = base.read(WHEEL_MEMBER)
        if b"Build:" in wheel:
            raise ValueError("Unexpected existing wheel build tag")
        wheel += b"Build: " + BUILD.encode("ascii") + b"\n"
        replacements = {SOURCE_MEMBER: changed, WHEEL_MEMBER: wheel}
        rows = list(csv.reader(io.StringIO(base.read(RECORD_MEMBER).decode("utf-8"))))
        seen: set[str] = set()
        for row in rows:
            if row[0] in replacements:
                seen.add(row[0])
                data = replacements[row[0]]
                row[1:] = [record_hash(data), str(len(data))]
        if seen != replacements.keys():
            raise ValueError(
                "Replacement source or wheel metadata is absent from RECORD"
            )
        record = io.StringIO(newline="")
        csv.writer(record).writerows(rows)
        replacements[RECORD_MEMBER] = record.getvalue().encode("utf-8")
        with zipfile.ZipFile(destination, "x") as output:
            for member in base.infolist():
                if member.filename in replacements:
                    output.writestr(member, replacements[member.filename])
                else:
                    with (
                        base.open(member) as reader,
                        output.open(member, "w") as writer,
                    ):
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    return {
        "upstream_wheel_sha256": BASE_SHA256,
        "wheel_sha256": file_sha256(destination),
        "source_sha256": hashlib.sha256(changed).hexdigest(),
        "wheel_filename": FILENAME,
        "changed_members": sorted(replacements),
        "native_build": "Original upstream CPU build, unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / SOURCE_MEMBER,
    )
    args = parser.parse_args()
    print(json.dumps(repackage(args.original, args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
