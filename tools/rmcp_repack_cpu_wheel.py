"""Repackage one pinned CPU wheel with the reviewed Dynamo and Python-owned registration changes.

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
BASE_SOURCES = {
    "torch/library.py": "adb13397e5604630cdec2a7cc7be38d78ace93759aedff74bdf0f7434e7c72b1",
    "torch/_library/utils.py": "53ce827f4fe5c83e1cc81461d5b687104dc462edd9893fd0f29d608fa9d7ce7b",
    "torch/_dynamo/trace_rules.py": "3e1725a8609e088e7e7a5bfc786a79bf6596f10efe06ae37b2c92d2dc6a1995a",
    "torch/_dynamo/variables/functions.py": "2bad7a82f1c2a9ce62674f9f1089f61f72d8ea32ae4135d4e309c8456222de1e",
    "torch/_dynamo/variables/torch.py": "9e5e8be5f35a590cba52b47d727cb8ecde5bffc12dac84a9fea5be2b4a2e5579",
}
BUILD = "6rmcp20261103"
FILENAME = f"torch-2.10.0+cpu-{BUILD}-cp312-cp312-manylinux_2_28_x86_64.whl"
WHEEL_MEMBER = "torch-2.10.0+cpu.dist-info/WHEEL"
RECORD_MEMBER = "torch-2.10.0+cpu.dist-info/RECORD"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
        b"="
    ).decode("ascii")


def repackage(
    original: Path, source_root: Path, destination: Path
) -> dict[str, object]:
    if file_sha256(original) != BASE_SHA256:
        raise ValueError(
            "Input must be the pinned upstream Linux x86_64 CPython 3.12 CPU wheel"
        )
    if destination.name != FILENAME:
        raise ValueError(f"Destination filename must be {FILENAME}")
    with zipfile.ZipFile(original) as base:
        replacements = {}
        source_hashes = {}
        for member, expected in BASE_SOURCES.items():
            baseline = base.read(member)
            if hashlib.sha256(baseline).hexdigest() != expected:
                raise ValueError(
                    f"Pinned upstream {member} differs from the reviewed baseline"
                )
            changed = (source_root / member).read_bytes()
            if changed == baseline:
                raise ValueError(f"The source change in {member} is missing")
            compile(changed, member, "exec")
            replacements[member] = changed
            source_hashes[member] = hashlib.sha256(changed).hexdigest()
        wheel = base.read(WHEEL_MEMBER)
        if b"Build:" in wheel:
            raise ValueError("Unexpected existing wheel build tag")
        wheel += b"Build: " + BUILD.encode("ascii") + b"\n"
        replacements[WHEEL_MEMBER] = wheel
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
        "source_sha256": source_hashes,
        "wheel_filename": FILENAME,
        "changed_members": sorted(replacements),
        "native_build": "Original upstream CPU build, unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    print(
        json.dumps(
            repackage(args.original, args.source_root, args.destination), indent=2
        )
    )


if __name__ == "__main__":
    main()
