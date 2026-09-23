"""Verify packaged checksums across Git checkouts on different platforms.

UTF-8 text is hashed with LF line endings; binary files are hashed byte-for-byte.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
BINARY_EXTENSIONS = {".pdf", ".zip", ".whl", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico"}


def checksum_bytes(path):
    data = path.read_bytes()
    if path.suffix.lower() in BINARY_EXTENSIONS or b"\0" in data:
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return data.replace(b"\r\n", b"\n")


def verify_ledger(ledger, base):
    errors = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = base / name
        if not path.is_file() or hashlib.sha256(checksum_bytes(path)).hexdigest() != expected:
            errors.append(str(path.relative_to(ROOT)))
    return errors


errors = verify_ledger(ROOT / "checksums.sha256", ROOT)
errors += verify_ledger(ROOT / "workflow_demo/tensorflow_flowers/checksums.sha256",
                        ROOT / "workflow_demo/tensorflow_flowers")
if errors:
    raise SystemExit("FAIL: " + ", ".join(dict.fromkeys(errors)))
print("PASS: all listed file checksums match (LF-normalized text, exact binary)")
