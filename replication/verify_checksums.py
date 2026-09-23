"""Verify the repository's file checksums without modifying any files."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
errors=[]
for line in (ROOT/'checksums.sha256').read_text().splitlines():
    expected,name=line.split('  ',1)
    p=ROOT/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append(name)
if errors:raise SystemExit('FAIL: '+', '.join(errors))
print('PASS: all listed file checksums match')
