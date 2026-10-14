"""Verify one finished server archive, extract locally and rebuild clickable links."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Path(sys.argv[1])
    with source.open(encoding="utf-8-sig") as stream:
        header = json.loads(next(stream))
        archive = ROOT / "docs" / header["filename"]
        assert archive.parent.resolve() == (ROOT / "docs").resolve()
        digest = hashlib.sha256()
        size = 0
        with archive.open("wb") as target:
            for line in stream:
                data = base64.b64decode(line.strip(), validate=True)
                digest.update(data)
                target.write(data)
                size += len(data)
    assert digest.hexdigest() == header["sha256"] and size == header["bytes"], "Archive transmission mismatch"
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        manifest = json.loads(bundle.read("delivery_manifest.json"))["files"]
        for name, expected in manifest.items():
            destination = (ROOT / name).resolve()
            assert destination.is_relative_to(ROOT), name
            data = bundle.read(name)
            assert hashlib.sha256(data).hexdigest() == expected, name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    subprocess.run([sys.executable, str(ROOT / "tools/report_exp009.py")], cwd=ROOT, check=True)
    report = ROOT / "docs/baseline_exp009_results_2026-10-14.md"
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n[本机009 TensorBoard](http://127.0.0.1:6007/)；原6006保留旧实验。\n")
    state = dict(header, verified_files=len(manifest), crc_valid=True, all_internal_hashes_verified=True,
                 received_at=datetime.now(timezone.utc).isoformat(), local_report=str(report))
    output = ROOT / "docs/baseline_exp009_final_transfer_verification.json"
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source.unlink()
    print("FINAL_DELIVERY_VERIFIED", json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
