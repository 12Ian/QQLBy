"""Restore a verified upstream Xuance release while retaining original files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

EXPECTED = "cd8e706619a5a8ed7cc33234d9879a4363196b9ecce5e19721e4be74170ead29"
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=str(ROOT / "outputs/bootstrap/xuance-1.4.4.tar.gz"))
    args = parser.parse_args()
    archive_path = Path(args.archive)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError("Unverified Xuance source archive")
    manifest = ROOT / "UPSTREAM_XUANCE.json"
    if manifest.exists():
        raise FileExistsError("Framework has already been restored; refusing to replace local extensions")
    backup = ROOT / "outputs/bootstrap/original_xuance"
    records = []
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[0] != "xuance-1.4.4" or parts[1] != "xuance":
                continue
            if not member.isfile():
                if member.issym() or member.islnk():
                    raise ValueError("Links are not allowed in the release archive")
                continue
            relative = Path(*parts[1:])
            target = (ROOT / relative).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError("Archive path escapes the checkout")
            content = archive.extractfile(member).read()
            original = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
            if target.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            records.append({"path": relative.as_posix(), "original_sha256": original,
                            "upstream_sha256": hashlib.sha256(content).hexdigest()})
        licenses = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name in {"LICENSE", "LICENSE.txt"}]
        if licenses:
            (ROOT / "LICENSE.xuance").write_bytes(archive.extractfile(licenses[0]).read())
    manifest.write_text(json.dumps({"version": "1.4.4", "source": "https://pypi.org/project/xuance/1.4.4/",
                                  "archive_sha256": digest, "files": records,
                                  "backup": "outputs/bootstrap/original_xuance"}, indent=2) + "\n", encoding="utf-8")
    print(f"Restored {len(records)} official files; {sum(r['original_sha256'] is not None for r in records)} originals backed up")


if __name__ == "__main__":
    main()
