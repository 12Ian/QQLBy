"""Apply only to a verified restoration of the parent source archive."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib,json,shutil

def main(destination):
    archive=Path(__file__).resolve().parent
    checkout=Path(destination).resolve()
    source=checkout/"xuance-master"
    assert (checkout/"restoration_verified.json").is_file(), "Restore the parent archive first"
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    parent=archive.parent/"lstm_matd3_capture_long_v1"/"source_manifest.json"
    assert all((source/n).is_file() and digest(source/n)==h for n,h in json.loads(parent.read_text()).items()), "Parent sources differ"
    for path in (archive/"overlay").rglob("*"):
        if path.is_file():
            target=source/path.relative_to(archive/"overlay")
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    manifest=json.loads((archive/"frozen_source_sha256.json").read_text())
    assert all((source/n).is_file() and digest(source/n)==h for n,h in manifest.items()), "Restored overlay differs"
    result=dict(passed=True,source_files=len(manifest),archive=str(archive))
    (checkout/"entity_attention_restoration_verified.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=="__main__":
    parser=ArgumentParser();parser.add_argument("--destination",required=True)
    main(parser.parse_args().destination)
