"""Restore an archived experiment's source in a new, separate checkout."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib,json,subprocess
def run(archive,destination):
    archive=Path(archive).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError("Destination must be new: "+str(destination))
    if not destination.parent.is_dir():raise FileNotFoundError(destination.parent)
    metadata=json.loads((archive/"archive.json").read_text())
    patch=archive/"source.patch"
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(patch)!=metadata["patch_sha256"]:raise ValueError("Archive patch checksum differs")
    repo=subprocess.check_output(["git","rev-parse","--show-toplevel"],cwd=archive,text=True).strip()
    subprocess.run(["git","cat-file","-e",metadata["base_commit"]+"^{commit}"],cwd=repo,check=True)
    subprocess.run(["git","clone","--shared","--no-checkout","--quiet",repo,str(destination)],check=True)
    subprocess.run(["git","checkout","--quiet",metadata["base_commit"]],cwd=destination,check=True)
    for check in [True,False]:
        args=["git","apply","--whitespace=nowarn"]
        if check:args.append("--check")
        subprocess.run(args+[str(patch)],cwd=destination,check=True)
    source=destination/"xuance-master"
    manifest=json.loads((archive/"source_manifest.json").read_text())
    wrong=[name for name,h in manifest.items() if not (source/name).is_file() or digest(source/name)!=h]
    if wrong:raise ValueError("Restored sources differ: "+repr(wrong[:12]))
    result=dict(archive=str(archive),destination=str(destination),base_commit=metadata["base_commit"],verified_source_files=len(manifest),passed=True)
    (destination/"restoration_verified.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=="__main__":
    p=ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args();run(a.archive,a.destination)
