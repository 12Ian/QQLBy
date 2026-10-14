"""Package the finished exp008 report; never starts or changes training."""
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
STEM = 'baseline_exp008_complete_report_2026-10-13'
RENDER = DOCS / 'exp008_verification/complete_report_render_checked'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

files = [f'docs/{STEM}{ext}' for ext in ['.docx', '.md', '.json']]
files += [f'docs/{name}' for name in [
    'baseline_exp008_final_curriculum_2026-10-13.png',
    'baseline_exp008_final_rewards_and_losses_2026-10-13.png',
    'baseline_exp008_final_retention_2026-10-13.png',
    'baseline_exp008_final_plot_data_2026-10-13.json',
    'baseline_exp008_spawns_2026-10-13.png',
    'baseline_exp008_spawns_2026-10-13.json',
    'baseline_exp008_videos_index_2026-10-13.md',
]]
files += ['tools/write_exp008_complete_report.py',
          'tools/plot_exp008_final_report.py', 'tools/package_exp008_report.py']
files += ['outputs/experiments/exp008/metadata.json',
          'outputs/experiments/exp008/source_snapshot.zip']

missing_links = []
link_count = 0
for rel in [f'docs/{STEM}.md', 'docs/baseline_exp008_videos_index_2026-10-13.md']:
    for target in re.findall(r'\]\((C:/[^)]+)\)', (ROOT / rel).read_text(encoding='utf-8')):
        link_count += 1
        if not Path(target).exists():
            missing_links.append(target)
assert not missing_links, missing_links
pages = sorted(RENDER.glob('page-*.png'))
assert len(pages) == 26, len(pages)
with ZipFile(DOCS / (STEM + '.docx')) as document:
    assert document.testzip() is None
    xml = document.read('word/document.xml').decode('utf-8')
    equation_count = xml.count('<m:oMath>')
    assert equation_count == 14, equation_count
    assert '<w:pBdr>' not in xml
metadata = json.loads((ROOT / 'outputs/experiments/exp008/metadata.json').read_text())
with ZipFile(ROOT / 'outputs/experiments/exp008/source_snapshot.zip') as snapshot:
    assert snapshot.testzip() is None
    for rel, expected in metadata['source_hashes'].items():
        assert hashlib.sha256(snapshot.read(rel)).hexdigest() == expected, rel

manifest = {
    'experiment': '008', 'document_pages': 26,
    'visual_review': 'All 26 pages inspected at original resolution; unchanged pages verified by PNG hashes after the final figure correction.',
    'editable_equations': equation_count, 'local_links_checked': link_count,
    'frozen_source_files_verified': len(metadata['source_hashes']),
    'file_sha256': {rel: sha(ROOT / rel) for rel in files},
    'qa_page_sha256': {p.name: sha(p) for p in pages},
    'archive_contains': 'Report, figures, source data, frozen source snapshot and report generation scripts. Evaluation models, videos and full training CSVs are in the separately verified archives.',
}
manifest_path = DOCS / 'baseline_exp008_complete_report_delivery_v1_manifest.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
files.append(str(manifest_path.relative_to(ROOT)).replace('\\', '/'))
archive = DOCS / 'baseline_exp008_complete_report_delivery_v1.zip'
with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=6) as package:
    for rel in files:
        package.write(ROOT / rel, rel)
with ZipFile(archive) as package:
    assert package.testzip() is None
print(json.dumps({'archive': archive.name, 'bytes': archive.stat().st_size,
                  'sha256': sha(archive), 'files': len(files),
                  'pages': 26, 'links': link_count, 'equations': equation_count}))
