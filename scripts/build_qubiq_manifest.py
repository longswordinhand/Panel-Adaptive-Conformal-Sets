from __future__ import annotations
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = {
    'train': RAW / 'train' / 'training_data_v3_QC',
    'valid': RAW / 'valid' / 'validation_data_qubiq2021',
    'test': RAW / 'test' / 'QUBIQ21_test',
}
TASKS = ['prostate','brain-growth','brain-tumor','kidney','pancreas','pancreatic-lesion']
CASE_RE = re.compile(r'^case')
SEG_RE = re.compile(r'(?:task(?P<task>\d+)_)?seg(?P<rater>\d+)\.nii\.gz$')
ALT_SEG_RE = re.compile(r'case[^/]*_seg(?P<rater>\d+)\.nii\.gz$')

def find_case_dirs(task_root: Path):
    out=[]
    for p in task_root.rglob('*'):
        if p.is_dir() and CASE_RE.match(p.name):
            if any(f.name == 'image.nii.gz' for f in p.iterdir() if f.is_file()):
                out.append(p)
    return sorted(out, key=lambda p: p.name)

rows=[]
for split, base in SPLITS.items():
    for dataset_task in TASKS:
        task_root = base / dataset_task
        if not task_root.exists():
            continue
        for case_dir in find_case_dirs(task_root):
            image = case_dir / 'image.nii.gz'
            seg_files=[]
            for p in sorted(case_dir.glob('*.nii.gz')):
                if p.name == 'image.nii.gz':
                    continue
                m=SEG_RE.search(p.name)
                if m:
                    task_id = m.group('task') or '01'
                    rater_id = m.group('rater')
                    seg_files.append((task_id, rater_id, p))
                    continue
                m=ALT_SEG_RE.search(p.name)
                if m:
                    seg_files.append(('01', m.group('rater'), p))
            by_task={}
            for task_id,rater_id,p in seg_files:
                by_task.setdefault(task_id,[]).append((rater_id,p))
            if not by_task:
                by_task={'NA':[]}
            for task_id, items in sorted(by_task.items()):
                rows.append({
                    'split': split,
                    'dataset_task': dataset_task,
                    'case_id': case_dir.name,
                    'segmentation_task': task_id,
                    'image_path': str(image.relative_to(ROOT)),
                    'n_annotations': len(items),
                    'rater_ids': ';'.join(r for r,_ in items),
                    'annotation_paths': ';'.join(str(p.relative_to(ROOT)) for _,p in items),
                })

csv_path=OUT/'qubiq2021_manifest.csv'
with csv_path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

summary={}
for r in rows:
    k=(r['split'],r['dataset_task'],r['segmentation_task'])
    s=summary.setdefault(k, {'cases':0,'annotation_counts':[]})
    s['cases'] += 1
    s['annotation_counts'].append(int(r['n_annotations']))
summary_rows=[]
for (split,task,seg_task),s in sorted(summary.items()):
    vals=s['annotation_counts']
    summary_rows.append({
        'split':split,'dataset_task':task,'segmentation_task':seg_task,'cases':s['cases'],
        'annotations_min':min(vals),'annotations_max':max(vals),
        'annotations_mean':round(sum(vals)/len(vals),3),
    })
with (OUT/'qubiq2021_summary.json').open('w') as f:
    json.dump(summary_rows,f,indent=2)

print(csv_path)
for r in summary_rows:
    print(r)
