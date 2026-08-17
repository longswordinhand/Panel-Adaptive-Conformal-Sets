#!/usr/bin/env python3
import argparse, hashlib, io, json, os, time
from pathlib import Path
import pandas as pd
import requests
from PIL import Image

DATASET='chehablab/NIHChestXR'; CONFIG='default'; SPLIT='train'
ROWS='https://datasets-server.huggingface.co/rows'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--expert-csv',default='data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv')
    ap.add_argument('--metadata-csv',default='data/public/nih_expert_labels/meta/Data_Entry_2017_v2020.csv')
    ap.add_argument('--out-dir',default='data/public/nih_expert_labels/images224')
    ap.add_argument('--manifest',default='data/public/nih_expert_labels/meta/images224_manifest.csv')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    exp=pd.read_csv(args.expert_csv)
    names=sorted(exp['Image ID'].astype(str).unique())
    if args.limit: names=names[:args.limit]
    meta=pd.read_csv(args.metadata_csv).reset_index().rename(columns={'index':'row_idx'})
    idx=dict(zip(meta['Image Index'].astype(str),meta['row_idx'].astype(int)))
    missing=[x for x in names if x not in idx]
    if missing: raise SystemExit(f'metadata missing {len(missing)} images: {missing[:5]}')
    s=requests.Session(); rows=[]
    for j,name in enumerate(names,1):
        row_idx=idx[name]; dst=out/(Path(name).stem+'.jpg')
        status='exists'; sha=''; w=h=0
        if dst.exists():
            b=dst.read_bytes(); sha=hashlib.sha256(b).hexdigest()
            try:
                with Image.open(io.BytesIO(b)) as im: w,h=im.size; im.verify()
            except Exception: status='corrupt-existing'
        if not dst.exists() or status=='corrupt-existing':
            status='downloaded'
            last=None
            for attempt in range(5):
                try:
                    r=s.get(ROWS,params={'dataset':DATASET,'config':CONFIG,'split':SPLIT,'offset':row_idx,'length':1},timeout=40)
                    r.raise_for_status(); d=r.json(); rr=d['rows'][0]
                    if int(rr['row_idx'])!=row_idx: raise RuntimeError('row index mismatch')
                    src=rr['row']['image']['src']
                    q=s.get(src,timeout=60); q.raise_for_status(); b=q.content
                    with Image.open(io.BytesIO(b)) as im:
                        im.load(); w,h=im.size
                        if (w,h)!=(224,224): raise RuntimeError(f'unexpected size {(w,h)}')
                    dst.write_bytes(b); sha=hashlib.sha256(b).hexdigest(); last=None; break
                except Exception as e:
                    last=e; time.sleep(min(2**attempt,8))
            if last is not None: status=f'ERROR:{type(last).__name__}:{last}'
        rows.append({'image_id':name,'row_idx':row_idx,'local_path':str(dst),'status':status,'width':w,'height':h,'sha256':sha})
        if j%50==0 or j==len(names): print(f'{j}/{len(names)}')
    m=pd.DataFrame(rows); Path(args.manifest).parent.mkdir(parents=True,exist_ok=True); m.to_csv(args.manifest,index=False)
    errs=m[~m['status'].isin(['exists','downloaded'])]
    print(json.dumps({'requested':len(names),'downloaded_or_exists':int(len(m)-len(errs)),'errors':int(len(errs)),'unique_sha':int(m['sha256'].nunique())},indent=2))
    if len(errs): print(errs[['image_id','status']].head(20).to_string(index=False)); raise SystemExit(2)
if __name__=='__main__': main()
