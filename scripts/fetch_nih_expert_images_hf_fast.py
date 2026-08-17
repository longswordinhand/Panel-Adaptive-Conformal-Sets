#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib, io, time
import pandas as pd, requests
from PIL import Image

EXPERT='data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv'
META='data/public/nih_expert_labels/meta/Data_Entry_2017_v2020.csv'
OUT=Path('data/public/nih_expert_labels/images224')
MAN='data/public/nih_expert_labels/meta/images224_manifest.csv'
ROWS='https://datasets-server.huggingface.co/rows'
DATASET='chehablab/NIHChestXR'
OUT.mkdir(parents=True,exist_ok=True)
exp=pd.read_csv(EXPERT); names=sorted(exp['Image ID'].astype(str).unique())
meta=pd.read_csv(META).reset_index().rename(columns={'index':'row_idx'})
idx=dict(zip(meta['Image Index'].astype(str),meta['row_idx'].astype(int)))

def fetch(name):
    row_idx=idx[name]; dst=OUT/(Path(name).stem+'.jpg')
    if dst.exists():
        try:
            b=dst.read_bytes();
            with Image.open(io.BytesIO(b)) as im: im.verify(); wh=im.size
            return dict(image_id=name,row_idx=row_idx,local_path=str(dst),status='exists',width=wh[0],height=wh[1],sha256=hashlib.sha256(b).hexdigest())
        except Exception:
            try: dst.unlink()
            except: pass
    err=''
    for attempt in range(3):
        try:
            r=requests.get(ROWS,params={'dataset':DATASET,'config':'default','split':'train','offset':row_idx,'length':1},timeout=(5,15))
            r.raise_for_status(); d=r.json(); rr=d['rows'][0]
            if int(rr['row_idx'])!=row_idx: raise RuntimeError('row mismatch')
            src=rr['row']['image']['src']
            q=requests.get(src,timeout=(5,20)); q.raise_for_status(); b=q.content
            with Image.open(io.BytesIO(b)) as im: im.load(); wh=im.size
            if wh!=(224,224): raise RuntimeError(f'size={wh}')
            tmp=dst.with_suffix('.tmp'); tmp.write_bytes(b); tmp.replace(dst)
            return dict(image_id=name,row_idx=row_idx,local_path=str(dst),status='downloaded',width=224,height=224,sha256=hashlib.sha256(b).hexdigest())
        except Exception as e:
            err=f'{type(e).__name__}:{e}'; time.sleep(0.5*(attempt+1))
    return dict(image_id=name,row_idx=row_idx,local_path=str(dst),status='ERROR:'+err,width=0,height=0,sha256='')

results=[]
with ThreadPoolExecutor(max_workers=12) as ex:
    fut={ex.submit(fetch,n):n for n in names}
    for j,f in enumerate(as_completed(fut),1):
        try: results.append(f.result())
        except Exception as e: results.append({'image_id':fut[f],'row_idx':idx[fut[f]],'local_path':'','status':'ERROR_TOP:'+repr(e),'width':0,'height':0,'sha256':''})
        if j%100==0: print(f'{j}/{len(names)}',flush=True)
m=pd.DataFrame(results).sort_values('image_id'); m.to_csv(MAN,index=False)
err=m[m.status.str.startswith('ERROR')]
print('requested',len(m),'ok',len(m)-len(err),'errors',len(err),'unique_sha',m.sha256.replace('',pd.NA).nunique(),flush=True)
if len(err): print(err[['image_id','status']].head(30).to_string(index=False),flush=True)
