#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms

ROOT=Path('.')
LAB=ROOT/'data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv'
IMG=ROOT/'data/public/nih_expert_labels/images224'
OUT=ROOT/'experiments/pr_rescue/nih_features'
OUT.mkdir(parents=True,exist_ok=True)
WEIGHTS=Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'

class DS(Dataset):
    def __init__(self,names):
        self.names=list(names)
        self.tf=transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.names)
    def __getitem__(self,i):
        name=self.names[i]; p=IMG/(Path(name).stem+'.jpg')
        with Image.open(p) as im: x=self.tf(im.convert('L'))
        return x,name

def main():
    torch.manual_seed(20260815); np.random.seed(20260815)
    df=pd.read_csv(LAB); all_names=sorted(df['Image ID'].astype(str).unique())
    names=[n for n in all_names if (IMG/(Path(n).stem+'.jpg')).exists()]
    missing=[n for n in all_names if n not in set(names)]
    if not WEIGHTS.exists(): raise FileNotFoundError(WEIGHTS)
    model=resnet50(weights=None)
    sd=torch.load(WEIGHTS,map_location='cpu',weights_only=True); model.load_state_dict(sd)
    model.fc=nn.Identity(); model.eval()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev)
    dl=DataLoader(DS(names),batch_size=64,shuffle=False,num_workers=4,pin_memory=(dev.type=='cuda'))
    feats=[]; seen=[]
    with torch.inference_mode():
        for bi,(x,nm) in enumerate(dl):
            y=model(x.to(dev,non_blocking=True)).float().cpu().numpy(); feats.append(y); seen.extend(list(nm))
            print('batch',bi+1,'/',len(dl),'n',len(seen),flush=True)
    X=np.concatenate(feats,axis=0).astype('float32')
    np.save(OUT/'resnet50_imagenet_features.npy',X)
    pd.DataFrame({'image_id':seen}).to_csv(OUT/'resnet50_imagenet_index.csv',index=False)
    meta={'n_features':int(X.shape[1]),'n_images':int(X.shape[0]),'device':str(dev),'weights':str(WEIGHTS),'missing_images':missing,'seed':20260815}
    (OUT/'resnet50_imagenet_meta.json').write_text(json.dumps(meta,indent=2))
    print(meta); print('finite',bool(np.isfinite(X).all()),'shape',X.shape,'mean',float(X.mean()),'sd',float(X.std()))

if __name__=='__main__': main()
