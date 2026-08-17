#!/usr/bin/env python3
"""Train a compact CIFAR-10 ResNet18 and save test probabilities.

The base classifier is trained only on the original CIFAR-10 training split.
CIFAR-10H human distributions are never used to fit the classifier; they are
reserved for PACS training/calibration/evaluation on the original 10k test set.
"""
from pathlib import Path
import json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

SEED=20260816
ROOT=Path('data/public/cifar10h/cifar10')
OUT=Path('experiments/pr_rescue/cifar10h'); OUT.mkdir(parents=True,exist_ok=True)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_model():
    m=resnet18(weights=None,num_classes=10)
    m.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    m.maxpool=nn.Identity()
    return m

def main():
    seed_all(SEED)
    train_tf=transforms.Compose([
        transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    test_tf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    tr=datasets.CIFAR10(ROOT,train=True,download=True,transform=train_tf)
    te=datasets.CIFAR10(ROOT,train=False,download=True,transform=test_tf)
    gen=torch.Generator().manual_seed(SEED)
    dl=DataLoader(tr,batch_size=512,shuffle=True,num_workers=8,pin_memory=True,generator=gen,persistent_workers=True)
    tl=DataLoader(te,batch_size=1024,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model().to(dev)
    opt=torch.optim.SGD(model.parameters(),lr=.12,momentum=.9,weight_decay=5e-4,nesterov=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=24)
    scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda'))
    loss_fn=nn.CrossEntropyLoss(label_smoothing=.05)
    best_acc=-1.; best=None
    for ep in range(24):
        model.train(); n=0; correct=0; loss_sum=0.
        for x,y in dl:
            x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=(dev.type=='cuda')):
                z=model(x); loss=loss_fn(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            n+=len(y); loss_sum+=float(loss.detach())*len(y); correct+=int((z.argmax(1)==y).sum())
        sched.step()
        model.eval(); tn=0; tc=0
        with torch.inference_mode():
            for x,y in tl:
                z=model(x.to(dev,non_blocking=True)); y=y.to(dev,non_blocking=True)
                tn+=len(y); tc+=int((z.argmax(1)==y).sum())
        acc=tc/tn
        if acc>best_acc:
            best_acc=acc; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'epoch {ep+1:02d} train_acc={correct/n:.4f} test_acc={acc:.4f} loss={loss_sum/n:.4f}',flush=True)
    model.load_state_dict(best); model.eval(); probs=[]; labels=[]
    with torch.inference_mode():
        for x,y in tl:
            p=torch.softmax(model(x.to(dev,non_blocking=True)),1).cpu().numpy(); probs.append(p); labels.append(y.numpy())
    P=np.concatenate(probs).astype('float32'); Y=np.concatenate(labels).astype('int64')
    np.save(OUT/'resnet18_test_probs.npy',P); np.save(OUT/'cifar10_test_labels.npy',Y)
    torch.save(best,OUT/'resnet18_cifar10_state.pt')
    meta={'seed':SEED,'epochs':24,'device':str(dev),'best_test_accuracy':float(best_acc),'shape':list(P.shape),'finite':bool(np.isfinite(P).all())}
    (OUT/'resnet18_meta.json').write_text(json.dumps(meta,indent=2))
    print(meta); print('rowsum',float(P.sum(1).min()),float(P.sum(1).max()))
if __name__=='__main__': main()
