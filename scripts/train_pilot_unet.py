#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multirater_conformal_seg.models.unet2d import UNet2D

CACHE = ROOT / "data/processed/prostate_512"
SPLITS = ROOT / "experiments/pilot/splits/prostate_5fold_patient_splits.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProstateConsensusDataset(Dataset):
    def __init__(self, case_ids: list[str], task: str, size: int, augment: bool, seed: int):
        self.task = task
        self.size = size
        self.augment = augment
        self.seed = seed
        self.items = []
        for case_id in case_ids:
            with np.load(CACHE / f"{case_id}.npz") as d:
                image = d["image"].astype(np.float32)
                mask = d[f"task{task}_consensus"].astype(np.float32)
            self.items.append((case_id, image, mask))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        case_id, image, mask = self.items[idx]
        x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        y = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y, size=(self.size, self.size), mode="nearest").squeeze(0)
        if self.augment:
            # Determinism is controlled by the DataLoader/process seed; num_workers=0.
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[2])
            gain = 0.9 + 0.2 * torch.rand(())
            bias = -0.05 + 0.1 * torch.rand(())
            x = torch.clamp(x * gain + bias, 0.0, 1.0)
            if torch.rand(()) < 0.5:
                x = torch.clamp(x + 0.015 * torch.randn_like(x), 0.0, 1.0)
        return case_id, x, y


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def hard_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


def train(args) -> tuple[UNet2D, list[dict]]:
    seed = args.seed + args.fold * 100 + int(args.task)
    set_seed(seed)
    split_df = pd.read_csv(SPLITS)
    fold_df = split_df[split_df.fold == args.fold]
    train_cases = sorted(fold_df.loc[fold_df.role == "train", "case_id"].tolist())
    if len(train_cases) != 33:
        raise RuntimeError(f"Expected 33 train cases, got {len(train_cases)}")

    ds = ProstateConsensusDataset(train_cases, args.task, args.input_size, augment=True, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=True, generator=generator, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Pilot training requires the verified CUDA environment")
    model = UNet2D(base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, dices = [], []
        for _, x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().item()))
            dices.append(hard_dice(logits.detach(), y))
        sched.step()
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "train_dice": float(np.mean(dices)),
               "lr": float(opt.param_groups[0]["lr"])}
        history.append(rec)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            print(json.dumps(rec), flush=True)
    return model, history


@torch.no_grad()
def predict(model: UNet2D, args, case_ids: list[str]) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    out = {}
    for case_id in case_ids:
        with np.load(CACHE / f"{case_id}.npz") as d:
            image = torch.from_numpy(d["image"].astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(image, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prob = torch.sigmoid(model(x))
        prob = F.interpolate(prob.float(), size=(512, 512), mode="bilinear", align_corners=False)
        out[case_id] = prob.squeeze().cpu().numpy().astype(np.float16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument("--task", required=True, choices=["01", "02"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    model, history = train(args)
    split_df = pd.read_csv(SPLITS)
    fold_df = split_df[split_df.fold == args.fold]
    pred_cases = sorted(fold_df.loc[fold_df.role.isin(["calibration", "test"]), "case_id"].tolist())
    probs = predict(model, args, pred_cases)

    model_dir = ROOT / "experiments/pilot/models" / args.tag / f"fold{args.fold}" / f"task{args.task}"
    pred_dir = ROOT / "experiments/pilot/predictions" / args.tag / f"fold{args.fold}" / f"task{args.task}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "history": history}, model_dir / "model.pt")
    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    for case_id, p in probs.items():
        np.save(pred_dir / f"{case_id}.npy", p)
    with open(pred_dir / "metadata.json", "w") as f:
        json.dump({"fold": args.fold, "task": args.task, "cases": pred_cases, "shape": [512, 512],
                   "dtype": "float16", "input_size": args.input_size, "tag": args.tag}, f, indent=2)
    print(f"MODEL={model_dir / 'model.pt'}", flush=True)
    print(f"PREDICTIONS={pred_dir}", flush=True)


if __name__ == "__main__":
    main()
