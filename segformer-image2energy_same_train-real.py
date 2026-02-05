import os
from pathlib import Path
from typing import Tuple, List, Optional
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF

import segmentation_models_pytorch as smp

# ---------------- Config ----------------
NUM_CLASSES = 4          # 能源分级：0..3
TARGET_SIZE = (512, 512) # 输入图像分辨率
LABEL_SIZE = (20, 20)    # 能源标签分辨率

DATA_ROOT = Path("~/tasks/buildingenergyconsumption").expanduser()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

CITY_LABEL_MAP = {
    "NewYorkCity": "energy_labels_blocks5x5_classes_new_merge",
    "Lyon": "energy_labels_blocks5x5_classes_2020_new_merge",
    "Boston": "energy_labels_blocks5x5_classes_2025_new_merge",
    "Busan": "energy_labels_blocks5x5_classes_2025_new_merge",
}

TRAIN_JSON = "train-5cities-merge_filtered-500.json"
VAL_JSON   = "test-5cities-merge_filtered-500.json"

REDUCE_SEED = 42
RATIOS = [0.2]

OUT_ROOT = Path("./runs_energy_segformer")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------- Utils ----------------
def resolve_sat_path(path_str: str, root_dir: Path) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    known_cities = list(CITY_LABEL_MAP.keys())
    for i, part in enumerate(p.parts):
        if part in known_cities:
            new_p = root_dir.joinpath(*p.parts[i:])
            if new_p.exists():
                return new_p
    return p

def load_data_from_jsonl(json_path: str, data_root: Path) -> List[Tuple[Path, Path]]:
    items = []
    p = Path(json_path)
    if not p.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return []

    print(f"[Loader] parsing {json_path} ...")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sat_str = rec.get("target")
                if not sat_str:
                    continue

                sat_path = resolve_sat_path(sat_str, data_root)

                city_raw = rec.get("city", "").replace(" ", "")
                if not city_raw:
                    for part in sat_path.parts:
                        if part in CITY_LABEL_MAP:
                            city_raw = part
                            break

                label_folder = CITY_LABEL_MAP.get(city_raw)
                if not label_folder:
                    continue

                nrg_path = data_root / city_raw / label_folder / sat_path.name
                if sat_path.exists() and nrg_path.exists():
                    items.append((sat_path, nrg_path))
            except Exception:
                continue

    print(f"[Loader] Loaded {len(items)} pairs.")
    return sorted(items, key=lambda x: str(x[0]))

# ---------------- Dataset ----------------
class UrbanEnergyDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Tuple[Path, Path]]):
        self.items = items
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            img = img.convert("RGB").resize(TARGET_SIZE[::-1], resample=Image.BILINEAR)
            x = TF.to_tensor(img)
            return (x - self.mean) / self.std

    def _load_label(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            img20 = img.convert("L").resize(LABEL_SIZE[::-1], resample=Image.NEAREST)
            raw_y = np.array(img20, dtype=np.int64)

            # 0..255 -> 0..3（与你原逻辑一致）
            y_mapped = np.zeros_like(raw_y)
            y_mapped[(raw_y > 0) & (raw_y <= 6)] = 1
            y_mapped[(raw_y > 6) & (raw_y <= 12)] = 2
            y_mapped[raw_y > 12] = 3

            return torch.from_numpy(y_mapped)

    def __getitem__(self, idx: int):
        x_path, y_path = self.items[idx]
        return self._load_image(x_path), self._load_label(y_path)

# -------------- Model --------------
class SegFormerEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Segformer(
            encoder_name="mit_b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if logits.shape[-2:] != LABEL_SIZE:
            logits = F.interpolate(logits, size=LABEL_SIZE, mode="bilinear", align_corners=False)
        return logits

# -------------- Metrics --------------
def confmat(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    p = pred.view(-1).cpu().numpy()
    t = target.view(-1).cpu().numpy()
    cm = np.bincount(NUM_CLASSES * t + p, minlength=NUM_CLASSES ** 2)
    return cm.reshape(NUM_CLASSES, NUM_CLASSES)

def metrics_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    prec = tp / np.clip(tp + fp, 1e-7, None)
    rec  = tp / np.clip(tp + fn, 1e-7, None)
    iou  = tp / np.clip(tp + fp + fn, 1e-7, None)
    acc  = tp.sum() / np.clip(cm.sum(), 1, None)
    return prec, rec, iou, acc

def dice_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    dice = (2 * tp) / np.clip(2 * tp + fp + fn, 1e-9, None)
    mDice = float(np.nanmean(dice))
    return dice, mDice

def estimate_class_weights(dataset: torch.utils.data.Dataset, max_samples: int = 1000) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    n = min(len(dataset), max_samples)
    print(f"[EstWeight] Scanning {n} samples...")
    for i in tqdm(range(n), ncols=100):
        _, y = dataset[i]
        yy = y.numpy().ravel()
        for c in range(NUM_CLASSES):
            counts[c] += (yy == c).sum()

    freq = counts / counts.sum().clip(1)
    inv = 1.0 / np.clip(freq, 1e-4, None)
    inv = inv / inv.mean()
    inv[0] *= 0.5  # background 降权（与你原策略一致）

    print("[ClassFreq]", counts, "-> weights:", np.round(inv, 3))
    return torch.tensor(inv, dtype=torch.float32)

# -------------- Loss (替换为你 ResNet 版本的写法) --------------
class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, weight: torch.Tensor = None,
                 smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.eps = eps
        if weight is None:
            weight = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)

        intersection = (probs * onehot).sum(dims)
        cardinality  = probs.sum(dims) + onehot.sum(dims)

        dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        loss_c = 1.0 - dice_c

        w = self.weight
        w = w / (w.mean() + self.eps)
        return (w * loss_c).mean()

# -------------- Runner --------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_one_ratio(reduce_ratio: float):
    tag = f"r{int(round(reduce_ratio * 100)):03d}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(REDUCE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Run {tag} (ratio={reduce_ratio}) on {device} ==========")

    train_items_full = load_data_from_jsonl(TRAIN_JSON, DATA_ROOT)
    val_items        = load_data_from_jsonl(VAL_JSON, DATA_ROOT)

    if len(train_items_full) == 0:
        raise RuntimeError(f"No training data loaded from {TRAIN_JSON}")

    if reduce_ratio < 1.0:
        n_pick = max(1, int(round(len(train_items_full) * reduce_ratio)))
        train_items = random.sample(train_items_full, n_pick)
        print(f"[TrainReduce] ratio={reduce_ratio} | {len(train_items_full)} -> {len(train_items)}")
    else:
        train_items = train_items_full
        print(f"[TrainFull] {len(train_items)} samples")

    tr_set  = UrbanEnergyDataset(train_items)
    val_set = UrbanEnergyDataset(val_items)

    class_weights = estimate_class_weights(tr_set, max_samples=1000).to(device)

    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=16, shuffle=True,
                                            num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False,
                                             num_workers=4, pin_memory=True)

    model = SegFormerEnergy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    criterion_ce   = nn.CrossEntropyLoss(weight=class_weights)
    criterion_dice = DiceLoss(NUM_CLASSES, weight=class_weights).to(device)

    # loss 权重（和你 ResNet 脚本一致的结构）
    ce_weight = 1.0
    dice_weight = 1.0

    best_mIoU = -1.0
    log_csv = out_dir / f"train_log_{tag}.csv"

    # CSV：加入每类 IoU + mDice
    with open(log_csv, "w", encoding="utf-8") as f:
        iou_headers = ",".join([f"iou_c{i}" for i in range(NUM_CLASSES)])
        f.write(f"epoch,train_loss,val_acc,mIoU,mDice,{iou_headers}\n")

    for epoch in range(1, 31):
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for x, y in tqdm(tr_loader, desc=f"{tag} | Ep {epoch} [Train]", ncols=100):
            x, y = x.to(device), y.to(device).long()
            logits = model(x)

            loss_ce = criterion_ce(logits, y)
            loss_dice = criterion_dice(logits, y)
            loss = ce_weight * loss_ce + dice_weight * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            n_seen += bs

        tr_loss = loss_sum / max(1, n_seen)

        # ---- Val ----
        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"{tag} | Ep {epoch} [Val]", ncols=100):
                x, y = x.to(device), y.to(device).long()
                pred = model(x).argmax(1)
                cm += confmat(pred, y)

        prec, rec, iou, acc = metrics_from_cm(cm)
        mIoU = float(iou.mean())
        dice_c, mDice = dice_from_cm(cm)

        print(f"\n{tag} | Epoch {epoch:02d} | Loss: {tr_loss:.4f} | Acc: {acc:.4f} | mIoU: {mIoU:.4f} | mDice: {mDice:.4f}")
        for c in range(NUM_CLASSES):
            print(f"  Class {c}: IoU={iou[c]:.4f} | Prec={prec[c]:.4f} | Rec={rec[c]:.4f}")
        print("-" * 30)

        with open(log_csv, "a", encoding="utf-8") as f:
            iou_str = ",".join([f"{val:.6f}" for val in iou])
            f.write(f"{epoch},{tr_loss:.6f},{acc:.6f},{mIoU:.6f},{mDice:.6f},{iou_str}\n")

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            torch.save(model.state_dict(), out_dir / f"segformer-best_mIoU_{tag}.pt")

def main():
    for r in RATIOS:
        run_one_ratio(r)

if __name__ == "__main__":
    main()
