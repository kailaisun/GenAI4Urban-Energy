import os

# 指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF  

import segmentation_models_pytorch as smp

# ---------------- Config ----------------
NUM_CLASSES = 5
TARGET_SIZE = (512, 512)

DATA_ROOT = Path("~/tasks/buildingenergyconsumption").expanduser()
TRAIN_JSON = "train-5cities-merge_filtered-500.json"
VAL_JSON = "test-5cities-merge_filtered-500.json"

CITY_LABEL_MAP = {
    "NewYorkCity": "NewYorkCity_2KM/height_image_merge/",
    "Lyon": "Lyon_2KM/height_image_merge/",
    "Boston": "Boston_2KM/height_image_merge/",
    "Busan": "Busan_2KM/height_image_merge/",
}

CITY_SUBDIR_MAP = {
    "NewYorkCity": "NewYorkCity_2KM",
    "Lyon": "Lyon_2KM",
    "Boston": "Boston_2KM",
    "Busan": "Busan_2KM",
}

HINT_FEAT_FOLDER = "hint_image_merge"
REDUCE_SEED = 42
OUT_ROOT = Path("./runs_height_improved")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------- Utilities ----------------
def resolve_city_and_paths(json_rec: dict, root_dir: Path) -> Optional[Tuple[Path, Path]]:
    sat_str = json_rec.get("target")
    if not sat_str: return None
    sat_path_obj = Path(sat_str)
    filename_stem = sat_path_obj.stem
    filename_full = sat_path_obj.name

    city_raw = json_rec.get("city", "")
    city_clean = city_raw.replace(" ", "")

    if not city_clean:
        for part in sat_path_obj.parts:
            if part in CITY_LABEL_MAP:
                city_clean = part
                break

    label_folder = CITY_LABEL_MAP.get(city_clean)
    subdir_name = CITY_SUBDIR_MAP.get(city_clean)
    if not label_folder or not subdir_name: return None

    feat_path = root_dir / city_clean / subdir_name / HINT_FEAT_FOLDER / (filename_stem + "_gt.npz")
    nrg_path = root_dir / city_clean / label_folder / (filename_stem + ".png")

    if not nrg_path.exists():
        temp_path = root_dir / city_clean / label_folder / filename_full
        if temp_path.exists():
            nrg_path = temp_path
        elif nrg_path.with_suffix(".tif").exists():
            nrg_path = nrg_path.with_suffix(".tif")

    return feat_path, nrg_path


def load_data_from_jsonl(json_path: str, data_root: Path) -> List[Tuple[Path, Path]]:
    items = []
    p = Path(json_path)
    if not p.exists(): return []
    print(f"[Loader] parsing {json_path} ...")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except:
                continue
            paths = resolve_city_and_paths(rec, data_root)
            if paths and paths[0].exists() and paths[1].exists():
                items.append(paths)
    return sorted(items, key=lambda x: str(x[0]))


def compute_quantile_thresholds(items, sample_ratio=0.5):

    print(f"\n[Thresholds] Calculating dynamic thresholds using {sample_ratio * 100}% of data...")
    pixel_values = []
    n_samples = max(10, int(len(items) * sample_ratio))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(items), size=n_samples, replace=False)

    for idx in tqdm(sample_indices, desc="Scanning pixels"):
        _, label_path = items[idx]
        try:
            with Image.open(label_path) as img:
                if img.mode != "L": img = img.convert("L")
                arr = np.array(img)
                valid = arr[arr > 0]
                if len(valid) > 5000: valid = np.random.choice(valid, 5000)
                if len(valid) > 0: pixel_values.append(valid)
        except:
            continue

    if not pixel_values: return [86, 109, 165]
    all_pixels = np.concatenate(pixel_values)
    thresholds = np.percentile(all_pixels, [25, 50, 75])
    print(f"  Computed Thresholds: {thresholds}")
    return thresholds.tolist()


# ---------------- Dataset with Augmentation ----------------
class UrbanHeightDataset(torch.utils.data.Dataset):
    def __init__(self, items, thresholds, target_size=TARGET_SIZE, is_train=True):
        self.items = items
        self.target_size = target_size
        self.thresholds = thresholds
        self.is_train = is_train  

    def __len__(self):
        return len(self.items)

    def _load_feat(self, p: Path) -> torch.Tensor:
        try:
            d = np.load(p)
            x = d.get("arr_0", d.get("feat", next(iter(d.values()))))
            x = np.asarray(x)
            if x.ndim == 4: x = np.squeeze(x)
            if x.ndim == 3 and x.shape[-1] == 4: x = np.transpose(x, (2, 0, 1))
            return torch.from_numpy(x).float().contiguous()
        except:
            return torch.zeros((4, self.target_size[0], self.target_size[1]), dtype=torch.float32)

    def _load_label(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            if img.mode != "L": img = img.convert("L")
            img = img.resize(self.target_size, resample=Image.NEAREST)
            raw_y = np.array(img, dtype=np.float32)

            y_mapped = np.zeros_like(raw_y, dtype=np.int64)
            t1, t2, t3 = self.thresholds
            y_mapped[(raw_y > 0) & (raw_y <= t1)] = 1
            y_mapped[(raw_y > t1) & (raw_y <= t2)] = 2
            y_mapped[(raw_y > t2) & (raw_y <= t3)] = 3
            y_mapped[(raw_y > t3)] = 4
            return torch.from_numpy(y_mapped)

    def __getitem__(self, idx):
        f, p = self.items[idx]
        feat = self._load_feat(f)
        label = self._load_label(p)


        if self.is_train:
            if random.random() > 0.5:
                feat = TF.hflip(feat)
                label = TF.hflip(label)

            if random.random() > 0.5:
                feat = TF.vflip(feat)
                label = TF.vflip(label)

            if random.random() > 0.5:
                rot = random.choice([90, 180, 270])
                k = rot // 90
                feat = torch.rot90(feat, k, [1, 2])
                label = torch.rot90(label, k, [0, 1])

        return feat, label


# ---------------- Model with Input BN ----------------
class SegFormerHeight(nn.Module):
    def __init__(self, in_chans, num_classes):
        super().__init__()

        self.input_bn = nn.BatchNorm2d(in_chans)

        self.model = smp.Segformer(
            encoder_name="mit_b3",
            encoder_weights="imagenet",
            in_channels=in_chans,
            classes=num_classes,
        )

    def forward(self, x):
        x = self.input_bn(x)
        out = self.model(x)
        if out.shape[-2:] != TARGET_SIZE:
            out = F.interpolate(out, TARGET_SIZE, mode="bilinear", align_corners=False)
        return out


class WeightedCombinedLoss(nn.Module):
    def __init__(self, weight=None, num_classes=5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = smp.losses.DiceLoss(mode='multiclass', classes=num_classes, from_logits=True)

    def forward(self, logits, target):
        return 0.6 * self.ce(logits, target) + 0.4 * self.dice(logits, target)


# ---------------- Tools ----------------
def confmat(pred, target, num_classes=NUM_CLASSES):
    p = pred.view(-1).cpu().numpy()
    t = target.view(-1).cpu().numpy()
    cm = np.bincount(num_classes * t + p, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def metrics_from_cm(cm):
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (tp + fp + fn)
        acc = tp.sum() / cm.sum()
    return np.nan_to_num(iou), acc


def estimate_class_weights(dataset):
    counts = np.zeros(NUM_CLASSES, np.int64)
    n = min(len(dataset), 2000)
    for i in tqdm(range(n), desc="[EstWeight]"):
        _, y = dataset[i]
        yy = y.numpy().ravel()
        for c in range(NUM_CLASSES): counts[c] += (yy == c).sum()
    freq = counts / counts.sum()
    w = 1 / np.clip(freq, 1e-4, None)
    w = w / w.mean()
    w[0] *= 0.5
    return torch.tensor(w, dtype=torch.float32)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Main ----------------
def run():
    tag = "height_improved_v2"
    ckpt_dir = OUT_ROOT / tag / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(REDUCE_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"========== Start {tag} ==========")

    train_items = load_data_from_jsonl(TRAIN_JSON, DATA_ROOT)
    val_items = load_data_from_jsonl(VAL_JSON, DATA_ROOT)

    thresholds = compute_quantile_thresholds(train_items)

    tr_set = UrbanHeightDataset(train_items, thresholds, TARGET_SIZE, is_train=True)
    val_set = UrbanHeightDataset(val_items, thresholds, TARGET_SIZE, is_train=False)

    batch_size = 16
    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    in_ch = tr_set[0][0].shape[0]
    model = SegFormerHeight(in_chans=in_ch, num_classes=NUM_CLASSES).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


    max_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    class_weights = estimate_class_weights(tr_set).to(device)
    print("Class Weights:", class_weights.cpu().numpy())
    criterion = WeightedCombinedLoss(weight=class_weights)

    best_mIoU = -1.0
    log_csv = OUT_ROOT / tag / "train_log.csv"
    with open(log_csv, "w") as f:
        f.write("epoch,loss,acc,mIoU,lr\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0;
        n_seen = 0
        for x, y in tqdm(tr_loader, desc=f"Ep {epoch} [Tr]", ncols=80):
            x = x.to(device);
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        tr_loss = loss_sum / n_seen

        # Val
        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Ep {epoch} [Val]", ncols=80):
                x = x.to(device);
                y = y.to(device)
                pred = model(x).argmax(1)
                cm += confmat(pred, y)

        iou, acc = metrics_from_cm(cm)
        mIoU = float(iou.mean())

        print(f"Ep {epoch} | Loss={tr_loss:.4f} | Acc={acc:.4f} | mIoU={mIoU:.4f} | LR={curr_lr:.2e}")
        print(f"  IoU: {np.round(iou, 3)}")

        with open(log_csv, "a") as f:
            f.write(f"{epoch},{tr_loss},{acc},{mIoU},{curr_lr}\n")

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            torch.save(model.state_dict(), ckpt_dir / f"best_mIoU_{best_mIoU:.4f}.pt")
            print("  [*] Saved Best mIoU")


if __name__ == "__main__":

    run()
