import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json, re, numpy as np
from PIL import Image
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
import random
import segmentation_models_pytorch as smp

# ---------------- Config ----------------
NUM_CLASSES = 4  # 能源分级：0..3
TARGET_SIZE = (512, 512)  # SegFormer 输入图像分辨率
LABEL_SIZE = (20, 20)  # 能源标签分辨率

# 显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 基础数据路径
DATA_ROOT = Path("~/tasks/buildingenergyconsumption").expanduser()
TRAIN_JSON = "train-5cities-merge_filtered-500.json"
VAL_JSON = "test-5cities-merge_filtered-500.json"

# 【新增】合成数据配置
AUX_IMG_DIR = "../../output_image/5city_test-one/5-city-lastlas"
AUX_LABEL_DIR = "../../output_image/5city_test-one/5-city-lastlas"

CITY_LABEL_MAP = {
    "NewYorkCity": "energy_labels_blocks5x5_classes_new_merge",
    "Lyon": "energy_labels_blocks5x5_classes_2020_new_merge",
    "Boston": "energy_labels_blocks5x5_classes_2025_new_merge",
    "Busan": "energy_labels_blocks5x5_classes_2025_new_merge",
}

# 【核心修改】合成数据比例搜索空间
# 例如：0.2 表示加入合成数据总量的 20% 注入训练集
AUX_RATE_LIST = [0.01,0.4,0.5, 1.0]
data_rate=0.4
REDUCE_SEED = 42
MAX_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 3e-4

OUT_ROOT = Path("./runs_energy_segformer_aux")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
Path("./runs_energy_segformer_aux/checkpoints_segformer").mkdir(parents=True, exist_ok=True)


# ---------------- Utils (路径解析与数据加载) ----------------

def resolve_sat_path(path_str: str, root_dir: Path) -> Path:
    p = Path(path_str)
    if p.exists(): return p
    known_cities = list(CITY_LABEL_MAP.keys())
    for i, part in enumerate(p.parts):
        if part in known_cities:
            new_p = root_dir.joinpath(*p.parts[i:])
            if new_p.exists(): return new_p
    return p


def load_data_from_jsonl(json_path: str, data_root: Path) -> List[Tuple[Path, Path]]:
    items = []
    p = Path(json_path)
    if not p.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return []
    print(f"[Loader] Parsing {json_path} ...")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                sat_str = rec.get("target")
                if not sat_str: continue
                sat_path = resolve_sat_path(sat_str, data_root)
                city_raw = rec.get("city", "").replace(" ", "")
                if not city_raw:
                    for part in sat_path.parts:
                        if part in CITY_LABEL_MAP:
                            city_raw = part;
                            break
                label_folder = CITY_LABEL_MAP.get(city_raw)
                if not label_folder: continue
                nrg_path = data_root / city_raw / label_folder / sat_path.name
                if sat_path.exists() and nrg_path.exists():
                    items.append((sat_path, nrg_path))
            except:
                continue
    return sorted(items, key=lambda x: str(x[0]))


def get_forbidden_stems(json_path: str) -> set:
    """提取验证集中的 ID，防止合成数据泄露"""
    stems = set()
    p = Path(json_path)
    if not p.exists(): return stems
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                target_path = rec.get("target")
                if target_path: stems.add(Path(target_path).stem)
            except:
                continue
    print(f"[Filter] Loaded {len(stems)} forbidden stems from validation set.")
    return stems


def get_train_keys(json_path: str) -> set:
    """辅助函数：从训练集 JSON 中提取 (城市名, ID) 组合"""
    keys = set()
    p = Path(json_path)
    if not p.exists():
        print(f"[Warning] Training JSON not found: {json_path}")
        return keys
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                # 清洗城市名（去除空格，保持与文件名一致）
                city = rec.get("city", "").replace(" ", "")
                target_path = rec.get("target")
                if target_path:
                    # 提取 target 的文件名作为 ID (例如: top33_left16_r2_d0)
                    id_str = Path(target_path).stem
                    keys.add((city, id_str))
            except:
                continue
    return keys


def scan_pairs_aux(aux_img_dir: str, aux_label_dir: str, exclude_json_path: str = None, train_json_path: str = None) -> \
List[Tuple[Path, Path]]:
    """
    扫描合成数据：
    1. 排除验证集泄露数据 (exclude_json_path)
    2. 【新增】必须在训练集中存在的图片 (train_json_path)
    """
    aux_img_dir, aux_label_dir = Path(aux_img_dir), Path(aux_label_dir)
    items = []

    # 1. 加载黑名单 (验证集 IDs，用于防泄露)
    forbidden_stems = get_forbidden_stems(exclude_json_path) if exclude_json_path else set()

    # 2. 加载白名单 (训练集 City + ID 组合)
    train_keys = get_train_keys(train_json_path) if train_json_path else set()
    if train_json_path:
        print(f"[Filter] Loaded {len(train_keys)} valid (City, ID) pairs from training set.")

    img_globs = ["*.jpg", "*.png", "*.PNG", "*.jpeg"]
    leak_count = 0
    not_in_train_count = 0

    for pat in img_globs:
        for f in sorted(aux_img_dir.glob(pat)):
            if "label" in f.name: continue

            # 正则匹配 stem 和样本序号 k
            # 示例: Lyon_2KM-hint_image_merge-top33_left28_r1_d0.png_sample0.jpg
            m = re.match(r"^(?P<stem>.+?)(?:\.energy)?\.png_sample(?P<k>\d+)\.", f.name)
            if not m: m = re.match(r"^(?P<stem>.+?)_sample(?P<k>\d+)\.", f.name)
            if not m: continue

            stem = m.group("stem")
            k = m.group("k")

            # --- 解析合成数据的城市和 ID ---
            # 根据格式，城市通常在第一个下划线前，ID 在最后一个横杠后
            try:
                city = stem.split('_')[0]  # 例如 Lyon
                id_str = stem.split('-')[-1]  # 例如 top33_left28_r1_d0
            except IndexError:
                continue

            # A. 泄露检查 (验证集黑名单)
            if any(bad_id in stem for bad_id in forbidden_stems):
                leak_count += 1
                continue

            # B. 【核心修改】训练集存在检查
            # 如果指定了训练集路径，则进行强匹配
            if train_json_path:
                if (city, id_str) not in train_keys:
                    not_in_train_count += 1
                    continue

            # 寻找对应的 Label 文件
            p_candidates = [
                # aux_label_dir / f"{stem}.energy.energy_sample{k}.label.png",
                aux_label_dir / f"{stem}.energy_sample{k}.label.png",
                aux_label_dir / f"{stem}_sample{k}_label.png"
            ]
            for p in p_candidates:
                if p.exists():
                    items.append((f, p))
                    break

    print(f"[AuxScan] Found {len(items)} valid synthetic pairs.")
    if train_json_path:
        print(f"[AuxScan] Filtered: {leak_count} leaks, {not_in_train_count} samples not found in training set.")

    return items


# ---------------- Dataset ----------------

class UrbanEnergyDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Tuple[Path, Path]]):
        self.items = items
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.items)

    def _load_image(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            img = img.convert("RGB").resize(TARGET_SIZE[::-1], resample=Image.BILINEAR)
            x = TF.to_tensor(img)
            return (x - self.mean) / self.std

    def _load_label(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            img20 = img.convert("L").resize(LABEL_SIZE[::-1], resample=Image.NEAREST)
            raw_y = np.array(img20, dtype=np.int64)
            y_mapped = np.zeros_like(raw_y)
            # print(np.max(raw_y))
            if np.max(raw_y)<=3:
                y_mapped = raw_y
            else:
                y_mapped[(raw_y > 0) & (raw_y <= 6)] = 1
                y_mapped[(raw_y > 6) & (raw_y <= 12)] = 2
                y_mapped[raw_y > 12] = 3
            return torch.from_numpy(y_mapped)

    def __getitem__(self, idx):
        f, p = self.items[idx]
        return self._load_image(f), self._load_label(p)


# -------------- Model & Loss --------------

class SegFormerEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Segformer(encoder_name="mit_b3", encoder_weights="imagenet",
                                   in_channels=3, classes=NUM_CLASSES)

    def forward(self, x):
        logits = self.model(x)
        if logits.shape[-2:] != LABEL_SIZE:
            logits = F.interpolate(logits, size=LABEL_SIZE, mode="bilinear", align_corners=False)
        return logits


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, weight: torch.Tensor = None, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.register_buffer("weight", weight if weight is not None else torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        cardinality = probs.sum(dims) + onehot.sum(dims)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth + 1e-7)
        return (self.weight * dice_loss).mean()


# -------------- Metrics --------------

def confmat(pred, target):
    p, t = pred.view(-1).cpu().numpy(), target.view(-1).cpu().numpy()
    mask = (t >= 0) & (t < NUM_CLASSES)
    cm = np.bincount(NUM_CLASSES * t[mask] + p[mask], minlength=NUM_CLASSES ** 2)
    return cm.reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    prec = tp / np.clip(tp + fp, 1e-7, None)
    rec = tp / np.clip(tp + fn, 1e-7, None)
    iou = tp / np.clip(tp + fp + fn, 1e-7, None)
    acc = tp.sum() / np.clip(cm.sum(), 1, None)
    return prec, rec, iou, acc


def estimate_class_weights(items: List):
    # 抽样估计类别权重
    dataset = UrbanEnergyDataset(items)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    n = min(len(dataset), 500)
    for i in range(n):
        _, y = dataset[i]
        for c in range(NUM_CLASSES): counts[c] += (y == c).sum().item()
    inv = 1.0 / np.clip(counts / counts.sum(), 1e-4, None)
    inv = inv / inv.mean()
    inv[0] *= 0.5  # 降低背景权重
    return torch.tensor(inv, dtype=torch.float32)


# -------------- Core Logic: Mixing --------------

def build_train_val_mixed(aux_rate: float):
    # 1. 加载真实验证集
    val_real = load_data_from_jsonl(VAL_JSON, DATA_ROOT)
    # 2. 加载真实训练集
    train_real = load_data_from_jsonl(TRAIN_JSON, DATA_ROOT)
    # 3. 扫描合成数据（带防泄露）
    aux_all = scan_pairs_aux(AUX_IMG_DIR, AUX_LABEL_DIR, VAL_JSON)

    aux_used = []
    if aux_rate > 0 and len(aux_all) > 0:
        n_pick = max(1, int(round(len(aux_all) * aux_rate)))
        rng = np.random.default_rng(REDUCE_SEED)
        idx = rng.choice(len(aux_all), size=n_pick, replace=False)
        aux_used = [aux_all[i] for i in idx]
        print(f"[Mix] Added {len(aux_used)} synthetic samples (Rate: {aux_rate})")

    if data_rate > 0 and len(train_real) > 0:
        n_pick = max(1, int(round(len(train_real) * data_rate)))
        rng = np.random.default_rng(REDUCE_SEED)
        idx = rng.choice(len(train_real), size=n_pick, replace=False)
        real_train = [train_real[i] for i in idx]
        print(f"[real](Rate: {data_rate})  {len(real_train)}  samples)")

    train_final = real_train + aux_used
    return train_final, val_real


# -------------- Runner --------------

def train_one_trial(aux_rate: float, train_items: List, val_items: List):
    tag = f"aux_{str(aux_rate).replace('.', 'p')}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_set = UrbanEnergyDataset(train_items)
    val_set = UrbanEnergyDataset(val_items)
    class_weights = estimate_class_weights(train_items).to(device)

    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SegFormerEnergy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_dice = DiceLoss(NUM_CLASSES, weight=class_weights)

    best_mIoU = -1.0
    log_csv = out_dir / f"train_log_mix-{tag}.csv"

    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_acc,mIoU\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in tqdm(tr_loader, desc=f"Rate {aux_rate} Ep {epoch} [Tr]"):
            x, y = x.to(device), y.to(device).long()
            logits = model(x)
            loss = criterion_ce(logits, y) + criterion_dice(logits, y)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            loss_sum += loss.item() * x.size(0)

        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).long()
                pred = model(x).argmax(1)
                cm += confmat(pred, y)

        prec, rec, iou, acc = metrics_from_cm(cm)
        mIoU = float(iou.mean())
        tr_loss = loss_sum / len(tr_set)

        print(f"Epoch {epoch} | Loss: {tr_loss:.4f} | Acc: {acc:.4f} | mIoU: {mIoU:.4f}")
        for c in range(NUM_CLASSES):
            print(f"  Class {c}: IoU={iou[c]:.4f} | Prec={prec[c]:.4f} | Rec={rec[c]:.4f}")
        print("-" * 30)
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{tr_loss:.6f},{acc:.6f},{mIoU:.6f}\n")

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            ckpt_path = Path("./runs_energy_segformer_aux/checkpoints_segformer") / f"segformer_best_mix-{tag}.pt"
            torch.save(model.state_dict(), ckpt_path)


def main():
    random.seed(REDUCE_SEED)
    np.random.seed(REDUCE_SEED)
    torch.manual_seed(REDUCE_SEED)

    for rate in AUX_RATE_LIST:
        print(f"\n{'=' * 30}\nStarting Trial: AUX_RATE = {rate}\n{'=' * 30}")
        train_items, val_items = build_train_val_mixed(rate)
        train_one_trial(rate, train_items, val_items)


if __name__ == "__main__":
    main()