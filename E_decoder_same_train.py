import os

# 指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json, re, numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random

# ---------------- Config ----------------
NUM_CLASSES = 4

# 【新增配置】 定义类别名称，用于打印日志
CLASS_NAMES = [
    "Background",
    "Building Energy 1",
    "Building Energy 2",
    "Building Energy 3"
]

# 数据根目录
DATA_ROOT = Path("~/tasks/buildingenergyconsumption").expanduser()

# 指定训练集和验证集的 JSON 路径
TRAIN_JSON = "train-5cities-merge_filtered-500.json"
VAL_JSON = "test-5cities-merge_filtered-500.json"

# 【核心配置 1】Label 文件夹名称 (Y)
CITY_LABEL_MAP = {
    "NewYorkCity": "energy_labels_blocks5x5_classes_new_merge",
    "Lyon": "energy_labels_blocks5x5_classes_2020_new_merge",
    "Boston": "energy_labels_blocks5x5_classes_2025_new_merge",
    "Busan": "energy_labels_blocks5x5_classes_2025_new_merge",
}

# # 【核心配置 1】Label 文件夹名称 (Y)
# CITY_LABEL_MAP = {
#     "NewYorkCity": "NewYorkCity_2KM/energy_image_merge",
#     "Lyon": "Lyon_2KM/energy_image_merge",
#     "Boston": "Boston_2KM/energy_image_merge",
#     "Busan": "Busan_2KM/energy_image_merge",
# }

# NewYorkCity/NewYorkCity_2KM/energy_image_merge/
# 【核心配置 2】隐特征中间层子文件夹名称 (X 的中间层)
CITY_SUBDIR_MAP = {
    "NewYorkCity": "NewYorkCity_2KM",
    "Lyon": "Lyon_2KM",
    "Boston": "Boston_2KM",
    "Busan": "Busan_2KM",
}

# 隐特征文件夹通用名称
HINT_FEAT_FOLDER = "hint_image_merge"

REDUCE_SEED = 42
RATIOS = [1.0]
OUT_ROOT = Path("./runs_hidden_2km")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------- Utils ----------------

def resolve_city_and_paths(json_rec: dict, root_dir: Path) -> Optional[Tuple[Path, Path]]:
    """根据 JSON 记录解析隐特征路径 (.npz) 和 标签路径 (.png/.tif)"""
    # 1. 获取原始文件名 (通过 target 字段)
    sat_str = json_rec.get("target")
    if not sat_str: return None

    sat_path_obj = Path(sat_str)
    filename_stem = sat_path_obj.stem  # 文件名不带后缀
    filename_full = sat_path_obj.name  # 文件名带后缀

    # 2. 确定城市名
    city_raw = json_rec.get("city", "")
    city_clean = city_raw.replace(" ", "")

    if not city_clean:
        for part in sat_path_obj.parts:
            if part in CITY_LABEL_MAP:
                city_clean = part
                break

    # 获取该城市的配置
    label_folder = CITY_LABEL_MAP.get(city_clean)
    subdir_name = CITY_SUBDIR_MAP.get(city_clean)

    if not label_folder or not subdir_name:
        return None

    # 3. 构造路径
    # (X) 隐特征路径: root / City / City_2KM / hint_image_merge / stem_gt.npz
    feat_path = root_dir / city_clean / subdir_name / HINT_FEAT_FOLDER / (filename_stem + "_gt.npz")

    # (Y) 标签路径: root / City / label_folder / full_filename
    nrg_path = root_dir / city_clean / label_folder / filename_full

    # print(f"feat_path: {feat_path}")
    # print(f"nrg_path: {nrg_path}")

    # 如果 Label .tif/.png 后缀不匹配的 fallback
    if not nrg_path.exists():
        nrg_path_png = nrg_path.with_suffix(".png")
        if nrg_path_png.exists():
            nrg_path = nrg_path_png

    return feat_path, nrg_path


def load_data_from_jsonl(json_path: str, data_root: Path) -> List[Tuple[Path, Path]]:
    """读取 JSONL 并构建 (Feature, Label) 路径对"""
    items = []
    p = Path(json_path)
    if not p.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return []

    print(f"[Loader] parsing {json_path} ...")

    valid_count = 0
    missing_count = 0
    skipped_city_count = 0

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            paths = resolve_city_and_paths(rec, data_root)
            if not paths:
                skipped_city_count += 1
                continue

            feat_p, label_p = paths

            if feat_p.exists() and label_p.exists():
                items.append((feat_p, label_p))
                valid_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:
                    if not feat_p.exists(): print(f"[Missing X] {feat_p}")
                    if not label_p.exists(): print(f"[Missing Y] {label_p}")

    print(
        f"[Loader] Loaded {valid_count} pairs. (Missing Files: {missing_count}, Skipped/Unknown Cities: {skipped_city_count})")
    return sorted(items, key=lambda x: str(x[0]))


# ---------------- Dataset ----------------
class UrbanEnergyDataset(torch.utils.data.Dataset):
    def __init__(self, items_override: List[Tuple[Path, Path]], target_size: Tuple[int, int] = (20, 20)):
        self.target_size = target_size
        self.items = items_override

    def __len__(self):
        return len(self.items)

    def _load_feat(self, p: Path) -> torch.Tensor:
        try:
            d = np.load(p)
            x = d.get("arr_0", d.get("feat", None))
            if x is None: x = next(iter(d.values()))
            x = np.asarray(x)
            if x.ndim == 4: x = np.squeeze(x)
            if x.ndim == 3:
                if x.shape[0] == 4:
                    pass
                elif x.shape[-1] == 4:
                    x = np.transpose(x, (2, 0, 1))
            return torch.from_numpy(x).float().contiguous()
        except Exception as e:
            print(f"Error loading {p}: {e}")
            return torch.zeros((4, 512, 512), dtype=torch.float32)

    def _load_label20(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            if img.mode != "L":
                img = img.convert("L")

            # 1. Resize (使用 NEAREST 避免插值产生原本不存在的小数或中间值)
            img20 = img.resize(self.target_size[::-1], resample=Image.NEAREST)
            raw_y = np.array(img20, dtype=np.int64)

            # 2. 创建一个全零的输出数组 (默认为 Class 0 Background)
            y_mapped = np.zeros_like(raw_y)

            # 3. 根据阈值进行分桶 (Binning)
            # Class 1: 0 < val <= T1
            mask_c1 = (raw_y > 0) & (raw_y <= 6)
            y_mapped[mask_c1] = 1

            # Class 2: T1 < val <= T2
            mask_c2 = (raw_y > 6) & (raw_y <= 12)
            y_mapped[mask_c2] = 2

            # Class 3: val > T2
            mask_c3 = (raw_y > 12)
            y_mapped[mask_c3] = 3

            return torch.from_numpy(y_mapped)

    def __getitem__(self, idx):
        f, p = self.items[idx]
        return self._load_feat(f), self._load_label20(p)


# -------------- ResNet18 Backbone (in_ch=4) --------------
def make_resnet18_dilated(in_ch=4):
    try:
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        pretrained = True
    except Exception:
        backbone = torchvision.models.resnet18(weights=None)
        pretrained = False

    old_w = backbone.conv1.weight.data.clone()
    backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False)

    with torch.no_grad():
        if pretrained:
            if in_ch >= 3:
                backbone.conv1.weight[:, :3] = old_w
                if in_ch > 3:
                    backbone.conv1.weight[:, 3:] = old_w.mean(dim=1, keepdim=True)
            else:
                backbone.conv1.weight[:] = old_w[:, :in_ch]
        else:
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

    backbone.maxpool = nn.Identity()
    backbone.layer2[0].conv1.stride = (2, 2)
    backbone.layer2[0].downsample[0].stride = (2, 2)
    backbone.layer3[0].conv1.stride = (2, 2)
    backbone.layer3[0].downsample[0].stride = (2, 2)
    backbone.layer4[0].conv1.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)
    for m in backbone.layer4.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2);
            m.padding = (2, 2)
    return backbone


# -------------- Model --------------
class ResNetSeg20(nn.Module):
    def __init__(self, num_classes=4, in_ch=4, head_ch=256):
        super().__init__()
        self.backbone = make_resnet18_dilated(in_ch=in_ch)
        self.decode = nn.Sequential(
            nn.Conv2d(512, head_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_ch), nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, head_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_ch), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(head_ch, num_classes, 1)

    def forward(self, x):
        x = self.backbone.conv1(x);
        x = self.backbone.bn1(x);
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x);
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x);
        x = self.backbone.layer4(x)
        x = self.decode(x)
        x = F.interpolate(x, size=(20, 20), mode="bilinear", align_corners=False)
        return self.classifier(x)


# -------------- Metrics / Loss --------------
def confmat(pred: torch.Tensor, target: torch.Tensor, num_classes=NUM_CLASSES) -> np.ndarray:
    p = pred.view(-1).cpu().numpy()
    t = target.view(-1).cpu().numpy()
    k = (t >= 0) & (t < num_classes)
    cm = np.bincount(num_classes * t[k] + p[k], minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def metrics_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    iou = tp / np.clip(tp + fp + fn, 1, None)
    acc = tp.sum() / np.clip(cm.sum(), 1, None)
    return prec, rec, iou, acc


def dice_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp;
    fn = cm.sum(1) - tp
    dice = (2 * tp) / np.clip(2 * tp + fp + fn, 1e-9, None)
    mDice = float(np.nanmean(dice))
    return dice, mDice


def estimate_class_weights(dataset: torch.utils.data.Dataset, max_samples: int = 2000):
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
    inv[0] *= 0.5
    print("[ClassFreq]", counts, "-> weights:", np.round(inv, 3))
    return torch.tensor(inv, dtype=torch.float32)


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, weight: torch.Tensor = None, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes;
        self.smooth = smooth;
        self.eps = eps
        if weight is None: weight = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        cardinality = probs.sum(dims) + onehot.sum(dims)
        dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        loss_c = 1.0 - dice_c
        w = self.weight;
        w = w / (w.mean() + self.eps)
        return (w * loss_c).mean()


# -------------- Runner --------------
def set_all_seeds(seed: int):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def run_one_ratio(reduce_ratio: float):
    tag = f"hidden_2km_r{int(round(reduce_ratio * 100)):03d}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(REDUCE_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Run {tag} (ratio={reduce_ratio}) on {device} ==========")

    train_items_full = load_data_from_jsonl(TRAIN_JSON, DATA_ROOT)
    val_items = load_data_from_jsonl(VAL_JSON, DATA_ROOT)

    if len(train_items_full) == 0:
        raise RuntimeError(f"No training data loaded from {TRAIN_JSON}")

    if reduce_ratio < 1.0:
        n_all = len(train_items_full)
        n_pick = max(1, int(round(n_all * reduce_ratio)))
        rng = np.random.default_rng(REDUCE_SEED)
        pick_idx = rng.choice(n_all, size=n_pick, replace=False)
        pick_idx.sort()
        train_items = [train_items_full[i] for i in pick_idx]
        print(f"[TrainReduce] ratio={reduce_ratio} | {n_all} -> {len(train_items)}")
    else:
        train_items = train_items_full
        print(f"[TrainFull] {len(train_items)} samples")

    batch_size = 16
    lr = 3e-4
    weight_decay = 1e-4
    max_epochs = 40
    ce_weight = 1.0
    dice_weight = 1.0
    num_workers = 4

    tr_set = UrbanEnergyDataset(items_override=train_items, target_size=(20, 20))
    class_weights = estimate_class_weights(tr_set).to(device)

    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)

    val_loader = None
    if len(val_items) > 0:
        val_set = UrbanEnergyDataset(items_override=val_items, target_size=(20, 20))
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=True)

    model = ResNetSeg20(num_classes=NUM_CLASSES, in_ch=4, head_ch=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_dice = DiceLoss(NUM_CLASSES, weight=class_weights).to(device)

    best_mIoU = -1.0;
    best_mDice = -1.0
    start_epoch = 1

    # 【新增功能】 尝试加载 Checkpoint
    # 优先加载 best_mIoU 模型
    resume_path = ckpt_dir / f"best_mIoU_{tag}.pt"

    if resume_path.exists():
        print(f"[Resume] Found checkpoint: {resume_path}")
        # 加载到 CPU 避免显存问题，然后再 load_state_dict 到 device
        checkpoint = torch.load(resume_path, map_location=device)

        # 1. 加载模型权重
        model.load_state_dict(checkpoint["model"])

        # 2. 尝试加载优化器状态 (如果上次保存了)
        if "optimizer" in checkpoint:
            print("[Resume] Loading optimizer state...")
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("[Resume] Warning: No optimizer state in checkpoint, starting optimizer fresh.")

        # 3. 恢复 epoch 和 最佳指标
        # 注意：如果加载的是 'best' 模型，它的 epoch 可能是之前的某一个 epoch (例如 35)，
        # 而不是最后训练的 epoch。如果你想从那个最优状态继续微调，这没有问题。
        # 我们从记录的 epoch + 1 开始。
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"[Resume] Resuming from Epoch {start_epoch}")

        best_mIoU = checkpoint.get("mIoU", -1.0)
        best_mDice = checkpoint.get("mDice", -1.0)
        print(f"[Resume] Previous Best mIoU: {best_mIoU:.4f} | Best mDice: {best_mDice:.4f}")

    else:
        print("[Resume] No checkpoint found, starting from scratch.")
        # 如果是新训练，创建/覆盖日志文件
        log_csv = out_dir / f"train_log_{tag}.csv"
        with open(log_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_acc,mIoU,mDice\n")

    log_csv = out_dir / f"train_log_{tag}.csv"

    # 【修改循环】 从 start_epoch 开始
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        loss_sum = 0.0;
        n_seen = 0
        for x, y in tqdm(tr_loader, desc=f"{tag} | Ep {epoch} [Tr]", ncols=100):
            x = x.to(device);
            y = y.to(device).long()
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

        if val_loader:
            model.eval()
            cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"{tag} | Ep {epoch} [Val]", ncols=100):
                    x = x.to(device);
                    y = y.to(device).long()
                    logits = model(x)
                    pred = logits.argmax(1)
                    cm += confmat(pred, y)

            # 获取所有类别的指标数组
            prec, rec, iou, acc = metrics_from_cm(cm)
            dice_c, mDice = dice_from_cm(cm)
            mIoU = float(iou.mean())

            # 打印总体指标
            print(
                f"{tag} | Ep {epoch:02d} | Loss={tr_loss:.4f} | Overall acc={acc:.4f} | mIoU={mIoU:.4f} | mDice={mDice:.4f}")

            # 打印逐类指标
            print("-" * 60)
            for c in range(NUM_CLASSES):
                c_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
                print(f"{c_name:<18}: P={prec[c]:.3f} R={rec[c]:.3f} IoU={iou[c]:.3f} Dice={dice_c[c]:.3f}")
            print("-" * 60)

            # 记录 CSV (使用 'a' 模式追加)
            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{tr_loss:.6f},{acc:.6f},{mIoU:.6f},{mDice:.6f}\n")

            # 【修改保存逻辑】 保存 optimizer 状态以便下次继续
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),  # 新增保存优化器状态
                "epoch": epoch,
                "mIoU": mIoU,
                "mDice": mDice
            }

            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save(save_dict, ckpt_dir / f"best_mIoU_{tag}.pt")
                print(f"[*] Saved best mIoU checkpoint: {mIoU:.4f}")

            if mDice > best_mDice:
                best_mDice = mDice
                torch.save(save_dict, ckpt_dir / f"best_mDice_{tag}.pt")
                print(f"[*] Saved best mDice checkpoint: {mDice:.4f}")

        else:
            print(f"{tag} | Ep {epoch:02d} | Loss={tr_loss:.4f}")


def main():
    for r in RATIOS:
        run_one_ratio(r)


if __name__ == "__main__":
    main()