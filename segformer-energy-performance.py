import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List

from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp

# ---------------- Config ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

NUM_CLASSES = 4
TARGET_SIZE = (512, 512)   # (H, W)
LABEL_SIZE  = (20, 20)     # (H, W)

DATA_ROOT = Path("~/tasks/buildingenergyconsumption").expanduser()
TRAIN_JSON = "train-5cities-merge_filtered-500.json"
VAL_JSON   = "test-5cities-merge_filtered-500.json"

CITY_LABEL_MAP = {
    "NewYorkCity": "energy_labels_blocks5x5_classes_new_merge",
    "Lyon": "energy_labels_blocks5x5_classes_2020_new_merge",
    "Boston": "energy_labels_blocks5x5_classes_2025_new_merge",
    "Busan": "energy_labels_blocks5x5_classes_2025_new_merge",
}


CHECKPOINT_PATH = Path("./runs_energy_segformer_aux/checkpoints_segformer/segformer_best_mix-aux_0p01.pt")

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

    return sorted(items, key=lambda x: str(x[0]))

# ---------------- Dataset ----------------
class UrbanEnergyDataset(torch.utils.data.Dataset):

    def __init__(self, items: List[Tuple[Path, Path]], return_raw: bool = False):
        self.items = items
        self.return_raw = return_raw
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            img = img.convert("RGB").resize(TARGET_SIZE[::-1], resample=Image.BILINEAR)
            x = TF.to_tensor(img)
            return (x - self.mean) / self.std

    def _load_label_and_raw(self, p: Path):
        with Image.open(p) as img:
            img20 = img.convert("L").resize(LABEL_SIZE[::-1], resample=Image.NEAREST)
            raw_y = np.array(img20, dtype=np.float32)  # log1p(EU)

            y_mapped = np.zeros_like(raw_y, dtype=np.int64)
            y_mapped[(raw_y > 0) & (raw_y <= 6)] = 1
            y_mapped[(raw_y > 6) & (raw_y <= 12)] = 2
            y_mapped[raw_y > 12] = 3

            return torch.from_numpy(y_mapped), torch.from_numpy(raw_y)

    def __getitem__(self, idx: int):
        x_path, y_path = self.items[idx]
        x = self._load_image(x_path)
        y_mask, raw = self._load_label_and_raw(y_path)
        if self.return_raw:
            return x, y_mask, raw
        return x, y_mask

# ---------------- Model ----------------
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

# ---------------- Calibration (log-mean) ----------------
def calibrate_rep_raw_logmean(loader, device):
    """
    rep_raw[c] = E[ log1p(EU) | class=c ]   (c=1..3)
    """
    print("[Calibration] Estimating representative log1p(EU) per class (mean in log-space)...")
    raw_sum = torch.zeros(NUM_CLASSES, device=device, dtype=torch.double)
    cnt = torch.zeros(NUM_CLASSES, device=device, dtype=torch.long)

    for _, mask, raw in tqdm(loader, desc="Scanning Train Set", ncols=100):
        mask = mask.to(device)
        raw = raw.to(device).float()  # log1p(EU)
        for c in range(1, NUM_CLASSES):
            m = (mask == c)
            if m.any():
                raw_sum[c] += raw[m].double().sum()
                cnt[c] += m.sum()

    rep_raw = torch.zeros(NUM_CLASSES, device=device, dtype=torch.double)
    for c in range(1, NUM_CLASSES):
        if cnt[c] > 0:
            rep_raw[c] = raw_sum[c] / cnt[c]
    return rep_raw.float()

# ---------------- Metrics ----------------
def nmbe_rmse_cvrmse(pred: np.ndarray, ref: np.ndarray):
    nmbe = (np.sum(pred - ref) / (np.sum(ref) + 1e-12)) * 100
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    cvrmse = (rmse / (np.mean(ref) + 1e-12)) * 100
    return nmbe, rmse, cvrmse

def load_checkpoint_flexible(model: nn.Module, ckpt_path: Path, device):
    """
    兼容两种保存：
      1) torch.save(model.state_dict(), path)
      2) torch.save({"model": model.state_dict(), ...}, path)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise RuntimeError("Unsupported checkpoint format")
    model.load_state_dict(state, strict=True)

# ---------------- Main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    train_items = load_data_from_jsonl(TRAIN_JSON, DATA_ROOT)
    val_items   = load_data_from_jsonl(VAL_JSON, DATA_ROOT)

    calib_set = UrbanEnergyDataset(train_items, return_raw=True)
    calib_loader = torch.utils.data.DataLoader(calib_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    val_set = UrbanEnergyDataset(val_items, return_raw=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 1) Calibration
    rep_raw = calibrate_rep_raw_logmean(calib_loader, device)  # log1p(EU)
    rep_eu  = torch.expm1(rep_raw)                              # EU
    print("rep_raw (log1p EU):", rep_raw.detach().cpu().numpy())
    print("rep_eu  (EU):      ", rep_eu.detach().cpu().numpy())

    # 2) Load SegFormer
    print(f"[Load] checkpoint: {CHECKPOINT_PATH}")
    model = SegFormerEnergy().to(device)
    load_checkpoint_flexible(model, CHECKPOINT_PATH, device)
    model.eval()

    # 3) Inference: district totals (Pred vs Ideal)
    all_pred_eu, all_ideal_eu = [], []

    with torch.no_grad():
        for x, y_mask, _raw in tqdm(val_loader, desc="Testing", ncols=100):
            x = x.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            pred_cls = model(x).argmax(dim=1)  # (B,20,20)

            pred_total_eu  = rep_eu[pred_cls.long()].view(x.size(0), -1).sum(1)   # (B,)
            ideal_total_eu = rep_eu[y_mask.long()].view(x.size(0), -1).sum(1)     # (B,)

            valid = ideal_total_eu > 1.0
            if valid.any():
                all_pred_eu.append(pred_total_eu[valid].cpu().numpy())
                all_ideal_eu.append(ideal_total_eu[valid].cpu().numpy())

    pred_eu = np.concatenate(all_pred_eu, axis=0)
    ideal_eu = np.concatenate(all_ideal_eu, axis=0)

    # 4) Metrics in EU space
    nmbe_eu, rmse_eu, cvrmse_eu = nmbe_rmse_cvrmse(pred_eu, ideal_eu)

    # 5) Metrics in log1p(total EU) space
    pred_log  = np.log1p(pred_eu)
    ideal_log = np.log1p(ideal_eu)
    nmbe_log, rmse_log, cvrmse_log = nmbe_rmse_cvrmse(pred_log, ideal_log)

    print("\n" + "=" * 70)
    print("FINAL ENERGY EVALUATION (SegFormer | District total only | Ref = IDEAL)")
    print("=" * 70)
    print("[EU space]        NMBE: {:>8.2f}% | RMSE: {:>12.4f} | CV(RMSE): {:>7.2f}%"
          .format(nmbe_eu, rmse_eu, cvrmse_eu))
    print("[log1p(EU_total)] NMBE: {:>8.2f}% | RMSE: {:>12.4f} | CV(RMSE): {:>7.2f}%"
          .format(nmbe_log, rmse_log, cvrmse_log))
    print("=" * 70)

if __name__ == "__main__":
    main()

