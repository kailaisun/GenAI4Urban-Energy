import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import json


DATA_ROOT = Path("~/tasks/buildingenergyconsumption").expanduser()
TRAIN_JSON = "train-5cities-merge_filtered-500.json"
CITY_LABEL_MAP = {
    "NewYorkCity": "energy_labels_blocks5x5_classes_new_merge",
    "Lyon": "energy_labels_blocks5x5_classes_2020_new_merge",
    "Boston": "energy_labels_blocks5x5_classes_2025_new_merge",
    "Busan": "energy_labels_blocks5x5_classes_2025_new_merge",
}
CITY_SUBDIR_MAP = {  
    "NewYorkCity": "NewYorkCity_2KM",
    "Lyon": "Lyon_2KM",
    "Boston": "Boston_2KM",
    "Busan": "Busan_2KM",
}
HINT_FEAT_FOLDER = "hint_image_merge"



def resolve_city_and_paths(json_rec: dict, root_dir: Path):
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
    nrg_path = root_dir / city_clean / label_folder / filename_full
    if not nrg_path.exists():
        nrg_path_png = nrg_path.with_suffix(".png")
        if nrg_path_png.exists(): nrg_path = nrg_path_png
    return feat_path, nrg_path


def load_data_from_jsonl(json_path: str, data_root: Path):
    items = []
    p = Path(json_path)
    if not p.exists(): return []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except:
                continue
            paths = resolve_city_and_paths(rec, data_root)
            if paths and paths[1].exists():
                items.append(paths)
    return items



def calculate_thresholds():
    train_items = load_data_from_jsonl(TRAIN_JSON, DATA_ROOT)

    valid_pixels = []


    for _, label_path in tqdm(train_items):
        try:
            with Image.open(label_path) as img:
                if img.mode != "L": img = img.convert("L")
                arr = np.array(img)

                non_zero = arr[arr > 0]

                if len(non_zero) > 0:
                    valid_pixels.append(non_zero)
        except Exception as e:
            print(f"Error reading {label_path}: {e}")


    all_values = np.concatenate(valid_pixels)

    # compute 33.3% and 66.6% 
    t1 = np.percentile(all_values, 33.33)
    t2 = np.percentile(all_values, 66.66)

    print("\n" + "=" * 40)
    print(f"no-zero pixel: {len(all_values)}")
    print(f"min: {all_values.min()}, max: {all_values.max()}, average: {all_values.mean():.2f}")
    print(f"-" * 40)
    print(f" T1 (33%): {t1}")
    print(f" T2 (66%): {t2}")
    print("=" * 40)


if __name__ == "__main__":

    calculate_thresholds()
