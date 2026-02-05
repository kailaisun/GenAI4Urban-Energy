import os, json, random
from pathlib import Path
from collections import defaultdict

# ========= 可配置 =========
BASE_PATH = Path(
    "/home/kailais/tasks/buildingenergyconsumption/"
)  # 数据集根目录

# 要处理的城市目录（必须与 BASE_PATH 下的子目录同名）
CITIES = ["New York City", "Busan", "Boston", "Lyon"]

BLOCK_SIZE = 5                         # 5×5 → 2km×2km
STRIDE_ROW_IDX = 2                     # 行锚点索引步长
STRIDE_COL_IDX = 2                     # 列锚点索引步长
OUT_TRAIN = "./train-5cities-merge-all-nouse.json"  # 输出训练集（所有城市合并）
OUT_TEST  = "./test-5cities-merge-all-nouse.json"   # 输出测试集
SEED = 0
# =========================

# 每个城市对应的 2KM 数据目录，例如：
# New York City -> /home/.../NewYorkCity/NewYorkCity_2KM
CITY_TILE_DIRS = {
    "New York City": BASE_PATH / "NewYorkCity" / "NewYorkCity_2KM",
    "Busan":         BASE_PATH / "Busan"       / "Busan_2KM",
    # "Hong Kong":     BASE_PATH / "HongKong"    / "HongKong_2KM",
    "Boston":        BASE_PATH / "Boston"      / "Boston_2KM",
    "Lyon":          BASE_PATH / "Lyon"        / "Lyon_2KM",
}

# 每个城市的 3way 过滤目录（在城市根目录下，而不是 2KM 目录里）
# CITY_FILTER_DIRS = {
#     "New York City": BASE_PATH / "NewYorkCity" / "NewYorkCity_2KM" / "positives_side_by_side_energy_height_5x5_2025_merge_filter",
#     "Busan":         BASE_PATH / "Busan"       / "Busan_2KM" / "positives_side_by_side_energy_height_5x5_2025_merge_filter",
#     "Hong Kong":     BASE_PATH / "HongKong"    / "HongKong_2KM" / "positives_side_by_side_energy_height_5x5_2025_merge_filter",
#     "Boston":        BASE_PATH / "Boston"      / "Boston_2KM" / "positives_side_by_side_energy_height_5x5_2025_merge_filter",
#     "Lyon":          BASE_PATH / "Lyon"        / "Lyon_2KM" / "positives_side_by_side_energy_height_5x5_2020_merge_filter",
# }

CITY_FILTER_DIRS = {
    "New York City": BASE_PATH / "NewYorkCity" / "NewYorkCity_2KM" / "satellite_image_merge",
    "Busan":         BASE_PATH / "Busan"       / "Busan_2KM" / "satellite_image_merge",
    # "Hong Kong":     BASE_PATH / "HongKong"    / "HongKong_2KM" / "hint_image_merge",
    "Boston":        BASE_PATH / "Boston"      / "Boston_2KM" / "satellite_image_merge",
    "Lyon":          BASE_PATH / "Lyon"        / "Lyon_2KM" / "satellite_image_merge",
}

def list_geojsons(city_tiles_path: Path):
    """
    在 city_tiles_path (比如 NewYorkCity_2KM) 下递归查找
    *density_landuse.geojson 文件。
    """
    return sorted(
        [
            p for p in city_tiles_path.rglob("*.geojson")
            if "density_landuse" in p.name
        ]
    )

def load_features(geojson_path: Path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("features", [])

def group_by_rd(features):
    """把 400m tile 按 (r,d) 分组；返回 {(r,d): {(row,col): props}}"""
    idx = defaultdict(dict)
    for ft in features:
        prop = ft.get("properties", {})
        r_tag = str(prop.get("r", "r0"))  # 'r0'
        d_tag = str(prop.get("d", "d0"))  # 'd0'
        row = int(prop["row"])
        col = int(prop["col"])
        idx[(r_tag, d_tag)][(row, col)] = prop
    return idx

def find_blocks_strided(rc_to_prop: dict, block_w=5, block_h=5,
                        stride_row_idx=3, stride_col_idx=3):
    """在 (row,col) 索引上步进，挑选 5×5 完整块的左上锚点 (top_row,left_col)。"""
    rows_sorted = sorted({r for r, _ in rc_to_prop.keys()})
    cols_sorted = sorted({c for _, c in rc_to_prop.keys()})
    blocks = []
    for i in range(0, len(rows_sorted), stride_row_idx):
        top_row = rows_sorted[i]
        # 向下（row-1）要能取满 block_h 行
        if any((top_row - dy) not in rows_sorted for dy in range(block_h)):
            continue
        for j in range(0, len(cols_sorted), stride_col_idx):
            left_col = cols_sorted[j]
            # 向右（col+1）要能取满 block_w 列
            if any((left_col + dx) not in cols_sorted for dx in range(block_w)):
                continue
            # 5×5 是否齐全
            full = True
            for dy in range(block_h):
                for dx in range(block_w):
                    if (top_row - dy, left_col + dx) not in rc_to_prop:
                        full = False
                        break
                if not full:
                    break
            if full:
                blocks.append((top_row, left_col))
    return blocks

def agg_metrics_for_block(rc_to_prop: dict, top_row:int, left_col:int, block_w:int, block_h:int):
    """
    聚合 25 格：
      - BCR = Σ(built_surface_total) / Σ(cellarea) × 100（%）
      - BVD = Σ(built_volume_total)  / Σ(cellarea)（m³/m²）
      - Road Density = 简单算术平均(mean) 的 road_density（km/km²）
    """
    sum_surface = 0.0      # m^2
    sum_volume  = 0.0      # m^3
    sum_area    = 0.0      # m^2
    rd_list     = []       # 收集每格 road_density（若缺失则忽略）

    for dy in range(block_h):
        for dx in range(block_w):
            prop = rc_to_prop[(top_row - dy, left_col + dx)]
            surface = float(prop.get("built_surface_total", 0.0))
            volume  = float(prop.get("built_volume_total",  0.0))
            area    = float(prop.get("cellarea",           160000.0))
            sum_surface += surface
            sum_volume  += volume
            sum_area    += area

            rd_val = prop.get("road_density", None)
            if rd_val is not None:
                try:
                    rd_list.append(float(rd_val))
                except Exception:
                    pass

    if sum_area <= 0:
        return None

    bcr = (sum_surface / sum_area) * 100.0          # %
    bvd = (sum_volume  / sum_area)                  # m^3/m^2
    road_density = (sum(rd_list) / len(rd_list)) if rd_list else 0.0  # 简单平均

    return bcr, bvd, road_density

def find_first_existing(base: Path, exts=(".jpg",".png",".tif",".tiff")):
    """根据 base（不带后缀）找第一个存在的文件，找不到返回 None。"""
    for ext in exts:
        p = base.with_suffix(ext)
        if p.exists():
            return str(p)
    return None

# === 每个城市自己的 3way 过滤目录 → 允许的 (row, col, r_tag, d_tag) 以及其重复次数 ===
def load_filter_key_counts(filter_dir: Path):
    """
    读取像 10_16_r0_d0_3way.png 或 10_16_r0_d0_3way_copy1.png 这样的文件名，
    返回一个 dict: {(top_row, left_col, 'r0', 'd0'): count, ...}
    其中 count 是该组合在过滤目录中出现的次数（包括 copy）。
    """
    key_counts = defaultdict(int)
    if filter_dir is None:
        return key_counts

    if not filter_dir.exists():
        print(f"[WARN] 过滤目录不存在：{filter_dir}（该城市不做过滤）")
        return key_counts

    total_files = 0
    for p in filter_dir.iterdir():
        if not p.is_file() or not p.name.endswith(".png"):
            continue

        stem = p.stem  # 例如 '10_16_r0_d0_3way' 或 '10_16_r0_d0_3way_copy1'
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        try:
            top_row = int(parts[0][3:])
            left_col = int(parts[1][4:])
            r_tag = parts[2]   # 'r0'
            d_tag = parts[3]   # 'd0'
        except ValueError:
            continue

        key_counts[(top_row, left_col, r_tag, d_tag)] += 1
        total_files += 1

    print(f"[INFO] 从目录 {filter_dir} 读取到 {total_files} 个 PNG，对应 {len(key_counts)} 个 unique tile 组合")
    return key_counts

def main():
    random.seed(SEED)
    all_rows = []

    for city in CITIES:
        # city="Hong Kong"
        print(f"\n===== Processing city: {city} =====")

        count_num = 0
        # 对应这个城市的 2KM 瓦片目录
        city_tiles_path = CITY_TILE_DIRS[city]

        sat_merge_dir    = city_tiles_path / "satellite_image_merge"
        hint_merge_dir   = city_tiles_path / "hint_image_merge"
        energy_merge_dir = city_tiles_path / "energy_image_merge"
        height_merge_dir = city_tiles_path / "height_image_merge"
        dem_merge_dir = city_tiles_path / "dem_image_merge"

        # 目录校验
        missing = []
        for d in [sat_merge_dir, hint_merge_dir, energy_merge_dir, height_merge_dir,dem_merge_dir]:
            if not d.exists():
                missing.append(str(d))
        if missing:
            print(f"[WARN] 缺少目录：{missing}（跳过 {city}）")
            continue

        # 该城市自己的过滤 key 和重复次数
        filter_dir = CITY_FILTER_DIRS.get(city)
        key_counts = load_filter_key_counts(filter_dir)  # dict: key -> dup_count

        geojson_files = list_geojsons(city_tiles_path)
        if not geojson_files:
            print(f"[WARN] 未找到 GeoJSON：{city_tiles_path}")
            continue

        for gj in geojson_files:
            feats = load_features(gj)
            if not feats:
                continue

            rd_index = group_by_rd(feats)  # {(r,d): {(row,col): props}}
            for (r_tag, d_tag), rc_map in rd_index.items():
                blocks = find_blocks_strided(
                    rc_map,
                    block_w=BLOCK_SIZE, block_h=BLOCK_SIZE,
                    stride_row_idx=STRIDE_ROW_IDX, stride_col_idx=STRIDE_COL_IDX
                )
                print(
                    f"[INFO] {city} {gj.name} 组 {r_tag}_{d_tag}："
                    f"{len(blocks)} 个 {BLOCK_SIZE}×{BLOCK_SIZE}"
                )

                for (top_row, left_col) in blocks:
                    key = (top_row, left_col, r_tag, d_tag)

                    if key_counts:
                        dup_count = key_counts.get(key, 0)
                        # print(dup_count)
                        if dup_count == 0:
                            # 不在过滤目录里 → 跳过
                            continue
                    else:
                        # 没有过滤目录 → 每个 block 只用一次
                        dup_count = 1

                    metrics = agg_metrics_for_block(
                        rc_map, top_row, left_col, BLOCK_SIZE, BLOCK_SIZE
                    )
                    if metrics is None:
                        continue
                    bcr, bvd, rd = metrics

                    base_name = f"top{top_row}_left{left_col}_{r_tag}_{d_tag}"
                    target = find_first_existing(sat_merge_dir / base_name)
                    source = find_first_existing(hint_merge_dir / base_name)
                    energy = find_first_existing(energy_merge_dir / base_name)
                    height = find_first_existing(height_merge_dir / base_name)
                    dem=find_first_existing(dem_merge_dir / base_name)
                    # print(dem)

                    if dem and source:
                        count_num=count_num+1
                    else:
                        continue

                    prompt = (
                        f"Satellite imagery of {city}. "
                        f"The Building Coverage Ratio in this area is {bcr:.2f} %. "
                        f"The Building Volume Density is {bvd:.2f} cubic meters per square meter. "
                        f"The Road Density is {rd:.2f} kilometers per square kilometer."
                    )

                    # 根据 dup_count 复制记录（香港会有 >1，其他城市一般都是 1）
                    for _ in range(dup_count):
                        all_rows.append({
                            "prompt": prompt,
                            "target": target,     # 2km 卫星图
                            "source": source,     # 2km hint 图
                            "energy": energy,     # 2km energy 图
                            "height": height,     # 2km height 图
                            "dem":dem,
                            "city": city,
                            "r": r_tag,
                            "d": d_tag,
                            "top_row": top_row,
                            "left_col": left_col,
                            "BCR": round(bcr, 4),
                            "BVD": round(bvd, 4),
                            "RoadDensity": round(rd, 4),
                        })
        print(count_num)

    # ========= 全城市合并后统一打乱 & 划分 =========
    random.shuffle(all_rows)
    split = int(len(all_rows) * 0.8)
    train_rows = all_rows[:split]
    test_rows  = all_rows[split:]

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT_TEST, "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    print(
        f"\n[DONE] mosaics (ALL cities): {len(all_rows)} | "
        f"train: {len(train_rows)} | test: {len(test_rows)}"
    )
    if all_rows:
        bcrs = [r["BCR"] for r in all_rows]
        bvds = [r["BVD"] for r in all_rows]
        rds  = [r["RoadDensity"] for r in all_rows]
        print(f"BCR%  range: {min(bcrs):.2f} ~ {max(bcrs):.2f}")
        print(f"BVD   range: {min(bvds):.2f} ~ {max(bvds):.2f}")
        print(f"RoadD range: {min(rds):.2f} ~ {max(rds):.2f}")

if __name__ == "__main__":
    main()
