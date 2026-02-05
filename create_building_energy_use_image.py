#
#
#
# # -*- coding: utf-8 -*-
# import os, numpy as np, pandas as pd, geopandas as gpd
# from shapely.geometry import Point
# import matplotlib.pyplot as plt
# from rasterio.transform import from_bounds
# from rasterio.features import rasterize
# from tqdm.auto import tqdm
# from concurrent.futures import ThreadPoolExecutor
#
# # -----------------------------
# # 基本参数
# # -----------------------------
# GRID_PATHs = [
#     "../urban_data/NYC_2KM/grid_r0_d0_density_landuse.geojson",
#     "../urban_data/NYC_2KM/grid_r0_d1_density_landuse.geojson",
#     "../urban_data/NYC_2KM/grid_r1_d0_density_landuse.geojson",
#     "../urban_data/NYC_2KM/grid_r1_d1_density_landuse.geojson"
# ]
#
# POINTS_CSV = "./building_energy_use/NYC_Building_Energy_and_Water_Data_Disclosure_for_Local_Law_84__2022-Present__20250908.csv"
# LON, LAT = "Longitude", "Latitude"
# METRICS = {"energy": "Site Energy Use (kBtu)"}   # 用于汇总
# metric_col = METRICS["energy"]
#
# OUT_DIR = "./energy_image_blocks4x4_p10p98"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# IMG_SIZE      = 512          # 输出影像大小（512×512）
# N_BLOCK       = 4            # 4×4 子块
# BLOCK_PIX     = IMG_SIZE // N_BLOCK  # 128
# ALL_TOUCHED   = True         # 掩膜触碰即算
# SAMPLE_N      = 400          # 估全局分位的抽样网格数（可设 None 用全部）
# VMIN_QUANT    = 0.10         # 全局 vmin 分位
# VMAX_QUANT    = 0.98         # 全局 vmax 分位
# GAMMA         = 0.85         # 伽马校正（<1 拉亮中间灰；=1 线性）
#
# # 数值域：'linear' / 'log1p' / 'sqrt'
# VALUE_SCALE = 'log1p'
# def apply_scale(arr: np.ndarray) -> np.ndarray:
#     if VALUE_SCALE == 'log1p':
#         return np.log1p(np.clip(arr, 0, None))
#     elif VALUE_SCALE == 'sqrt':
#         return np.sqrt(np.clip(arr, 0, None))
#     else:
#         return arr
#
# print("Loading grid …")
# grids = [gpd.read_file(p) for p in GRID_PATHs]
# grid = pd.concat(grids, ignore_index=True)
# grid = gpd.GeoDataFrame(grid, crs=grids[0].crs)
# assert grid.crs is not None
# target_crs = grid.crs
#
# print("Loading points …")
# df = pd.read_csv(POINTS_CSV)
# pts_ori = gpd.GeoDataFrame(
#     df, geometry=[Point(xy) for xy in zip(df[LON], df[LAT])], crs="EPSG:4326"
# ).to_crs(target_crs)
#
# # 只取 2023 且能耗>0
# pts = pts_ori[pts_ori.get('Calendar Year', np.nan) == 2023].copy()
# if metric_col in pts.columns:
#     pts[metric_col] = pd.to_numeric(pts[metric_col], errors="coerce")
#     pts = pts[np.isfinite(pts[metric_col]) & (pts[metric_col] > 0)]
# pts = pts[["geometry", metric_col]].copy()
#
# # 空间索引
# sindex = pts.sindex
#
# def query_points_in_poly_with_weight(gdf_points, sindex, poly, metric_col):
#     bbox_ids = list(sindex.intersection(poly.bounds))
#     if not bbox_ids:
#         return np.empty((0, 3))
#     cand = gdf_points.iloc[bbox_ids]
#     cand = cand[cand.geometry.intersects(poly)]  # 包含边界
#     if cand.empty:
#         return np.empty((0, 3))
#     x = cand.geometry.x.values
#     y = cand.geometry.y.values
#     w = cand[metric_col].to_numpy()
#     m = np.isfinite(w) & (w > 0)
#     if not m.any():
#         return np.empty((0, 3))
#     return np.c_[x[m], y[m], w[m]]
#
# def polygon_mask_on_transform(poly, transform, width, height, all_touched=True):
#     return rasterize(
#         [(poly, 1)],
#         out_shape=(height, width),
#         transform=transform,
#         fill=0,
#         all_touched=all_touched,
#         dtype="uint8",
#     )
#
# def blocks4x4_sum_for_cell(poly, pts_xyz, transform, img_size=IMG_SIZE, n_block=N_BLOCK):
#     blocks = np.zeros((n_block, n_block), dtype=float)
#     if pts_xyz.shape[0] == 0:
#         return blocks
#     Tinv = ~transform
#     cols, rows = Tinv * (pts_xyz[:, 0], pts_xyz[:, 1])
#     bx = np.floor(cols / (img_size / n_block)).astype(int)
#     by = np.floor(rows / (img_size / n_block)).astype(int)
#     in_img = (bx >= 0) & (bx < n_block) & (by >= 0) & (by < n_block)
#     if not in_img.any():
#         return blocks
#     np.add.at(blocks, (by[in_img], bx[in_img]), pts_xyz[in_img, 2])
#     return blocks
#
# def upsample_blocks_to_image(blocks, block_pix=BLOCK_PIX):
#     return np.kron(blocks, np.ones((block_pix, block_pix), dtype=float))
#
# def estimate_global_range_blocks(grid, pts, sindex, metric_col,
#                                  size=IMG_SIZE, sample_n=SAMPLE_N,
#                                  qmin=VMIN_QUANT, qmax=VMAX_QUANT):
#     """
#     同一数值域（线性/对数等）里估全局 [vmin, vmax]（分位数法）
#     """
#     sample = grid if (sample_n is None or sample_n >= len(grid)) \
#                   else grid.sample(min(sample_n, len(grid)), random_state=0)
#     all_vals = []
#     for _, cell in sample.iterrows():
#         poly = cell.geometry
#         T = from_bounds(*poly.bounds, size, size)
#         pts_xyz = query_points_in_poly_with_weight(pts, sindex, poly, metric_col)
#         blocks = blocks4x4_sum_for_cell(poly, pts_xyz, T, img_size=size, n_block=N_BLOCK)
#         blocks = apply_scale(blocks)                   # ★ 与出图一致
#         vals = blocks[np.isfinite(blocks) & (blocks > 0)]
#         if vals.size:
#             all_vals.append(vals.ravel())
#     if not all_vals:
#         return 0.0, 1e-9
#     all_vals = np.concatenate(all_vals)
#     vmin = float(np.quantile(all_vals, qmin))
#     vmax = float(np.quantile(all_vals, qmax))
#     if not np.isfinite(vmin): vmin = 0.0
#     if (not np.isfinite(vmax)) or (vmax <= vmin): vmax = vmin + 1e-9
#     return vmin, vmax
#
# def save_png_gray_with_range(Z, out_path, vmin, vmax, gamma=1.0):
#     Z0 = Z.copy().astype(float)
#     Z0[~np.isfinite(Z0)] = 0.0
#     if vmax <= vmin: vmax = vmin + 1e-9
#     Z0 = (Z0 - vmin) / (vmax - vmin)
#     Z0 = 1-np.clip(Z0, 0.0, 1.0)
#     if gamma != 1.0:
#         Z0 = np.power(Z0, gamma)
#     img = (Z0 * 255).astype(np.uint8)
#     plt.imsave(out_path, img, cmap="gray_r", vmin=0, vmax=255)
#
# print("Estimating GLOBAL [vmin, vmax] via percentiles …")
# GLOBAL_VMIN, GLOBAL_VMAX = estimate_global_range_blocks(
#     grid, pts, sindex, metric_col,
#     size=IMG_SIZE, sample_n=SAMPLE_N,
#     qmin=VMIN_QUANT, qmax=VMAX_QUANT
# )
# print("GLOBAL_VMIN:", GLOBAL_VMIN)
# print("GLOBAL_VMAX:", GLOBAL_VMAX)
# print("VALUE_SCALE:", VALUE_SCALE, "| GAMMA:", GAMMA)
#
# def process_grid_cell(row_tuple):
#     _, cell = row_tuple
#     poly = cell.geometry
#     T = from_bounds(*poly.bounds, IMG_SIZE, IMG_SIZE)
#
#     # 4×4 总和
#     pts_xyz = query_points_in_poly_with_weight(pts, sindex, poly, metric_col)
#     blocks = blocks4x4_sum_for_cell(poly, pts_xyz, T, img_size=IMG_SIZE, n_block=N_BLOCK)
#
#     # 数值域变换（与估计范围一致）
#     blocks = apply_scale(blocks)
#     print(blocks)
#
#
#     # 上采样到 512×512 并掩膜
#     Z = upsample_blocks_to_image(blocks, block_pix=BLOCK_PIX)
#     mask = polygon_mask_on_transform(poly, T, IMG_SIZE, IMG_SIZE, all_touched=ALL_TOUCHED)
#     Z = np.where(mask == 1, Z, 0.0)
#
#     # 保存
#     fname = f"{cell['row']}_{cell['col']}_{cell['r']}_{cell['d']}.png"
#     save_png_gray_with_range(Z, os.path.join(OUT_DIR, fname),
#                              vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX, gamma=GAMMA)
#     return 1
#
# print("Rendering 4x4 block images …")
# with ThreadPoolExecutor(max_workers=32) as ex:
#     for _ in tqdm(ex.map(process_grid_cell, grid.iterrows()),
#                   total=len(grid),
#                   desc="Rendering (p10–p98 stretch)"):
#         pass
#
# print("Done.")
#

# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image  # 用于保存整型标签PNG（不被colormap缩放）

# -----------------------------
# 基本参数
# -----------------------------
GRID_PATHs = [
    "../urban_data/NYC_2KM/grid_r0_d0_density_landuse.geojson",
    "../urban_data/NYC_2KM/grid_r0_d1_density_landuse.geojson",
    "../urban_data/NYC_2KM/grid_r1_d0_density_landuse.geojson",
    "../urban_data/NYC_2KM/grid_r1_d1_density_landuse.geojson"
]

POINTS_CSV = "./building_energy_use/NYC_Building_Energy_and_Water_Data_Disclosure_for_Local_Law_84__2022-Present__20250908.csv"
LON, LAT = "Longitude", "Latitude"
METRICS = {"energy": "Site Energy Use (kBtu)"}
metric_col = METRICS["energy"]

OUT_DIR = "./energy_labels_blocks4x4_classes"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE      = 512
N_BLOCK       = 4
BLOCK_PIX     = IMG_SIZE // N_BLOCK
ALL_TOUCHED   = True
SAMPLE_N      = 400        # 用于估计全局阈值的抽样cell数（设 None 用全部）
VALUE_SCALE   = 'log1p'    # 'linear' / 'log1p' / 'sqrt'

def apply_scale(arr: np.ndarray) -> np.ndarray:
    if VALUE_SCALE == 'log1p':
        return np.log1p(np.clip(arr, 0, None))
    elif VALUE_SCALE == 'sqrt':
        return np.sqrt(np.clip(arr, 0, None))
    else:
        return arr

print("Loading grid …")
grids = [gpd.read_file(p) for p in GRID_PATHs]
grid = pd.concat(grids, ignore_index=True)
grid = gpd.GeoDataFrame(grid, crs=grids[0].crs)
target_crs = grid.crs

print("Loading points …")
df = pd.read_csv(POINTS_CSV, low_memory=False)
pts_ori = gpd.GeoDataFrame(
    df, geometry=[Point(xy) for xy in zip(df[LON], df[LAT])], crs="EPSG:4326"
).to_crs(target_crs)

# 只取 2023 且能耗>0
pts = pts_ori[pts_ori.get('Calendar Year', np.nan) == 2023].copy()
pts[metric_col] = pd.to_numeric(pts[metric_col], errors="coerce")
pts = pts[np.isfinite(pts[metric_col]) & (pts[metric_col] > 0)]
pts = pts[["geometry", metric_col]].copy()

# 空间索引
sindex = pts.sindex

def query_points_in_poly_with_weight(gdf_points, sindex, poly, metric_col):
    bbox_ids = list(sindex.intersection(poly.bounds))
    if not bbox_ids: return np.empty((0, 3))
    cand = gdf_points.iloc[bbox_ids]
    cand = cand[cand.geometry.intersects(poly)]  # 包含边界
    if cand.empty: return np.empty((0, 3))
    x = cand.geometry.x.values
    y = cand.geometry.y.values
    w = cand[metric_col].to_numpy()
    m = np.isfinite(w) & (w > 0)
    if not m.any(): return np.empty((0, 3))
    return np.c_[x[m], y[m], w[m]]

def polygon_mask_on_transform(poly, transform, width, height, all_touched=True):
    return rasterize(
        [(poly, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )

def blocks_sum_for_cell(poly, pts_xyz, transform, n_block=N_BLOCK, img_size=IMG_SIZE):
    """n_block×n_block 权重总和"""
    blocks = np.zeros((n_block, n_block), dtype=float)
    if pts_xyz.shape[0] == 0: return blocks
    Tinv = ~transform
    cols, rows = Tinv * (pts_xyz[:, 0], pts_xyz[:, 1])
    bx = np.floor(cols / (img_size / n_block)).astype(int)
    by = np.floor(rows / (img_size / n_block)).astype(int)
    in_img = (bx >= 0) & (bx < n_block) & (by >= 0) & (by < n_block)
    if not in_img.any(): return blocks
    np.add.at(blocks, (by[in_img], bx[in_img]), pts_xyz[in_img, 2])
    return blocks

def upsample_blocks_to_image(blocks, block_pix=BLOCK_PIX):
    """上采样到 512×512"""
    return np.kron(blocks, np.ones((block_pix, block_pix), dtype=float))

# ---------- 全局阈值：把正值分成尽量均衡的 3 档 ----------
def estimate_positive_tertiles(grid, pts, sindex, metric_col, size=IMG_SIZE, sample_n=SAMPLE_N):
    """
    收集所有cell的正值（按VALUE_SCALE变换后），计算 1/3 和 2/3 分位数，返回 (t1, t2)。
    """
    sample = grid if (sample_n is None or sample_n >= len(grid)) else \
             grid.sample(min(sample_n, len(grid)), random_state=0)
    pos_vals = []
    for _, cell in sample.iterrows():
        poly = cell.geometry
        T = from_bounds(*poly.bounds, size, size)
        pts_xyz = query_points_in_poly_with_weight(pts, sindex, poly, metric_col)
        blocks = blocks_sum_for_cell(poly, pts_xyz, T, n_block=N_BLOCK, img_size=size)
        b = apply_scale(blocks)
        b = b[np.isfinite(b) & (b > 0)]
        if b.size:
            pos_vals.append(b.ravel())
    if not pos_vals:
        # 没有正值，避免崩：给出虚阈值
        return 1.0, 2.0
    vals = np.concatenate(pos_vals)
    t1 = float(np.quantile(vals, 1/3))
    t2 = float(np.quantile(vals, 2/3))
    # 防同值：若 t1==t2，轻微扰动
    if not np.isfinite(t1) or not np.isfinite(t2) or t2 <= t1:
        eps = 1e-6
        t1 = float(np.quantile(vals, 0.33 - 0.01))
        t2 = float(np.quantile(vals, 0.67 + 0.01))
        if t2 <= t1: t2 = t1 + eps
    return t1, t2

print("Estimating global tertiles on positive values …")
T1, T2 = estimate_positive_tertiles(grid, pts, sindex, metric_col)
print(f"GLOBAL TERTILES (scaled domain): T1={T1:.6f}, T2={T2:.6f} | VALUE_SCALE={VALUE_SCALE}")

def save_label_ids_png(label_ids: np.ndarray, out_path: str):
    """
    直接保存 uint8 的类别ID图（0,1,2,3），不走 colormap，避免被缩放。
    """
    img = Image.fromarray(label_ids.astype(np.uint8), mode='L')
    img.save(out_path)

def process_grid_cell(row_tuple):
    _, cell = row_tuple
    poly = cell.geometry
    T = from_bounds(*poly.bounds, IMG_SIZE, IMG_SIZE)

    # 1) 4×4 总和 → 数值域变换
    pts_xyz = query_points_in_poly_with_weight(pts, sindex, poly, metric_col)
    blocks = blocks_sum_for_cell(poly, pts_xyz, T, n_block=N_BLOCK, img_size=IMG_SIZE)
    blocks_scaled = apply_scale(blocks)

    # 2) 上采样到 512×512，并生成 polygon 掩膜
    Z = upsample_blocks_to_image(blocks_scaled, block_pix=BLOCK_PIX)   # 连续值（含0）
    mask = polygon_mask_on_transform(poly, T, IMG_SIZE, IMG_SIZE, all_touched=ALL_TOUCHED)

    # 3) 分级：0 类=数值==0 或 多边形外；正值按 tertiles 切成 1/2/3 类
    labels = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)  # 默认 0 类
    inside = (mask == 1)

    pos = (Z > 0) & inside
    if np.any(pos):
        labels[pos & (Z <= T1)] = 1
        labels[pos & (Z >  T1) & (Z <= T2)] = 2
        labels[pos & (Z >  T2)] = 3
    # 多边形外保持 0

    # 4) 保存
    fname = f"{cell['row']}_{cell['col']}_{cell['r']}_{cell['d']}.png"
    save_label_ids_png(labels, os.path.join(OUT_DIR, fname))
    return 1

print("Rendering class-label images (0/1/2/3) …")
with ThreadPoolExecutor(max_workers=32) as ex:
    for _ in tqdm(ex.map(process_grid_cell, grid.iterrows()),
                  total=len(grid),
                  desc=f"Labels ({N_BLOCK}x{N_BLOCK}, tertiles)"):
        pass

print("Done. Saved to:", OUT_DIR)
