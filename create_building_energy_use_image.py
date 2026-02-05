
# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image  

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
SAMPLE_N      = 400        
VALUE_SCALE   = 'log1p'  

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


pts = pts_ori[pts_ori.get('Calendar Year', np.nan) == 2023].copy()
pts[metric_col] = pd.to_numeric(pts[metric_col], errors="coerce")
pts = pts[np.isfinite(pts[metric_col]) & (pts[metric_col] > 0)]
pts = pts[["geometry", metric_col]].copy()


sindex = pts.sindex

def query_points_in_poly_with_weight(gdf_points, sindex, poly, metric_col):
    bbox_ids = list(sindex.intersection(poly.bounds))
    if not bbox_ids: return np.empty((0, 3))
    cand = gdf_points.iloc[bbox_ids]
    cand = cand[cand.geometry.intersects(poly)]  
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


def estimate_positive_tertiles(grid, pts, sindex, metric_col, size=IMG_SIZE, sample_n=SAMPLE_N):
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
        return 1.0, 2.0
    vals = np.concatenate(pos_vals)
    t1 = float(np.quantile(vals, 1/3))
    t2 = float(np.quantile(vals, 2/3))
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
    img = Image.fromarray(label_ids.astype(np.uint8), mode='L')
    img.save(out_path)

def process_grid_cell(row_tuple):
    _, cell = row_tuple
    poly = cell.geometry
    T = from_bounds(*poly.bounds, IMG_SIZE, IMG_SIZE)

    pts_xyz = query_points_in_poly_with_weight(pts, sindex, poly, metric_col)
    blocks = blocks_sum_for_cell(poly, pts_xyz, T, n_block=N_BLOCK, img_size=IMG_SIZE)
    blocks_scaled = apply_scale(blocks)

    Z = upsample_blocks_to_image(blocks_scaled, block_pix=BLOCK_PIX)  
    mask = polygon_mask_on_transform(poly, T, IMG_SIZE, IMG_SIZE, all_touched=ALL_TOUCHED)

    labels = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8) 
    inside = (mask == 1)

    pos = (Z > 0) & inside
    if np.any(pos):
        labels[pos & (Z <= T1)] = 1
        labels[pos & (Z >  T1) & (Z <= T2)] = 2
        labels[pos & (Z >  T2)] = 3


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
