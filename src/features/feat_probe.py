"""
Feature Engineering — Part 4: Probe / Speed Features
src/features/feat_probe.py

Snaps GPS probe readings to road segments, then computes per-segment:
  - probe_count        : total valid readings
  - speed_mean         : mean speed (km/h)
  - speed_std          : speed standard deviation
  - speed_p10          : 10th percentile speed
  - speed_p90          : 90th percentile speed
  - pct_below_20kmh    : % readings below 20 km/h (congestion)

Strategy:
  - Process one monthly Parquet at a time.
  - Vectorized nearest-neighbor join (sjoin_nearest) with max radius.
  - Welford's online algorithm for running mean/std.
  - Reservoir-style sampling for percentiles.

Output: data/processed/features/feat_probe.parquet
"""

import os
import glob
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from tqdm import tqdm

CONFIG_PATH = "configs/data_sources.yaml"
SNAP_RADIUS_M = 30      # metres
BATCH_SIZE    = 500_000  # rows per batch


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    seg  = cfg["road_segments"]
    feat = cfg["features"]
    return {
        "segments":    seg["output"],
        "probe_dir":   "data/processed/probe_bangkok",
        "output_dir":  feat["output_dir"],
        "crs":         seg["projected_crs"],
        "speed_low":   feat["probe_speed_low_kmh"],
        "snap_radius": SNAP_RADIUS_M,
    }


def process_file(path, segments, cfg, accum):
    """
    Read one probe Parquet, snap to segments, update accumulators.
    accum: dict sid -> {n, mean, M2, speeds_sample, below}
    """
    df = pd.read_parquet(path, columns=["lat", "lon", "speed", "gps_valid"])
    # Basic filtering: valid GPS and realistic speeds
    df = df[(df["gps_valid"] == 1) & (df["speed"] > 0) & (df["speed"] < 200)].copy()
    if df.empty:
        return

    # Process in large batches to balance speed and memory
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start : start + BATCH_SIZE].copy()
        
        # 1. Convert to GDF
        gdf = gpd.GeoDataFrame(
            chunk, 
            geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"]), 
            crs="EPSG:4326"
        ).to_crs(cfg["crs"])

        # 2. Vectorized Snap (Nearest within radius)
        snapped = gpd.sjoin_nearest(
            gdf,
            segments,
            max_distance=cfg["snap_radius"],
            how="inner"
        )
        # Bug fix: drop duplicates in case a point matched multiple equidistant segments
        snapped = snapped[~snapped.index.duplicated(keep="first")]

        # 3. Update Statistics
        for sid, grp in snapped.groupby("segment_id"):
            speeds = grp["speed"].values.astype(float)
            if sid not in accum:
                accum[sid] = {"n": 0, "mean": 0.0, "M2": 0.0, "speeds": [], "below": 0}
            
            a = accum[sid]
            for s in speeds:
                a["n"] += 1
                delta = s - a["mean"]
                a["mean"] += delta / a["n"]
                a["M2"] += delta * (s - a["mean"])
            a["below"] += int(np.sum(speeds < cfg["speed_low"]))
            
            # Update sample for percentiles
            if len(a["speeds"]) < 1000:
                a["speeds"].extend(speeds[:1000 - len(a["speeds"])].tolist())


def main():
    cfg = load_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    output_path = os.path.join(cfg["output_dir"], "feat_probe.parquet")
    state_file = os.path.join(cfg["output_dir"], "feat_probe_state.pkl")
    accum = {}
    processed_files = set()

    print(f"--- Feature Engineering: Probe Data ---")

    # Overwrite protection: If final output exists and we aren't resuming, ask first.
    if os.path.exists(output_path) and not os.path.exists(state_file):
        print(f"\n[!] WARNING: Final output already exists at: {output_path}")
        confirm = input("    Do you want to overwrite it and start from scratch? (y/n): ")
        if confirm.lower() != 'y':
            print("    Aborting to protect existing data.")
            return
    
    # Check for existing checkpoint
    if os.path.exists(state_file):
        print(f"Loading checkpoint from {state_file}...")
        with open(state_file, "rb") as f:
            state = pickle.load(f)
            accum = state["accum"]
            processed_files = state["processed_files"]
        print(f"  Resuming: {len(processed_files)} months already completed.")

    print("Loading road segments...")
    segments = gpd.read_file(cfg["segments"])[["segment_id", "geometry"]]
    
    probe_files = sorted(glob.glob(os.path.join(cfg["probe_dir"], "*.parquet")))
    if not probe_files:
        print(f"Error: No probe files found in {cfg['probe_dir']}")
        return

    # Filter out already processed files
    to_process = [f for f in probe_files if os.path.basename(f) not in processed_files]
    
    if not to_process:
        print("All files already processed according to checkpoint.")
    else:
        print(f"Processing {len(to_process)} remaining monthly files...")
        for path in tqdm(to_process, desc="Months"):
            process_file(path, segments, cfg, accum)
            
            # Save checkpoint after each month
            processed_files.add(os.path.basename(path))
            with open(state_file, "wb") as f:
                pickle.dump({"accum": accum, "processed_files": processed_files}, f)

    # --- Aggregate and Format ---
    print("\nFinalizing segment features...")
    results = []
    for sid, a in accum.items():
        n = a["n"]
        if n == 0: continue
        
        variance = a["M2"] / n if n > 1 else 0.0
        speeds = np.array(a["speeds"])
        
        results.append({
            "segment_id": sid,
            "probe_count": n,
            "speed_mean": round(a["mean"], 2),
            "speed_std": round(np.sqrt(variance), 2),
            "speed_p10": round(float(np.percentile(speeds, 10)), 2),
            "speed_p90": round(float(np.percentile(speeds, 90)), 2),
            "pct_below_20kmh": round((a["below"] / n) * 100, 2)
        })

    feat_df = pd.DataFrame(results)
    full_df = pd.DataFrame({"segment_id": segments["segment_id"].unique()})
    final_df = full_df.merge(feat_df, on="segment_id", how="left")
    final_df["probe_count"] = final_df["probe_count"].fillna(0).astype(int)

    output_path = os.path.join(cfg["output_dir"], "feat_probe.parquet")
    final_df.to_parquet(output_path, index=False)
    
    # Cleanup checkpoint on success
    if os.path.exists(state_file):
        os.remove(state_file)
        print("Final output saved. Checkpoint file removed.")
    
    print(f"Success → {output_path}")


if __name__ == "__main__":
    main()
