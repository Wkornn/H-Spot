import subprocess
import sys
import argparse
import time
import os
import yaml

DATA_CONFIG_PATH = "configs/data_sources.yaml"
PIPE_CONFIG_PATH = "configs/pipeline.yaml"

def load_yaml(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

def run_script(script_path, output_files=None, force=False):
    """
    Runs a script only if its output_files don't exist, or if force is True.
    """
    if output_files and not force:
        if isinstance(output_files, str):
            output_files = [output_files]
        
        if all(os.path.exists(f) for f in output_files):
            print(f">>> Skipping: {script_path} (Output already exists)")
            return True

    print(f"\n>>> Running: {script_path}")
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        elapsed = time.time() - start_time
        print(f">>> Success: {script_path} (Took {elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR: {script_path} failed with exit code {e.returncode}")
        return False

def main():
    # 1. Load Configurations
    p_cfg = load_yaml(PIPE_CONFIG_PATH).get("pipeline", {})
    d_cfg = load_yaml(DATA_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="H-Spot Bangkok: Config-Driven Data Pipeline")
    
    # Execution Groups (Defaults come from pipeline.yaml)
    parser.add_argument("--ingest",     action="store_true", default=p_cfg.get("ingest", False),     help="Run Phase 1: Ingestion")
    parser.add_argument("--preprocess", action="store_true", default=p_cfg.get("preprocess", False), help="Run Phase 2: Preprocessing")
    parser.add_argument("--features",   action="store_true", default=p_cfg.get("features", False),   help="Run Phase 3: Feature Engineering")
    parser.add_argument("--matrix",     action="store_true", default=p_cfg.get("matrix", False),     help="Run Phase 4: Final Matrix")
    
    # Overrides & Flags
    parser.add_argument("--force",      action="store_true", default=p_cfg.get("force", False),      help="Force re-run (overwrite existing)")
    parser.add_argument("--all",        action="store_true", help="Run ALL phases (overrides config)")
    parser.add_argument("--probe",      action="store_true", default=p_cfg.get("run_probe", False),  help="Include probe data processing (slow)")

    args = parser.parse_args()

    # 1. Phase: Ingestion
    if args.ingest or args.all:
        print("\n--- Phase 1: Ingestion ---")
        run_script("src/ingestion/load_osm_data.py", 
                   output_files=d_cfg.get("boundary", {}).get("bangkok"), 
                   force=args.force)
        run_script("src/ingestion/load_accident_data.py", force=args.force)

    # 2. Phase: Preprocessing
    if args.preprocess or args.all:
        print("\n--- Phase 2: Preprocessing ---")
        acc_out = d_cfg.get("accidents", {}).get("clean_parquet")
        run_script("src/ingestion/preprocess_accidents.py", output_files=acc_out, force=args.force)
        
        seg_cfg = d_cfg.get("road_segments", {})
        run_script("src/geospatial/spatial/segment_roads.py", 
                   output_files=[seg_cfg.get("output"), seg_cfg.get("intersections_output")], 
                   force=args.force)

    # 3. Phase: Feature Engineering
    if args.features or args.all:
        print("\n--- Phase 3: Feature Engineering ---")
        f_dir = d_cfg.get("features", {}).get("output_dir")
        
        run_script("src/features/feat_road.py",      output_files=os.path.join(f_dir, "feat_road.parquet"),      force=args.force)
        run_script("src/features/feat_accidents.py", output_files=os.path.join(f_dir, "feat_accidents.parquet"), force=args.force)
        run_script("src/features/feat_spatial.py",   output_files=os.path.join(f_dir, "feat_spatial.parquet"),   force=args.force)
        # Only run probe if explicitly enabled in config/flags
        probe_out = os.path.join(f_dir, "feat_probe.parquet")
        if args.probe or args.all:
            run_script("src/features/feat_probe.py", output_files=probe_out, force=args.force)

    # 4. Phase: Final Matrix
    if args.matrix or args.all or args.features:
        print("\n--- Phase 4: Final Matrix ---")
        matrix_out = os.path.join(d_cfg.get("features", {}).get("output_dir"), "feature_matrix.parquet")
        run_script("src/features/build_feature_matrix.py", output_files=matrix_out, force=args.force)

    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()
