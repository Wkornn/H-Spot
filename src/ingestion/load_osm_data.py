import osmnx as ox
import yaml
import os


def fetch_and_save(place, tags, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Fetching {os.path.basename(output_path)}...")
    gdf = ox.features_from_place(place, tags=tags)

    whitelist = ['amenity', 'building', 'highway', 'landuse', 'shop', 'name', 'oneway', 'lanes']
    
    existing_cols = [c for c in whitelist if c in gdf.columns]
    gdf = gdf[existing_cols + ['geometry']]

    for col in gdf.columns:
        if col != 'geometry':
            gdf[col] = gdf[col].astype(str)

    gdf.columns = [c.replace(":", "_").replace(" ", "_") for c in gdf.columns]

    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved → {output_path} ({len(gdf)} features)")


def load_config(config_path="configs/data_sources.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    place = config["osm"]["place"]

    for layer in config["osm"]["layers"]:
        print(f"\n=== Layer: {layer['name']} ===")
        fetch_and_save(place, layer["tags"], layer["output_path"])


if __name__ == "__main__":
    main()
