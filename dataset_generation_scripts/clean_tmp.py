import polars as pl
import json
import glob
import os

MUSEUMS_DIR_TO_CLEAN = "../multi_theme_museums"

def get_room_ids(filename):
    
    with open(filename, 'r') as f:
        museum = json.load(f)
    museum = sorted(museum["room_ids"])

    return museum

def clean():
    dataset_file = pl.read_csv("./merged_painting_dataset/refined_dataset.csv", separator=";")
    categories = dataset_file.columns
    categories.remove("path")

    # Create DataFrame with schema
    museums = pl.DataFrame(
        {"file_path": glob.glob(os.path.join(MUSEUMS_DIR_TO_CLEAN, '**/*.json'), recursive=True)}
    )

    # Apply `get_room_ids` function to each file path
    museums = museums.with_columns(
        room_ids=museums["file_path"].map_elements(get_room_ids)
    )

    # Remove duplicates
    museums_to_keep = museums.unique(subset=["room_ids"])
    museums_to_delete = museums.filter(~museums["file_path"].is_in(museums_to_keep["file_path"]))

    # Print and remove files
    n = len(museums_to_delete)
    print(f"Removing {n} museums")
    for file in museums_to_delete["file_path"]:
        # os.remove(file)
        print(file)


if __name__ == "__main__":
    clean()