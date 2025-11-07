import os
import json
import pandas as pd
import numpy as np
from scipy.special import comb
import argparse

ROOMS_DIR = "./rooms"
SPLIT = "test"
DATASET_FILE = pd.read_csv(f"./encoding_scripts/refined_dataset_{SPLIT}.csv", sep=";")

def main():
    # Define allowed values
    allowed_values = {'type_wikiart', 'type_semart', 'style', 'timeframe', 'TECHNIQUE', 'SCHOOL', 'author'}

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Parse command line arguments for allowed values.")
    parser.add_argument(
        '--fields',
        type=str,
        required=False,
        help="Accepted values are 'type_wikiart', 'type_semart', 'style', 'timeframe', 'TECHNIQUE', 'SCHOOL', 'author'. Separate multiple values with a comma (,)."
    )
    parser.add_argument(
        '--combined',
        action=argparse.BooleanOptionalAction
    )

    # Parse arguments
    args = parser.parse_args()

    # Split input by ';' and check validity
    if args.fields:
        fields = args.fields.split(',')
        invalid_fields = [field for field in fields if field not in allowed_values]

        if invalid_fields:
            print(f"Invalid field(s) detected: {', '.join(invalid_fields)}. Accepted values are: {', '.join(allowed_values)}")
        else:
            categories = fields
    else:
        categories = list(DATASET_FILE.columns)
        categories.remove("path")

    categories.sort()

    print("Generating rooms for the following categories:", categories)
    generate_rooms(categories, args.combined)

def generate_rooms(categories, combined=False):

    if combined:
        categories = [categories]

    min_paintings_per_room = 4
    max_paintings_per_room = 10

    for cat in categories:
        
        if combined:
            cat_dir = '-'.join(cat)
        else:
            cat_dir = cat
            cat = [cat]

        if not os.path.exists(os.path.join(ROOMS_DIR, cat_dir)):
            os.mkdir(os.path.join(ROOMS_DIR, cat_dir))
        
        classes = DATASET_FILE[cat].dropna().drop_duplicates(ignore_index=True)

        for i, val in classes.iterrows():

            #if combined:
            val_dir = '-'.join(val)
            #else:
            #    val_dir = val

            if not os.path.exists(os.path.join(ROOMS_DIR, cat_dir, val_dir)):
                os.mkdir(os.path.join(ROOMS_DIR, cat_dir, val_dir))
            
            num_rooms_per_value = 250
            subset = DATASET_FILE.reset_index().merge(classes[i:i+1]).set_index("index")[["path"]]

            if subset.size == 0:
                print(f"No paintings available for {cat_dir}:{val_dir}")
                continue
            num_rooms_per_value = min(num_rooms_per_value, int(comb(len(subset), max_paintings_per_room)))
            num_rooms_per_value = max(num_rooms_per_value, 1)
            for j in range(num_rooms_per_value):
                #print(f"Category: {cat} val {n+1}/{len(classes)} room {i+1}/{1000}", end='\r')
                n = np.random.randint(min_paintings_per_room, max_paintings_per_room+1)
                try:
                    room = subset.sample(n) 
                except: # iff there are few paintings to sample
                    room = subset

                additional_info = DATASET_FILE.iloc[room.index].drop(columns=["path"]+cat)
                additional_info = {col_name: additional_info[col_name].dropna().unique().tolist() for col_name in additional_info.columns}
                room_json = {
                    "id": f"room_{cat_dir}_{val_dir}_{j}.json",
                    "categories": dict(zip(cat,val)),
                    "secondary_categories": additional_info,
                    "paintings": list(room["path"]),
                }
                with open(os.path.join(ROOMS_DIR,cat_dir, val_dir, f"room_{cat_dir}_{val_dir}_{j}.json"), 'w') as f:
                    json.dump(room_json, f, indent=4)
            print(f"Created {num_rooms_per_value} rooms for {cat_dir}:{val_dir}")
        print()

if __name__ == "__main__":
    main()
