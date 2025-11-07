import os
import argparse
import json
import random
import glob
import pandas as pd
import numpy as np
from scipy.special import comb
import clean_tmp

ROOMS_DIR = "./rooms"
MUSEUMS_DIR = "./museums"
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

    # Split input by ',' and check validity
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

    print("Generating museums for the following categories:", categories)
    generate_museums(categories, args.combined)

def generate_museums(categories, combined):

    # musei monotematici
    #categories = list(DATASET_FILE.columns)
    #categories = os.listdir(ROOMS_DIR)
    #categories.remove("path")

    if combined:
        categories = [categories]

    min_rooms_per_museum = 4
    max_rooms_per_museum = 10
    avg_rooms_per_museum = (min_rooms_per_museum + max_rooms_per_museum)//2
    avg_paintings_per_room = 7

    for cat in categories:

        if combined:
            cat_dir = '-'.join(cat)
        else:
            cat_dir = cat
            cat = [cat]

        if not os.path.exists(os.path.join(MUSEUMS_DIR, SPLIT, cat_dir)):
            os.mkdir(os.path.join(MUSEUMS_DIR, SPLIT, cat_dir))

        #classes = DATASET_FILE[cat].dropna().unique()
        classes = DATASET_FILE[cat].dropna().drop_duplicates(ignore_index=True)

        for i, val in classes.iterrows():

            val_dir = '-'.join(val)

            if not os.path.exists(os.path.join(MUSEUMS_DIR, SPLIT, cat_dir, val_dir)):
                os.mkdir(os.path.join(MUSEUMS_DIR, SPLIT, cat_dir, val_dir))

            n_created_museums = 0

            subset = pd.merge(DATASET_FILE, classes[i:i+1])[["path"]] # subset of all paintings tagged with the classes we are currently considering

            if subset.size == 0:
                print(f"No paintings available for {cat_dir}:{val_dir}")
                continue

            num_museums_per_value = 250
            #num_museums_per_value = min(num_museums_per_value, int(comb(len(subset), max_rooms_per_museum)))
            num_museums_per_value = min(num_museums_per_value, int(comb(len(subset), avg_paintings_per_room * avg_rooms_per_museum)))
            num_museums_per_value = max(num_museums_per_value, 1)
            
            for j in range(num_museums_per_value):   

                n_rooms_in_museum = np.random.randint(min_rooms_per_museum, max_rooms_per_museum+1)
                tmp_subset = subset["path"].copy()
                rooms = os.listdir(os.path.join(ROOMS_DIR, SPLIT, cat_dir, val_dir))
                
                n_sampled_rooms = 0
                sampled_rooms = []
                sampled_rooms_ids = []
                samples_tried = 0

                while n_sampled_rooms < n_rooms_in_museum and rooms: # rooms is the remaining number of rooms from which we can sample
                    sample_file = random.choice(rooms)
                    sample_file_path = os.path.join(ROOMS_DIR, SPLIT, cat_dir, val_dir, sample_file)
                    with open(sample_file_path, 'r') as f:
                        sample = json.load(f)
                    room_paths = pd.Series(sample["paintings"])

                    if room_paths.isin(tmp_subset).all():
                        sampled_rooms.append(sample)
                        sampled_rooms_ids.append(sample["id"])
                        tmp_subset = tmp_subset[~tmp_subset.isin(room_paths)].dropna()
                        n_sampled_rooms += 1
                    
                    samples_tried +=1

                    rooms.remove(sample_file)
                #print(f"Sampled {samples_tried} rooms")
                museum_json = {
                    "id": f"museum_{cat_dir}_{val_dir}_{j}",
                    "categories":dict(zip(cat,val)),
                    "room_ids": sampled_rooms_ids,
                    "rooms": sampled_rooms,
                }
                with open(os.path.join(MUSEUMS_DIR, SPLIT, cat_dir, val_dir, f"museum_{cat_dir}_{val_dir}_{j}.json"), 'w') as f:
                    json.dump(museum_json, f, indent=4)
                
                n_created_museums += 1
                #if rooms == []:
                #    print("Consumed all rooms") # if we sampled all the rooms before filling a museums, it means we cannot really make any more museums
                #    break


            print(f"Created {n_created_museums} museum for {cat}:{val}")
            print()
    
    print("Cleaning duplicate museums")
    clean_tmp.clean()

def get_room_ids(filename):
    
    with open(filename, 'r') as f:
        museum = json.load(f)
    museum = sorted(museum["room_ids"])

    return museum

def clean():
    dataset_file = pd.read_csv("./merged_painting_dataset/refined_dataset.csv", sep=";")
    categories = list(dataset_file.columns)
    categories.remove("path")

    museums = pd.DataFrame(glob.iglob(os.path.join(MUSEUMS_DIR, SPLIT, '**/*.json'), recursive=True), columns=["file_path"])
    museums["room_ids"] = museums["file_path"].map(get_room_ids)

    museums_to_keep = museums.drop_duplicates(subset=["room_ids"])
    museums_to_delete = museums.drop(museums_to_keep.index)

    n = len(museums_to_delete)
    print(f"Removing {n} museums")
    for file in museums_to_delete["file_path"]:
        #os.remove(file)
        print(file)

if __name__ == "__main__":
    main()
