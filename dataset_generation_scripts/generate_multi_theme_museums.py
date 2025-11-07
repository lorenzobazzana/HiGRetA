import os
import argparse
import json
import random
import glob
import pandas as pd
from clean_tmp import clean
import numpy as np
from scipy.special import comb

ROOMS_DIR = "./rooms"
MUSEUMS_DIR = "./multi_theme_museums"
SPLIT = "test"
DATASET_FILE = pd.read_csv(f"./encoding_scripts/refined_dataset_{SPLIT}.csv", sep=";")

def main():
    # Define allowed values
    allowed_values = {'type_wikiart', 'type_semart', 'style', 'timeframe', 'TECHNIQUE', 'SCHOOL', 'author', 'author-type_semart', 'author-type_wikiart', 'timeframe-type_semart', 'timeframe-type_wikiart', 'style-type_semart', 'style-type_wikiart'}

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Parse command line arguments for allowed values.")
    parser.add_argument(
        '--fields',
        type=str,
        required=False,
        help="Accepted values are 'type_wikiart', 'type_semart', 'style', 'timeframe', 'TECHNIQUE', 'SCHOOL', 'author'. Separate multiple values with a comma (,)."
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
        categories = list(allowed_values)
        

    categories.sort()

    print("Generating museums for the following categories:", categories)
    generate_museums(categories)

def generate_museums(categories):

    # musei multitematici
    # categories e' l'insieme di stanze che consideriamo (folder contenenti le stanze)

    min_rooms_per_museum = 4
    max_rooms_per_museum = 10

    if not os.path.exists(os.path.join(MUSEUMS_DIR, SPLIT)):
        os.mkdir(os.path.join(MUSEUMS_DIR, SPLIT))

    n_created_museums = 0

    #subset = pd.merge(DATASET_FILE, classes[i:i+1])[["path"]] # subset of all paintings tagged with the classes we are currently considering
    
    rooms = []
    for cat in categories:
        rooms.extend(glob.glob(os.path.join(ROOMS_DIR, SPLIT, cat, "**/*.json"), recursive=True))

    num_museums_to_generate = int(5e4)
    painting_list = DATASET_FILE["path"]

    for j in range(num_museums_to_generate):     
        print(f"Generating museum {j}/{num_museums_to_generate} ({100*j/num_museums_to_generate:.2f}%)", end='\r')
        n_rooms_in_museum = np.random.randint(min_rooms_per_museum, max_rooms_per_museum+1)
        tmp_subset = painting_list.copy()
        tmp_rooms = rooms.copy()

        n_sampled_rooms = 0
        sampled_rooms = []
        sampled_rooms_ids = []
        samples_tried = 0
        sampled_categories = dict()

        while n_sampled_rooms < n_rooms_in_museum and tmp_rooms: # rooms is the remaining number of rooms from which we can sample
            sample_file = random.choice(tmp_rooms)
            with open(sample_file, 'r') as f:
                sample = json.load(f)
            room_paths = pd.Series(sample["paintings"])

            if room_paths.isin(tmp_subset).all():
                sampled_rooms.append(sample)
                sampled_rooms_ids.append(sample["id"])
                for key in sample["categories"].keys(): # add room categories to museum
                    if key in sampled_categories.keys():
                        sampled_categories[key].add(sample["categories"][key])
                    else:
                        sampled_categories[key] = {sample["categories"][key]}
                tmp_subset = tmp_subset[~tmp_subset.isin(room_paths)].dropna()
                n_sampled_rooms += 1
            
            samples_tried +=1

            tmp_rooms.remove(sample_file)

        museum_json = {
            "id": f"museum_multitheme_{j}",
            "categories":{key: list(val) for key, val in sampled_categories.items()},
            "room_ids": sampled_rooms_ids,
            "rooms": sampled_rooms,
        }
        with open(os.path.join(MUSEUMS_DIR, SPLIT, f"museum_multitheme_{j}.json"), 'w') as f:
            json.dump(museum_json, f, indent=4)
        
        n_created_museums += 1


    print()
    print(f"Created {n_created_museums} museums")
    print()
    
    print("Cleaning duplicate museums")
    clean()

if __name__ == "__main__":
    main()
