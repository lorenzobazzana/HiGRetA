import json
import os
import glob
import argparse
import lz4.frame
import pickle
import pandas as pd

MUSEUMS_DIR = "./multi_theme_museums"
SEMART_DESCRIPTION_FILE = "./SemArtVideoArtGenDataset/semart_refined.csv"
WIKIART_DESCRIPTION_FILE = "./janus_descriptions/wikiart_described.csv"
VIDEO_DESCRIPTION_FILE = "./SemArtVideoArtGenDataset/metadata.json"

semart_descs = pd.read_csv(SEMART_DESCRIPTION_FILE)[["path", "DESCRIPTION"]].rename(columns={"DESCRIPTION":"description"})
wikiart_descs = pd.read_csv(WIKIART_DESCRIPTION_FILE, sep=';')[["path", "description"]]
painting_descs = pd.concat([semart_descs, wikiart_descs])

with open(VIDEO_DESCRIPTION_FILE, 'r') as f:
    video_descs = json.load(f) 
video_descs = {painting["painting_file"].replace(".jpg", ".mp4"):painting for painting in video_descs}


def clean_style(name):
    if name == "Mannerism_Late_Renaissance":
        name = "Mannerism/Late Renaissance"
    elif name == "Ukiyo_e":
        name = "Ukiyo-e"
    else:
        name = name.replace("_", " ")
    
    return name

def clean_type(name):
    if name == "still_life":
        name = "still-life"
    elif name == "sketch_and_study":
        name = "sketch and study"
    else:
        name = name.replace("_painting", "")
    
    return name

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def add_painting_and_video_descriptions(room):
    
    paintings = room["paintings"]

    tmp_paintings = painting_descs[painting_descs["path"].isin(paintings)].reset_index()#["DESCRIPTION"]
    tmp_paintings["path"] = pd.Categorical(tmp_paintings["path"], categories=paintings, ordered=True)
    tmp_paintings = tmp_paintings.sort_values("path")
    tmp_paintings = [f"Painting {i+1}: " + el for i, el in enumerate(tmp_paintings["description"].to_list())]

    videos = room["videos"]

    tmp_videos = [video_descs[el]["description"] for el in videos]
    tmp_videos = [f"Video {i+1}: " + el for i, el in enumerate(tmp_videos)]

    desc = room["description"]
    desc = desc + ' '.join(tmp_paintings) + ' '.join(tmp_videos)
    room["description"] = desc

    return room

def generate_description(museum_data, multiple_themes):

    if multiple_themes:
        base_template = "This museum contains rooms with diverse themes."
    else:
        base_template = "This museum is focused on{type} artworks{style}{author}{timeframe}{TECHNIQUE}{SCHOOL}."
        template_fill = {
            "type": "",
            "style": " in the style of {}",
            "author": " by {}",
            "timeframe": " dated {}",
            "TECHNIQUE": " created using {}",
            "SCHOOL": " from the {} school",
        }

        cat = museum_data["categories"]
        for key, val in cat.items():
            if key == "type_wikiart" or key == "type_semart":
                val = clean_type(val)
                template_fill["type"] = " " + val
            else:
                if key == "style":
                    val = clean_style(val)

                template_fill[key] = template_fill[key].format(val)

        for key in list(set(template_fill.keys()) - set(cat.keys())):
            if key == "type":
                continue
            template_fill[key] = ""

        base_template = base_template.format(**template_fill)

    n_rooms_desc = f' It is divided in {str(len(museum_data["room_ids"]))} rooms.'

    museum_data["rooms"] = [add_painting_and_video_descriptions(room) for room in museum_data["rooms"]]

    room_descriptions = [room["description"] for room in museum_data["rooms"]]
    room_descriptions = list(map(lambda x: '.'.join(x.split('.')[1:]), room_descriptions))
    room_descriptions = [desc.replace("collection", f"{ordinal(i+1)} room", 1) for i,desc in enumerate(room_descriptions)]
    
    return base_template + n_rooms_desc + ''.join(room_descriptions)

def main():

    parser = argparse.ArgumentParser(description="Parse command line arguments for allowed values.")
    parser.add_argument(
        '--multiple-themes',
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    multiple_themes = args.multiple_themes if args.multiple_themes else False

    museum_list = glob.iglob(os.path.join(MUSEUMS_DIR, '**/*.json'), recursive=True)
    n_museums = len(list(museum_list))
    
    museum_list = glob.iglob(os.path.join(MUSEUMS_DIR, '**/*.json'), recursive=True)
    for i, museum in enumerate(museum_list):
        print(f"Described {i/n_museums*100:.2f}% of the museums ({i}/{n_museums})", end="\r")
        with open(museum, 'r') as f:
            #print(museum)
            museum_json = json.load(f)
        
        museum_json["description"] = generate_description(museum_json, multiple_themes)

        compressed = lz4.frame.compress(pickle.dumps(museum_json))
        with open(museum, 'wb') as f:
            #json.dump(museum_json, f, indent=4)
            f.write(compressed)
    print(f"Described {100:.2f}% of the museums ({n_museums}/{n_museums})", end="\r")
    print()

if __name__ == "__main__":
    main()