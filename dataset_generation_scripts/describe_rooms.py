import os
import json
import glob


# for wikiart_classes we just replace _ with a space, and get rid of "painting"

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

def generate_description(room_data):
    base_template_primary = "This room is focused on{type} artworks{style}{author}{timeframe}{TECHNIQUE}{SCHOOL}. "
    base_template_secondary = "The collection includes{style}{type} paintings{timeframe}{author}{TECHNIQUE}{SCHOOL}."

    template_fill_1 = {
        "type": "",
        "style": " in the style of {}",
        "author": " by {}",
        "timeframe": " dated {}",
        "TECHNIQUE": " created using {}",
        "SCHOOL": " from the {} school",
    }

    template_fill_2 = {
        "style": " {} style",
        "type": " {}",
        "timeframe": " from {}",
        "author": ", featuring works by renowned artists such as {}",
        "TECHNIQUE": ", created using techniques like {}",
        "SCHOOL": " and influenced by painting schools such as {}",
    }

    # Filling the first category template
    cat = room_data["categories"]

    for key, val in cat.items():
        if key == "type_wikiart" or key == "type_semart":
            val = clean_type(val)
            template_fill_1["type"] = " " + val
        else:
            if key == "style":
                val = clean_style(val)

            template_fill_1[key] = template_fill_1[key].format(val)

    for key in list(set(template_fill_1.keys()) - set(cat.keys())):
        if key == "type":
            continue
        template_fill_1[key] = ""

    base_template_primary = base_template_primary.format(**template_fill_1)

    # Filling the second category template
    cat = room_data["secondary_categories"]
    if "type_wikiart" in cat.keys() and "type_semart" in cat.keys():
        type = set(cat["type_semart"]) | set([clean_type(el) for el in cat["type_wikiart"]])
    elif "type_wikiart" in cat.keys():
        type = [clean_type(el) for el in cat["type_wikiart"]]
    else:
        type = cat["type_semart"]

    template_fill_2["type"] = template_fill_2["type"].format(", ".join(type))


    for key, val in cat.items():
        if key == "style":
            val = [clean_style(el) for el in val]
        elif key == "type_wikiart" or key == "type_semart":
            continue#val = [clean_type(el) for el in val]

        template_fill_2[key] = template_fill_2[key].format(", ".join(val)) if val != [] else ""

    for key in list(set(template_fill_2.keys()) - set(cat.keys())):
        if key == "type":
            continue
        template_fill_2[key] = ""

    base_template_primary = base_template_primary.format(**template_fill_1)
    base_template_secondary = base_template_secondary.format(**template_fill_2)
    return base_template_primary + base_template_secondary

def main():
    ROOMS_DIR = "./rooms"
    for split in ["train", "test", "val"]:
        room_list = glob.iglob(os.path.join(ROOMS_DIR, split, '**/*.json'), recursive=True)
        n_rooms = len(list(room_list))
        
        room_list = glob.iglob(os.path.join(ROOMS_DIR, split, '**/*.json'), recursive=True)
        for i, room in enumerate(room_list):
            print(f"Described {i/n_rooms*100:.2f}% of the rooms ({i}/{n_rooms})", end="\r")
            with open(room, 'r') as f:
                #print(room)
                room_json = json.load(f)
            
            room_json["description"] = generate_description(room_json)
            with open(room, 'w') as f:
                json.dump(room_json, f, indent=4)

if __name__ == "__main__":
    main()