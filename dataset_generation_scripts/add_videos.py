import json
import glob
import os
import open_clip
import torch
import random
import time
import pandas as pd
from functools import reduce
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 107327830

MUSEUM_PATH = ["../museums/", "../multi_theme_museums"]
VIDEO_PATH = "./SemArtVideoArtGenDataset"
SEMART_PATH = "./SemArt/Images"
WIKIART_PATH = "./WikiArt/wikiart"

semart_raw = pd.read_csv(os.path.join(VIDEO_PATH, "semart_refined.csv"), encoding="latin-1", sep=",")
dataset_file = pd.read_csv("./dataset_splits/refined_dataset_test.csv", sep=";", dtype="string")

videos_list = glob.glob(os.path.join(VIDEO_PATH, "videos/**/*"), recursive=True)
videos_list = list(map(lambda x: x.split("/")[-1].replace(".mp4",".jpg"), videos_list))
semart_raw = semart_raw[semart_raw["path"].isin(videos_list)]
semart_raw = semart_raw.merge(dataset_file["path"], on="path")

def format_elapsed_time(seconds):
    days, seconds = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, seconds = divmod(seconds, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(seconds, 60)  # 60 seconds in a minute
    
    parts = []
    if days > 0:
        parts.append(f"{days:.0f}d")
    if hours > 0:
        parts.append(f"{hours:.0f}h")
    if minutes > 0:
        parts.append(f"{minutes:.0f}m")
    if seconds > 0 or not parts:  # Always include seconds, even if 0, if no other parts exist
        parts.append(f"{seconds:.0f}s")
    
    return " ".join(parts)

# paintings: paintings in a room
# only called for rooms that don't have semart paintings in them
def check_similarity(model, device, preprocess, tokenizer, paintings, semart_paintings, n_paintings_to_select=1):

    if n_paintings_to_select > len(semart_paintings):
        return semart_paintings.index


    semart_image_names = semart_paintings["path"].tolist()
    painting_desc = semart_paintings["DESCRIPTION"].tolist()

    images = torch.stack([preprocess(Image.open(os.path.join(WIKIART_PATH, image))) for image in paintings])
    semart_images = torch.stack([preprocess(Image.open(os.path.join(SEMART_PATH, image))) for image in semart_image_names])
    semart_images = semart_images.to(device)

    text = tokenizer(painting_desc)
    text = text.to(device)

    images = images.to(device)
    
        
    tmp_images = [torch.cat([semart_painting_i.unsqueeze(0), images]) for semart_painting_i in semart_images]
    with torch.no_grad(), torch.amp.autocast("cuda:1"):
        image_features = torch.stack([model.encode_image(image).T for image in tmp_images])
        text_features = model.encode_text(text).unsqueeze(1)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = 100.0 * torch.bmm(text_features, image_features).flatten(1)
        
        text_probs = text_probs / text_probs[:,0:1]
        mean = text_probs.mean(dim=-1)
        _, selected_painting = torch.topk(mean, n_paintings_to_select)
    #print(f"Most fitting painting for room: {selected_painting} - {semart_image_names[selected_painting]}")


    return semart_paintings.iloc[selected_painting.tolist()].index

def sample_semart_paintings(room_data, used_paintings, n_paintings_to_sample=7):

    room_cats = room_data["categories"]
    #museum_paintings = [room["paintings"] for room in museum_data["rooms"]] # divided by room

    cond = [semart_raw[col].isin(values if type(values) == list else [values]) for col, values in room_cats.items()]
    cond = reduce(lambda x,y: x | y, cond)
    relevant_paintings = semart_raw[cond]

    # select only the paintings that are not already present in the museum
    relevant_paintings = relevant_paintings.loc[relevant_paintings.index.difference(used_paintings.index)]

    if relevant_paintings.empty:
        relevant_paintings = semart_raw.sample(n_paintings_to_sample)
        paintings_not_strictly_relevant = True
    else:    
        relevant_paintings = relevant_paintings.sample(min(n_paintings_to_sample, len(relevant_paintings)))
        paintings_not_strictly_relevant = False

    return relevant_paintings, paintings_not_strictly_relevant

def sample_videos(paintings, used_paintings, n_videos_to_sample=1):

    semart_paintings_in_room = semart_raw[semart_raw["path"].isin(filter(lambda x: '/' not in x, paintings))]
    n_videos_to_sample = min(n_videos_to_sample, len(semart_paintings_in_room))

    paintings_to_sample = semart_paintings_in_room.loc[semart_paintings_in_room.index.difference(used_paintings.index)]
    
    if paintings_to_sample.empty:
        relevant_paintings = semart_raw.sample()
    else:    
        relevant_paintings = paintings_to_sample.sample(min(n_videos_to_sample, len(paintings_to_sample)))

    return relevant_paintings


def main(path):

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    museums = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
    n_museums = len(museums)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    start_time = time.time()

    for i, museum in enumerate(museums):        
        with open(museum, 'r') as f:
            museum_data = json.load(f)


        paintings = pd.DataFrame({"room_paintings":[room["paintings"] for room in museum_data["rooms"]]}) # divided by room
        rooms_without_semart_paintings = paintings[paintings["room_paintings"].apply(lambda x: all('/' in s for s in x))]
        rooms_with_semart_paintings = paintings.drop(rooms_without_semart_paintings.index)


        museum_used_paintings = pd.DataFrame()
        for room_idx, room_paintings in rooms_with_semart_paintings.itertuples():

            room_data = museum_data["rooms"][room_idx]
            n_videos_in_room = random.randint(2,3)

            selected_paintings = sample_videos(room_data["paintings"], museum_used_paintings, n_videos_in_room)
            museum_data["rooms"][room_idx]["videos"] = selected_paintings["path"].map(lambda x: x.split('.')[0] + ".mp4").to_list()
            
            museum_used_paintings = pd.concat([museum_used_paintings, selected_paintings])
            museum_used_paintings = museum_used_paintings.sort_index()

        for room_idx, room_paintings in rooms_without_semart_paintings.itertuples():

            room_data = museum_data["rooms"][room_idx]
            semart_paintings, paintings_not_strictly_relevant = sample_semart_paintings(room_data, museum_used_paintings)
            n_videos_in_room = 1 if paintings_not_strictly_relevant else random.randint(2,3)

            selected_paintings = check_similarity(model, device, preprocess, tokenizer, room_paintings, semart_paintings, n_videos_in_room)
            selected_paintings = semart_paintings.loc[selected_paintings]

            museum_data["rooms"][room_idx]["videos"] = selected_paintings["path"].map(lambda x: x.split('.')[0] + ".mp4").to_list()
            
            museum_used_paintings = pd.concat([museum_used_paintings, selected_paintings])
            museum_used_paintings = museum_used_paintings.sort_index()

        with open(museum, 'w') as f:
            json.dump(museum_data, f, indent=2)

        partial_time = time.time() - start_time
        estimated_time = (partial_time / i) * (n_museums - i) if i > 5 else -1
        print(f"Added videos to {i+1} museums; completing in {format_elapsed_time(estimated_time) if estimated_time > -1 else '(estimating...)'}            ", end='\r', flush=True)
    
    print(f"Added videos to {n_museums} museums {''.join([' ' for _ in range(25)])}", end="\r")
    print()
    print("Done")
    end_time = time.time()
    
    elapsed_time = end_time - start_time

    print("Time elapsed: " + format_elapsed_time(elapsed_time))


if __name__ == "__main__":
    for museum_dir in MUSEUM_PATH:
        for split in ["train", "test", "val"]:
            main(path=os.path.join(museum_dir, split))


