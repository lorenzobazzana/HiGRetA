import torch
import open_clip
import pandas as pd
import tqdm
import os
import itertools
import glob
import lz4
import pickle
from torch.utils.data import Dataset, DataLoader

class MuseumDescDataset(Dataset):
    def __init__(self, museums_path, multiple_themes):
        super().__init__()
        museums_list = glob.glob(os.path.join(museums_path, "**/*"), recursive=True)
        self.museums_list = [museum for museum in museums_list if os.path.isfile(museum)]

        self.multiple_themes = multiple_themes

    def __len__(self):
        return len(self.museums_list)
    
    def clean_style(self, name):
        if name == "Mannerism_Late_Renaissance":
            name = "Mannerism/Late Renaissance"
        elif name == "Ukiyo_e":
            name = "Ukiyo-e"
        else:
            name = name.replace("_", " ")
    
        return name

    def clean_type(self, name):
        if name == "still_life":
            name = "still-life"
        elif name == "sketch_and_study":
            name = "sketch and study"
        else:
            name = name.replace("_painting", "")
        
        return name

    def __getitem__(self, index):
        museum_name = self.museums_list[index]
        with lz4.frame.open(museum_name, 'rb') as f:
            museum_data = pickle.loads(f.read())

        if self.multiple_themes:
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
                    val = self.clean_type(val)
                    template_fill["type"] = " " + val
                else:
                    if key == "style":
                        val = self.clean_style(val)

                    template_fill[key] = template_fill[key].format(val)

            for key in list(set(template_fill.keys()) - set(cat.keys())):
                if key == "type":
                    continue
                template_fill[key] = ""

            base_template = base_template.format(**template_fill)

        n_rooms_desc = f' It is divided in {str(len(museum_data["room_ids"]))} rooms.'

        return museum_name, base_template + n_rooms_desc



tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip.eval()

def collate_fn(batch):
    batch_room_names, batch_descr = zip(*batch)
    #batch_descr = tokenizer(batch_descr)

    batch_descr = [tokenizer(painting_desc.strip('.').split('.')) for painting_desc in batch_descr]

    descr_lens = [len(x) for x in batch_descr]

    flattened_room_descr = torch.cat(batch_descr)

    return batch_room_names, flattened_room_descr, descr_lens

save_base_path = "../../encodings/descriptions/multi_theme_museums"
#save_base_path = "../../encodings/descriptions/museums"
museums_path = "../../multi_theme_museums"
#museums_path = "../../museums"

ds = MuseumDescDataset(museums_path, multiple_themes=True)
dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8, collate_fn=collate_fn)
#names, descs, descs_len = next(iter(dl))
#print(len(descs), descs_len)
device = torch.device("cuda:0")
clip = clip.to(device)


for names, descs, descs_len in tqdm.tqdm(dl, total=len(ds)//128):
    descs = descs.to(device)
    descs = clip.encode_text(descs)
    rebatched_descs = []
    last_index = 0
    for j in range(len(descs_len)):
        rebatched_descs.append(descs[last_index:last_index+descs_len[j], :])
        last_index = descs_len[j]
    for name, tens in zip(names, rebatched_descs):
        tens = tens.detach().clone().cpu()

        save_path = os.path.join(name.replace(museums_path, save_base_path).replace(".jpg", ".pth")) 
        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))
        torch.save(tens, save_path)