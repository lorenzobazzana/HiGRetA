import torch
import open_clip
import pandas as pd
import tqdm
import os
import itertools
import glob
import json
from torch.utils.data import Dataset, DataLoader

class RoomDescDataset(Dataset):
    def __init__(self, rooms_path):
        super().__init__()
        rooms_list = glob.glob(os.path.join(rooms_path, "**/*"), recursive=True)
        self.rooms_list = [room for room in rooms_list if os.path.isfile(room)]

    def __len__(self):
        return len(self.rooms_list)
    
    def __getitem__(self, index):
        room_name = self.rooms_list[index]
        with open(room_name, 'r') as f:
            room = json.load(f)
        return room_name, room["description"]

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


save_base_path = "../encodings/descriptions/rooms"
rooms_path = "../../rooms"

ds = RoomDescDataset(rooms_path)
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

        save_path = os.path.join(name.replace(rooms_path, save_base_path).replace(".jpg", ".pth")) 
        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))
        torch.save(tens, save_path)