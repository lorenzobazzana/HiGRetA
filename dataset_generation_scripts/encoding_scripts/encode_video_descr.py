import torch
import open_clip
import json
import tqdm
import os
import itertools
from torch.utils.data import Dataset, DataLoader

class PaintingDescDataset(Dataset):
    def __init__(self, painting_file):
        super().__init__()
        with open(painting_file, 'r') as f:
            file = json.load(f)
        self.file = [(el["painting_file"], el["description"]) for el in file]

    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, index):
        return self.file[index]

tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip.eval()

def collate_fn(batch):
    batch_painting_names, batch_descr = zip(*batch)

    batch_descr = [tokenizer(painting_desc.strip('.').split('.')) for painting_desc in batch_descr]

    descr_lens = [len(x) for x in batch_descr]

    flattened_museum_descr = torch.cat(batch_descr)

    return batch_painting_names, flattened_museum_descr, descr_lens



save_base_path = "../../encodings/descriptions/videos"
video_descr_file = "../../SemArtVideoArtGenDataset/metadata.json"
ds = PaintingDescDataset(video_descr_file)
dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8, collate_fn=collate_fn)
device = torch.device("cuda:2")
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

        save_path = os.path.join(save_base_path, name.replace(".jpg", ".pth")) 
        torch.save(tens, save_path)


