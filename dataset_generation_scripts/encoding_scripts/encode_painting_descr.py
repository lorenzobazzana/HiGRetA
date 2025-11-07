# %%
import torch
import open_clip
import pandas as pd
import tqdm
import os
import itertools
from torch.utils.data import Dataset, DataLoader

# GPU usabili 5,6,7

class PaintingDescDataset(Dataset):
    def __init__(self, painting_file, sep=","):
        super().__init__()
        self.file = pd.read_csv(painting_file, sep=sep)

    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, index):
        return self.file["path"][index], self.file["DESCRIPTION"][index]

tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip.eval()

def collate_fn(batch):
    batch_painting_names, batch_descr = zip(*batch)
    #batch_descr = tokenizer(batch_descr)

    batch_descr = [tokenizer(painting_desc.strip('.').split('.')) for painting_desc in batch_descr]

    descr_lens = [len(x) for x in batch_descr]

    flattened_museum_descr = torch.cat(batch_descr)

    return batch_painting_names, flattened_museum_descr, descr_lens



# %%
save_base_path = "../encodings/descriptions/paintings/semart"
wikiart_painting_file = "SemArtVideoArtGenDataset/semart_refined.csv"#"janus_tmp/wikiart_described.csv"
ds = PaintingDescDataset(wikiart_painting_file, sep=",")
dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8, collate_fn=collate_fn)
#names, descs, descs_len = next(iter(dl))
#print(len(descs), descs_len)
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
        #if not os.path.exists(os.path.join(save_base_path, name.split('/')[0])):
        #    os.mkdir(os.path.join(save_base_path, name.split('/')[0]))
        torch.save(tens, save_path)


