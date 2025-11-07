import pandas as pd
import os
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDatasetWikiart(Dataset):
    def __init__(self, images_dir, data_file, transform=None):
        super().__init__()
        
        self.annotations = pd.read_csv(data_file)
        self.dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        img_path = os.path.join(self.dir, self.annotations["path"][index])
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        
        style = int(self.annotations["style_label"][index])
        genre = int(self.annotations["genre_label"][index])

        return img, style, genre
    



class ImageDatasetSemArt(Dataset):
    def __init__(self, images_dir, data_file, class_file_technique, class_file_type, class_file_school, class_file_timeframe, transform=None):
        super().__init__()
        
        self.annotations = pd.read_csv(data_file)
        self.dir = images_dir
        self.class_file = {
            "TECHNIQUE": pd.read_csv(class_file_technique, index_col=0),
            "TYPE": pd.read_csv(class_file_type, index_col=0),
            "SCHOOL": pd.read_csv(class_file_school, index_col=0),
            "TIMEFRAME": pd.read_csv(class_file_timeframe, index_col=0)
        }
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        img_path = os.path.join(self.dir, self.annotations["IMAGE_FILE"][index])
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        
        out = []
        for tag in ["TECHNIQUE", "TYPE", "SCHOOL", "TIMEFRAME"]:
            img_class = self.annotations[tag][index]
            img_class = self.class_file[tag].loc[img_class].iloc[0]
            out.append(int(img_class))

        return img, out
    

class ImageDatasetMixed(Dataset):
    def __init__(self, images_dir, data_file, type_wikiart_class_file, type_semart_class_file, style_class_file, timeframe_class_file, transform=None):
        super().__init__()
        
        self.annotations = pd.read_csv(data_file, sep=';')
        self.annotations["path"] = self.annotations["path"].map(lambda x: "wikiart/"+x if "/" in x else "semart/"+x)
        self.dir = images_dir
        self.class_file = {
            "type_wikiart": pd.read_csv(type_wikiart_class_file, index_col=0),
            "type_semart": pd.read_csv(type_semart_class_file, index_col=0),
            "style": pd.read_csv(style_class_file, index_col=0),
            "timeframe": pd.read_csv(timeframe_class_file, index_col=0)
        }
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        img_path = os.path.join(self.dir, self.annotations["path"][index])
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        
        out = []
        for tag in ["type_wikiart", "type_semart", "style", "timeframe"]:
            img_class = self.annotations[tag][index]
            img_class = self.class_file[tag].loc[img_class].iloc[0]
            out.append(int(img_class))

        return img, out
