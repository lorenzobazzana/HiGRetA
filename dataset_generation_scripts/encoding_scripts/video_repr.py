import os
import sys
import argparse
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from transformers import CLIPVisionModelWithProjection

from visionTransformer import CustomViT


VIDEOS_DIR = "../../SemArtVideoArtGenDataset/videos"
ENCODINGS_DIR = "../../encodings/videos"
# Code to convert one video to few images.  
def video2image(video_path, frame_rate=1.0, size=224):
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),            
            CenterCrop(size),
            #lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(n_px)
    
    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = np.zeros([3, size, size], dtype=np.float32) 
        print("ERROR: problem reading video file: ", video_path)
    else:
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(start_sec*fps, end_sec*fps, interval))
        ret = True     
        images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)
            
        for i, idx in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES , idx)
            ret, frame = cap.read()    
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             
            last_frame = i
            images[i,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))
            
        images = images[:last_frame+1]
    cap.release()
    video_frames = torch.tensor(images)
    return video_frames

def main():    

    parser = argparse.ArgumentParser(description="Parse command line arguments for allowed values.")
    parser.add_argument(
        '--repr',
        type=str,
        required=False,
        help="Accepted values are \"clip\" (default) and \"art\"."
    )
    allowed_values = ["clip", "art"]
    args = parser.parse_args()
    if args.repr:
        _repr = args.repr.split(',')
        invalid_fields = [field for field in _repr if field not in allowed_values]

        if invalid_fields:
            print(f"Invalid field(s) detected: {', '.join(invalid_fields)}. Accepted values are: {', '.join(allowed_values)}")
            Exception()
        elif "clip" in _repr and "art" in _repr:
            print("Cannot have both representations in command arguments. Defaulting to \"clip\"")
            enc = "clip"
        else:
            enc = _repr[0]
    else:
        enc = "clip"


    if enc == "clip":
        model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
    else:
        fine_tuned_weights = torch.load("../encoding_scripts/trained models/trained_model_ViT-B-16_artistic_repr.pth", weights_only=False)
        model = CustomViT(image_size=224, 
                                patch_size=16, 
                                num_layers=12, 
                                num_heads=12, 
                                hidden_dim=768, 
                                mlp_dim=3072, 
                                num_classes=[10, 10, 27, 22],
                                weights_name=fine_tuned_weights)
        
    model = model.eval()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    video_list = glob.glob(VIDEOS_DIR+"/**/*", recursive=True)
    for video_path in tqdm(video_list, total=len(video_list)):
        if os.path.isdir(video_path):
            continue
        video = video2image(video_path).to(device)
        if enc == "clip":
            visual_output = model(video)
            visual_output = visual_output["image_embeds"]
        else:
            _, visual_output = model(video, True)

        # Normalizing the embeddings and calculating mean between all embeddings. 
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = torch.mean(visual_output, dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        visual_output = visual_output.detach().clone().cpu()

        if enc == "clip":
            enc_dir = "clip_general"
        else:
            enc_dir = "artistic_repr"
        save_path = os.path.join(ENCODINGS_DIR, enc_dir, video_path.split("/")[-1].replace(".mp4", ".pth"))
        torch.save(visual_output, save_path)

if __name__ == "__main__":
    main()