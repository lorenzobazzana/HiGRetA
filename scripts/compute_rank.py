import os

import torch
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

def cosine_sim(im, s):
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim

def pairwise(desc_tens, scene_tens, chunk_size=1000):
    dataset_len = desc_tens.shape[0]
    pairwise_full = np.empty((dataset_len, dataset_len))

    with torch.no_grad():
        for i in tqdm(range(dataset_len//chunk_size + 1)):
            
            initial_index = i * chunk_size
            final_index = (i + 1) * chunk_size
            if final_index > dataset_len:
                final_index = dataset_len
            
            chunk_pairwise = cosine_sim(desc_tens[initial_index:final_index, :], scene_tens)
            pairwise_full[initial_index:final_index, :] = chunk_pairwise.cpu().numpy()

    return pairwise_full

def compute_recall(sorted_pairwise):

    dataset_len = sorted_pairwise.shape[0]

    expected_values = np.arange(dataset_len)
    
    r1 = np.sum(expected_values == sorted_pairwise[:, 0]) / dataset_len * 100# Rank @1
    r5 = np.sum([ev in sorted_pairwise[i, :5] for i, ev in enumerate(expected_values)]) / dataset_len * 100 # Rank @5
    r10 = np.sum([ev in sorted_pairwise[i, :10] for i, ev in enumerate(expected_values)]) / dataset_len * 100# Rank @10

    return r1, r5, r10

def compute_rank(sorted_pairwise):
    return np.diag(sorted_pairwise.argsort())


def compute_metrics(pairwise):
    indexes_ds = np.argsort(-pairwise, axis=-1)
    ds1, ds5, ds10 = compute_recall(indexes_ds)
    rank_ds = compute_rank(indexes_ds) + 1
    median_rank_ds = np.median(rank_ds)
    mrr_ds = np.mean(1/rank_ds)
    del indexes_ds, rank_ds

    indexes_sd = np.argsort(-pairwise.T, axis=-1)
    sd1, sd5, sd10 = compute_recall(indexes_sd)
    rank_sd = compute_rank(indexes_sd) + 1
    median_rank_sd = np.median(rank_sd)
    mrr_sd = np.mean(1/rank_sd)
    del indexes_sd, rank_sd

    out = {
        "ds1": ds1,
        "ds5": ds5,
        "ds10": ds10,
        "sd1": sd1,
        "sd5": sd5,
        "sd10": sd10,
        "median_rank_ds": median_rank_ds,
        "median_rank_sd": median_rank_sd,
        "mrr_ds": mrr_ds,
        "mrr_sd": mrr_sd,
    }

    return out

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True, help="File to save results to.")
    parser.add_argument("--embedding-dir", required=True, help="Directory with the embeddings to use for ranking computation.")
    args = parser.parse_args()

    results_save_file = args.result_file
    embedding_dir = args.embedding_dir

    architectures = os.listdir(embedding_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for arch in architectures:
        arch_name = arch.split('/')[-1]
        print(f"Model: {arch_name}")
        desc_tensor = torch.load(os.path.join(arch, "desc.pth")).to(device)
        scene_tensor = torch.load(os.path.join(arch, "scene.pth")).to(device)

        print("Computing pairwise distances...")
        distances = pairwise(desc_tensor, scene_tensor)
        print("Computing metrics...")
        metrics = compute_metrics(distances)

        if os.path.exists(results_save_file):
            results = pd.read_csv(results_save_file, index_col=0)
            results.loc[arch_name] = metrics
        else:
            results = pd.DataFrame(metrics, index=[arch_name])

        results.to_csv(results_save_file)

        del distances


if __name__ == "__main__":
    main()
