import torch
import numpy as np
import faiss

def evaluate(output_description, output_scene):

    dataset_len = len(output_description)

    output_description = output_description.detach().numpy()
    output_scene = output_scene.detach().numpy()
    expected_values = np.arange(dataset_len)

    faiss.normalize_L2(output_description)
    faiss.normalize_L2(output_scene)

    index = faiss.IndexFlatIP(512)

    index.train(output_scene)
    index.add(output_scene)
    _, indexes_d2s = index.search(output_description, 10)

    ds1 = np.sum(expected_values == indexes_d2s[:, 0]) / dataset_len * 100# Rank @1
    ds5 = np.sum([ev in indexes_d2s[i, :5] for i, ev in enumerate(expected_values)]) / dataset_len * 100 # Rank @5
    ds10 = np.sum([ev in indexes_d2s[i, :10] for i, ev in enumerate(expected_values)]) / dataset_len * 100# Rank @10
    del index, indexes_d2s

    index = faiss.IndexFlatIP(512)

    index.train(output_description)
    index.add(output_description)
    _, indexes_s2d = index.search(output_scene, 10)

    sd1 = np.sum(expected_values == indexes_s2d[:, 0]) / dataset_len * 100# Rank @1
    sd5 = np.sum([ev in indexes_s2d[i, :5] for i, ev in enumerate(expected_values)]) / dataset_len * 100# Rank @5
    sd10 = np.sum([ev in indexes_s2d[i, :10] for i, ev in enumerate(expected_values)]) / dataset_len * 100# Rank @10
    del index, indexes_s2d

    return ds1, ds5, ds10, sd1, sd5, sd10
