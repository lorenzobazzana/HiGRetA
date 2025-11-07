import torch
import os
import faiss
import h5py
import sys
import lz4
import pickle
import itertools
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from HiGRetA.higreta import DescriptorDataset, museumDataset, HiGRetA_GAT, HiGRetA_GCN

def cosine_sim(im, s):
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim

def create_meta_graph(edges, n_nodes, meta_graph_nodes=[3, 1]): 
    
    last_level_nodes = 0
    for k in meta_graph_nodes:
        new_nodes_start = n_nodes
        new_nodes_end = n_nodes + k
        # Add edges from the current level to the next meta-graph nodes
        edges.extend([ 
            [i, j] for i in range(last_level_nodes, new_nodes_start) 
                    for j in range(new_nodes_start, new_nodes_end)
        ])
        # Add bidirectional connections between new meta-graph nodes
        edges.extend(zip(range(new_nodes_start, new_nodes_end - 1), 
                         range(new_nodes_start + 1, new_nodes_end)))
        edges.extend(zip(range(new_nodes_start + 1, new_nodes_end), 
                         range(new_nodes_start, new_nodes_end - 1)))

        # Update node counts
        last_level_nodes = new_nodes_start
        n_nodes = new_nodes_end

    return edges


def load_or_create_cluster_file(cluster_file, n_clusters, train_painting_file, video_file, base_encoding_dir, video_dir):
    if os.path.exists(cluster_file):
        with h5py.File(cluster_file, 'r') as h5file:
            clusters = h5file["clusters"][:]
            saved_n_clusters = h5file["clusters"].attrs["n_clusters"]

    if not os.path.exists(cluster_file) or saved_n_clusters != n_clusters:
        print("Creating descriptor dataset...")
        
        tmp_dataset = DescriptorDataset(train_painting_file, base_encoding_dir, video_file, video_dir)
        tmp_dataloader = DataLoader(tmp_dataset, num_workers=8, batch_size=128)
        descriptor_tensor = torch.tensor([])
        num_batches = len(tmp_dataloader)

        for i, batch in enumerate(tmp_dataloader):
            print(f"Loading batch {i+1}/{num_batches}", end='\r')
            descriptor_tensor = torch.cat([descriptor_tensor, batch], 0)
        print()

        kmeans = faiss.Kmeans(descriptor_tensor.shape[1], n_clusters, verbose=False, nredo=16, max_points_per_centroid=descriptor_tensor.shape[0])
        print("Fitting KMeans...")
        kmeans.train(descriptor_tensor)
        print("Done!")
        clusters = kmeans.centroids

        with h5py.File(cluster_file, 'w') as h5file:
            cluster_dataset = h5file.create_dataset("clusters", data=clusters)
            cluster_dataset.attrs["n_clusters"] = n_clusters

    return clusters

def load_or_create_dataset(dataset_file, dataset_type, museums_dirs, painting_file, base_encoding_dir):
    if os.path.exists(dataset_file):
            print(f"Loading {dataset_type} dataset from disk...")
            with lz4.frame.open(dataset_file, 'rb') as f:
                    data = pickle.loads(f.read())
                    dataset = object.__new__(museumDataset) # Instantiate empty dummy dataset, and then update it with data
                    dataset.__dict__.update(data["state"])
            print("Done!")
    else:
        print(f"Creating {dataset_type} dataset...")
        dataset = museumDataset(dataset_type, museums_dirs, painting_file, base_encoding_dir)
        
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        with lz4.frame.open(dataset_file, 'wb') as f:
            pickle.dump({"state": dataset.__dict__}, f)
        print("Dataset saved!")

    return dataset


def collate_fn(batch, uses_meta_graph, uses_netvlad, meta_graph_nodes):
    batch_rooms, batch_descr, batch_room_descr, batch_graphs = zip(*batch)

    if uses_netvlad:
        batch_rooms = [[
                torch.stack(room).permute(2,1,0).unsqueeze(0) for room in museum # required to prepare data shape for netVlad
            ] for museum in batch_rooms
        ]
    else:
        batch_rooms = [[
                torch.stack(room) for room in museum # this is in the no netvlad case
            ] for museum in batch_rooms
        ]

    n_nodes = [len(museum) for museum in batch_rooms]
    if uses_meta_graph:
        batch_graphs = [create_meta_graph(graph, nodes, meta_graph_nodes) for graph, nodes in zip(batch_graphs, n_nodes)]
        batch_graphs = [torch.tensor(graph) for graph in batch_graphs]
    else:
        # FOR BASELINE/NO METAGRAPH ONLY
        # add self loops
        add_self_loops = lambda edges, n: edges + [[i,i] for i in range(n)]
        batch_graphs = [torch.tensor(add_self_loops(graph, n), dtype=torch.int64) for graph, n in zip(batch_graphs, n_nodes)]

    batch_room_descr = list(itertools.chain.from_iterable(batch_room_descr)) # flattening

    museum_descr_lens = [len(x) for x in batch_descr]
    room_descr_lens = [len(x) for x in batch_room_descr]

    packed_museum_descr = pack_padded_sequence(pad_sequence(batch_descr, batch_first=True),
                                                            torch.tensor(museum_descr_lens),
                                                            batch_first=True,
                                                            enforce_sorted=False)
        
    packed_room_descr = pack_padded_sequence(pad_sequence(batch_room_descr, batch_first=True),
                                            torch.tensor(room_descr_lens),
                                            batch_first=True,
                                            enforce_sorted=False)
    

    return n_nodes, batch_rooms, batch_graphs, packed_museum_descr, packed_room_descr


def HiGRetA_builder(build_gat, use_mg, model_name, in_channels, hidden_dim, out_channels, dropout_p=0.2):

    if build_gat:
        return HiGRetA_GAT(name=model_name,
                            in_channels=in_channels, 
                            hidden_dim=hidden_dim, 
                            out_channels=out_channels,
                            use_mg=use_mg,
                            dropout_p=dropout_p)
    else:       
        return HiGRetA_GCN(name=model_name,
                            in_channels=in_channels, 
                            hidden_dim=hidden_dim, 
                            out_channels=out_channels,
                            use_mg=use_mg,
                            dropout_p=dropout_p)
