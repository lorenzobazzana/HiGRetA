import os

import sys
import h5py
import argparse
import torch
import itertools
import pickle
import yaml
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from HiGRetA.netvlad import NetVLAD
from HiGRetA.higreta import *
from HiGRetA.tester import Tester
from scripts.utils import *

CONFIG_FILE = os.path.join(ROOT_DIR, "config.yaml")


def main():

    # Load config -----
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    base_encoding_dir = config["base_encoding_dir"]
    dataset_save_dir = config["dataset_save_dir"]
    museums_test_dirs = config["museums_test_dirs"]

    n_clusters = config["n_clusters"]
    cluster_file = config["cluster_file"].format(n_clusters=n_clusters)

    descriptor_size = config["descriptor_size"]
    
    out_feature_size = config["out_feature_size"]
    higreta_hidden_dim = config["gcn_hidden_dim"]

    test_painting_file = config["test_painting_file"]

    model_save_path = config["model_save_path"]

    out_save_path = config["out_save_path"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    meta_graph_nodes = config["meta_graph_nodes"]
    out_feature_size = config["out_feature_size"]

    device_name = config["device"]

    #-------------------

    with h5py.File(cluster_file, 'r') as h5file:
        clusters = h5file["clusters"][:]
        saved_n_clusters = h5file["clusters"].attrs["n_clusters"]

    # Set up command line argument parsing.
    parser = argparse.ArgumentParser(description="Load model weights and run evaluation.")
    parser.add_argument("model_name", help="Name of the model to load (e.g., 'my_model')")
    parser.add_argument("--graph_attention",help="Choose between GAT and classical GCN", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--two-rnns", help="Whether the model requires a separate rnn for museums and rooms", action=argparse.BooleanOptionalAction)
    parser.add_argument("--netvlad", help="Whether the model uses NetVLAD for room aggregation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--meta-graph", help="Whether the model uses a meta-graph structure", action=argparse.BooleanOptionalAction)
    parser.add_argument("--rnns-in-series", help="Whether the model uses two rnns in series", action=argparse.BooleanOptionalAction) # rewrite this better
    args = parser.parse_args()

    # Use the provided model name
    model_name = args.model_name
    graph_attention = args.graph_attention
    two_rnns = args.two_rnns
    uses_meta_graph = args.meta_graph
    uses_netvlad = args.netvlad
    rnns_in_series = args.rnns_in_series

    n_added_nodes = sum(meta_graph_nodes)

    # Read/create test dataset file
    test_dataset_file = os.path.join(dataset_save_dir, "test.lz4")
    test_dataset = load_or_create_dataset(test_dataset_file, "val", museums_test_dirs, test_painting_file, base_encoding_dir)


    collate_with_parameters = lambda x: collate_fn(x, uses_meta_graph, uses_netvlad, meta_graph_nodes)
    test_dataloader = DataLoader(test_dataset, batch_size, num_workers=num_workers, collate_fn=collate_with_parameters, shuffle=False)

    # Model pipeline
    if uses_netvlad:
        vlad_module = NetVLAD(n_clusters, descriptor_size, alpha=50, clusters=clusters)
        room_aggregation_out_channels = n_clusters * descriptor_size
    else:
        vlad_module = None
        room_aggregation_out_channels = config["descriptor_size"]

    
    higreta_gcn = HiGRetA_builder(use_mg=graph_attention,
                                    use_mg=uses_meta_graph,
                                    name=model_name,
                                    in_channels=room_aggregation_out_channels, 
                                    hidden_dim=higreta_hidden_dim, 
                                    out_channels=out_feature_size)


    higreta_rnn_museums = HiGRetA_GRU(num_features=512, hidden_size=out_feature_size, is_bidirectional=True)

    if two_rnns:
        higreta_rnn_rooms = HiGRetA_GRU(num_features=512, hidden_size=out_feature_size, is_bidirectional=True)
    else:
        higreta_rnn_rooms = higreta_rnn_museums

    device = torch.device(device_name)
    vlad_module = vlad_module.to(device)
    higreta_gcn = higreta_gcn.to(device)
    higreta_rnn_rooms = higreta_rnn_rooms.to(device)
    higreta_rnn_museums = higreta_rnn_museums.to(device)

    # Load checkpoint
    model_path = os.path.join(model_save_path, model_name)
    vlad_weights = torch.load(os.path.join(model_path, "vlad.pth"))
    gcn_weights = torch.load(os.path.join(model_path, "higreta_gcn.pth"))

    vlad_module.load_state_dict(vlad_weights)
    higreta_gcn.load_state_dict(gcn_weights)

    rnn_museums_weights = torch.load(os.path.join(model_path, "higreta_rnn_museums.pth"))
    higreta_rnn_museums.load_state_dict(rnn_museums_weights)
    
    if two_rnns:
        rnn_rooms_weights = torch.load(os.path.join(model_path, "higreta_rnn_rooms.pth"))
        higreta_rnn_rooms.load_state_dict(rnn_rooms_weights)


    hyper_params = {
        "batch_size": batch_size,
        "n_clusters": n_clusters,
        "n_added_nodes": n_added_nodes,
        "room_aggregation_out_channels": room_aggregation_out_channels,
        "out_feature_size": out_feature_size,
        "two_rnns": two_rnns,
        "uses_meta_graph": uses_meta_graph,
        "uses_netvlad": uses_netvlad,
        "rnns_in_series": rnns_in_series   
    }

    ###############################

    tester = Tester(vlad_module,
                    higreta_gcn,
                    higreta_rnn_museums,
                    higreta_rnn_rooms,
                    test_dataloader,
                    device,
                    hyper_params)

    scene, desc = tester.test()
    

    model_out_dir = os.path.join(out_save_path, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    torch.save(scene.detach().cpu(), os.path.join(model_out_dir, "scene.pth"))
    torch.save(desc.detach().cpu(), os.path.join(model_out_dir, "desc.pth"))

if __name__ == "__main__":
    main()