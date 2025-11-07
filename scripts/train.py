import os

import sys
import torch
import tqdm
import argparse
import wandb
import random
import yaml
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from HiGRetA.netvlad import NetVLAD
from HiGRetA.higreta import *
from HiGRetA.trainer import Trainer
from scripts.utils import *
from scripts.eval import evaluate

CONFIG_FILE = os.path.join(ROOT_DIR, "config.yaml")


def main():

    # Load config -----
    print(f"Using config file {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    base_encoding_dir = config["base_encoding_dir"]
    dataset_save_dir = config["dataset_save_dir"]
    museums_test_dirs = config["museums_test_dirs"]

    n_clusters = config["n_clusters"]
    cluster_file = config["cluster_file"].format(n_clusters=n_clusters)

    out_feature_size = config["out_feature_size"]
    higreta_hidden_dim = config["gcn_hidden_dim"]

    descriptor_size = config["descriptor_size"]
    out_feature_size = config["out_feature_size"]

    train_painting_file = config["train_painting_file"]
    val_painting_file = config["val_painting_file"]
    
    video_file = config["video_file"] 
    video_dir = config["video_dir"]

    model_save_path = config["model_save_path"]

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    meta_graph_nodes = config["meta_graph_nodes"]

    train_museum_dir = config["museum_dir"]["train"]
    train_multi_theme_dir = config["multi_theme_museum_dir"]["train"]
    
    val_museum_dir = config["museum_dir"]["val"]
    val_multi_theme_dir = config["multi_theme_museum_dir"]["val"]
    
    device_name = config["device"]
    
    lr = config["lr"]
    momentum = config["momentum"]

    gamma = config["gamma"]
    steps = config["steps"]

    amortized_loss_freq = config["amortized_loss_freq"]
    num_epochs = config["num_epochs"]
    #-------------------

        # Set up command line argument parsing.
    parser = argparse.ArgumentParser(description="Train a model from scratch, and save model checkpoints.")
    parser.add_argument("model_name", help="Name of the model to load (e.g., 'my_model')")
    parser.add_argument("--room-alignment",help="Whether the model should contrastively align representations at both room and museum level", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-attention",help="Choose between GAT and classical GCN", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--two-rnns", help="Whether the model requires a separate rnn for museums and rooms.", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--netvlad", help="Whether the model uses NetVLAD for room aggregation.", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--meta-graph", help="Whether the model uses a meta-graph structure.", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rnns-in-series", help="Whether the model uses two rnns in series.", action=argparse.BooleanOptionalAction, default=False) # rewrite this better
    parser.add_argument("--wandb", help="Record run metrics using wandb.", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()

    # Use the provided model name
    model_name = args.model_name
    graph_attention = args.graph_attention
    room_alignment = args.room_alignment
    two_rnns = args.two_rnns
    uses_meta_graph = args.meta_graph
    uses_netvlad = args.netvlad
    rnns_in_series = args.rnns_in_series
    use_wandb = args.wandb

    n_added_nodes = sum(meta_graph_nodes)

    # Dataset creation and loading
    museums_train_dirs = [train_museum_dir, train_multi_theme_dir]
    museums_val_dirs = [val_museum_dir, val_multi_theme_dir]

    clusters = load_or_create_cluster_file(cluster_file, n_clusters, train_painting_file, video_file, base_encoding_dir, video_dir)


    train_dataset_file = os.path.join(dataset_save_dir, "train.lz4")
    train_dataset = load_or_create_dataset(train_dataset_file, "train", museums_train_dirs, train_painting_file, base_encoding_dir)

    val_dataset_file = os.path.join(dataset_save_dir, "val.lz4")
    val_dataset = load_or_create_dataset(val_dataset_file, "val", museums_val_dirs, val_painting_file, base_encoding_dir)

    collate_with_parameters = lambda x: collate_fn(x, uses_meta_graph, uses_netvlad, meta_graph_nodes)

    train_dataloader = DataLoader(train_dataset, batch_size, num_workers=num_workers, collate_fn=collate_with_parameters, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, num_workers=num_workers, collate_fn=collate_with_parameters, shuffle=True)
    
    #############

    # Model creation
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    if uses_netvlad:
        vlad_module = NetVLAD(n_clusters, descriptor_size, alpha=50, clusters=clusters)
        room_aggregation_out_channels = n_clusters * descriptor_size
    else:
        vlad_module = None
        room_aggregation_out_channels = descriptor_size

    higreta_gcn = HiGRetA_builder(build_gat=graph_attention,
                                    use_mg=uses_meta_graph,
                                    model_name=model_name,
                                    in_channels=room_aggregation_out_channels, 
                                    hidden_dim=higreta_hidden_dim, 
                                    out_channels=out_feature_size)
    

    higreta_rnn_museums = HiGRetA_GRU(name=model_name, num_features=512, hidden_size=out_feature_size, is_bidirectional=True)

    if two_rnns:
        higreta_rnn_rooms = HiGRetA_GRU(name=model_name, num_features=512, hidden_size=out_feature_size, is_bidirectional=True)
    else:
        higreta_rnn_rooms = higreta_rnn_museums


    device = torch.device(device_name)

    if uses_netvlad:
        vlad_module = vlad_module.to(device)
    higreta_gcn = higreta_gcn.to(device)
    higreta_rnn_rooms = higreta_rnn_rooms.to(device)
    higreta_rnn_museums = higreta_rnn_museums.to(device)

    loss_fn = LossContrastive("GCN+GRU")

    params = list(vlad_module.parameters() if uses_netvlad else []) + list(higreta_gcn.parameters()) + list(higreta_rnn_museums.parameters()) + (list(higreta_rnn_rooms.parameters()) if two_rnns else [])
    wd = 1e-5 if graph_attention else 0 # weight decay for gat only!
    optimizer = optim.Adam(params, lr, weight_decay=wd) 


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=steps, gamma=gamma)

    hyper_params = {
        "batch_size": batch_size,
        "n_clusters": n_clusters,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "amortized_loss_freq": amortized_loss_freq,
        "n_added_nodes": n_added_nodes,
        "room_aggregation_out_channels": room_aggregation_out_channels,
        "out_feature_size": out_feature_size,
        "room_alignment": room_alignment,
        "two_rnns": two_rnns,
        "uses_meta_graph": uses_meta_graph,
        "uses_netvlad": uses_netvlad,
        "rnns_in_series": rnns_in_series   
    }

    ################################################

    trainer = Trainer(vlad_module,
                      higreta_gcn,
                      higreta_rnn_museums,
                      higreta_rnn_rooms,
                      loss_fn,
                      optimizer,
                      scheduler,
                      dataloaders,
                      device=device,
                      hyper_params=hyper_params)
    trainer.train(use_wandb)

    model_dir = os.path.join(model_save_path, model_name)
    trainer.save_checkpoint(model_dir)
    
if __name__ == "__main__":
    main()