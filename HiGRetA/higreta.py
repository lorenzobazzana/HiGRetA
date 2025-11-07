import torch
import json
import os
import glob
import lz4.frame
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geom
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool, LayerNorm


class museumDataset(Dataset):
    def __init__(self, dataset_type, museum_json_path, painting_csv_path, painting_representations_path):
        
        if isinstance(museum_json_path, list):
            museums = []
            for path in museum_json_path:
                museums += glob.glob(os.path.join(path, "**/*"), recursive=True)
        else:
            museums = glob.glob(os.path.join(museum_json_path, "**/*"), recursive=True)
        self.museums_path = museum_json_path
        self.representations_base_path = painting_representations_path
        self.museums = [f for f in museums if os.path.isfile(f)]
        
        painting_csv = pd.read_csv(painting_csv_path, sep=";")

        # Get the list of video representations
        video_repr_path = os.path.join(painting_representations_path, "videos/clip_general")
        available_videos = set(os.listdir(video_repr_path))  # Use a set for faster lookups

        painting_pth_files = painting_csv["path"].str.replace(".jpg", ".pth", regex=False)
        valid_videos = painting_csv[painting_pth_files.isin(available_videos)].reset_index()["path"]
        videos_list = valid_videos.str.replace(".jpg", ".pth", regex=False)
        
        # Generate clip representations
        print("Loading CLIP general representations...")
        clip_repr = [torch.load(self.get_representation_path(painting_representations_path, "clip_general", painting), weights_only=True) for painting in painting_csv["path"]]
        print("Done!")

        # Generate art representations
        print("Loading artistic representations...")
        art_repr = [torch.load(self.get_representation_path(painting_representations_path, "artistic_repr", painting), weights_only=True) for painting in painting_csv["path"]]
        print("Done!")
        
        combined_repr = [torch.cat([clip, art],-1) for clip,art in zip(clip_repr, art_repr)]
        self.painting_representations = {idx:combined_repr[i] for i,idx in painting_csv["path"].items()}

        # Generate video representations
        print("Loading video representations...")
        video_clip_repr = [torch.load(os.path.join(painting_representations_path, "videos/clip_general", video), weights_only=True) for video in videos_list]
        video_art_repr = [torch.load(os.path.join(painting_representations_path, "videos/artistic_repr", video), weights_only=True) for video in videos_list]
        combined_repr = [torch.cat([clip, art],-1) for clip,art in zip(video_clip_repr, video_art_repr)]
        self.video_representations = {idx.replace(".pth", ".mp4"):combined_repr[i] for i,idx in videos_list.items()}
        print("Done!")

        descriptions_path = os.path.join(painting_representations_path, "descriptions")

        # Painting descriptions
        print("Loading painting descriptions ecodings...")
        painting_descr = [torch.load(self.get_representation_path(descriptions_path, "paintings", painting), weights_only=True) for painting in painting_csv["path"]]
        self.painting_descr = {idx:painting_descr[i] for i,idx in painting_csv["path"].items()}
        print("Done!")

        # Video descriptions
        print("Loading video descriptions ecodings...")
        video_descr = [torch.load(os.path.join(descriptions_path, "videos", video), weights_only=True) for video in videos_list]
        self.video_descr = {idx.replace(".pth", ".mp4"):video_descr[i] for i,idx in videos_list.items()}
        print("Done!")

        # Room descriptions
        print("Loading room descriptions ecodings...")
        rooms_dir = os.path.join(descriptions_path, "rooms", dataset_type, "**/*")
        rooms_list = glob.glob(rooms_dir, recursive=True)
        self.room_descr = {room.split('/')[-1]:torch.load(room, weights_only=True) for room in rooms_list if os.path.isfile(room)}
        print("Done!")

        # Museum descriptions
        print("Loading museum descriptions encodings...")
        # Monothematic museums
        museums_dir = os.path.join(descriptions_path, "museums", dataset_type, "**/*")
        museums_list = glob.glob(museums_dir, recursive=True)
        self.museum_descr = {museum.split('/')[-1].replace(".lz4", ""):torch.load(museum, weights_only=True) for museum in museums_list if os.path.isfile(museum)}
        # Multi-theme museums
        museums_dir = os.path.join(descriptions_path, "multi_theme_museums", dataset_type, "**/*")
        museums_list = glob.glob(museums_dir, recursive=True)
        self.museum_descr |= {museum.split('/')[-1].replace(".lz4", ""):torch.load(museum, weights_only=True) for museum in museums_list if os.path.isfile(museum)}
        print("Done!")

    def __len__(self):
        return len(self.museums)
    
    def get_representation_path(self, base_path, encoding_type, painting):
        dataset_type = "wikiart" if "/" in painting else "semart"
        # Replace .jpg with .pth in the file name
        painting_pth = painting.replace(".jpg", ".pth")

        return os.path.join(base_path, encoding_type, dataset_type, painting_pth)
    

    def parse_graph_string(self, graph_str):
        graph_str = graph_str.replace("(", "").replace(")", "").replace("[", "").replace("]","")
        edges = np.fromstring(graph_str, sep=',', dtype=int).reshape(-1, 2)
        bidirectional_edges = np.vstack([edges, edges[:, ::-1]])
        return bidirectional_edges.tolist()
    
    def __getitem__(self, index):
        museum = self.museums[index]
        with lz4.frame.open(museum, 'rb') as f:
            #museum = json.load(f)
            museum = pickle.loads(f.read())

        # Images and video encodings    
        rooms = [(room['paintings'], room['videos']) for room in museum["rooms"]]
        rooms = [
                    [self.painting_representations[painting] for painting in room_paintings] + 
                    [self.video_representations[video] for video in room_videos]
                    for room_paintings, room_videos in rooms
                ]
        
        # Description encodings
        room_descr = [
            torch.cat(
                [self.room_descr[room["id"]]] + 
                [self.painting_descr[painting] for painting in room["paintings"]] + 
                [self.video_descr[video] for video in room["videos"]],
            dim=0) 
            for room in museum["rooms"]
            ]
        
        museum_descr = torch.cat([self.museum_descr[museum["id"]]] + room_descr, dim=0)

        # Graph
        graph = museum["corridors"]
        graph = self.parse_graph_string(graph)

        return rooms, museum_descr, room_descr, graph

class DescriptorDataset(Dataset):
    def __init__(self, image_file, base_dir, video_file, video_dir):
        self.file = pd.read_csv(image_file, sep=";")
        self.base_dir = base_dir

        videos_df = pd.read_csv(os.path.join(video_dir, video_file))
        videos_df["path"] = videos_df["available_videos"].str.replace(".mp4", ".jpg")

        self.videos = videos_df.merge(self.file["path"], on="path").drop(columns="available_videos")
        self.video_dir = base_dir + "/videos"

        self.n_paintings = len(self.file)
        self.n_videos = len(self.videos)

    def __len__(self):
        return self.n_paintings + self.n_videos
    
    def __getitem__(self, index):

        if index < self.n_paintings:
            filename = self.file["path"][index]
            prefix = "wikiart/" if '/' in filename else "semart/"
            filename = prefix + filename
            dir_path = self.base_dir
        else:
            filename = self.videos["path"][index - self.n_paintings]
            dir_path = self.video_dir

        filename = filename.replace(".jpg", ".pth")
        clip_repr = torch.load(os.path.join(dir_path, "clip_general", filename), weights_only=True)
        art_repr = torch.load(os.path.join(dir_path, "artistic_repr", filename), weights_only=True)
        
        return torch.cat([clip_repr, art_repr], -1).flatten()

class LossContrastive:
    def __init__(self, name, patience=15, delta=.001, verbose=True):
        self.train_losses = []
        self.validation_losses = []
        self.name = name
        self.counter_patience = 0
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.verbose = verbose

    def compute_loss(self, pairwise_distances, margin=.25, margin_tensor=None):
        batch_size = pairwise_distances.shape[0]
        diag = pairwise_distances.diag().view(batch_size, 1)
        pos_masks = torch.eye(batch_size).bool().to(pairwise_distances.device)
        d1 = diag.expand_as(pairwise_distances)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            cost_s = (margin_tensor + pairwise_distances - d1).clamp(min=0)
        else:
            cost_s = (margin + pairwise_distances - d1).clamp(min=0)
        cost_s = cost_s.masked_fill(pos_masks, 0)
        cost_s = cost_s / (batch_size * (batch_size - 1))
        cost_s = cost_s.sum()

        d2 = diag.t().expand_as(pairwise_distances)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            cost_d = (margin_tensor + pairwise_distances - d2).clamp(min=0)
        else:
            cost_d = (margin + pairwise_distances - d2).clamp(min=0)
        cost_d = cost_d.masked_fill(pos_masks, 0)
        cost_d = cost_d / (batch_size * (batch_size - 1))
        cost_d = cost_d.sum()

        return (cost_s + cost_d) / 2


class HiGRetA_GRU(nn.Module):
    def __init__(self, name, num_features, hidden_size, is_bidirectional):
        super(HiGRetA_GRU, self).__init__()

        self.name = name
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True, bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        _, h_n = self.gru(x)

        if self.is_bidirectional:
            return h_n.mean(0)
        else:
            return h_n.squeeze(0)

class HiGRetA_GAT(nn.Module):
    def __init__(self, name, in_channels, hidden_dim, out_channels, use_mg, dropout_p=0.2):
        super().__init__()

        self.name = name
        self.use_mg = use_mg
        self.dropout = nn.Dropout(p=dropout_p)

        self.layer_norm1 = LayerNorm(hidden_dim*4)
        self.layer_norm2 = LayerNorm(hidden_dim*4)

        self.conv1 = GATConv(in_channels, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=4, concat=True)

        self.conv3 = GCNConv(hidden_dim*4, out_channels)


    def forward(self, graph_nodes, graph_edges, batch_info, n_nodes, return_node_features=False):

        last_node_indices = torch.cumsum(torch.bincount(batch_info), dim=0) - 1

       
        # Convolution -> LayerNorm -> ReLU -> Dropout
        graph_nodes = self.conv1(graph_nodes, graph_edges).relu()
        graph_nodes = self.layer_norm1(graph_nodes)  
        graph_nodes = self.dropout(graph_nodes)

        # Convolution -> LayerNorm -> ReLU -> Dropout
        graph_nodes = self.conv2(graph_nodes, graph_edges).relu()
        graph_nodes = self.layer_norm2(graph_nodes) 
        graph_nodes = self.dropout(graph_nodes)

        # Convolution
        graph_nodes = self.conv3(graph_nodes, graph_edges)


        last_node_features = graph_nodes[last_node_indices]

        if return_node_features:
            batch_nodes = []
            for i in range(batch_info.max().item() + 1):  # Iterate over all graphs in the batch
                current_graph_nodes = (batch_info == i).nonzero(as_tuple=True)[0]  # Nodes belonging to graph `i`
                specific_indices = current_graph_nodes[:n_nodes[i]] 
                batch_nodes.append(graph_nodes[specific_indices])

            if self.use_mg:
                return last_node_features, batch_nodes 
            else:
                return global_mean_pool(graph_nodes, batch_info), batch_nodes
        else:
            if self.use_mg:
                return last_node_features
            else:
                return global_mean_pool(graph_nodes, batch_info)


class HiGRetA_GCN(nn.Module):
    def __init__(self, name, in_channels, hidden_dim, out_channels, use_mg, dropout_p=0.2):
        super().__init__()

        self.name = name
        self.use_mg = use_mg
        self.dropout = nn.Dropout(p=dropout_p)

        self.batch_norm = BatchNorm(hidden_dim)
        
        self.conv1 = GCNConv(in_channels, hidden_dim) 
        self.conv2 = GCNConv(hidden_dim, hidden_dim) 
        self.conv3 = GCNConv(hidden_dim, out_channels)

        

    def forward(self, graph_nodes, graph_edges, batch_info, n_nodes, return_node_features=False):

        last_node_indices = torch.cumsum(torch.bincount(batch_info), dim=0) - 1

        graph_nodes = self.conv1(graph_nodes, graph_edges).relu()

        graph_nodes = self.batch_norm(graph_nodes)
        graph_nodes = self.dropout(graph_nodes)
        graph_nodes = self.conv2(graph_nodes, graph_edges).relu()
        graph_nodes = self.dropout(graph_nodes)

        graph_nodes = self.conv3(graph_nodes, graph_edges)

        last_node_features = graph_nodes[last_node_indices]

        if return_node_features:
            batch_nodes = []
            for i in range(batch_info.max().item() + 1):  # Iterate over all graphs in the batch
                current_graph_nodes = (batch_info == i).nonzero(as_tuple=True)[0]  # Nodes belonging to graph `i`
                specific_indices = current_graph_nodes[:n_nodes[i]] 
                batch_nodes.append(graph_nodes[specific_indices])

            if self.use_mg:
                return last_node_features, batch_nodes 
            else:
                return global_mean_pool(graph_nodes, batch_info), batch_nodes
        else:
            if self.use_mg:
                return last_node_features
            else:
                return global_mean_pool(graph_nodes, batch_info)