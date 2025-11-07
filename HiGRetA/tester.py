import os
import sys
import torch
import tqdm
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from scripts.utils import *

class Tester():
    def __init__(self, vlad_module, higreta_gcn, higreta_rnn_museums, higreta_rnn_rooms, test_dataloader, device, hyper_params):
        self.vlad_module = vlad_module
        self.higreta_gcn = higreta_gcn
        self.higreta_rnn_museums = higreta_rnn_museums
        self.higreta_rnn_rooms = higreta_rnn_rooms

        self.test_dataset = test_dataloader.dataset
        self.test_dataloader = test_dataloader
        self.device = device

        self.hyper_params = hyper_params


    def inference(self, n_nodes, batch_rooms, batch_graphs, packed_museum_descr, packed_room_descr, n_added_nodes, room_aggregation_out_channels):

        if self.hyper_params["uses_netvlad"]:
            aggregated_rooms = [torch.stack([self.vlad_module(room.to(self.device)) for room in museum]).flatten(1) for museum in batch_rooms] # every element of the list is a museum
        else: # mean
            aggregated_rooms = [torch.stack([room.to(self.device).mean(0) for room in museum]).flatten(1) for museum in batch_rooms]

        batch_graphs = [graph.to(self.device) for graph in batch_graphs]                        

        if self.hyper_params["uses_meta_graph"]: # prepare additional nodes
            _additional_nodes = [torch.zeros((n_added_nodes, room_aggregation_out_channels), device=self.device) for _ in range(len(batch_rooms))]
            batch_rooms = [Data(x=torch.cat([museum, _new_nodes]), edge_index=room_edges.T.contiguous()) for museum, room_edges, _new_nodes in zip(aggregated_rooms, batch_graphs, _additional_nodes)] # TODO: edit this part with if
        else:
            batch_rooms = [Data(x=museum, edge_index=room_edges.T.contiguous()) for museum, room_edges in zip(aggregated_rooms, batch_graphs)]

        batch_rooms = Batch.from_data_list(batch_rooms)

        museum_features, room_features = self.higreta_gcn(batch_rooms.x, batch_rooms.edge_index, batch_rooms.batch, n_nodes, return_node_features=True)
        room_features = torch.cat(room_features)

        
        packed_room_descr = packed_room_descr.to(self.device)
        room_descr_embeddings = self.higreta_rnn_rooms(packed_room_descr)

        if self.hyper_params["rnns_in_series"]:
        ##%%%%%%%%%%%%% Two RNNs in series
            cs = np.cumsum([0]+n_nodes)
            packed_museum_descr = [room_descr_embeddings[cs[i]:cs[i+1], :] for i in range(len(cs)-1)]
            packed_museum_descr = pack_padded_sequence(pad_sequence(packed_museum_descr, batch_first=True),
                                                    torch.tensor(n_nodes),
                                                    batch_first=True,
                                                    enforce_sorted=False)
        ##%%%%%%%%%%%%%

        packed_museum_descr = packed_museum_descr.to(self.device)
        museum_descr_embeddings = self.higreta_rnn_museums(packed_museum_descr)

        return museum_features, museum_descr_embeddings, room_features, room_descr_embeddings


    def test(self):

        batch_size = self.hyper_params["batch_size"]
        higreta_out_channels = self.hyper_params["out_feature_size"]
        n_added_nodes = self.hyper_params["n_added_nodes"]
        room_aggregation_out_channels = self.hyper_params["room_aggregation_out_channels"]

        if self.hyper_params["uses_netvlad"]:
            self.vlad_module.eval()
        

        self.higreta_gcn.eval()
        self.higreta_rnn_museums.eval()

        if self.hyper_params["rnns_in_series"]:
            self.higreta_rnn_rooms.eval()

        dataset_len = len(self.test_dataset)
        output_scene = torch.empty(dataset_len, higreta_out_channels)
        output_description = torch.empty_like(output_scene)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):

                museum_features, museum_descr_embeddings, _, _ = self.inference(*batch, n_added_nodes, room_aggregation_out_channels)

                initial_index = i * batch_size
                final_index = (i + 1) * batch_size
                if final_index > dataset_len:
                    final_index = dataset_len

                output_scene[initial_index:final_index, :] = museum_features
                output_description[initial_index:final_index, :] = museum_descr_embeddings

        return output_scene, output_description