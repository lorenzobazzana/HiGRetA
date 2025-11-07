import os
import sys
import torch
import random
import wandb
import tqdm
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from scripts.utils import *
from scripts.eval import evaluate

class Trainer:
    def __init__(self, vlad_module, higreta_gcn, higreta_rnn_museums, higreta_rnn_rooms, loss, optimizer, scheduler, dataloaders, device, hyper_params):

        self.vlad_module = vlad_module
        self.higreta_gcn = higreta_gcn
        self.higreta_rnn_museums = higreta_rnn_museums
        self.higreta_rnn_rooms = higreta_rnn_rooms

        self.loss_fn = loss

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_dataset = dataloaders["train"].dataset
        self.val_dataset = dataloaders["val"].dataset
        self.dataloaders = dataloaders
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

    def partial_validation(self): # TODO: put this somewhere else?

        subset_len = 20000
        val_subset = random.sample(range(len(self.val_dataset)), subset_len)
        sampler = SubsetRandomSampler(val_subset)
        dl = DataLoader(self.val_dataset, batch_size=96, sampler=sampler, collate_fn=self.dataloaders["val"].collate_fn, num_workers=8)

        out_feature_size = self.hyper_params["out_feature_size"]
        output_description = torch.empty(subset_len, out_feature_size)
        output_scene = torch.empty(subset_len, out_feature_size)

        if self.hyper_params["uses_netvlad"]:
            self.vlad_module.eval()
        self.higreta_gcn.eval()
        self.higreta_rnn_museums.eval()
        self.higreta_rnn_rooms.eval()

        n_added_nodes = self.hyper_params["n_added_nodes"]
        room_aggregation_out_channels = self.hyper_params["room_aggregation_out_channels"] if self.hyper_params["uses_netvlad"] else 1280 # TODO put this into some form of parameter


        with torch.no_grad():
            for i, batch in enumerate(dl):

                museum_features, museum_descr_embeddings, _, _ = self.inference(*batch, n_added_nodes, room_aggregation_out_channels)

                initial_index = i * self.hyper_params["batch_size"]
                final_index = (i + 1) * self.hyper_params["batch_size"]
                if final_index > subset_len:
                    final_index = subset_len

                output_scene[initial_index:final_index, :] = museum_features
                output_description[initial_index:final_index, :] = museum_descr_embeddings

        if self.hyper_params["uses_netvlad"]:
            self.vlad_module.train()
        self.higreta_gcn.train()
        self.higreta_rnn_museums.train()
        self.higreta_rnn_rooms.train()


        return evaluate(output_description, output_scene)
    

    def train(self, use_wandb=False):

        print("Starting training")
        
        self.best_models = {
            "ds5": 0,
            "netvlad": None,
            "gcn": None,
            "gru_museums": None,
            "gru_rooms": None
        }

        # For uncertainty weighted loss
        sigma_museums = nn.Parameter(torch.rand(1, device=self.device), requires_grad=True)
        sigma_rooms = nn.Parameter(torch.rand(1, device=self.device), requires_grad=True)
        self.optimizer.add_param_group({"params": [sigma_museums, sigma_rooms]})

        if use_wandb:
            wandb.init(
                project = "Museums retrieval",
                config = self.hyper_params
            )
            wandb.run.log_code('.',exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("museums") or os.path.relpath(path, root).startswith("rooms") or os.path.relpath(path, root).startswith("multi_theme"))
            if self.hyper_params["uses_netvlad"]:
                wandb.watch(self.vlad_module, log="gradients", log_freq=self.hyper_params["amortized_loss_freq"], idx=0, log_graph=True)
            wandb.watch(self.higreta_gcn, log="gradients", log_freq=self.hyper_params["amortized_loss_freq"], idx=1, log_graph=True)
            wandb.watch(self.higreta_rnn_museums, log="gradients", log_freq=self.hyper_params["amortized_loss_freq"], idx=5, log_graph=True)
            if self.hyper_params["two_rnns"]:
                wandb.watch(self.higreta_rnn_rooms, log="gradients", log_freq=self.hyper_params["amortized_loss_freq"], idx=2, log_graph=True)

        num_epochs = self.hyper_params["num_epochs"]
        batch_size = self.hyper_params["batch_size"]
        amortized_loss_freq = self.hyper_params["amortized_loss_freq"]

        n_added_nodes = self.hyper_params["n_added_nodes"]
        room_aggregation_out_channels = self.hyper_params["room_aggregation_out_channels"]
        out_feature_size = self.hyper_params["out_feature_size"]

        for epoch in range(num_epochs):
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for phase in ["train", "val"]:

                print(f"Phase: {phase}")

                if phase == "train":
                    if self.hyper_params["uses_netvlad"]:
                        self.vlad_module.train()
                    self.higreta_gcn.train()
                    self.higreta_rnn_rooms.train()
                    self.higreta_rnn_museums.train()
                else:
                    if self.hyper_params["uses_netvlad"]:
                        self.vlad_module.eval()
                    self.higreta_gcn.eval()
                    self.higreta_rnn_rooms.eval()
                    self.higreta_rnn_museums.eval()

                amortized_loss = 0
                total_loss = 0

                dataset_len = len(self.dataloaders[phase].dataset)
                num_batches = dataset_len//batch_size

                if phase == "train":
                    sampler_for_evaluation = set(random.sample(range(num_batches-1), num_batches//8))
                    subset_size = len(sampler_for_evaluation)*batch_size
                else:
                    sampler_for_evaluation = set(range(num_batches))
                    subset_size = dataset_len
                
                output_description = torch.empty(subset_size, out_feature_size)
                output_scene = torch.empty(subset_size, out_feature_size)
                j = 0

                

                for i, batch in tqdm.tqdm(enumerate(self.dataloaders[phase]), total=num_batches):
                    
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        
                        museum_features, museum_descr_embeddings, room_features, room_descr_embeddings = self.inference(*batch, n_added_nodes, room_aggregation_out_channels)

                        sim_museums = cosine_sim(museum_features, museum_descr_embeddings)

                        # Contrastive loss for standard embeddings
                        if self.hyper_params["room_alignment"]:
                            sim_rooms = cosine_sim(room_features, room_descr_embeddings)
                            loss = 0.5 / (sigma_museums ** 2) * self.loss_fn.compute_loss(sim_museums) + 0.5 / (sigma_rooms ** 2) * self.loss_fn.compute_loss(sim_rooms) + torch.log(1 + sigma_museums ** 2) + torch.log(1 + sigma_rooms ** 2)
                        else:
                            loss = self.loss_fn.compute_loss(sim_museums)



                        if torch.isnan(loss):
                            print()
                            print("Warning! Loss is NaN. Batch_number:", i)
                            print()
                            raise Exception("Loss is NaN")
                        else:
                            amortized_loss += loss.item()
                            total_loss += loss.item()

                        if i in sampler_for_evaluation:
                            initial_index = j * batch_size
                            final_index = (j + 1) * batch_size
                            if final_index > dataset_len:
                                final_index = dataset_len

                            output_scene[initial_index:final_index, :] = museum_features.cpu()
                            output_description[initial_index:final_index, :] = museum_descr_embeddings.cpu()
                            j += 1


                        if i > 0 and i % amortized_loss_freq == 0 and use_wandb:
                            wandb.log({
                                f"{phase}/amortized_loss": amortized_loss/amortized_loss_freq
                            })
                            amortized_loss = 0


                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                        if phase == "train" and i > 0 and i % (num_batches//4) == 0:
                            partial_recall = self.partial_validation() 
                            if use_wandb:
                                wandb.log({"partial/"+key:val for key, val in zip(["T2S_R@1", "T2S_R@5", "T2S_R@10", "S2T_R@1", "S2T_R@5", "S2T_R@10"], partial_recall)})

                if phase == "train":
                    self.scheduler.step()
                    
                
                # VALIDATION PROCEDURE
                print("Evaluating rank...")
                ds1, ds5, ds10, sd1, sd5, sd10 = evaluate(output_description, output_scene)
                

                epoch_loss = total_loss / dataset_len
                
                if phase == "val" and ds5 > self.best_models["ds5"]:
                    self.best_models["ds5"] = ds5
                    if self.hyper_params["uses_netvlad"]:
                        self.best_models["netvlad"] = deepcopy(self.vlad_module.state_dict())
                    self.best_models["gcn"] = deepcopy(self.higreta_gcn.state_dict())
                    self.best_models["gru_museums"] = deepcopy(self.higreta_rnn_museums.state_dict())
                    if self.hyper_params["two_rnns"]:
                        self.best_models["gru_rooms"] = deepcopy(self.higreta_rnn_rooms.state_dict())
             
                print(f"{phase} epoch loss: {epoch_loss}")
                if use_wandb:
                    wandb.log({
                        f'{phase}/epoch_loss': epoch_loss,
                        f"{phase}/T2S_R@1": ds1, 
                        f"{phase}/T2S_R@5": ds5, 
                        f"{phase}/T2S_R@10": ds10,
                        f"{phase}/S2T_R@1": sd1, 
                        f"{phase}/S2T_R@5": sd5, 
                        f"{phase}/S2T_R@10": sd10,
                        "scheduler_lr": self.scheduler.get_last_lr()[0]
                    })

                del output_description, output_scene
        
        if use_wandb:
            wandb.finish()

    def save_checkpoint(self, save_path):

        cpu = lambda state_dict: {k: v.cpu() for k, v in state_dict.items()}
        os.makedirs(save_path, exist_ok=True)

        if self.hyper_params["uses_netvlad"]:
            torch.save(cpu(self.best_models["netvlad"]), os.path.join(save_path,          "vlad.pth"))
        torch.save(cpu(self.best_models["gcn"]), os.path.join(save_path,         "higreta_gcn.pth"))
        torch.save(cpu(self.best_models["gru_museums"]), os.path.join(save_path, "higreta_rnn_museums.pth"))
        if self.hyper_params["two_rnns"]:
            torch.save(cpu(self.best_models["gru_rooms"]), os.path.join(save_path,   "higreta_rnn_rooms.pth"))
