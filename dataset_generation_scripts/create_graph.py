import os
import glob
import json
import random
import lz4.frame
import pickle
import networkx as nx

MUSEUM_PATH = ["../museums/", "../multi_theme_museums"]

def create_corridors(museum_data):

    rooms = museum_data["rooms"]

    partial_tree = []
    base_rooms = []
    remaining_rooms = set(range(len(rooms)))
    while len(remaining_rooms) > 0:
        base_room_idx = random.choice(list(remaining_rooms))
        base_room_categories = set(rooms[base_room_idx]["categories"].items())
        
        #remaining_rooms.remove(base_room_idx)
        similar_rooms = [room_idx for room_idx in remaining_rooms if len(set(rooms[room_idx]["categories"].items()).intersection(base_room_categories)) != 0] # at least one common category
        #similar_rooms.append(base_room_idx)
        
        tree = nx.random_unlabeled_tree(len(similar_rooms)) # labels from 0 to len(...)-1 (we also consider base_room_idx)
        tree = nx.relabel_nodes(tree, {i:j for i,j in zip(range(len(similar_rooms)), similar_rooms)}, copy=True)
        
        base_rooms.append(base_room_idx)
        partial_tree.append(tree)
        remaining_rooms -= set(similar_rooms)
    
    base_rooms_tree = nx.random_unlabeled_tree(len(base_rooms))
    base_rooms_tree = nx.relabel_nodes(base_rooms_tree, {i:j for i,j in zip(range(len(base_rooms)), base_rooms)}, copy=True)
    if len(partial_tree) > 1:
        partial_tree = nx.union_all(partial_tree)
        partial_tree = nx.compose(partial_tree, base_rooms_tree)
    else:
        partial_tree = partial_tree[0]

    # add a couple of random edges so that it's not necessarily a tree
    if len(rooms) > 5:
        n_edges_to_add = random.randint(0,2)
        for _ in range(n_edges_to_add):
            node_a = random.choice(list(partial_tree.nodes()))
            node_b = random.choice(list(partial_tree.nodes()))
            while node_b == node_a:
                node_b = random.choice(list(partial_tree.nodes()))
            tree.add_edge(node_a, node_b)

    return str(list(partial_tree.edges))
    

def main(path):

    museum_list = glob.glob(os.path.join(path, '**/*.lz4'), recursive=True)
    n_museums = len(museum_list)
    

    for i, museum_file in enumerate(museum_list):
        print(f"Created corridors for {i/n_museums*100:.2f}% of the museums ({i}/{n_museums})", end="\r")
        with lz4.frame.open(museum_file, 'r') as f:
            #museum = json.load(f)
            museum = pickle.loads(f.read())
        
        museum["corridors"] = create_corridors(museum)
        compressed = lz4.frame.compress(pickle.dumps(museum))
        with open(museum_file, 'wb') as f:
            f.write(compressed)
    print(f"Created corridors for {100:.2f}% of the museums ({n_museums}/{n_museums})", end="\r")
    print()

    return

if __name__ == "__main__":
    for museum_dir in MUSEUM_PATH:
        for split in ["train", "test", "val"]:
            main(path=os.path.join(museum_dir, split))