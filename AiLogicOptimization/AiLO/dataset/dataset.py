import os
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
import random
from tqdm import tqdm
from copy import deepcopy
import torch
from torch_geometric.data import Data, InMemoryDataset
from collections import defaultdict

from .utils import *

data_dir = '' # the path to store data
graphml_tool = '' # the path to the aig2graphml tool
csv_dir = os.path.join(data_dir, 'csv') 
des_class = '' # 'EPFL' or 'Core'
data_dir = os.path.join(data_dir, des_class)

def nodes_analyse_num(G, node_types_of_interest ):    
    nodes = {node_type: 0 for node_type in node_types_of_interest}
    node_type_count = defaultdict(int)
    for node, data in G.nodes(data=True):
        node_type = data.get('type')
        if node_type in nodes: 
            nodes[node_type] += 1  

    return nodes

def edges_analyse_num(G,edge_types_of_interest):
    edges = {edge_type: 0 for edge_type in edge_types_of_interest}

    for u, v, attr in G.edges(data=True):
        edge_type = attr.get('type')
        if edge_type in edges:  
            edges[edge_type] += 1  

    return edges

def load_mean_std(csv_dir):
    df = pd.read_csv(csv_dir)
    mean_ ,std_ = df['mean'].iloc[0], df['std'].iloc[0]
    return mean_, std_

def Graphml_to_Data(G, node_types=node_types):
    H = nx.DiGraph()

    H.add_nodes_from(G.nodes(data=True))
    cnt = len(G.nodes)
    for u, v, data in G.edges(data=True):
        if data['type'] in ['not', 'buf']:
            ty = data['type']
            new_node_id = f"{cnt}"
            H.add_node(new_node_id, type=data['type'])
            H.add_edge(u, new_node_id, **data)
            H.add_edge(new_node_id, v, **data)
            cnt += 1
        else:
            H.add_edge(u, v, **data)
        
    node_attrs = nx.get_node_attributes(H, 'type')
    num_node_types = len(node_types)  
    x = torch.zeros((len(H.nodes)+1, num_node_types), dtype=torch.long)  

    for node, attr in node_attrs.items():
        if attr == 'zero':
            continue  
        type_idx = node_types[attr]  
        x[int(node), type_idx] = 1  

    edge_index = torch.tensor(list(map(lambda x: (int(x[0]), int(x[1])), H.edges())), dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    node = nodes_analyse_num(H, node_types)
    data.pi_num = torch.tensor(node['pi'],dtype=torch.int)
    data.po_num = torch.tensor(node['po'],dtype=torch.int)
    data.and_num = torch.tensor(node['and'],dtype=torch.int)
    data.buf_num = torch.tensor(node['buf'],dtype=torch.int)
    data.not_num = torch.tensor(node['not'],dtype=torch.int)
    return data

def convert_aig_to_graphml(aig_file_path, graphml_path, tool_path):
    command = f"{tool_path} {aig_file_path} {graphml_path}"
    subprocess.run(command, shell=True, check=True)

class QoR_Dataset(InMemoryDataset):
    def __init__(self, root, data_dir, designs, target,transform=None, pre_transform=None, force_reload=False, empty=False):
        self.root = root
        self.data_dir = data_dir
        self.designs = designs
        self.target = target
        self.data_list = []

        super(QoR_Dataset, self).__init__(root, transform, pre_transform, None) 

        if not force_reload and self._check_processed_files_exist():
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif not empty:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    def _check_processed_files_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.target}.pt']

    def process(self):
        for design in self.designs:
            des_dir = os.path.join(self.data_dir, design)
            graphml_dir = os.path.join(des_dir, f'{design}.graphml')
            G = nx.read_graphml(graphml_dir)
            data = Graphml_to_Data(G, node_types)

            abc_dir = os.path.join(des_dir, 'character.csv')
            if self.target == 'area':
                nor_dir = os.path.join(des_dir, 'des_area.csv')
            elif self.target == 'delay':
                nor_dir = os.path.join(des_dir, 'des_delay.csv')
            else:
                raise ValueError(f"Invalid target: {self.target}")
            abc_df = pd.read_csv(abc_dir)
            mean_, std_ = load_mean_std(nor_dir)

            for i in tqdm(range(len(abc_df)),desc=f'processing {design}'):
                data_temp = deepcopy(data)
                opt_seq = eval(abc_df.iloc[i]['opt_seq'])
                if len(opt_seq) < 20:
                    opt_seq = np.pad(opt_seq, (0, 20 - len(opt_seq)), 'constant', constant_values=(0))
                data_temp.opt_seq = torch.tensor(opt_seq, dtype=torch.int)
                if self.target == 'area':
                    nor = (abc_df.iloc[i]['area'] - mean_) / std_
                elif self.target == 'delay':
                    nor = (abc_df.iloc[i]['delay'] - mean_) / std_
                data_temp.target = torch.tensor(nor, dtype=torch.float)
                self.data_list.append(data_temp)
        random.shuffle(self.data_list)
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])
    
def get_command_number(command_name):
    return OptDict_reverse.get(command_name, None)

def convert_line_to_label(line):
    commands = line.split(';')
    labels = []
    for command in commands:
        if command:  
            labels.append(get_command_number(command))
    return labels

class DSEDataset(InMemoryDataset):
    def __init__(self, dataset, design, target,transform=None, pre_transform=None, force_reload=False, empty=False):
        self.root = dataset
        if target == 'test':
            self.data_dir = os.path.join(self.root, 'eval.csv')
        elif target == 'train':
            self.data_dir = os.path.join(self.root, 'character.csv')
        elif target == 'dse':
            self.data_dir = os.path.join(self.root, 'scripts.csv')
        else:
            raise ValueError(f"Invalid target: {self.target}")
        self.design = design
        self.target = target
        self.graphml_dir = os.path.join(self.root, f'{self.design}.graphml')
        self.transform, self.pre_transform = transform, pre_transform
        self.data_list = []

        super(DSEDataset, self).__init__(dataset, transform, pre_transform)
        if not force_reload and self._check_processed_files_exist():
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif not empty:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    def _check_processed_files_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.target}evaluate.pt']

    def process(self):
        # data_list = []
        G = nx.read_graphml(self.graphml_dir)
            
        data = Graphml_to_Data(G, node_types)
        data_df = pd.read_csv(self.data_dir)
        for i in tqdm(range(len(data_df)),desc=f'processing {self.design}'):
            data_temp = deepcopy(data)
            opt_seq = eval(data_df.iloc[i]['opt_seq'])
            if len(opt_seq) < 20:
                opt_seq = np.pad(opt_seq, (0, 20 - len(opt_seq)), 'constant', constant_values=(0))
            data_temp.opt_seq = torch.tensor(opt_seq, dtype=torch.int)
            self.data_list.append(data_temp)
        print(f"Total data objects created: {len(self.data_list)}")
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])

def nsga_Data(InMemoryDataset):
    def __init__(self, dataset, design,transform=None, pre_transform=None, force_reload=False, empty=False):
        self.root = dataset
        self.design = design
        self.graphml_dir = os.path.join(self.root, f'{self.design}.graphml')
        self.transform, self.pre_transform = transform, pre_transform
        self.data_list = []

        super(DSEDataset, self).__init__(dataset, transform, pre_transform)
        if not force_reload and self._check_processed_files_exist():
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif not empty:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    def _check_processed_files_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.target}evaluate.pt']

    def process(self):
        G = nx.read_graphml(self.graphml_dir)
            
        data = Graphml_to_Data(G, node_types, edge_types)
        node = nodes_analyse_num(G, node_types)
        edge = edges_analyse_num(G, edge_types)
        data.pi_num = torch.tensor([node['pi']],dtype=torch.int)
        data.po_num = torch.tensor([node['po']],dtype=torch.int)
        data.and_num = torch.tensor([node['and']],dtype=torch.int)
        data.buf_num = torch.tensor([edge['buf']],dtype=torch.int)
        data_df = pd.read_csv(self.data_dir)
        for i in tqdm(range(len(data_df)),desc=f'processing {self.design}'):
            data_temp = deepcopy(data)
            opt_seq = eval(data_df.iloc[i]['opt_seq'])
            if len(opt_seq) < 20:
                opt_seq = np.pad(opt_seq, (0, 20 - len(opt_seq)), 'constant', constant_values=(0))
            data_temp.opt_seq = torch.tensor(opt_seq, dtype=torch.int)
            self.data_list.append(data_temp)
        print(f"Total data objects created: {len(self.data_list)}")

        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])

def apply(res_dir, data_dir, design, aig_dir, graphml_tool, target):
    
    des_dir = os.path.join(data_dir, des_class, design)
    aig_dir = os.path.join(data_dir,f'benchmark/EPFL/{design}.aig')
    graphml_dir = os.path.join(des_dir, f'{design}.graphml')
    if os.path.exists(graphml_dir)==False:
        convert_aig_to_graphml(aig_dir, graphml_dir, graphml_tool)
            
    
if __name__ == '__main__':
    target = 'area'
    aig_dir = os.path.join(data_dir,f'benchmark/EPFL')
    cnt = 1
    for design in designs:
        res_dir = os.path.join(data_dir,des_class, f'design{cnt}')
        apply(res_dir, data_dir, design, aig_dir, graphml_tool, target)
        cnt += 1
        
        