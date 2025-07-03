import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE
from util import *
from torch_geometric.data import Data


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # cached only for transductive learning
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VAE(object):
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        edges, features, labels, number_classes, idx_train, idx_val, idx_test = load_data()

        # path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
        # dataset = Planetoid(path, dataset)
        # number_classes = dataset.num_classes

        # data1 = dataset[0].to(device)

        # self define data
        data = Data(x=features, edge_index=edges.T, y=labels)

        # GNN model
        max_layer = 3
        input_dim = data.num_node_features
        hid_dim = 16
        ouput_dim = number_classes

        self.model = GAE(GCNEncoder(input_dim, ouput_dim).to(device))
        print(self.model)
        # self.model = GCN_NET(
        #     max_layer, input_dim, hid_dim, ouput_dim).to(device)
        self.data = data.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        # z = self.model.encode(x, train_pos_edge_index)
        z = self.model.encode(self.data.x, self.data.edge_index)
        # recon_loss
        loss = self.model.recon_loss(z, self.data.edge_index)
        # if args.variational:
        #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def embedding(self):
        with torch.no_grad:
            return self.model.encode(self.data.x, self.data.edge_index)

    def accuracy_rate(self):
        self.model.eval()
        pred = self.model.encode(self.data.x, self.data.edge_index)
        pred = pred.max(1)[1]
        acc = pred.eq(self.data.y).sum().item() / len(self.data.y)
        return acc
