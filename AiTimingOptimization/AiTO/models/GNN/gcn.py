import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from util import *
from torch_geometric.data import Data
import os.path as osp
from torch_geometric.datasets import Planetoid, PPI


class GCN_NET(torch.nn.Module):

    def __init__(self, max_layer, feature_dim, hidden_dim, out_dim):
        self.hidden = []
        super(GCN_NET, self).__init__()
        # shape (feature_dim, hidden_dim)
        self.conv1 = GCNConv(feature_dim, hidden_dim, cached=True)
        for i in range(max_layer - 2):  # include input and output layer
            self.hidden.append(GCNConv(hidden_dim, hidden_dim, cached=True))
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=False)  # shape (hidden_dim, output)
        self.conv = GCNConv(feature_dim, out_dim)  # shape (hidden_dim, output)

        self.linear = torch.nn.Linear(out_dim, out_dim, bias = True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # print(x)
        for hid in self.hidden:
            x = F.relu(hid(x, edge_index))
            x = F.dropout(x, training=self.training)
        self.embedding = self.conv2(x, edge_index)
        # return x
        return F.log_softmax(self.embedding, dim=1)
        # return self.linear(self.embedding)
        # return self.embedding

class GCN(object):
    def __init__(self, max_layer, feature_dim, hidden_dim, out_dim):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # path = osp.join(osp.dirname(osp.realpath(__file__)),
        #                 '../../data', "Cora")
        # dataset = Planetoid(path, "Cora")
        # number_classes = dataset.num_classes

        # data = dataset[0].to(device)

        # self define data
        edges, features, topo_sort = init_data()
        data = Data(x=features, edge_index=edges.T)
        # graph_showing(data)

        self.topo_sort = topo_sort

        # GNN model
        self.max_layer = max_layer
        # input_dim = data.num_node_features
        self.input_dim = data.num_node_features
        # hid_dim = 64
        self.hid_dim = hidden_dim
        self.ouput_dim = out_dim

        self.model = GCN_NET(
            self.max_layer, self.input_dim, self.hid_dim, self.ouput_dim).to(device)
        print(self.model)
        self.data = data.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def next_state_transition(self):
        num_nodes = self.data.num_nodes
        topo_sort1 = ahead_one(self.topo_sort)
        one_one_map = zip(self.topo_sort, topo_sort1)
        topo_map = {i: j for i, j in one_one_map}
        index = torch.LongTensor([self.topo_sort, topo_sort1])
        val = torch.FloatTensor(np.ones(len(self.topo_sort)))
        next_state_transition = torch.sparse.FloatTensor(
        index, val, (num_nodes, num_nodes))
        return next_state_transition

    def train(self, pred):
        self.model.train()
        # pred = self.model(self.data)
        y = self.data.y
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

    def train(self):
        self.model.train()
        pred = self.model(self.data)
        y = self.data.y
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

    def embedding(self):
        self.model.train()
        self.optimizer.zero_grad()
        # with torch.no_grad():
        #     return self.model(self.data)
        return self.model(self.data)

    def accuracy_rate(self, pred):
        pred = np.argmax(pred.detach().numpy(), axis=1)
        acc = torch.from_numpy(pred).eq(self._label).sum().numpy()
        return acc

    def accuracy_rate(self):
        self.model.eval()
        pred = self.model(self.data)
        pred = pred.max(1)[1]
        acc = pred.eq(self.data.y).sum().item() / len(self.data.y)
        return acc
