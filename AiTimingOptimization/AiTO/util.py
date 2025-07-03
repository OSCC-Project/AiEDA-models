from cProfile import label
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from copy import deepcopy


def graph_showing(data):
    '''
    args:
         data: torch_geometric.data.Data
    '''
    G = nx.DiGraph()
    edge_index = data['edge_index'].t()
    edge_index = np.array(edge_index.cpu())
    # edge_index = edge_index[0:100]
    # print(edge_index)

    for edge in edge_index:
        if 633 in edge:
            print(edge)

    G.add_edges_from(edge_index)
    nx.draw(G, node_size=50, with_labels=True, font_size=5)
    plt.draw()  # pyplot draw()
    plt.savefig("graph.pdf")
    plt.show()


def interval_mapping(data, map_min, map_max):
    data_min = -1
    data_max = 1
    return list([map_min+((map_max - map_min) / (data_max - data_min)) * (i - data_min) for i in data])


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


# def load_data(path="./data/cora/", dataset="cora"):
#     print('Loading {} dataset...'.format(dataset))

#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])

#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)

#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     # features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))

#     idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # return adj, features, labels, idx_train, idx_val, idx_test


def load_data(path="./data/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int64).reshape(edges_unordered.shape)

    di_edges = edges[:, [1, 0]]
    edges = np.concatenate([edges, di_edges])
    edges = torch.from_numpy(edges)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)


    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    number_classes = labels.shape[1]
    labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return edges, features, labels, number_classes, idx_train, idx_val, idx_test

def init_data():
    idx_features_labels = np.genfromtxt("/home/wuhongxi/iEDA-test/iEDA/src/iTO/src/module/extractor/feature.txt",
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)
    # features = sp.csr_matrix(idx_features_labels[0:13, :], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("/home/wuhongxi/iEDA-test/iEDA/src/iTO/src/module/extractor/graph.txt",
                                    dtype=np.int32)
    edges = np.array(list(edges_unordered),
                     dtype=np.int64).reshape(edges_unordered.shape)


    #  for topological sorts
    DG = nx.DiGraph()
    DG.add_edges_from(edges)
    circle = list(nx.simple_cycles(DG))
    for c in circle:
        # if len(c) == 2:
        if DG.has_edge(c[0], c[1]):
            DG.remove_edge(c[0], c[1])
    circle = list(nx.simple_cycles(DG))

    topo_sort = list(nx.topological_sort(DG))

    # 变成无向图
    # di_edges = edges[:, [1, 0]]
    # edges = np.concatenate([edges, di_edges])
    # 自环
    # self_loop = np.array([[i, i] for i in np.arange(217)])
    # edges = np.concatenate([edges, self_loop])

    edges = torch.from_numpy(edges)

    # features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    return edges, features, topo_sort

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def ahead_one(topo_sort):
    topo_sort1 = deepcopy(topo_sort)
    topo_sort1.pop(0)
    topo_sort1.append(topo_sort1[-1])
    return topo_sort1