import os
import pathlib
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors


DATA_FILE = 'C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\data\\packet_size_features_labels.csv'
data_df = pd.read_csv(DATA_FILE)
vecs = data_df.loc[:data_df.columns[1:10]].values
k = 5

nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(vecs)
distances, indices = nbrs.kneighbors(vecs)

G = nx.Graph()
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:
        G.add_edge(i, j)


edge_index = torch.tensor(list(G.edges)).t().contiguous()
x = torch.tensor(vecs, dtype=torch.float)


class GCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        return x


class GCNRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.squeeze()


mdl = GCNRegressor(in_channels=128, hidden_channels=64)
criterion = torch.nn.MSELoss()


optimizer = torch

