# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:42:25 2021

@author: prctha
"""

from torch.nn import Linear, TripletMarginLoss, Sigmoid, BCELoss, ModuleList
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import GCNConv, GraphConv, NNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import time
from pylab import zeros, arange, subplots, plt, savefig

import torch


# conv types with edges
# NNConv - node features and edge features in message passing funciton ### doesn't seem to update edge_features
# PNAConv - similar to above
# CGConv - similar to above


class GCNTriplet(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, nb_layers=3, convtype="GCNConv"):
        super(GCNTriplet, self).__init__()
        # torch.manual_seed(12345)
        self.nb_layers = nb_layers
        conv_layers = []
        for i in range(nb_layers):
            if i == 0:
                if convtype == "GCNConv":
                    conv_layers.append(GCNConv(dataset.num_node_features, hidden_channels))
                elif convtype == "GraphConv":
                    conv_layers.append(GraphConv(dataset.num_node_features, hidden_channels))
                elif convtype == "NNConv":
                    nn = Seq(
                        Lin(dataset.num_edge_features, hidden_channels),
                        ReLU(),
                        Lin(hidden_channels, dataset.num_node_features * hidden_channels)
                    )
                    conv_layers.append(NNConv(dataset.num_node_features,
                                              hidden_channels,
                                              nn=nn,
                                              aggr='mean'))
            elif i == nb_layers - 1:
                if convtype == "GCNConv":
                    conv_layers.append(GCNConv(hidden_channels, hidden_channels))
                elif convtype == "GraphConv":
                    conv_layers.append(GraphConv(hidden_channels, hidden_channels))
                elif convtype == "NNConv":
                    nn = Seq(
                        # doesn't seem to update edge_features
                        Lin(dataset.num_edge_features, hidden_channels),
                        ReLU(),
                        Lin(hidden_channels, hidden_channels * hidden_channels)
                    )
                    conv_layers.append(NNConv(hidden_channels,
                                              hidden_channels,
                                              nn=nn,
                                              aggr='mean'))
            else:
                if convtype == "GCNConv":
                    conv_layers.append(GCNConv(hidden_channels, hidden_channels))
                elif convtype == "GraphConv":
                    conv_layers.append(GraphConv(hidden_channels, hidden_channels))
                elif convtype == "NNConv":
                    nn = Seq(
                        Lin(dataset.num_edge_features, hidden_channels),
                        ReLU(),
                        Lin(hidden_channels, hidden_channels * hidden_channels)
                    )
                    conv_layers.append(NNConv(hidden_channels,
                                              hidden_channels,
                                              nn=nn,
                                              aggr='mean'))
        self.conv_layers = ModuleList(conv_layers)
        self.lin0 = Linear(hidden_channels, int(hidden_channels/2))
        if nb_layers == 0:
            # map from distance to "similarity score"
            self.lin = Linear(2*dataset.num_node_features, 1)
        else:
            # map from distance to "similarity score" ## reduced from lin0
            self.lin = Linear(hidden_channels, 1)
        self.sig = Sigmoid()
        self.convtype = convtype

    def conv_pass(self, x, edge_index, batch, edge_attr=None):

        for i in range(self.nb_layers):
            # 1. Obtain node embeddings
            if edge_attr == None:
                x = self.conv_layers[i](x, edge_index)
            else:
                x = self.conv_layers[i](x, edge_index, edge_attr)

            if i == self.nb_layers - 1:
                # 2. Readout layer
                x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                # 3.
                x = F.dropout(x, p=0.5, training=self.training)
            else:
                x = x.relu()

        if self.nb_layers == 0:
            x = global_add_pool(x, batch)  # mean

        x = self.lin0(x)
        return x

    def forward(self, **kwargs):

        if self.convtype in ['GCNConv', 'GraphConv']:  # node features only
            x0 = self.conv_pass(kwargs['x0'], kwargs['edge_index0'], kwargs['batch0'])
            x1 = self.conv_pass(kwargs['x1'], kwargs['edge_index1'], kwargs['batch1'])
            x2 = self.conv_pass(kwargs['x2'], kwargs['edge_index2'], kwargs['batch2'])
        elif self.convtype in ['NNConv']:  # node and edge features
            x0 = self.conv_pass(kwargs['x0'], kwargs['edge_index0'],
                                kwargs['batch0'], edge_attr=kwargs['edge_attr0'])
            x1 = self.conv_pass(kwargs['x1'], kwargs['edge_index1'],
                                kwargs['batch1'], edge_attr=kwargs['edge_attr1'])
            x2 = self.conv_pass(kwargs['x2'], kwargs['edge_index2'],
                                kwargs['batch2'], edge_attr=kwargs['edge_attr2'])
        # check is triplet is already correct (not used for loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32)  # .to(device)
        correct_score = torch.zeros([1], dtype=torch.int32)  # .to(device)
        dist_p = F.pairwise_distance(x0, x1)
        dist_n = F.pairwise_distance(x0, x2)

        y1 = torch.cat((x0, x1), 1)
        y2 = torch.cat((x0, x2), 1)

        y1 = self.lin(y1)
        y2 = self.lin(y2)
        score_p = self.sig(y1)
        score_n = self.sig(y2)

        for i in range(0, len(dist_p)):
            if (dist_n[i] - dist_p[i] > 0):  # mabye add margin of error
                correct[0] += 1

        for i in range(0, len(score_p)):
            if (score_p[i] - score_n[i] > 0):  # mabye add margin of error
                correct_score[0] += 1

        return x0, x1, x2, correct, score_p, score_n, correct_score
