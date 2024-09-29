# import os.path as osp
# import os
# from typing import List, Optional, Tuple, Union
# import json
# import time
from collections import defaultdict, Counter
import numpy as np
import copy
from copy import deepcopy
from sklearn.metrics import f1_score
from typing import Callable, List, Optional
import torch
import dionysus as dion
import networkx as nx
from torch import Tensor
from torch_geometric.nn import GINEConv, GATConv, GCNConv, NNConv, EdgeConv, SAGEConv, GINConv
from torch_geometric.utils import homophily, degree

from torch.nn import Embedding
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
# import torch_geometric.transforms as T
# from torch_geometric.typing import Adj, OptPairTensor, Size
# from torch_geometric.utils import negative_sampling
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader, NodeLoader, NeighborLoader, NeighborSampler
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import subgraph, bipartite_subgraph
# from torch_geometric.nn.conv import MessagePassing
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
#
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from .utils import (pout,
                    homophily_edge_labels,
                    edge_index_from_adjacency,
                    node_degree_statistics,
                    l2_normalize_features,
                    min_max_normalize_features,
                    mean_normalize_features,
                    standardize_features,
                    MaskedBatchNorm)
#profiling tools
# from guppy import hpy
# from memory_profiler import profile
# from memory_profiler import memory_usage
from typing import Union, List,Optional
from sklearn.metrics import accuracy_score
from sklearn import metrics
from torch.cuda.amp import autocast

from .conv import SAgeConv

from .experiments.metrics import optimal_metric_threshold

NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d,
    'MaskedBatchNorm': MaskedBatchNorm
}

class FeedForwardModule(nn.Module):
    def __init__(self, dim, dim_out, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim_out)
        self.dropout_2 = nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.linear_1.reset_parameters()
        self.linear_2.reset_parameters()
    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x
def gin_mlp_factory(gin_mlp_type: str, dim_in: int, dim_out: int):
    if gin_mlp_type == 'lin':
        return nn.Linear(dim_in, dim_out)

    elif gin_mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )

    elif gin_mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )
    elif gin_mlp_type == 'lin_bn_gelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.GELU(),
            nn.Linear(dim_in, dim_out)
        )
    elif gin_mlp_type == 'lin_gelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LayerNorm(dim_in),
            nn.GELU(),
            nn.Linear(dim_in, dim_out)
        )
    else:
        raise ValueError("Unknown gin_mlp_type!")

class combine():
    def __init__(self):
        a = 1
    def mean(self, tensor1, tensor2, padding_value=0, equivalent_nodes=None, device=None):

        # Determine the shape of the combined tensor
        max_rows = max(tensor1.size(0), tensor2.size(0))
        embedding_dim = tensor1.size(1)

        # Initialize the combined tensor with padding_value
        combined_tensor = torch.full((max_rows, embedding_dim), padding_value,dtype=torch.float).to(device)

        for i in range(tensor1.size(0)):
            if equivalent_nodes is not None:
                if i in equivalent_nodes:
                    j = equivalent_nodes[i]
                    # Combine embeddings (e.g., by averaging, concatenating, etc.)
                    combined_embedding = (tensor1[i] + tensor2[j]) / 2
                    combined_tensor[i] = combined_embedding
                else:
                    combined_tensor[i] = tensor1[i]
            else:
                # Combine embeddings (e.g., by averaging, concatenating, etc.)
                combined_embedding = (tensor1[i] + tensor2[i]) / 2
                combined_tensor[i] = combined_embedding

        for j in range(tensor2.size(0)):
            if equivalent_nodes is not None:
                if j not in equivalent_nodes.values():
                    combined_tensor[j] = tensor2[j]
            else:
                if j > tensor1.size(0):
                    combined_tensor[j] = tensor2[j]

        return combined_tensor
    def mean_comp(self, *tensors, padding_value=0, equivalent_nodes=None, device=None):

        # Determine the shape of the combined tensor

        max_rows = 0
        embedding_dim=0
        for tensor in tensors:
            m = tensor.size(0)
            if m > max_rows:
                max_rows=m
            embedding_dim = tensor.size(1)

        # Initialize the combined tensor with padding_value
        combined_tensor = torch.full((max_rows, embedding_dim), padding_value,dtype=torch.float).to(device)
        tensor_count = 0
        for tensor in tensors:
            tensor_count+=1
            for i in range(tensor.size(0)):
                if equivalent_nodes is not None:
                    if i in equivalent_nodes:
                        j = equivalent_nodes[i]
                        # Combine embeddings (e.g., by averaging, concatenating, etc.)
                        combined_embedding = (combined_tensor[i] + tensor[i])
                        combined_tensor[i] = combined_embedding
                    else:
                        combined_tensor[i] = tensor[i]
                else:
                    # Combine embeddings (e.g., by averaging, concatenating, etc.)
                    combined_embedding = (combined_tensor[i] + tensor[i])
                    combined_tensor[i] = combined_embedding
            #
        for i in range(combined_tensor.size(0)):
            if equivalent_nodes is not None:
                if i in equivalent_nodes:
                    j = equivalent_nodes[i]
                    # Combine embeddings (e.g., by averaging, concatenating, etc.)
                    combined_tensor[i] = combined_tensor[i] / tensor_count
                else:
                    combined_tensor[i] = combined_tensor[i]
            else:
                # Combine embeddings (e.g., by averaging, concatenating, etc.)
                combined_tensor[i] = combined_tensor[i] / tensor_count
            # for j in range(tensor2.size(0)):
            #     if equivalent_nodes is not None:
            #         if j not in equivalent_nodes.values():
            #             combined_tensor[j] = tensor2[j]
            #     else:
            #         if j > tensor1.size(0):
            #             combined_tensor[j] = tensor2[j]

        return combined_tensor

class DummyEdgeFilterFunction():
    def __init__(self):
        self.x = None
        self.data=None
        self.y=None
        self.dataset=None

    def assign_edge_filter_values(self, graph, split_values, split_percents):
        edge_weights_dict = {}
        num_edges = graph.edge_index.size(1)
        split_ranges = [percent*num_edges for percent in split_percents[:-1]]
        check_len = num_edges - split_ranges[-1]*num_edges
        last_value = split_ranges[-1]*num_edges + check_len
        split_ranges.append(last_value)
        weights = []
        for value, split_range in zip(split_values,split_ranges):
            weights += [value for i in np.arange(int(split_range))]

        graph.edge_weights = torch.from_numpy(np.array( weights ))
        graph.edge_attr = torch.from_numpy(np.array( weights ))
        print(graph.edge_weights)
        return graph

class SubGraphFilterConv(nn.Module):
    def  __init__(self, in_dim,
                 dim_hidden,
                 out_dim,
                 num_layers,
                 dropout=0,
                 use_batch_norm = False,
                 normalization=None):

        super().__init__()

        self.num_feats = in_dim
        self.dim_hidden = dim_hidden
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.normalization = normalization

        self.model_nbr_msg_aggr = torch.nn.ModuleList()
        # GraphSage, ego- neighborhood embedding seperation performs better
        self.model_nbr_msg_aggr.append(SAGEConv(in_channels=self.num_feats,
                                          out_channels=self.dim_hidden))
        # self.bns.append(nn.BatchNorm1d(n_2)) self.bns = nn.ModuleList()

        self.batch_norms = []
        batch_norm_layer = normalization(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        for _ in range(self.num_layers - 2):
            self.model_nbr_msg_aggr.append(SAGEConv(in_channels=self.dim_hidden,
                                              out_channels=self.dim_hidden))
            # batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            batch_norm_layer = normalization(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)
        self.model_nbr_msg_aggr.append(SAGEConv(in_channels=self.dim_hidden,
                                                out_channels=self.dim_hidden))
        batch_norm_layer = normalization(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        # self.linear_in = FeedForwardModule(dim=self.num_feats,
        #                                     dim_out=self.dim_hidden,
        #                                     input_dim_multiplier=1,
        #                                     hidden_dim_multiplier=1,
        #                                     dropout=self.dropout)
        self.linear_out = FeedForwardModule(dim=self.dim_hidden,
                                            dim_out=self.num_feats,
                                            input_dim_multiplier=1,
                                            hidden_dim_multiplier=1,
                                            dropout=self.dropout)

        self.normalization_out = normalization(self.num_feats) if self.use_batch_norm else nn.Identity(self.num_feats)

        self.dropout = nn.Dropout(self.dropout)
        self.act = nn.GELU()
        # self.jump = JumpingKnowledge(mode='cat')
        # self.probability = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)  # nn.Softmax(dim=1) #nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        for embedding_layer in self.model_nbr_msg_aggr:
            embedding_layer.reset_parameters()
        self.linear_out.reset_parameters()

    def set_device(self, device):
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]
        self.to(device)

    def forward(self, x, adjs, filtration_function_in, filtration_function_hidden, filtration_function_out, single_sample=False):

        num_targets = 0
        edge_index = adjs[0][0]

        # x = self.linear_in(x)
        #
        # x = self.normalization_in(x)

        x_residual = x

        for i, (edge_index, _, size) in enumerate(adjs):

            num_targets = size[1]
            edge_index = edge_index

            x = self.model_nbr_msg_aggr[i](x, edge_index)
            if i!=self.num_layers-1:
                x = self.act(x)
                x = self.dropout(x)
            x = self.batch_norms[i](x)

        x = self.linear_out(x)

        x = self.normalization_out(x)

        x = x + x_residual

        x , filtration_value = filtration_function_out(x=x, #(x_source, x[: num_targets]),
                                                          x_target=x[: num_targets],
                                                          edge_index=edge_index,
                                                          degree=None,
                                                          single_sample=single_sample)


        # scale new embeddings by filtration coefficient
        x = filtration_value * x
        return x, filtration_value


class FiltrationGraphHierarchy():
    def __init__(self,
                 graph,
                 persistence: Optional[Union[float, List[float]]],
                 filtration = None):

        self.graph = graph
        self.thresholds = persistence
        self.filtration = filtration
        self.num_classes = len(graph.y.unique())
        num_targets = 1 if self.num_classes == 2 else self.num_classes

        pout((" %%%%% %%%% %%%% %%%% %%% %%%% ", "PERFORMING FILTRATION OF GRAPH"))
        self.graphs, self.supergraph_to_subgraph_mappings, self.sub_to_sup_mappings = self.process_graph_filtrations(data=self.graph,
                                                                         thresholds=self.thresholds,
                                                                         )

        # pout(("Total graphs: ", len(self.graphs), "%%%%%%%"))
        self.graph_levels = np.flip(np.arange(len(self.graphs) + 1))
        graph = self.graphs[0]
        # self.graphlevelloader = self.get_graph_dataloader(graph,
        #                                                   shuffle=True,
        #                                                   num_neighbors=self.num_neighbors[:self.num_layers])
        self.graph_level = 0

        for graph_num, g in enumerate(self.graphs):
            pout(("GRAPH NUMBER ", graph_num))
            pout(("GRAPH NUMBER NODES ", g.num_nodes))
            self.graph_statistics(g)

    def graph_statistics(self, graph):
        edge_homophily = homophily(graph.edge_index, graph.y, method='edge')
        node_homophily = homophily(graph.edge_index, graph.y, method='node')
        # edge_insensitive_homophily = homophily(graph.edge_index, graph.y, method='edge_insensitive')
        pout(("node_homophily ", node_homophily))
        pout(("edge_homophily ",edge_homophily))
        # graph.edge_homophily = edge_homophily
        # graph.node_homophily = node_homophily
        #
        # pout(("edge_insensitive_homophily ",edge_insensitive_homophily))
    def expand_labels(self, labels):
        neg_labels = ~labels
        labels = labels.type(torch.FloatTensor)  # labels.type(torch.FloatTensor)
        neg_labels = neg_labels.type(torch.FloatTensor)
        labels = [neg_labels, labels]
        # if not as_logit:
        return torch.stack(labels, dim=1)

    # def get_graph_dataloader(self, graph, shuffle=True, num_neighbors=[-1]):
    #     # pout(("graph data edge index",graph.edge_index))
    #     # Compute the degree of each node
    #     if num_neighbors==[-1]:
    #         max_degree, avg_degree, degrees = node_degree_statistics(graph)
    #         # pout((" MAX DEGREE ", max_degree," AVERAGE DEGREE ", avg_degree))
    #         if avg_degree > 25:
    #             self.val_batch_size = 8
    #         else:
    #             self.val_batch_size = self.batch_size
    #         batch_size = self.val_batch_size
    #         pout(("NOW USING NEW VALIDATION BATCH SIZE AND NUMBER NEIGHBORS"))
    #         pout(("VAL BATCH_SIZE", self.val_batch_size))
    #     else:
    #         batch_size = self.batch_size
    #
    #     if graph.num_nodes < 1200:
    #         num_workers=1
    #     else:
    #         num_workers=4
    #
    #     if shuffle:
    #         neighborloader = NeighborLoader(data=graph,
    #                               batch_size=self.batch_size,
    #                               num_neighbors=self.num_neighbors[: self.num_layers],
    #                                         subgraph_type='induced', #for undirected graph
    #                               # directed=False,#True,
    #                               shuffle=shuffle,
    #                             num_workers=num_workers
    #                                         ) #for graph in self.graphs]
    #         neighborsampler = NeighborSampler(
    #             graph.edge_index,
    #             # node_idx=train_idx,
    #             # directed=False,
    #             sizes=self.num_neighbors[: self.num_layers],
    #             batch_size=self.batch_size,
    #             # subgraph_type='induced',  # for undirected graph
    #             # directed=False,#True,
    #             shuffle=shuffle,
    #             num_workers=num_workers
    #         )
    #     else:
    #         neighborloader = NeighborLoader(data=graph,
    #                               batch_size=batch_size,
    #                               num_neighbors=num_neighbors,
    #                                         subgraph_type='induced', #for undirected graph
    #                               # directed=False,#True,
    #                               shuffle=shuffle
    #                                         # num_workers=8
    #                                         ) #for graph in self.graphs]
    #
    #         neighborsampler = NeighborSampler(
    #             graph.edge_index,
    #             # node_idx=train_idx,
    #             # directed=False,
    #             sizes=self.num_neighbors[: self.num_layers],
    #             batch_size=batch_size,
    #             # subgraph_type='induced',  # for undirected graph
    #             # directed=False,#True,
    #             shuffle=shuffle,
    #             # num_workers=8,
    #         )
    #     return neighborsampler

    def initialize_from_subgraph(self, subgraph, supergraph, graph_level, node_mappings):
        pout(("graph level ", graph_level, "graph levelS ", self.graph_levels, " node mappings length ",
              len(node_mappings)))
        subgraph_mapping = node_mappings[0]
        supgraph_mapping = node_mappings[1]
        supsubsub_mapping = {global_id: sub_id for sub_id, global_id in subgraph_mapping.items()}
        supsub_mapping = {global_id: sub_id for sub_id, global_id in supgraph_mapping.items()}

        # for node, i in subgraph_mapping.items():

        new_node_features = supergraph.x.clone().detach().cpu().numpy()
        for node, embedding in subgraph.items():
            global_id = supsubsub_mapping[node]
            new_node_features[supgraph_mapping[global_id]] = embedding  # supsub_mapping[global_id]] = embedding
        supergraph.x = torch.tensor(new_node_features)
        return supergraph

    def pyg_to_dionysus(self, data):
        # Extract edge list and weights
        data = data.clone()
        edge_index = data.edge_index.t().cpu().numpy()
        edge_weight = data.edge_weights.cpu().numpy()
        filtration = dion.Filtration()
        for i, (u, v) in enumerate(edge_index):
            filtration.append(dion.Simplex([u, v], edge_weight[i]))
        filtration.sort()
        return filtration

    def filtration_to_networkx(self, filtration, data, clone_data=False, node_mapping=True):
        # if clone_data:
        # data = copy.copy(data)#.copy()
        data = data.clone()
        node_mapping_true = node_mapping

        edge_emb = data.edge_attr.cpu().numpy()
        y = data.y.cpu().numpy()
        G = nx.Graph()
        for simplex in filtration:
            u, v = simplex
            G.add_edge(u, v, weight=simplex.data, embedding=edge_emb[simplex])
            G.nodes[u]['y'] = y[u]  # , features=data.x[u])#.tolist() if data.x is not None else {})
            G.nodes[v]['y'] = y[v]
            G.nodes[u]['features'] = data.x[u].tolist() if data.x is not None else {}
            G.nodes[v]['features'] = data.x[v].tolist() if data.x is not None else {}

        # if node_mapping_true:
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        return G, node_mapping

    def create_filtered_graphs(self, filtration, thresholds, data, clone_data=False, nid_mapping=None):
        # if clone_data:
        # data = copy.copy(data)#.copy()
        data = data.clone()



        node_mappings = []
        graphs = []

        for threshold in thresholds:
            G = nx.Graph()
            edge_emb = data.edge_attr.cpu().numpy()
            y = data.y.cpu().numpy()
            for simplex in filtration:
                if simplex.data >= threshold:
                    # copy simplex for new simplicial level
                    u, v = simplex
                    # add edge, incident nodes, labels, and features to subgraph
                    G.add_edge(u, v, weight=simplex.data, embedding=edge_emb[simplex])
                    G.nodes[u]['y'] = y[u]  # , features=data.x[u])#.tolist() if data.x is not None else {})
                    G.nodes[v]['y'] = y[v]
                    G.nodes[u]['features'] = data.x[u].tolist() if data.x is not None else {}
                    G.nodes[v]['features'] = data.x[v].tolist() if data.x is not None else {}
                # else:
                #     u, v = simplex
                #     # add edge, incident nodes, labels, and features to subgraph
                #     G.add_edge(u, v, weight=simplex.data, embedding=edge_emb[simplex])
                #     G.nodes[u]['y'] = y[u]  # , features=data.x[u])#.tolist() if data.x is not None else {})
                #     G.nodes[v]['y'] = y[v]
                #     G.nodes[u]['features'] = data.x[u].tolist() if data.x is not None else {}
                #     G.nodes[v]['features'] = data.x[v].tolist() if data.x is not None else {}
            graphs.append(G)

            # if nid_mapping is None:
            # if threshold != 0.0:
            node_mapping = {node: i for i, node in enumerate(G.nodes())}#np.sort(G.nodes()))}

            # else:
            #     node_mapping = {node:node for node in np.sort(G.nodes())} # must also do sup to int bc train/val/test splits
            # if threshold == 0:
            #     pout(("SANITY CHECK FOR NODE MAPPINGS ", "CHECKING IF GLOBAL ID MAP EQUAL TO ITER"))
            #     pout(("IS EQUAL: ",node_mapping.values() == node_mapping.keys()))
            #     pout(("Keys sample: ", [k for k in node_mapping.keys()][:10]))
            #     pout(("values sample: ", [k for k in node_mapping.values()][:10]))
            # else:
            #     node_mapping = {node: nid_mapping[node] for i, node in enumerate(G.nodes())}
            node_mappings.append(node_mapping)
        return graphs, node_mappings

    def pyg_to_networkx(self, data, clone_data=False):
        # if clone_data:
        data = data.clone()
        # Initialize a directed or undirected graph based on your need
        G = nx.DiGraph() if data.is_directed() else nx.Graph()

        # Add nodes along with node features if available
        for i in range(data.num_nodes):
            node_features = data.x[i].tolist() if data.x is not None else {}
            G.add_node(i, features=node_features)

        # Add edges along with edge attributes if available
        edge_index = data.edge_index.t().cpu().numpy()
        if data.edge_attr is not None:
            edge_attributes = data.edge_attr.cpu().numpy()
            for idx, (source, target) in enumerate(edge_index):
                G.add_edge(source, target, weight=edge_attributes[idx])
        else:
            for source, target in edge_index:
                G.add_edge(source, target)

        return G

    def nx_to_pyg(self, graph, supergraph_to_subgraph_mapping, graph_level=None):
        graph = copy.copy(graph)

        target_type = torch.long if self.num_classes > 2 else torch.float
        # Mapping nodes to contiguous integers
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        int_mapping = {v: u for u, v in node_mapping.items()}
        # Convert edges to tensor format
        edge_list = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges], dtype=torch.float)
        # edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.nodes(data=True)], dtype=torch.float)

        num_nodes = graph.number_of_nodes()
        node_features = torch.tensor([attr['features'] for node, attr in graph.nodes(data=True)], dtype=torch.float)
        y = torch.tensor([attr['y'] for node, attr in graph.nodes(data=True)], dtype=target_type)
        # node_embeddings = torch.tensor([graph.nodes[i]['embeddings'] for i in range(num_nodes)], dtype=torch.float)

        # x = torch.tensor([graph[u]['features'] for u in graph.nodes], dtype=torch.float)
        # y = torch.tensor([graph[u]['y'] for u in graph.nodes], dtype=torch.float)
        # edge_emb = torch.tensor([graph[u]['embedding'] for u in graph.nodes], dtype=torch.float)

        data = Data(x=node_features,
                    edge_index=edge_index,
                    y=y,
                    edge_attr=edge_attr,
                    # edge_embedding=edge_emb,
                    num_nodes=int(graph.number_of_nodes()))

        n_ids = torch.arange(int(graph.number_of_nodes()))
        filter_values = {n_id: float(1.0) for i, n_id in enumerate(n_ids)}
        data.node_filter_values = filter_values

        return data

    #
    #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # from pytorch_geometric.utils import subgraph
    # subgraph(sub_nid, sub_edge_index, sub_edge_attr
    def process_graph_filtrations(self, thresholds, data=None, filtration=None):
        # Convert PyG data to Dionysus filtration
        if data is not None and filtration is None:
            filtration = self.pyg_to_dionysus(data)
        filtration = self.filtration if filtration is None else filtration

        # full graph
        full_graph, full_node_mapping = self.filtration_to_networkx(filtration=filtration,
                                                               data=data,
                                                               clone_data=True,
                                                               node_mapping=True)

        # sort in decending order to high homopholous graphs first
        # if 0.0 not in thresholds:
        #     thresholds.append(0.0)
        thresholds = np.flip(np.sort(thresholds))
        # Create filtered graphs
        filtered_graphs, sup_to_sub_idx_mappings = self.create_filtered_graphs(filtration=filtration,
                                                                     thresholds=thresholds,
                                                                     data=data,
                                                                     clone_data=True)
        # nid_mapping=node_mapping)

        filtered_graphs.append(full_graph)
        sup_to_sub_idx_mappings.append(full_node_mapping)

        for m in sup_to_sub_idx_mappings:
            pout(("subgraph map length ", len(m)))
        # Convert back to PyG data objects
        pyg_graphs = [self.nx_to_pyg(graph,
                                     supergraph_to_subgraph_mapping=sup_to_sub_idx_mappings[i])
                      for i, graph in enumerate(filtered_graphs)]
        sub_to_sup_mappings = []
        for subidx_map in sup_to_sub_idx_mappings:
            sub_to_sup_mappings.append({sub_id: global_id for global_id,sub_id in subidx_map.items()})

        return pyg_graphs, sup_to_sub_idx_mappings, sub_to_sup_mappings



####################################################
#
#
class SubGraphFilterFunction(torch.nn.Module):
    r"""
    A node level learnable filtration function used for Hierarchical Joint Training
    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~HierGNN.SubLevelGraphFiltration` takes in a subgraph
    :obj:`dataset` and uses a graph isomorphism network GIN-e of type :obj:`gin_mlp_type'
    to produce a node embedding of dimension :obj:`gin_dimension' with learnable
    epsilon parameter as discussed in "How Powerful are Graph Neural Nets" Xu et al. The node embedding is then
    passed to :obj:`gin_mlp_type' multi-layer perceptron with output dimension 1.
    This output is then a lernable factor to multiply the final subgraophs node embedding by.
    """
    def __init__(self,
                 max_node_deg,
                 dim_in,
                 dim_out,
                 eps: float = 0.5, # we initialize epsilon based on
                 # number of graph levels so all initially contribute equally
                 cat_seperate=False,
                 use_node_degree=None,
                 set_node_degree_uninformative=None,
                 use_node_feat=None,
                 gin_number=None,
                 gin_dimension=None,
                 gin_mlp_type=None,
                 use_batch_norm=True,
                 normalization=None,
                 **kwargs
                 ):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        self.normalization = normalization

        dim = gin_dimension

        max_node_deg = max_node_deg
        num_node_feat = dim_in

        # if set_node_degree_uninformative and use_node_degree:
        #     self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        # if use_node_degree:
        #     self.embed_deg = nn.Embedding(max_node_deg + 1, dim)
        # else:
        #     self.embed_deg = None

        # self.embed_feat = nn.Embedding(num_node_feat, dim, dtype=torch.float) if use_node_feat else None

        # self.embed_feat= nn.Embedding.from_pretrained(data.x,
        #                                                     freeze=False).requires_grad_(True) if use_node_feat else None
        # self.edge_embeddings.weight.data.copy_(train_data.edge_attr)
        # self.edge_embeddings.weight.requires_grad = True
        # self.edge_embeddings#.to(device)

        #dim_input = dim * ((self.embed_deg is not None) + (self.embed_feat is not None))
        self.cat = 2 if cat_seperate else 1
        dims = [self.cat * dim_in] + (gin_number) * [dim]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.GELU() # torch.nn.functional.leaky_relu
        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            # pout(("n1 ", n_1, " n2 ",n_2))
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, eps=eps,train_eps=True))
            batch_norm_layer = normalization(n_2) if self.use_batch_norm else nn.Identity(n_2)
            self.bns.append(batch_norm_layer)

        b1 = normalization(dim) if self.use_batch_norm else nn.Identity(dim)
        b2 = normalization(dim_out) if self.use_batch_norm else nn.Identity(dim_out)

        self.fc = nn.Sequential(
            nn.Linear(sum(dims), dim),
            # nn.Linear(dim_out, dim),
            b1,#nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim_out),
            b2,#nn.BatchNorm1d(dim_in),
            nn.GELU(),
            nn.Linear(dim_out, 1),
            nn.Sigmoid()
        )
        self.reset_parameters()
        self.reset_model_parameters(self.fc)
        self.reset_model_parameters(self)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def reset_model_parameters(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def set_device(self, device):
        self.bns.to(device)
        self.convs.to(device)
        self.fc.to(device)
        self.to(device)

    def forward(self, x, x_target, edge_index, degree=None, single_sample = False, seperate=False ):

        node_deg = degree#batch.node_deg
        # node_feat = batch.node_lab

        edge_index = edge_index # batch.edge_index

        # tmp = [e(n_x) for e, n_x in
        #        zip([self.embed_deg, self.embed_feat], [node_deg, node_idx])
        #        if e is not None]
        #
        # tmp = torch.cat(tmp, dim=1)
        # # tmp = torch.cat(batch,dim=1)
        # z = [tmp]
        if self.cat == 2:
            tmp = [x,x_target]
            tmp = torch.cat(tmp,dim=1)
        else:
            tmp = x

        # out = []
        # if isinstance(x, Tensor):
        #     z = [x_target]
        # else:
        #     z = [x[1]]#[x]
        z=[x_target]
        for conv, bn in zip(self.convs, self.bns):
            x = conv((x,x_target),edge_index)#z[-1], edge_index)
            # if not single_sample:
            x = bn(x)
            x = self.act(x)
            z.append(x)
        # x = z[-1]
        filt_feat_val = z[-1]
        # x = self.global_pool_fn(x, batch.batch)
        # z=z[1:]
        # z = [z[0][1]]+z[1:]
        x_cat = torch.cat(z, dim=1)
        ret = self.fc(x_cat)
        # ret = ret.squeeze()
        return x, ret#.view(-1,1)


class MultiScaleGraphFiltrationSampler:
    def __init__(self,
                 super_data,
                 subset_data,
                 subset_samplers,
                 subset_to_super_mapping,
                 super_to_subset_mapping,
                 batch_size,
                 shuffle=False):
        self.super_data = super_data
        self.subset_data = subset_data
        self.subset_samplers = subset_samplers
        self.subset_to_super_mapping = subset_to_super_mapping
        self.super_to_subset_mapping = super_to_subset_mapping #{super_idx : sub_idx for sub_idx, super_idx in subset_to_super_mapping.items()}
        self.batch_size = batch_size
        self.num_nodes = super_data.num_nodes
        self.current_idx = 0
        self.shuffle = shuffle

        if self.shuffle:
            self.node_indices = torch.randperm(self.num_nodes)  # Randomly permute the node indices
        else:
            self.node_indices = torch.arange(self.num_nodes)  # Consecutive node indices
    

    def __iter__(self):
        self.current_idx = 0  # Reset the index for a new iteration
        if self.shuffle:
            self.node_indices = torch.randperm(self.num_nodes)  # Shuffle for each epoch
        else:
            # Sample nodes from the super-graph without shuffling
            self.node_indices = torch.arange(self.num_nodes)
        return self
    def __next__(self):
        if self.current_idx >= self.num_nodes:
            raise StopIteration
        # pout(("current idx ", self.current_idx))
        end_idx = min(self.current_idx + self.batch_size, self.num_nodes)
        sampled_indices = self.node_indices[self.current_idx:end_idx]
        self.current_idx = end_idx

        global_super_indices = [self.subset_to_super_mapping[-1][i] for i in sampled_indices.tolist()]
        global_super_indices_sorted = np.sort(global_super_indices)
        # pout(("end idx ", self.current_idx))

        # # Get valid nodes for each subset graph

        valid_subset_gids = []
        for i,mapping in enumerate(self.super_to_subset_mapping[:-1]):
            valid_nodes = [n for n in global_super_indices_sorted if n in mapping.keys()]
            # valid_global_nodes =np.sort([mapping[n] for n in valid_nodes])
            valid_subset_gids.append(valid_nodes)
            #valid_nodes_sub_idx = [self.super_to_subset_mapping[i][n] for n in valid_nodes]
            #valid_subset_nodes.append(torch.tensor(valid_nodes_sub_idx))
            """ Should I resample subgraph nodes to ensure same sized batches, or add dummy nodes?"""
        valid_subset_gids.append(global_super_indices_sorted)

        #sort nodes so self same nodes are in the same index
        valid_subset_gids_aligned = self.align_multiple_lists(*valid_subset_gids)

        valid_subset_nodes = []
        multilevel_sizes = []
        for i, valid_sub_gid in enumerate(valid_subset_gids_aligned):
            valid_subset_nodes.append([self.super_to_subset_mapping[i][nid] for nid in valid_sub_gid])


        # # Get valid nodes for each subset graph
        # valid_subset_nodes = []
        # valid_nodes_sub_idx = []
        # for i,mapping in enumerate(self.super_to_subset_mapping):
        #     valid_nodes = [n for n in valid_global_indices if n in mapping.keys()]
        #     # valid_nodes_sorted = np.sort(valid_nodes)
        #     valid_nodes_sub_idx = [mapping[n] for n in valid_nodes]
        #     valid_nodes_sub_idx_sorted = np.sort(valid_nodes_sub_idx)
        #     valid_subset_nodes.append(valid_nodes_sub_idx_sorted)
        #     multilevel_sizes.append(len(valid_nodes_sub_idx_sorted))
        #     pout((f"valid subset sorted {valid_nodes_sub_idx}"))
            # pout((f"valid nodes {valid_nodes}",f"valid ndoes sub idx {valid_nodes_sub_idx} valid nodes size {len(valid_nodes_sub_idx)}"))
            # pout((f"valid subset nodes {valid_subset_nodes}"))
        # Custom neighborhood sampling for each subset graph
        super_nodes_sorted = valid_subset_nodes[-1]
        # subset_samples = []
        # for i, data in enumerate(self.subset_data):
        #     if len(valid_subset_nodes[i]) > 0:  # Ensure there are valid nodes to sample
        #         sub_edge_index, sub_n_id, sub_mask = bipartite_subgraph(subset=valid_subset_nodes[i],
        #                                             edge_index=data.edge_index,
        #                                             edge_attr=data.edge_attr,
        #                                             relabel_nodes=True,
        #                                             return_edge_mask=True)
        #         mask = torch.logical_or(torch.isin(data.edge_index[0], valid_subset_nodes[i]),
        #                                 torch.isin(data.edge_index[1], valid_subset_nodes[i]))
        #         filtered_edge_index = data.edge_index[:,mask]
        #         # Map the original indices to consecutive indices starting from zero
        #         unique_nodes = torch.unique(filtered_edge_index)
        #         node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}
        #         # Relabel the filtered edge_index
        #         relabel_edge_index = torch.tensor([
        #             [node_mapping[node.item()] for node in filtered_edge_index[0]],
        #             [node_mapping[node.item()] for node in filtered_edge_index[1]]
        #         ], dtype=torch.long)
        #
                #         pout((f"sub edge index {sub_edge_index} sub edge index size {sub_edge_index.size()} sub_n_id {sub_n_id} sub nid size {sub_n_id.size()}"))
        #         pout((f" sub edge index attempt 2 {filtered_edge_index}"))
        #         pout((f"sub 2 size {filtered_edge_index.size()}"))
        #         pout((f"relabel_edge_index {relabel_edge_index} relabeled size {relabel_edge_index.size()}"))
        #         match = ""
        #         for n in valid_nodes_sub_idx:
        #             match += str(n in sub_edge_index[0] or n in sub_edge_index[1]) + " ... "
        #         pout(("nodes return are in edge index "+ match))
        #         subset_samples.append((valid_subset_nodes[i].size(0), sub_n_id, [sub_edge_index]))
        #     else:
        #         subset_samples.append((0, torch.tensor([]), []))  # Handle empty batches

        # # Get neighborhood information from each subset graph

        #  Note: `batch` returns a tuple (node, row, col, batch_size, n_id, e_id)
        #                                 batch_size, n_id, adjs, edge_index, thing, size
        #                                 batch_size, n_id, edge_index, thing, size
        # supergraph_samples = self.subset_samplers[-1].sample(valid_subset_nodes[-1])
        #
        # pout((f"sample samples length {len(supergraph_samples)}"))
        # pout(supergraph_samples)
        # pout(("type elm",[type(elm) for elm in supergraph_samples]))
        # pout(("adj last elm in sample", supergraph_samples[-1]))
        # pout(("last elm length",len(supergraph_samples[-1])))
        #
        # super_batch_size, super_n_id, super_adjs = supergraph_samples

        # supergraph_nodes_sorted, supergraph_indices_sorted  = torch.sort(super_n_id)
        # supergraph_samples_sorted= self.subset_samplers[-1].sample(supergraph_indices_sorted)#.tolist())

        # supergraph_adj = []
        # for super_edge_index, super_thing, super_size in super_adjs:
        #     supergraph_sampled_edge_index = super_edge_index
        #     supergraph_edge_index_remapped = self.remap_subgraph_edge_index(supergraph_sampled_edge_index,
        #                                                                     supergraph_sampled_nodes,
        #                                                                     supergraph_samples_sorted)
        #     super_adj_layer = (supergraph_edge_index_remapped, super_thing, super_size)
        #     supergraph_adj.append(super_adj_layer)

        # global_sampled_nids_sorted = [self.subset_to_super_mapping[-1][nid] for nid in valid_subset_nodes[-1]]#supergraph_indices_sorted.tolist()]

        # supergraph_samples = (super_batch_size, supergraph_samples_sorted, supergraph_adj)

        subset_samples = []
        sampled_indices_ordered = []
        for i, sampler in enumerate(self.subset_samplers):
            if len(valid_subset_nodes[i]) > 0:  # Ensure there are valid nodes to sample .sample_from_nodes()
                # subgraph_samples = sampler.collate_fn(valid_subset_nodes[i])
                # global_subgraph_indices = [self.subset_to_super_mapping[i][nid] for nid in valid_subset_nodes[i]]
                # global_subgraph_indices_sorted = np.sort(global_super_indices)
                # # valid_global_indices = [n for n in global_super_indices_sorted if
                # #                         n in self.super_to_subset_mapping[-1].keys()]
                #
                # # if nid in self.subset_to_super_mapping[i].keys()]
                # _, global_nid_subgraph = self.align_lists(global_super_indices_sorted, global_subgraph_indices)
                # valid_subset_nodes_i = [self.super_to_subset_mapping[i][nid] for nid in global_nid_subgraph]

                subgraph_samples = sampler.sample(valid_subset_nodes[i])
                # sub_batch_size, sub_n_id, sub_adjs = subgraph_samples
                # subgraph_sampled_nodes = sub_n_id

                # subgraph_samples_global_order = [self.super_to_subset_mapping[i][gid]\
                #                                  for gid in global_sampled_nids_sorted]
                # #torch.sort(subgraph_samples.n_id)
                # subgraph_nids_ordered = subgraph_samples_global_order + [nid for nid in subgraph_sampled_nodes\
                #                                                             if nid not in subgraph_samples_global_order]
                #
                # subgraph_samples_ordered = sampler.sample(subgraph_nids_ordered)
                # subgraph_adj = []
                # for sub_edge_index, sub_thing, sub_size in sub_adjs:
                #     subgraph_sampled_edge_index = sub_edge_index
                #     subgraph_edge_index_remapped = self.remap_subgraph_edge_index(subgraph_sampled_edge_index,
                #                                                                     subgraph_sampled_nodes,
                #                                                                     subgraph_samples_ordered)
                #     sub_adj_layer = (subgraph_edge_index_remapped, sub_thing, sub_size)
                #     subgraph_adj.append(sub_adj_layer)
                #
                # subgraph_samples = (sub_batch_size, subgraph_samples_ordered, subgraph_adj)

                subset_samples.append(subgraph_samples)#_ordered)
                sampled_indices_ordered.append(valid_subset_nodes[i])#subgraph_nids_ordered)
                multilevel_sizes.append(len(valid_subset_nodes))
            else:
                subset_samples.append((0, torch.tensor([]), []))  # Handle empty batches
                multilevel_sizes.append(0)


        # subset_samples.append(supergraph_samples)

        return sampled_indices_ordered, subset_samples, multilevel_sizes



    def align_multiple_lists(self, *lists):
        # Count the frequency of each element across all lists
        element_count = Counter()
        for lst in lists:
            element_count.update(set(lst))

        # Sort elements by their frequency in descending order
        sorted_elements = sorted(element_count, key=lambda x: -element_count[x])

        # Initialize dictionaries to keep track of elements that have been placed
        placed_elements = {i: [] for i in range(len(lists))}
        remaining_elements = {i: [] for i in range(len(lists))}

        # Place common elements at the start of each list in the same order
        for element in sorted_elements:
            for i, lst in enumerate(lists):
                if element in lst:
                    placed_elements[i].append(element)
                else:
                    remaining_elements[i].append(element)

        # Combine placed elements and remaining elements
        aligned_lists = []
        for i in range(len(lists)):
            # Add elements that were in the original list but not in sorted_elements
            remaining_elements[i] = [x for x in lists[i] if x not in placed_elements[i]]
            aligned_list = placed_elements[i] + remaining_elements[i]
            aligned_lists.append(aligned_list)

        return aligned_lists

    # Function to remap edge_index based on sorted nodes
    def remap_subgraph_edge_index(self, edge_index, original_nodes, sorted_nodes):
        mapping = {original_nodes[i].item(): sorted_nodes[i].item() for i in range(len(original_nodes))}
        remapped_edge_index = torch.zeros_like(edge_index)
        remapped_edge_index[0] = torch.tensor([mapping[node.item()] for node in edge_index[0]], dtype=torch.long)
        remapped_edge_index[1] = torch.tensor([mapping[node.item()] for node in edge_index[1]], dtype=torch.long)
        return remapped_edge_index

    # def remap_edge_index(self, sampler, adjs, sorted_nodes):
    #     remapped_adjs = []
    #     n_id = sorted_nodes
    #     for edge_index, e_id, size in adjs:
    #
    #         adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
    #         e_id = adj_t.storage.value()
    #         size = adj_t.sparse_sizes()[::-1]
    #         if self.__val__ is not None:
    #             adj_t.set_value_(self.__val__[e_id], layout='coo')
    #
    #         if self.is_sparse_tensor:
    #             adjs.append(Adj(adj_t, e_id, size))
    #         else:
    #             row, col, _ = adj_t.coo()
    #             edge_index = torch.stack([col, row], dim=0)
    #             adjs.append(EdgeIndex(edge_index, e_id, size))
    #
    #     adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]

class HierSGNN(torch.nn.Module):
    #fp = open('./run_logs/memory_profiler.log', 'w+')
    # @profile#stream=fp)
    def __init__(self,
                 args,
                 data,
                 # split_masks,
                 processed_dir,
                 out_dim = 1,
                 train_data=None,
                 test_data = None,
                 # in_channels,
                 # hidden_channels,
                 # out_channels
                 experiment=None,
                 exp_input_dict=None
                 ):
        super().__init__()

        self.experiment = experiment
        self.exp_input_dict = exp_input_dict
        if experiment is not None:
            if experiment == "seq_init_ablation":
                self.init_type = "seq_init"
                self.experimental_results = []
            if experiment == "fixed_init_ablation":
                self.init_type = "fixed_init"
                self.experimental_results = []

        # train_idx = split_masks["train"]
        # train_idx = split_masks["train"]
        # self.split_masks = split_masks
        self.save_dir = processed_dir

        self.type_model = args.type_model
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden

        self.num_classes = out_dim#args.num_classes
        self.num_feats = data.x.shape[-1]#args.num_feats
        self.batch_size = args.batch_size
        self.steps=0
        self.dropout = args.dropout

        self.epochs = args.epochs

        self.device = args.device

        self.data = data

        self.edge_index = data.edge_index

        heterophily_num_class = 2
        self.num_classes = out_dim#args.num_classes
        self.multi_label = True if self.num_classes > 1 else False

        self.out_dim = out_dim
        self.num_targets = self.out_dim
        self.cat = 1

        self.c1 = 0
        self.c2 = 0

        self.inf_threshold = args.inf_threshold
        self.threshold_pred = False

        self.weight_decay = args.weight_decay

        self.use_batch_norm = args.use_batch_norm


        #old_imp="""batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)


        self.edge_emb_mlp = torch.nn.ModuleList()

        self.batch_norms = []
        """batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)
        self.edge_emb_mlp.append(GCNConv(self.cat * self.num_feats , self.dim_hidden))

        for _ in range(self.num_layers - 2):
            self.edge_emb_mlp.append(GCNConv(self.dim_hidden, self.dim_hidden))
            #old_imp=batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            
            batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)

        batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        self.edge_pred_mlp = nn.Linear(self.dim_hidden, self.out_dim)
        """

        # GraphSage, ego- neighborhood embedding seperation performs better
        self.edge_emb_mlp.append(SAGEConv(in_channels=self.num_feats,
                                          out_channels=self.dim_hidden))
                                          # dropout=args.dropout,
                                          # hidden_dim_multiplier=1))

        batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        # construct MLP classifier
        for _ in range(self.num_layers-1):
            self.edge_emb_mlp.append(SAGEConv(in_channels=self.dim_hidden,
                                              out_channels=self.dim_hidden))
                                              #   dropout=args.dropout,
                                              # hidden_dim_multiplier=1))
            # batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)

        # self.edge_emb_mlp.append(SAgeConv(self.dim_hidden,
        #                                   self.out_dim,
        #                                   dropout=args.dropout,
        #                                   hidden_dim_multiplier=1))

        batch_norm_layer = nn.LayerNorm(self.out_dim) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        self.edge_pred_mlp = nn.Linear(self.dim_hidden,# * self.num_layers,
                                       self.out_dim)

        self.act = nn.GELU()

        self.probability = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1) #nn.Softmax(dim=1) #nn.Sigmoid()

        self.dropout = nn.Dropout(self.dropout)

        self.jump = JumpingKnowledge(mode='cat')

        num_neighbors = args.num_neighbors
        if num_neighbors is None:
            self.num_neighbors = [25, 25, 10, 5, 5, 5, 5, 5, 5, 5]
        else:
            if len(num_neighbors) < self.num_layers:
                self.num_neighbors = num_neighbors
                last_hop_nbrs = num_neighbors[-1]
                add_hop_neighbors = [last_hop_nbrs] * (self.num_layers-(len(num_neighbors)-1))
                self.num_neighbors.extend(add_hop_neighbors)
            else:
                self.num_neighbors = num_neighbors

        self.thresholds = args.persistence

        # self.graphs, self.node_mappings = self.process_graph_filtrations(data=train_data,
        #                                              thresholds=self.thresholds,
        #                                              filtration=None)
        #
        # pout(("%%%%%%%" "FINISHED CREATED GRAPH HIERARCHY","Total graphs: ", len(self.graphs),"%%%%%%%"))
        # self.graph_levels = np.flip(np.arange(len(self.graphs)+1))
        # graph = self.graphs[0]
        # self.graphlevelloader = self.get_graph_dataloader(graph,
        #                                                   shuffle=True,
        #                                                   num_neighbors=self.num_neighbors[:self.num_layers])
        # self.graph_level = 0

        self.hierarchicalgraphloader = FiltrationGraphHierarchy(graph=train_data,
                                                           persistence=self.thresholds,
                                                           filtration=None)

        self.graphs, self.sub_to_sup_mappings, self.supergraph_to_subgraph_mappings = (self.hierarchicalgraphloader.graphs,
                                                                          self.hierarchicalgraphloader.sub_to_sup_mappings,
                                                                          self.hierarchicalgraphloader.supergraph_to_subgraph_mappings)
        pout(("%%%%%%%" "FINISHED CREATED GRAPH HIERARCHY", "Total graphs: ", len(self.graphs), "%%%%%%%"))
        self.graph_levels = np.flip(np.arange(len(self.graphs) + 1))
        graph = self.graphs[0]
        self.established_graph_density = False
        self.graphlevelloader = self.get_graph_dataloader(graph,
                                                          shuffle=True,
                                                          num_neighbors=self.num_neighbors[:self.num_layers])
        self.graph_level = 0

        self.graph_lift_intervals = [int(args.epochs / np.max(self.graph_levels)) * graph_level for graph_level in
                          range(len(self.graphs))[1:]]

        pout(("Graph Lifting Epochs: ", self.graph_lift_intervals))




        self.reset_parameters()

    def reset_parameters(self):
        for embedding_layer in self.edge_emb_mlp:
            embedding_layer.reset_parameters()
        self.edge_pred_mlp.reset_parameters()

    def expand_labels(self, labels):
        neg_labels = ~labels
        labels = labels.type(torch.FloatTensor)  # labels.type(torch.FloatTensor)
        neg_labels = neg_labels.type(torch.FloatTensor)
        labels = [neg_labels, labels]
        # if not as_logit:
        return torch.stack(labels, dim=1)
    def get_graph_dataloader(self, graph, shuffle=True, num_neighbors=[-1]):

        if graph.num_nodes < 1200:
            num_workers = 0
        else:
            num_workers = 8

        if shuffle:
            neighborloader = NeighborLoader(data=graph,
                                  batch_size=self.batch_size,
                                  num_neighbors=self.num_neighbors[: self.num_layers],
                                            subgraph_type='induced', #for undirected graph
                                  # directed=False,#True,
                                  shuffle=shuffle,
                                num_workers=num_workers
                                            ) #for graph in self.graphs]
            neighborsampler = NeighborSampler(
                graph.edge_index,
                # node_idx=train_idx,
                # directed=False,
                sizes=self.num_neighbors[: self.num_layers],
                batch_size=self.batch_size,
                # subgraph_type='induced',  # for undirected graph
                # directed=False,#True,
                shuffle=shuffle,
                num_workers=num_workers,
                # directed=False
            )
        else:

            # pout(("Batch Size in Edge Inference ", self.batch_size))

            neighborloader = NeighborLoader(data=graph,
                                  batch_size=self.batch_size,
                                  num_neighbors=[-1,-1],
                                            subgraph_type='induced', #for undirected graph
                                  # directed=False,#True,
                                  shuffle=shuffle
                                            # num_workers=8
                                            ) #for graph in self.graphs]

            neighborsampler = NeighborSampler(
                graph.edge_index,
                # node_idx=train_idx,
                # directed=False,
                sizes=[-1],#self.num_neighbors[: self.num_layers],#num_neighbors,
                batch_size=self.batch_size,
                # subgraph_type='induced',  # for undirected graph
                # directed=False,#True,
                shuffle=shuffle,
                # num_workers=1
            )
        return neighborsampler


    def l2_loss(self, weight, factor):
        return factor * torch.square(weight).sum()

    def forward_nonsep(self, data):
        # batch_size, n_id, adjs in self.train_loader:
        # for i, (edge_index, _, size) in enumerate(adjs):
        # for i, (edge_index, _, size) in enumerate(adjs):
        # x_target = x[: size[1]]  # Target nodes are always placed first.
        # x = self.convs[i]((x, x_target), edge_index)
        #     x_target = x[: size[1]]  # Target nodes are always placed first.
        #     x = self.convs[i]((x, x_target), edge_index)
        x, edge_index = data.x, data.edge_index
        x = self.edge_emb_mlp[0](x, edge_index)
        x = self.dropout(x)
        x = self.act(x)
        for i,embedding_layer in enumerate(self.edge_emb_mlp):
            if i ==0:
                x_res = x
                continue
            x_hidden = x_res
            x_res = self.batch_norms[i](x_res)
            x_res = embedding_layer(x_res, edge_index)
            x_res = x_hidden + x_res
            # if i != self.num_layers - 1:
            x_res = self.dropout(x_res)
            x_res = self.act(x_res)


        # x = x + x_res  # torch.cat([x,x_res],axis=1)
        x = self.batch_norms[-1](x)
        x = self.edge_pred_mlp(x).squeeze(1)
        # x = self.sig(x)
        # x = torch.squeeze(x)
        return x#F.log_softmax( x , dim=1 )



    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        xs = []
        # for i, (edge_index, _, size) in enumerate(adjs):#
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            # if i != self.num_layers - 1:
            # x = self.dropout(x)
            # x_target = x[:batch_size]
            x = self.edge_emb_mlp[i]((x,x_target), edge_index)#(x, x_target), edge_index)
            x = self.dropout(x)


            x = self.batch_norms[i](x)

            # if i != self.num_layers - 1:
            x = self.act(x)

            xs.append(x)

        # x = self.jump(xs)

        x = self.edge_pred_mlp(x)#[batch_size])#[batch])
        # x = x_source + x
        x = self.batch_norms[-1](x)

        return self.probability(x).squeeze(1)

    #fp = open('./run_logs/training_memory_profiler.log', 'w+')
    # @profile#stream=fp)
    def train_net(self, input_dict, exp_input_dict=None):
        return self.hierarchical_successive_train_net(input_dict, exp_input_dict=exp_input_dict)


    def aggregate_edge_attr(self, edge_attr, edge_index):
        edge_attr_target = edge_attr[edge_index]
        # pout(("target edge attr", edge_attr_target, "target edge shape", edge_attr_target.shape))
        # torch.mean(torch.max(a, -1, True)[0], 1, True)
        return torch.max(edge_attr_target, -1, True)[0]

    # @profile
    @torch.no_grad()
    def inference(self, input_dict):
        return self.node_inference(input_dict)
    
    def hierarchical_successive_train_net(self, input_dict, exp_input_dict=None,
                                          filtration=None, thresholds=[.5, 1.0]):

        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        grad_scalar = input_dict["grad_scalar"]
        scheduler = input_dict["scheduler"]

        total_epochs = input_dict["total_epochs"]
        epoch = input_dict['epoch']
        eval_steps = input_dict['eval_steps']
        input_steps = input_dict["steps"]

        val_data = input_dict["val_data"]
        val_input_dict = {"data": val_data,
                            "device": device,
                          "dataset": input_dict["dataset"],
                          "loss_op":loss_op}
        #
        """ NOTE ON WHAT NEEDS TO BE DONE:
                create a hierarchical set of neioghborhood loaders for each each leve
                of the graph hierarchy based on persistent filtration BUT ALSO for each
                collection of nodes comprising the training set, validations set, and 
                graph in it's entirety """


        # global_edge_index = data.edge_index.t()

        # hierarchicalgraphloader = FiltrationHierarchyGraphLoader(graph=data,
        #                                                          persistence=self.thresholds,
        #                                                          num_neighbors=self.num_neighbors,
        #                                                          num_classes=self.num_classes,
        #                                                          filtration=None,
        #                                                          batch_size=self.batch_size,
        #                                                          num_layers=self.num_layers)
        #
        # self.graphs, self.node_mappings = hierarchicalgraphloader.graphs, hierarchicalgraphloader.node_mappings
        # pout(("%%%%%%%" "FINISHED CREATED GRAPH HIERARCHY", "Total graphs: ", len(self.graphs), "%%%%%%%"))
        # self.graph_levels = np.flip(np.arange(len(self.graphs) + 1))
        # graph = self.graphs[0]
        # self.graphlevelloader = self.get_graph_dataloader(graph,
        #                                                   shuffle=True,
        #                                                   num_neighbors=self.num_neighbors[:self.num_layers])
        # self.graph_level = 0


        total_loss = total_correct = 0
        total_val_loss = 0

        self.train()
        self.training = True

        data = self.graphs[self.graph_level]  # input_dict["train_data"]
        # data = data.to(device)
        if self.experiment:
            og_data = data.clone()
        # Compute the degree of each node
        if epoch == 0:
            max_degree, avg_degree, degrees = node_degree_statistics(data)
            pout((" MAX DEGREE ", max_degree," AVERAGE DEGREE ", avg_degree))
            # if avg_degree > 25:
            #     num_neighbors = 25
            #     self.val_batch_size = self.batch_size
            # else:
            #     num_neighbors = -1
            #     self.val_batch_size = self.batch_size

        approx_thresholds = np.arange(0.0, 1.0, 0.1)

        # length_training_batches = data.y.size()[0]
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]

        total_training_points = 0
        predictions = []
        all_labels = []
        # for graph_level, trainloader in enumerate(self.graphLoaders):#_size, n_id, adjs in self.train_loader:

        # for batch in self.graphlevelloader:
        for batch_size, n_id, adjs in self.graphlevelloader:
            # batch=batch.to(device)
            # batch_size = batch.batch_size




            optimizer.zero_grad()

            # x = batch.x
            # edge_index = batch.edge_index
            target_nid = n_id[:batch_size]
            # target_eid = batch.e_id[:batch_size]
            # x_target = x[batch.batch]#:batch_size]#x_target_id]
            # y = batch.y[:batch_size]
            #
            # # adjs = [adj.to(device) for adj in adjs]
            # n_id = batch.n_id
            adjs = [adj.to(device) for adj in adjs]

            x = data.x[n_id].to(device)
            y = data.y[n_id[:batch_size]].to(device)  # .float()
            # edge_index = data.edge_index[:,batch.e_id]#.edge_index
            # x = data.x[n_id]#.to(device)
            # x_target = x[target_nid]#
            # y = data.y[:batch_size].to(device)  # .float()
            # y =  data.y[target_nid]#.to(device)#.float()
            # # edge_index = adjs.edge_index
            #
            with autocast():
                out = self(x, adjs)[:batch_size]
            # out = out[:batch_size]#target_nid]#x_target_id]#adjs)#edge_index)
            # # out = self(batch)
            # # y = batch.y#.squeeze()
            # # y = batch.y.unsqueeze(-1)
            # # y = y.type_as(out)

            loss = loss_op(out, y)

            #
            #
            #
            grad_scalar.scale(loss).backward()
            grad_scalar.step(optimizer)
            grad_scalar.update()
            # optimizer.zero_grad()
            #
            #
            #
            total_training_points += batch_size#x.size(0)

            # back prop
            # loss.backward()
            # optimizer.step()

            total_loss += loss.item()



            if self.num_targets != 1:  # (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                #total_correct += out.argmax(dim=-1).eq(y.argmax(dim=-1)).float().item()
                total_correct += float(out.argmax(axis=1).eq(y).sum())
                all_labels.extend(y)  # ground_truth)
                predictions.extend(out.argmax(axis=1))
            else:
                predictions.extend(out)#preds)
                all_labels.extend(y)#ground_truth)
                # # total_correct += (out.long() == thresh_out).float().sum().item()#int(out.eq(y).sum())
                # total_correct += (y == thresh_out).float().sum().item()  # int(out.eq(y).sum())
                # # approx_acc = (y == thresh_out).float().mean().item()

            del adjs, batch_size, n_id, loss, out, x, y
            torch.cuda.empty_cache()


        predictions = torch.tensor(predictions)
        all_labels = torch.tensor(all_labels)
        # all_labels=torch.cat(all_labels,dim=0)

        num_classes = len(all_labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if self.num_targets != 1:
            # approx_acc = total_correct/all_labels.numel()
            approx_acc = (predictions == all_labels).float().mean().item()

            del predictions, all_labels

        else:
            total_correct = int(predictions.eq(all_labels).sum())
            all_labels = all_labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            approx_acc = total_correct/all_labels.shape[0]
            # approx_acc = 0
            # train_opt_thresh, approx_acc = optimal_metric_threshold(y_probs=predictions,
            #                                                                     y_true=all_labels,
            #                                                                     metric=accuracy_score,
            #                                                                     metric_name='accuracy',
            #                                                           num_targets=num_targets,
            #                                                         thresholds=approx_thresholds)

        if epoch in self.graph_lift_intervals or epoch % eval_steps == 0 and epoch != 0:
            with torch.no_grad():
                self.eval()
                val_pred, val_loss, val_ground_truth = self.inference(val_input_dict)
            self.training = True
            self.train()
            # val_out, val_loss, val_labels = self.inference(val_input_dict)
            # if scheduler is not None:
            #     scheduler.step(val_loss)
            # num_classes = len(all_labels.unique())
            # num_targets = 1 if num_classes == 2 else num_classes
            if self.num_targets != 1:
                # predictions = predictions
                # approx_acc = (predictions == all_labels).float().mean().item()

                val_pred = val_pred
                val_acc = (val_pred == val_ground_truth).float().mean().item()
                # val_acc = (val_pred.argmax(axis=-1) == val_ground_truth).float().mean().item()
                print("Epoch: ", epoch,  f" Validation ACC: {val_acc:.4f}")
                val_roc = 0
                del val_pred, val_ground_truth

            else:
                # if False:  # because too slow
                val_pred = val_pred.detach().cpu().numpy()
                val_ground_truth = val_ground_truth.detach().cpu().numpy()
                # val_acc = accuracy_score(val_ground_truth, val_pred)
                val_optimal_threshold, val_acc = optimal_metric_threshold(y_probs=val_pred,
                                                                                    y_true=val_ground_truth,
                                                                                    metric=accuracy_score,
                                                                                    metric_name='accuracy',
                                                                          num_targets=num_targets,
                                                                          thresholds=approx_thresholds)

                val_thresh, val_roc = optimal_metric_threshold(val_pred,
                                                                 val_ground_truth,
                                                                 metric=metrics.roc_auc_score,
                                                                 metric_name='ROC AUC',
                                                               thresholds=approx_thresholds)

                # all_labels = all_labels.detach().cpu().numpy()
                # predictions = predictions.detach().cpu().numpy()
                train_opt_thresh, approx_acc = optimal_metric_threshold(y_probs=predictions,
                                                                                    y_true=all_labels,
                                                                                    metric=accuracy_score,
                                                                                    metric_name='accuracy',
                                                                          num_targets=num_targets,
                                                                        thresholds=approx_thresholds)

                print("Epoch: ", epoch, f" Validation ACC: {val_acc:.4f}",
                      f" Validation ROC: {val_roc:.4f}")

                total_val_loss += val_loss

                # if scheduler is not None:
                #     scheduler.step(val_loss)

                del val_loss, val_pred, val_ground_truth

        #  # .item(


            #del y, batch_size, n_id, loss, out, l2_loss, l2_factor, edge_embedding, batch

        if epoch in self.graph_lift_intervals and epoch != total_epochs:#% (total_epochs // self.data_size) == 0 and epoch != 0:
            pout(("%%%%%%"))
            pout(("Moving up graph level hierarchy for successive training"))
            pout(("Epoch ", epoch, " of ", total_epochs ))
            pout(("%%%%%%"))
            pout(("RESULTS SUBGRAPH: ", self.graph_level))
            pout((f"Loss: {total_loss / total_training_points}",
                  f"approx_acc: {approx_acc}",
                  "(total_val_loss, val_acc, val_roc)", f"({total_val_loss}", f"{val_acc}", f"{val_roc})"))
            self.graph_level += 1
            # Save embeddings for each node in the last graph
            subgraph_embeddings = {}
            with torch.no_grad():
                for batch_size, n_id, adjs in self.graphlevelloader:#batch in self.graphlevelloader:
                    # batch.to(device)
                    # n_id = batch.n_id
                    for nid in n_id.detach().cpu().numpy():
                        n_embedding = data.x[nid].detach().cpu().numpy()

                        if self.experiment == "fixed_init_ablation":
                            n_embedding = og_data.x[nid].detach().cpu().numpy()

                        subgraph_embeddings[nid] = n_embedding
                    # for i, node in zip(n_id.detach().cpu().numpy(), data.x[n_id].detach().cpu().numpy()):#zip(e_id.detach().cpu().numpy()
                    #     subgraph_embeddings[i] = node#node] = out[i]#.detach().cpu().numpy()
                if self.experiment is not None:
                    pout(("RUNNING TEST ON SUBLEVEL GRAPH FOR EXPERIMENT:"))
                    pout((self.experiment))
                    pout(("Graph Level:"))
                    pout((self.graph_level-1))
                    test_acc, test_f1, test_roc = self.run_experiment(exp_input_dict)
                    pout(("Experiment Test Accuracy: ",test_acc,
                          "Experiment Test F1: ", test_f1,
                          "Experiment Test AUC:", test_roc))
                    self.experimental_results.append("(test_acc, test_f1, test_roc)")
                    self.experimental_results.append((test_acc, test_f1, test_roc))

            self.graphs[self.graph_level] = self.initialize_from_subgraph(
                subgraph_embeddings,
                self.graphs[self.graph_level],
                graph_level=self.graph_level,
                supergraph_to_subgraph_mappings=[self.supergraph_to_subgraph_mappings[self.graph_level-1],
                               self.supergraph_to_subgraph_mappings[self.graph_level]],
            subgraph_to_supergraph_mappings=[self.sub_to_sup_mappings[self.graph_level-1],
                                             self.sub_to_sup_mappings[self.graph_level]])



            self.graphlevelloader = self.get_graph_dataloader(self.graphs[self.graph_level],
                                                              shuffle=True,
                                                              num_neighbors=self.num_neighbors[:self.num_layers]
                                                              )
            # x = scatter(data.x, data.batch, dim=0, reduce='mean')




        torch.cuda.empty_cache()

        if scheduler is not None:
            scheduler.step(total_loss / total_training_points)

        if epoch % eval_steps == 0 and epoch != 0:
            return total_loss/total_training_points , approx_acc, (total_val_loss, val_acc, val_roc)
        else:
            return total_loss / total_training_points, approx_acc, (666, 666,666)

    @torch.no_grad()
    def node_inference(self, input_dict):
        # input dict input_dict = {"data": self.data, "y": self.y, "device": self.device, "dataset": self.dataset}
        self.eval()
        self.training = False
        device = input_dict["device"]
        # x = input_dict["x"].to(device)
        data = input_dict["data"]
        pout(("Graph Statistics node inference"))
        self.hierarchicalgraphloader.graph_statistics(data)
        # for validation testing
        loss_op = input_dict["loss_op"]

        # labels = data.y.to(device)

        inference_loader = self.get_graph_dataloader(data,
                                                     shuffle=False,
                                                     num_neighbors=[-1])
        """NeighborLoader(
            data=data,  # copy.copy(data),#.edge_index,
            input_nodes=None,
            # directed=True,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],
            batch_size=self.batch_size,
            shuffle=False,
        )"""


        self.batch_norms = [bn.to(device) for bn in self.batch_norms]

        train_sample_size = 0
        edges = []
        node_ids = []
        edge_weights = []
        # edge_embeddings = []
        node_pred_dict = {}
        x_pred = []
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            # for batch in inference_loader:
            for batch_size, n_id, adjs in inference_loader:
                # batch=batch.to(device)
                # batch_size = batch.batch_size
                # n_id = batch.n_id
                adjs= [adjs.to(device)]
                # optimizer.zero_grad()

                # adjs = [adj.to(device) for adj in adjs]

                x = data.x[n_id].to(device)
                y = data.y[n_id[:batch_size]].to(device)#.float()
                # y = batch.y

                with autocast():
                    out = self(x, adjs)[:batch_size]


                # out = self(batch)
                # y = batch.y#.squeeze()
                # y = y.type_as(out)

                loss = loss_op(out, y)#.to(torch.float))

                total_loss += loss.item()#.item()#float(loss.item())/batch.batch_size

                if self.num_classes > 2:
                    pred = out.argmax(axis=1)#self.sig(out)
                else:
                    pred = out#.argmax(dim=-1)
                # for nid, p in zip(n_id.detach().cpu().numpy(), out.detach().cpu().numpy()):
                #     # node_ids.append(batch.n_id.cpu().numpy())
                #     node_pred_dict[nid] = p#[0]#.append

                train_sample_size += batch_size#.cpu().float()
                all_preds.extend(pred.cpu().numpy())#pred)#.cpu().numpy())
                all_labels.extend(y.cpu().numpy())#F.one_hot(batch.y, num_classes=2))#.cpu().numpy()

                del y, adjs, loss, out, x
                torch.cuda.empty_cache()


        # data.node_preds = torch.tensor([node_pred_dict[i] for i in range(len(node_pred_dict))],
        #                                dtype=torch.float)

        # all_preds_np = np.concatenate(all_preds)
        # all_labels_np = np.concatenate(all_labels)
        # Compute F1 score
        # f1 = f1_score(all_labels_np, all_preds_np, average='micro')

        # all_preds = torch.cat(all_preds,dim=0)
        # all_labels = torch.cat(all_labels,dim=0)

        # all_preds = torch.tensor(all_preds)
        total_loss = total_loss/train_sample_size
        # if self.num_classes <= 2:
        #     all_preds = torch.tensor(all_preds)
        #     all_labels = torch.tensor(all_labels)
        # else:
        #     all_preds = torch.cat(all_preds, dim=0)
        #     all_labels = torch.cat(all_labels, dim=0)

        # all_labels = [item for sublist in all_labels for item in sublist]
        # all_preds = [item for sublist in all_preds for item in sublist]

        return torch.tensor(all_preds), total_loss, torch.tensor(all_labels) # torch.from_numpy(np.cat(x_pred,axis=0))##torch.stack(torch.tensor(x_pred).tolist(),dim=0)  # _all

    def initialize_from_subgraph(self,
                                 subgraph,
                                 supergraph,
                                 graph_level,
                                 supergraph_to_subgraph_mappings,
                                 subgraph_to_supergraph_mappings):
        pout(("graph level ", graph_level, "graph levelS ", self.graph_levels, " node mappings length ",
              len(supergraph_to_subgraph_mappings)))
        sup_to_sub_lower = supergraph_to_subgraph_mappings[0] # mapping of supergrpah to lower level subgraph
        sup_to_sub_higher = supergraph_to_subgraph_mappings[1] # mapping supergraph idx to higher level subgraph
        sub_to_sup_lower = subgraph_to_supergraph_mappings[0] #mapping of lower level graph to supergraph idx
        # {sub_id: global_id for global_id,sub_id in subgraph_mapping.items()}
        # supsub_mapping = {global_id: sub_id for sub_id, global_id in supgraph_mapping.items()}


        # for node, i in subgraph_mapping.items():

        new_node_features = supergraph.x.clone().detach().cpu().numpy()
        for node, embedding in subgraph.items():
            global_id = sub_to_sup_lower[node]
            new_node_features[sup_to_sub_higher[global_id]] = embedding#supsub_mapping[global_id]] = embedding
        supergraph.x = torch.tensor(new_node_features)
        return supergraph

    # def pyg_to_dionysus(self, data):
    #     # Extract edge list and weights
    #     data = data.clone()
    #     edge_index = data.edge_index.t().cpu().numpy()
    #     edge_weight = data.edge_weights.cpu().numpy()
    #     filtration = dion.Filtration()
    #     for i, (u, v) in enumerate(edge_index):
    #         filtration.append(dion.Simplex([u, v], edge_weight[i]))
    #     filtration.sort()
    #     return filtration
    #
    # def filtration_to_networkx(self,filtration, data, clone_data=False, node_mapping=True):
    #     if clone_data:
    #         data = data.clone()
    #     node_mapping_true = node_mapping
    #
    #     edge_emb = data.edge_attr.cpu().numpy()
    #     y = data.y.cpu().numpy()
    #     G = nx.Graph()
    #     for simplex in filtration:
    #         u, v = simplex
    #         G.add_edge(u, v, weight=simplex.data, embedding=edge_emb[simplex])
    #         G.nodes[u]['y'] = y[u]  # , features=data.x[u])#.tolist() if data.x is not None else {})
    #         G.nodes[v]['y'] = y[v]
    #         G.nodes[u]['features'] = data.x[u].tolist() if data.x is not None else {}
    #         G.nodes[v]['features'] = data.x[v].tolist() if data.x is not None else {}
    #
    #     # if node_mapping_true:
    #     node_mapping = {node: i for i, node in enumerate(G.nodes())}
    #     return G, node_mapping
    # def create_filtered_graphs(self,filtration, thresholds, data, clone_data=False, nid_mapping=None):
    #     if clone_data:
    #         data = data.clone()
    #
    #     edge_emb = data.edge_attr.cpu().numpy()
    #     y = data.y.cpu().numpy()
    #
    #
    #     node_mappings = []
    #     graphs = []
    #
    #     for threshold in thresholds:
    #         G = nx.Graph()
    #
    #         for simplex in filtration:
    #             if simplex.data >= threshold:
    #                 u, v = simplex
    #                 G.add_edge(u, v, weight=simplex.data, embedding=edge_emb[simplex])
    #                 G.nodes[u]['y'] = y[u]#, features=data.x[u])#.tolist() if data.x is not None else {})
    #                 G.nodes[v]['y'] = y[v]
    #                 G.nodes[u]['features'] = data.x[u].tolist() if data.x is not None else {}
    #                 G.nodes[v]['features'] = data.x[v].tolist() if data.x is not None else {}
    #         graphs.append(G)
    #
    #         # if nid_mapping is None:
    #         node_mapping = {node: i for i, node in enumerate(G.nodes())}
    #         # else:
    #         #     node_mapping = {node: nid_mapping[node] for i, node in enumerate(G.nodes())}
    #         node_mappings.append(node_mapping)
    #     return graphs, node_mappings
    #
    # def pyg_to_networkx(self,data, clone_data=False):
    #     if clone_data:
    #         data = data.clone()
    #     # Initialize a directed or undirected graph based on your need
    #     G = nx.Graph()#nx.DiGraph() if data.is_directed() else nx.Graph()
    #
    #     # Add nodes along with node features if available
    #     for i in range(data.num_nodes):
    #         node_features = data.x[i].tolist() if data.x is not None else {}
    #         G.add_node(i, features=node_features)
    #
    #     # Add edges along with edge attributes if available
    #     edge_index = data.edge_index.t().cpu().numpy()
    #     if data.edge_attr is not None:
    #         edge_attributes = data.edge_attr.cpu().numpy()
    #         for idx, (source, target) in enumerate(edge_index):
    #             G.add_edge(source, target, weight=edge_attributes[idx])
    #     else:
    #         for source, target in edge_index:
    #             G.add_edge(source, target)
    #
    #     return G
    # def nx_to_pyg(self, graph, node_mapping = True, graph_level = None):
    #
    #     target_type = torch.long if self.num_classes > 1 else torch.float
    #     # Mapping nodes to contiguous integers
    #     node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    #
    #     int_mapping = {v:u for u,v in node_mapping.items()}
    #     # Convert edges to tensor format
    #     edge_list = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
    #     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    #
    #     # edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges], dtype=torch.float)
    #     #edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.nodes(data=True)], dtype=torch.float)
    #
    #     num_nodes = graph.number_of_nodes()
    #     node_features = torch.tensor([attr['features'] for node, attr in graph.nodes(data=True)], dtype=torch.float)
    #     y = torch.tensor([attr['y'] for node, attr in graph.nodes(data=True)], dtype=target_type)
    #     # node_embeddings = torch.tensor([graph.nodes[i]['embeddings'] for i in range(num_nodes)], dtype=torch.float)
    #
    #     #x = torch.tensor([graph[u]['features'] for u in graph.nodes], dtype=torch.float)
    #     #y = torch.tensor([graph[u]['y'] for u in graph.nodes], dtype=torch.float)
    #     # edge_emb = torch.tensor([graph[u]['embedding'] for u in graph.nodes], dtype=torch.float)
    #
    #     data = Data(x=node_features,
    #                 edge_index=edge_index,
    #                 y=y,
    #                 edge_attr=edge_attr,
    #                 # edge_embedding=edge_emb,
    #                 num_nodes=graph.number_of_nodes())
    #     return data
    #

    @torch.no_grad()
    def run_experiment(self, input_dict=None):
        # pout(("In test net", "multilabel?", self.multi_label))
        self.eval()
        test_input_dict = input_dict

        test_out, loss, y_test = self.inference(test_input_dict)

        if self.multi_label == True:
            test_thresh, test_acc = optimal_metric_threshold(test_out,
                                                             y_test,
                                                             metric=metrics.accuracy_score,
                                                             metric_name='accuracy',
                                                             num_targets=self.num_classes)

            test_f1 = 0
            test_roc = 0
        if self.multi_label == False:
            y_true_test = y_test.cpu().numpy()
            y_score_test = test_out.cpu().numpy()
            test_thresh, test_acc = optimal_metric_threshold(y_score_test,
                                                             y_true_test,
                                                             metric=accuracy_score,
                                                             metric_name='accuracy')
            test_thresh, test_f1 = optimal_metric_threshold(y_score_test,
                                                            y_true_test,
                                                            metric=metrics.f1_score)
            all_thresh, test_roc = optimal_metric_threshold(y_score_test,
                                                            y_true_test,
                                                            metric=metrics.roc_auc_score,
                                                            metric_name='ROC AUC')
        train_roc = 0
        all_roc = 0
        train_f1 = 0
        all_f1 = 0
        all_out, loss, y_all = 0, 0, 0
        all_acc = 0
        train_acc = 0

        return  test_acc, test_f1, test_roc
def check_node_maps( n_i, n_j, sub_to_super_i, sub_to_super_j):
    pout(("similarity of global maps"))
    n_i_super = []
    n_j_super = []
    for i in n_i:
        n_i_super.append(sub_to_super_i[int(i.detach().cpu())])
    for j in n_j:
        n_j_super.append(sub_to_super_j[j])#int(j.detach().cpu())])
    sim_i_j = []
    for i in n_i_super:
        sim = i in n_j_super
        sim_i_j.append(sim)
    pout((sim_i_j))

class HierJGNN(torch.nn.Module):
    def __init__(self,
                 args,
                 data,
                 in_channels=None, # defaults to number of features in data.x
                 dim_hidden=None,  # must be passed in args if not directly
                 out_dim=None,     # defaults to number of unique labels i.e. num_classes
                 processed_dir=None,
                 train_data=None,
                 val_data=None,
                 test_data=None,
                 filtration_function=None,
                 normalization=None,
                 # hidden_channels,
                 # out_channels
                 ):
        super(HierJGNN, self).__init__()
        # base params
        self.save_dir = processed_dir
        self.type_model = args.type_model
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.steps=0
        self.dropout = args.dropout
        self.epochs = args.epochs
        self.device = args.device
        self.data = data
        self.edge_index = data.edge_index
        heterophily_num_class = 2
        self.cat = 1
        self.c1 = 0
        self.c2 = 0
        self.inf_threshold = args.inf_threshold
        self.threshold_pred = False
        self.weight_decay = args.weight_decay
        self.use_batch_norm = args.use_batch_norm
        pout(("Using Batch Normalization: ",self.use_batch_norm))
        self.thresholds = args.persistence
        self.val_data = val_data
        self.test_data = test_data
        #using base options expand samples to number of layers (hops)
        num_neighbors = args.num_neighbors
        if num_neighbors is None:
            self.num_neighbors = [25, 25, 10, 5, 5, 5, 5, 5, 5, 5]
        else:
            if len(num_neighbors) < self.num_layers:
                self.num_neighbors = num_neighbors
                last_hop_nbrs = num_neighbors[-1]
                add_hop_neighbors = [last_hop_nbrs] * (self.num_layers-(len(num_neighbors)-1))
                self.num_neighbors.extend(add_hop_neighbors)
            else:
                self.num_neighbors = num_neighbors
        #set degree information of data
        max_degree, avg_degree, degrees = node_degree_statistics(train_data)
        pout((" MAX DEGREE ", max_degree, " AVERAGE DEGREE ", avg_degree))
        train_data.max_node_degree = max_degree
        num_classes = len(train_data.y.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        # set dimensions
        self.out_dim = num_targets if not out_dim else out_dim
        self.num_classes = num_targets if not out_dim else out_dim
        self.dim_hidden = args.dim_hidden if not dim_hidden else dim_hidden
        self.num_feats = data.x.shape[-1] if not in_channels else in_channels
        self.dim_gin = args.dim_gin if args.dim_gin is not None else self.dim_hidden
        self.dim_multiscale_filter_conv = args.dim_multiscale_filter_conv if args.dim_multiscale_filter_conv is not None else self.dim_hidden
        ###################################################################################
        #         Hierarchical Joint Training requires the topological persistence graph hierarchy
        #         at inference time in order to employ the neighborhood message aggregating models
        #                 trained for each levelset graph in the graph sequence.
        ###################################################################################
        self.filtration_function = None# filtration_function

        ####################################################################################
        #         Compute persistence filtration hierarchy for leve-set graphs
        #             Collect Neighborhood samplers for each graphlevel
        ####################################################################################
        # compute persistence filtration for graph hierarchy
        filtration_graph_hierarchy = FiltrationGraphHierarchy(graph=train_data,
                                                              persistence=self.thresholds,
                                                              filtration=None)
        self.graphs, self.sub_to_sup_mappings, self.sup_to_sub_mapping = (filtration_graph_hierarchy.graphs,
                                                                          filtration_graph_hierarchy.sub_to_sup_mappings,
                                                                          filtration_graph_hierarchy.supergraph_to_subgraph_mappings)
        pout(("%%%%%%%" "FINISHED CREATED GRAPH HIERARCHY", "Total graphs: ", len(self.graphs), "%%%%%%%"))
        self.graph_levels = np.flip(np.arange(len(self.graphs) + 1))
        # get neighborhood loaders for each sublevel graph
        self.sublevel_graph_loaders = [self.get_graph_dataloader(graph,
                                                                 shuffle=False,
                                                                 num_neighbors=self.num_neighbors[:self.num_layers],
                                                                 batch_size=self.batch_size,
                                                                 num_workers=0)\
                                       for graph in self.graphs]
        self.super_graph = self.graphs[-1]
        #hierarchical graph neighborhood sampler
        self.hierarchical_graph_sampler = MultiScaleGraphFiltrationSampler(super_data=self.super_graph,
                                                                           subset_data=self.graphs,#[:-1],
                                                                           subset_samplers=self.sublevel_graph_loaders,
                                                                           subset_to_super_mapping=self.sub_to_sup_mappings,
                                                                           super_to_subset_mapping=self.sup_to_sub_mapping,
                                                                           batch_size=self.batch_size,
                                                                           shuffle=True)
        # validation graph hierarchy
        val_graph_hier = FiltrationGraphHierarchy(graph=val_data,persistence=self.thresholds,filtration=None)
        val_graphs, val_sub_to_sup_mappings, val_sup_to_sub_mapping = (val_graph_hier.graphs,
                                                                       val_graph_hier.sub_to_sup_mappings,
                                                                       val_graph_hier.supergraph_to_subgraph_mappings)
        val_subgraph_loaders = [self.get_graph_dataloader(graph,
                                                       shuffle=False,
                                                       num_neighbors=[-1],#-1],
                                                       batch_size=self.batch_size,
                                                       num_workers=0) for graph in val_graphs]
        val_super_graph = val_graphs[-1]
        val_hier_graph_sampler = MultiScaleGraphFiltrationSampler(super_data=val_super_graph,
                                                                  subset_data=val_graphs,
                                                                  subset_samplers=val_subgraph_loaders,
                                                                  subset_to_super_mapping=val_sub_to_sup_mappings,
                                                                  super_to_subset_mapping=val_sup_to_sub_mapping,
                                                                  batch_size=self.batch_size,
                                                                  shuffle=True)

        self.validation_dict={"graph_hierarchy":val_graph_hier,
                              "graphs":val_graphs,
                              "sub_to_sup_map":val_sub_to_sup_mappings,
                              "sup_to_sub_map":val_sup_to_sub_mapping,
                              "subgraph_loaders":val_subgraph_loaders,
                              "super_graph":val_super_graph,
                              "hierarchical_graph_sampler":val_hier_graph_sampler}
        #  Test setting
        test_graph_hier = FiltrationGraphHierarchy(graph=test_data,
                                                   persistence=self.thresholds,
                                                   filtration=None)
        test_graphs, test_sub_to_sup_mappings, test_sup_to_sub_mapping =(test_graph_hier.graphs,
                                                                         test_graph_hier.sub_to_sup_mappings,
                                                                         test_graph_hier.supergraph_to_subgraph_mappings)

        test_subgraph_loaders = [self.get_graph_dataloader(graph,
                                                           shuffle=False,
                                                           num_neighbors=[-1],#-1],
                                                           batch_size=self.batch_size,
                                                           num_workers=0)
                                 for graph in test_graphs]
        test_super_graph = test_graphs[-1]
        test_hier_graph_sampler = MultiScaleGraphFiltrationSampler(super_data=test_super_graph,
                                                                   subset_data=test_graphs,
                                                                   subset_samplers=test_subgraph_loaders,
                                                                   subset_to_super_mapping=test_sub_to_sup_mappings,
                                                                   super_to_subset_mapping=test_sup_to_sub_mapping,
                                                                   batch_size=self.batch_size,
                                                                   shuffle=False)

        self.testing_dict ={"graph_hierarchy":test_graph_hier,
                            "graphs":test_graphs,
                            "sub_to_sup_map":test_sub_to_sup_mappings,
                            "sup_to_sub_map":test_sup_to_sub_mapping,
                            "subgraph_loaders":test_subgraph_loaders,
                            "super_graph":test_super_graph,
                            "hierarchical_graph_sampler":test_hier_graph_sampler}
        #############################################################################################
        #
        #                              Model Design
        #
        ############################################################################################
        #
        # Normalalization in and out to prevent overfitting
        # Normalizaation of features

        self.normalizers = {}
        self.normalizers["in"] = nn.LayerNorm(self.num_feats)

        self.normalizers["hid"] = NORMALIZATION["LayerNorm"](self.dim_hidden)

        self.normalizers["comb"] = NORMALIZATION["MaskedBatchNorm"](self.dim_hidden*len(self.graphs))

        # self.normalizers["comb"] = NORMALIZATION["MaskedBatchNorm"](self.dim_hidden*len(self.graphs))

        self.normalizers["out"] = NORMALIZATION["MaskedBatchNorm"](self.dim_hidden*len(self.graphs))
        ######################################################################################
        #           Define message passing / neighborhood aggration scheme for each
        #             graph levelset for learned node embeddings per-graph level
        ######################################################################################
        self.levelset_modules = [SubGraphFilterConv(in_dim=self.num_feats,
                                                    dim_hidden=self.dim_multiscale_filter_conv,
                                                    out_dim=self.dim_hidden,  #outputs node_embedding of dimension dim hidden
                                                    num_layers=self.num_layers,
                                                    dropout=self.dropout,
                                                    use_batch_norm=self.use_batch_norm,
                                                    normalization=NORMALIZATION["LayerNorm"]) ]
                                 #for graph in self.graphs]

        #####################################################################################
        #     per sublevel graph filtration function, learns attention factor for learned
        #     node embeddings from respective sublevel neighborhoods before combining for
        #           all embeddings for hierarchical representation of node
        #####################################################################################
        # define learnable epsilon of GIN s.t. each graphs node embedding contributes equally
        epsilon = float(1.0 / len(self.graphs))
        # filttation function for each subgraph.
        # each subgraph has pair of filter functions for (dim in, hidden him)
        # self.levelset_graph_filtration_functions = [(SubLevelGraphFiltration(max_node_deg=int(max_degree),
        #                                                                     dim_in=self.num_feats,# Real valued filtration factor
        #                                                                     dim_out=self.dim_hidden,
        #                                                                     eps=epsilon,
        #                                                                     use_node_degree=False,
        #                                                                     set_node_degree_uninformative=False,
        #                                                                     use_node_feat=True,
        #                                                                     gin_number=1,
        #                                                                     gin_dimension=self.dim_hidden,
        #                                                                     gin_mlp_type ='lin_bn_lrelu_lin'),
        # Real valued filtration factor
        # self.levelset_graph_filtration_functions_in = [[SubGraphFilterFunction(max_node_deg=int(max_degree),
        #                                                                       dim_in=self.num_feats,
        #                                                                        dim_out=self.dim_hidden,
        #                                                                       eps=epsilon,
        #                                                                       use_node_degree=False,
        #                                                                       set_node_degree_uninformative=False,
        #                                                                       use_node_feat=True,
        #                                                                       gin_number=1,
        #                                                                       gin_dimension=self.dim_hidden,  # as factor
        #                                                                       gin_mlp_type ='lin_gelu_lin')]
        #                                                for graph in self.graphs]
        # self.levelset_graph_filtration_functions_hidden = [[SubGraphFilterFunction(max_node_deg=int(max_degree),
        #                                                                           dim_in=self.dim_hidden,
        #                                                                           dim_out=self.dim_hidden,
        #                                                                           eps=epsilon,
        #                                                                           use_node_degree=False,
        #                                                                           set_node_degree_uninformative=False,
        #                                                                           use_node_feat=True,
        #                                                                           gin_number=1,
        #                                                                           gin_dimension=self.dim_hidden,
        #                                                                           gin_mlp_type ='lin_gelu_lin')
        #                                                    for _ in range(self.num_layers - 1)]
        #                                                    for graph in self.graphs]
        # self.levelset_graph_filtration_functions_hidden = [ filt_in + filt_hid for filt_in,filt_hid in zip(self.levelset_graph_filtration_functions_in,
        #                                                                                                          self.levelset_graph_filtration_functions_hidden)]
        self.levelset_graph_filtration_functions_out = [SubGraphFilterFunction(max_node_deg=int(max_degree),
                                                                               dim_in=self.num_feats,
                                                                               dim_out=self.dim_hidden,
                                                                               eps=epsilon,
                                                                               use_node_degree=False,
                                                                               set_node_degree_uninformative=False,
                                                                               use_node_feat=True,
                                                                               gin_number=1,
                                                                               gin_dimension=self.dim_gin,
                                                                               gin_mlp_type ='lin_gelu_lin',
                                                                               use_batch_norm=self.use_batch_norm,
                                                                               normalization=NORMALIZATION["LayerNorm"])
                                                        for graph in self.graphs]

        # MLP out layer combining (concatenating) each nodes',
        # at each graph level in the hierarchy's,
        # learned embedding representation
        # self.hypergraph_node_embedding = torch.nn.Linear(self.dim_hidden * len(self.graphs) ,
        #                                                    self.out_dim)

        #
        # multiscale aggregator for all embeddings from
        # across mmultiscale neioghborhood message passing
        # GraphSage, ego- neighborhood embedding seperation performs better
        #
        # self.multiscale_nbr_msg_aggr = []
        # self.multiscale_nbr_msg_aggr.append(SAGEConv(in_channels=self.num_feats,#dim_hidden * (len(self.graphs)+1),
        #                                         out_channels=self.dim_hidden))
        # # self.bns.append(nn.BatchNorm1d(n_2)) self.bns = nn.ModuleList()
        self.batch_norms = []
        batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)
        for _ in range(self.num_layers - 1):
            # self.multiscale_nbr_msg_aggr.append(SAGEConv(in_channels=self.dim_hidden ,
            #                                   out_channels=self.dim_hidden))
            # batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)


        # # Target supergraph neiughborhood within neighborhood embedding
        # #
        # self.target_super_nbr_aggr = []
        # self.target_super_nbr_aggr.append(SAGEConv(in_channels=self.num_feats,
        #                                         out_channels=self.dim_hidden))
        # # # self.bns.append(nn.BatchNorm1d(n_2)) self.bns = nn.ModuleList()
        # self.target_batch_norms = [nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)]
        # for _ in range(self.num_layers - 1):
        #     self.target_super_nbr_aggr.append(SAGEConv(in_channels=self.dim_hidden ,
        #                                       out_channels=self.dim_hidden))
        #     self.target_batch_norms = self.target_batch_norms + [nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)]

        # self.Combiner = combine()
        # self.combine = lambda tensor1,tensor2,device: self.Combiner.mean(tensor1,tensor2,device=device)
        # self.combine_comp = lambda *tensors, device: self.Combiner.mean_comp(*tensors, device=device)

        # self.feed_forward = nn.Sequential(
        #     nn.Linear(self.dim_hidden * len(self.graphs), self.dim_hidden),
        #     # nn.Linear(dim_out, dim),
        #     nn.LayerNorm(self.dim_hidden),  # nn.BatchNorm1d(dim),
        #     # nn.GELU(),
        #     # nn.Linear(self.dim_hidden, self.dim_hidden)
        # )
        self.feed_forward = FeedForwardModule(dim=self.dim_hidden * len(self.graphs),
                                              dim_out=self.dim_hidden * len(self.graphs),
                                              input_dim_multiplier=1,
                                              hidden_dim_multiplier=2,
                                              dropout=self.dropout)

        self.lin_out = nn.Linear(self.dim_hidden * len(self.graphs), self.out_dim)#self.dim_hidden*len(self.graphs), self.out_dim)

        self.act = nn.GELU()
        # self.multiscale_nbr_aggr = nn.Linear(self.dim_hidden * len(self.graphs), self.out_dim) #nn.Sequential( *multiscale_node_embedding )
        # self.multiscale_nbr_aggr = nn.Linear(self.dim_hidden , self.out_dim)

        self.probability = nn.Sigmoid() if self.out_dim == 1 else nn.Softmax(dim=1) #nn.Softmax(dim=1) #nn.Sigmoid()

        self.reset_parameters()
        self.reset_model_parameters(self.feed_forward)
        self.reset_model_parameters(self)
        # for record keeping
        # self.print_class_attr_and_methods()
    def reset_parameters(self):
        self.lin_out.reset_parameters()
        # for l in self.target_batch_norms:
        #     l.reset_parameters()
        # for l in self.multiscale_nbr_msg_aggr:
        #     l.reset_parameters()
        for l in self.levelset_graph_filtration_functions_out:
            l.reset_parameters()
        for l in self.levelset_modules:
            l.reset_parameters()

    def reset_model_parameters(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    def print_class_attr_and_methods(self):
        attributes_and_methods = dir(self)

        # Iterate over the attributes and methods
        for name in attributes_and_methods:
            # Skip private and special attributes/methods
            if not name.startswith('__'):
                # Use getattr() to get the attribute/method
                attribute = getattr(self, name)
                print(f'{name}: {attribute}')

                # Check if it's a callable (i.e., a method)
                if callable(attribute):
                    print(f'  {name} is a method')
                else:
                    print(f'  {name} is an attribute with value: {attribute}')

    def get_graph_dataloader(self, graph, shuffle=False, num_neighbors=[-1], num_workers=None, batch_size=1):
        if not num_workers:
            if graph.num_nodes < 1200:
                num_workers = 0
            else:
                num_workers = 8
        neighborloader = NeighborLoader(data=graph,
                              batch_size=self.batch_size,
                              num_neighbors=self.num_neighbors[: self.num_layers],
                                        subgraph_type='induced', #for undirected graph
                              # directed=False,#True,
                              shuffle=shuffle,
                            num_workers=num_workers
                                        ) #for graph in self.graphs]
        neighborsampler = NeighborSampler(
            graph.edge_index,
            # node_idx=train_idx,
            # directed=False,
            sizes=num_neighbors,
            batch_size=batch_size,
            # subgraph_type='induced',  # for undirected graph
            # directed=True,
            shuffle=shuffle,
            num_workers=num_workers
        )
        return neighborsampler



    def forward(self, subset_xs, subset_adjs, super_xs, super_adj, multilevel_sizes, num_empty=0):
        subset_outs = []
        subset_node_filter_values = []
        empty_count = len(self.levelset_modules) - len(subset_xs)
        max_sample_size = 0
        for i, (subset_x, subset_adj) in enumerate(zip(subset_xs, subset_adjs)):
            single_sample = False
            if subset_x.size(0) <= 1: # invalid if empty (0) and can't use batch norm if one element
                if subset_x.size(0) == 0:
                    padding = torch.zeros(subset_xs[-1].size(0),
                                         self.dim_hidden).to(subset_x.device)
                    subset_outs.append(padding)
                    continue  # Skip if no valid nodes
                else:
                    single_sample = True

            subset_x = self.normalizers["in"](subset_x)

            subset_x, node_filter_values = self.levelset_modules[0](subset_x,
                                                                        subset_adj,
                                                                        filtration_function_in=None,#self.levelset_graph_filtration_functions_in[i],         #filtration function of input node reps
                                                                        filtration_function_hidden=None,#self.levelset_graph_filtration_functions_hidden[0],
                                                                        filtration_function_out=self.levelset_graph_filtration_functions_out[i],
                                                                        single_sample=single_sample) #filtration function of hidden reps
            subset_x = self.normalizers["hid"](subset_x)

            subset_outs.append(subset_x)

            subset_node_filter_values.append(node_filter_values)


            if max_sample_size < subset_x.size(0):
                max_sample_size = subset_xs[-1].size(0) #subset_x.size(0)

        uninformative_subset = []
        for i in range(num_empty):
            padding = torch.zeros(max_sample_size,
                                 self.dim_hidden).to(subset_outs[-1].device)
            uninformative_subset.append(padding)
            subset_outs = subset_outs + uninformative_subset

        # We only want to concatenate nodes that survive
        # topological filtration across a sequence of multiple
        # simplification graphs in the filtration hierarchy
        # different  number of neighbors can be in supersamples
        padded_subset_outs = []
        for i, out in enumerate(subset_outs):
            # multilevel_size = multilevel_sizes[i]
            if out.size(0) > max_sample_size:
                out = out[:max_sample_size]
            target_size = subset_adjs[-1][-1][2][1]
            num_super_samp = subset_xs[-1].size(0)

            # out = out[:multilevel_size]         # should i truncate to only same ndes?
            if out.size(0) < max_sample_size:
                padding_size = max_sample_size - out.size(0)
                padding = torch.zeros(padding_size, out.size(1)).to(out.device)
                out = torch.cat([out, padding], dim=0)
            out = out[:target_size]
            padded_subset_outs.append(out)

        # supx = subset_xs[-1]
        # for i, (edge_index, _, size) in enumerate(subset_adjs[-1]):#super_adj):
        #     x_target = supx[:size[1]]
        #     supx = self.multiscale_nbr_msg_aggr[i]((supx,x_target),edge_index)
        #     supx = self.batch_norms[i](supx)
        #     if i != len(subset_adjs[-1])-1:
        #         supx = self.act(supx)

        # # now combine aggregated accross graph level
        # # message passing embeddings with supergraph between nbr
        # # level messaeg passing embedding
        # # x = self.combine_comp(*(x,super_xs), device=super_xs.device)
        x = torch.cat(padded_subset_outs, dim=1)

        padding_mask = x != 0.0
        x = self.normalizers["comb"](x, padding_mask)

        x = self.feed_forward(x)

        x = self.normalizers["out"](x, padding_mask)

        x = self.lin_out(x)

        return self.probability(x).squeeze(1), subset_node_filter_values

    def train_net(self, input_dict):
        return self.hierarchical_joint_train_net(input_dict)
    def hierarchical_joint_train_net(self, input_dict, assign_filter_values=True):
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        grad_scalar = input_dict["grad_scalar"]
        scheduler = input_dict["scheduler"]

        total_epochs = input_dict["total_epochs"]
        epoch = input_dict['epoch']
        eval_steps = input_dict['eval_steps']
        input_steps = input_dict["steps"]

        val_data = input_dict["val_data"]
        val_input_dict = {"type":"validation",
                          "device": device,
                          "loss_op": loss_op}
        self.validation_dict["device"] = device
        self.validation_dict["loss_op"] = loss_op

        self.train()
        self.training = True

        approx_thresholds = np.arange(0.0, 1.0, 0.1)

        total_loss, total_correct, total_training_points = 0, 0, 0
        predictions, all_labels = [], []

        for module in self.levelset_modules:
            module.set_device(device)
        # self.levelset_graph_filtration_functions = [(f_in.set_device(device),
        #                                              f_hid.set_device(device))
        #                                             for f_in, f_hid in self.levelset_graph_filtration_functions]
        # for filtration_function in self.levelset_graph_filtration_functions_in:
        #     filtration_function.set_device(device)
        # for filtration_function_layers in self.levelset_graph_filtration_functions_hidden:
        #     for filtration_function in filtration_function_layers:
        #         filtration_function.set_device(device)
        for filtration_function in self.levelset_graph_filtration_functions_out:
            filtration_function.set_device(device)
        for normer in self.normalizers.keys():
            if normer != "pad":
                self.normalizers[normer].to(device)
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]
        # self.target_batch_norms = [bn.to(device) for bn in self.target_batch_norms]
        # self.multiscale_nbr_msg_aggr = [layer.to(device) for layer in self.multiscale_nbr_msg_aggr]
        # self.target_super_nbr_aggr = [layer.to(device) for layer in self.target_super_nbr_aggr]
        node_filter_values = [{} for i in self.graphs]

        for node_indices, subset_samples, multilevel_sizes in self.hierarchical_graph_sampler:
            # if node_indices.numel() == 0:
            #     break  # End of epoch
            optimizer.zero_grad()
            loss = 0

            subset_bs = []
            subset_adjs = []
            subset_xs = []
            subset_ys = []
            skip_count = 0
            subset_nids = []
            for i, (batch_size, n_id, adjs) in enumerate(subset_samples):
                if len(n_id) == 0:
                    skip_count+=1
                    continue  # Skip empty batches
                adjs = [adj.to(device) for adj in adjs]
                subset_adjs.append(adjs)
                # can also set n_id to sampled indices
                # n_id = n_id[:batch_size]#node_indices[i]
                subset_nids.append(n_id)
                subset_xs.append(self.graphs[i].x[n_id].to(device))
                subset_ys.append(self.graphs[i].y[n_id].to(device))
                subset_bs.append(batch_size)
                total_training_points += batch_size

            super_xs = subset_xs[-1]#self.graphs[-1].x[node_indices].to(device)
            super_nid = subset_nids[-1]
            super_adj = subset_samples[-1][2]
            super_adj = [adj.to(device) for adj in super_adj]
            # pout((f"length of sampled indices vs lenfth of indices sampled from sampler", f"length sampled {len(node_indices)}", f"subset xs {subset_xs[-1].size()}"))



            with autocast():
                out, subset_node_filter_values = self(subset_xs=subset_xs,
                                               subset_adjs=subset_adjs,
                                               super_xs=super_xs,
                                               super_adj=super_adj,
                                               multilevel_sizes=multilevel_sizes,
                                               num_empty=skip_count)
            if assign_filter_values:
                for sub_i, filter_values in enumerate(subset_node_filter_values.detach().cpu().numpy()):
                # self.graphs[i].
                    node_filter_values[sub_i][subset_nids[sub_i].detach().cpu().numpy()] = subset_node_filter_values[sub_i].detach().cpu().numpy()
                # for edge, p in zip(e_id.detach().cpu().numpy(), out.detach().cpu().numpy()):
                #     source, target = global_edge_index[edge]
                #     edges.append((source, target))
                #     edge_weights_dict[(source, target)] = p  # [0]#[p]# edge_weights_dict[edge] = [p]#.append

            # y = self.super_graph.y[node_indices].to(device)#n_id[:batch_size]].to(device)
            # out_size = out.size(0)
            # num_sampled = np.min([out_size, self.batch_size])
            supergraph_bs = subset_bs[-1]
            y = self.graphs[-1].y[super_nid].to(device)#torch.tensor(subset_ys).to(device)
            y=y[:supergraph_bs]
            out=out#[:supergraph_bs]
            loss = loss_op(out, y)

            grad_scalar.scale(loss).backward()
            grad_scalar.step(optimizer)
            grad_scalar.update()
            # optimizer.zero_grad()


            total_loss += loss.item()

            if self.num_classes > 2:  # (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                #total_correct += out.argmax(dim=-1).eq(y.argmax(dim=-1)).float().item()
                total_correct += float(out.argmax(axis=1).eq(y).sum())
                all_labels.extend(y)  # ground_truth)
                predictions.extend(out.argmax(axis=1))
            else:
                predictions.extend(out)#preds)
                all_labels.extend(y)#ground_truth)
                # # total_correct += (out.long() == thresh_out).float().sum().item()#int(out.eq(y).sum())
                # total_correct += (y == thresh_out).float().sum().item()  # int(out.eq(y).sum())
                # # approx_acc = (y == thresh_out).float().mean().item()

            del adjs, batch_size, n_id, loss, out, \
                super_adj, super_xs, super_nid, \
                subset_xs, subset_adjs, subset_bs, y, \
                node_indices, subset_samples
            torch.cuda.empty_cache()

        predictions = torch.tensor(predictions)
        all_labels = torch.tensor(all_labels)
        # all_labels=torch.cat(all_labels,dim=0)

        num_classes = len(all_labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets != 1:
            # approx_acc = total_correct/all_labels.numel()
            approx_acc = (predictions == all_labels).float().mean().item()

            del predictions, all_labels

        else:
            total_correct = int(predictions.eq(all_labels).sum())
            all_labels = all_labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            approx_acc = total_correct/all_labels.shape[0]
            # approx_acc = 0
            # train_opt_thresh, approx_acc = optimal_metric_threshold(y_probs=predictions,
            #                                                                     y_true=all_labels,
            #                                                                     metric=accuracy_score,
            #                                                                     metric_name='accuracy',
            #                                                           num_targets=num_targets,
            #                                                         thresholds=approx_thresholds)

        if epoch % eval_steps == 0 and epoch != 0:
            with torch.no_grad():
                self.eval()
                self.training = False
                val_pred, val_loss, val_ground_truth = self.inference(val_input_dict)
            self.training = True
            self.train()
            # val_out, val_loss, val_labels = self.inference(val_input_dict)
            # if scheduler is not None:
            #     scheduler.step(val_loss)
            # num_classes = len(all_labels.unique())
            # num_targets = 1 if num_classes == 2 else num_classes
            if num_targets != 1:
                # predictions = predictions
                # approx_acc = (predictions == all_labels).float().mean().item()

                val_pred = val_pred
                val_acc = (val_pred == val_ground_truth).float().mean().item()
                # val_acc = (val_pred.argmax(axis=-1) == val_ground_truth).float().mean().item()
                print("Epoch: ", epoch,  f" Validation ACC: {val_acc:.4f}")
                val_roc = 0
                del val_pred, val_ground_truth

            else:
                # if False:  # because too slow
                val_pred = val_pred.detach().cpu().numpy()
                val_ground_truth = val_ground_truth.detach().cpu().numpy()
                # val_acc = accuracy_score(val_ground_truth, val_pred)
                val_optimal_threshold, val_acc = optimal_metric_threshold(y_probs=val_pred,
                                                                                    y_true=val_ground_truth,
                                                                                    metric=accuracy_score,
                                                                                    metric_name='accuracy',
                                                                          num_targets=num_targets,
                                                                          thresholds=approx_thresholds)

                val_thresh, val_roc = optimal_metric_threshold(val_pred,
                                                                 val_ground_truth,
                                                                 metric=metrics.roc_auc_score,
                                                                 metric_name='ROC AUC',
                                                               thresholds=approx_thresholds)

                # all_labels = all_labels.detach().cpu().numpy()
                # predictions = predictions.detach().cpu().numpy()
                train_opt_thresh, approx_acc = optimal_metric_threshold(y_probs=predictions,
                                                                                    y_true=all_labels,
                                                                                    metric=accuracy_score,
                                                                                    metric_name='accuracy',
                                                                          num_targets=num_targets,
                                                                        thresholds=approx_thresholds)

                print("Epoch: ", epoch, f" Validation ACC: {val_acc:.4f}",
                      f" Validation ROC: {val_roc:.4f}")

                # if scheduler is not None:
                #     scheduler.step(val_loss)

                del val_pred, val_ground_truth

        #
        # get average learned filter value assigned to nodes within each subgraph
        #
        pout(( "AVERAGE NODE FILTER FUNCTION VALUE FOR EACH SUBLEVEL GRAPH"))
        for i,subgraph_node_filter_values in enumerate(node_filter_values):
            #subgraph_node_filter_vals = list(self.graphs[i].node_filter_values.values())#.detach().cpu().numpy()
            pout(("SUBGRAPH "))
            pout((i, "AVERAGE NODE FILTER VALUE"))
            pout(("GRAPH ",i," AVG ", np.mean(subgraph_node_filter_values)))

        torch.cuda.empty_cache()

        if scheduler is not None:
            scheduler.step(total_loss / total_training_points)

        if epoch % eval_steps == 0 and epoch != 0:
            return total_loss/total_training_points , approx_acc, (val_loss, val_acc, val_roc)
        else:
            return total_loss / total_training_points, approx_acc, (666, 666, 666)



    def inference(self, input_dict):
        r"""
        Given input data, applies Hierarchical Filtration Function to infer
        filtration alues to edge, performes topological filtration for persistence graph hierarchy,
        and infers with trained HJT model
        :param input_dict:
        :return:
        """
        self.eval()
        self.training = False

        device = input_dict["device"]
        loss_op = input_dict["loss_op"]

        # total_epochs = input_dict["total_epochs"]
        # epoch = input_dict['epoch']
        # eval_steps = input_dict['eval_steps']
        # input_steps = input_dict["steps"]


        ###############################################################
        # if a filtration function hasn't been applied to the data,
        # e.g. edge inference for filter function value assignment
        # (logistic prediction of an edge as homopholous for pers-
        # sistence hierarchy.
        ##############################################################
        # if self.filtration_function:
        #     _, _, _, data = self.graph_hierarchy_filter_function(input_dict,
        #                                                          assign_edge_weights=True)
        if input_dict["type"] == "new":
            pout(("%%%%%%%%%%%%%","Computing Inference Hierarchy","%%%%%%%%%%%%%"))
            # compute persistence filtration for graph hierarchy
            data = input_dict["data"]
            pout(("Performing Filtration On Inference Dataset"))
            filtration_graph_hierarchy = FiltrationGraphHierarchy(graph=data,
                                                                  persistence=self.thresholds,
                                                                  filtration=None)
            pout(("Filtration on Inference Done"))
            graphs, sub_to_sup_mappings, sup_to_sub_mapping = (filtration_graph_hierarchy.graphs,
                                                               filtration_graph_hierarchy.sub_to_sup_mappings,
                                                               filtration_graph_hierarchy.supergraph_to_subgraph_mappings)

            pout(("FINISHED CREATED INFERENCE GRAPH HIERARCHY", "Total graphs: ", len(graphs)))
            graph_levels = np.flip(np.arange(len(graphs) + 1))
            # get neighborhood loaders for each sublevel graph
            sublevel_graph_loaders = [self.get_graph_dataloader(graph,
                                                                shuffle=False,
                                                                num_neighbors=[-1],
                                                                num_workers=0,
                                                                batch_size=self.batch_size) for graph in graphs]
            super_graph = graphs[-1]
            # hierarchical graph neighborhood sampler
            hierarchical_graph_sampler = MultiScaleGraphFiltrationSampler(super_data=super_graph,
                                                                          subset_data= graphs,
                                                                          subset_samplers=sublevel_graph_loaders,
                                                                          subset_to_super_mapping=sub_to_sup_mappings,
                                                                          super_to_subset_mapping=sup_to_sub_mapping,
                                                                          batch_size=self.batch_size,
                                                                          shuffle=False)
        elif input_dict["type"]=="validation":
            pout(("%%%%%%%%%%%%%","Inferring on Validation Hierarchy","%%%%%%%%%%%%%"))
            input_dict = self.validation_dict
            filtration_graph_hierarchy = input_dict["graph_hierarchy"]
            val_graph_hier = input_dict["graph_hierarchy"]
            graphs = input_dict["graphs"]
            sub_to_sup_mappings = input_dict["sub_to_sup_map"]
            sup_to_sub_mapping = input_dict["sup_to_sub_map"]
            sublevel_graph_loaders = input_dict["subgraph_loaders"]
            super_graph = input_dict["super_graph"]

            hierarchical_graph_sampler = input_dict["hierarchical_graph_sampler"]

        elif input_dict["type"] == "test":
            pout(("%%%%%%%%%%%%%","Inferring on Test Hierarchy","%%%%%%%%%%%%%"))
            input_dict = self.testing_dict
            filtration_graph_hierarchy = input_dict["graph_hierarchy"]
            val_graph_hier = input_dict["graph_hierarchy"]
            graphs = input_dict["graphs"]
            sub_to_sup_mappings = input_dict["sub_to_sup_map"]
            sup_to_sub_mapping = input_dict["sup_to_sub_map"]
            sublevel_graph_loaders = input_dict["subgraph_loaders"]
            super_graph = input_dict["super_graph"]

            hierarchical_graph_sampler = input_dict["hierarchical_graph_sampler"]
        else:
            raise NotImplementedError(
                """please specify 'type' (validation, test, new) in input dict
                for data source or, if 'new', to compute with data in input dict""")

        for module in self.levelset_modules:
            module.set_device(device)
        # self.levelset_graph_filtration_functions = [(f_in.set_device(device),
        #                                              f_hid.set_device(device))
        #                                             for f_in, f_hid in self.levelset_graph_filtration_functions]
        # for filtration_function in self.levelset_graph_filtration_functions_in:
        #     filtration_function.set_device(device)
        # for filtration_function_layers in self.levelset_graph_filtration_functions_hidden:
        #     for filtration_function in filtration_function_layers:
        #         filtration_function.set_device(device)
        for filtration_function in self.levelset_graph_filtration_functions_out:
            filtration_function.set_device(device)
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]
        # self.target_batch_norms = [bn.to(device) for bn in self.target_batch_norms]
        # self.multiscale_nbr_msg_aggr = [layer.to(device) for layer in self.multiscale_nbr_msg_aggr]

        approx_thresholds = np.arange(0.0, 1.0, 0.1)

        total_loss, total_correct, total_training_points= 0, 0, 0
        predictions, labels = [], []

        # self.batch_norms = [bn.to(device) for bn in self.batch_norms]
        with torch.no_grad():
            for node_indices, subset_samples, multilevel_sizes in hierarchical_graph_sampler:


                subset_bs = []
                subset_adjs = []
                subset_xs = []
                subset_ys = []
                subset_nids = []
                skip_count = 0
                for i, (batch_size, n_id, adjs) in enumerate(subset_samples):
                    if len(n_id) == 0:
                        skip_count += 1
                        continue  # Skip empty batches
                    # adjs = [adj.to(device) for adj in adjs]
                    adjs = [adjs.to(device)]
                    subset_adjs.append(adjs)
                    # n_id = n_id[:batch_size]  #
                    subset_nids.append(n_id)
                    subset_xs.append(graphs[i].x[n_id].to(device))
                    subset_ys.append(graphs[i].y[n_id].to(device))
                    subset_bs.append(batch_size)

                super_xs = subset_xs[-1]  # self.graphs[-1].x[node_indices].to(device)
                super_adj = subset_adjs[-1]#subset_samples[-1][2]
                # super_adj = [adj.to(device) for adj in super_adj]
                super_nid = subset_nids[-1]

                with autocast():
                    out, filtration_values = self(subset_xs=subset_xs,
                                                   subset_adjs=subset_adjs,
                                                   super_xs=super_xs,
                                                   super_adj=super_adj,
                                                   multilevel_sizes=multilevel_sizes,
                                                   num_empty=skip_count)[:subset_bs[-1]]

                supergraph_bs = subset_bs[-1]
                total_training_points += supergraph_bs
                y = graphs[-1].y[super_nid[:supergraph_bs]].to(device)  # torch.tensor(subset_ys).to(device)
                # y = y
                # y = torch.tensor(subset_ys).to(device)
                # y = y[:supergraph_bs]
                # out = out[:supergraph_bs]

                # pout(("ys in inference hjt ", y))

                loss = loss_op(out, y)
                total_loss += loss.item()
                # for metrics
                if self.num_classes > 2:
                    pred = out.argmax(axis=1)
                else:
                    pred = out
                predictions.extend(pred.detach().cpu().numpy())
                labels.extend(y.detach().cpu().numpy())

                # del adjs, batch_size, n_id, loss, out,\
                #     subset_xs, subset_adjs, y,\
                #     node_indices, subset_samples
                del subset_adjs, y, out, \
                    super_nid, super_xs, super_adj, \
                    subset_xs, subset_bs, subset_ys, subset_nids
                torch.cuda.empty_cache()

        total_loss = total_loss / float(total_training_points)
        # pout(("PREDS ", predictions, "LABELS:", labels))
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)
        return predictions, total_loss, labels

    def sanity_check_nid_assignments(self,
                                     subset_nids,
                                     graphs,
                                     sub_to_sup_mappings,
                                     check_training=True,
                                     check_validation=False):

        if check_training:
            feat_sim = []
            # sim_mapped_ids = []
            # super_nid_from_super = [self.sub_to_sup_mappings[-1][int(gid.detach().cpu())] for gid in super_nid]
            sim_nid = []
            for i, xs in enumerate(subset_nids):
                print(xs)
                # check_node_maps(xs, super_xs, self.sub_to_sup_mappings[i], self.sub_to_sup_mappings[-1])
                gnids = []
                feats = []
                feats.append(graphs[i].x[xs].detach().cpu())
                for k,sub_idx in enumerate(xs):
                    # print(sub_idx)
                    gid = sub_to_sup_mappings[i][int(sub_idx.detach().cpu())]
                    gnids.append(gid)


                    # super_nid_from_j = self.sup_to_sub_mapping[-1][gid]
                    # sim_mapped_ids.append(super_nid_from_j in super_nid)
                    # sim_mapped_to_sup_to_sup.append(super_nid_from_j in super_nid_from_super)
                    # super_x_from_j = self.graphs[-1].x[super_nid_from_j]
                    # x_with_j = self.graphs[i].x[int(sub_idx.detach().cpu())]
                    # feat_sim.append(super_x_from_j == x_with_j)
                sim_nid.append(gnids)
                feat_sim.append(feats)



            print("simnid", sim_nid)
            id_eq = []
            for x,y in zip(sim_nid[0], sim_nid[1]):
                id_eq.append(x==y)#(x == sim_nid[0] for x in sim_nid))
            print("ids and index equal", id_eq)
            # print("id set equal", len(set(sim_nid[0]+[s[1:] for s in sim_nid]))==1)
            all_nids = sim_nid[0]
            for s in sim_nid[1:]:
                all_nids.extend(s)
            print(" set compare")
            print("number NOT matching ids: all id collection length - set all id collection: ",
                  len(all_nids) - len(set(all_nids)))
            print("set all:")
            print(set(all_nids))
            print("length compare")

            for l,s in enumerate(sim_nid):
                print("graph level ",l, " number nids: ",len(s))
            # print("feats equal: ", np.all(np.array(feat_sim)==feat_sim[0]))
            # print("feat set equal ", len(set(feat_sim[0] + [f[1:] for f in feat_sim])) == 1)
            # print("feasim", feat_sim)

            # pout(( "similarity of node feaurtes from maps",feat_sim,"mapped ids correct ",sim_mapped_ids))
        if check_validation:
            # y = self.super_graph.y[node_indices].to(device)#n_id[:batch_size]].to(device)
            # out_size = out.size(0)
            # num_sampled = np.min([out_size, self.batch_size])
            feat_sim = []
            # sim_mapped_ids = []
            # super_nid_from_super = [self.sub_to_sup_mappings[-1][int(gid.detach().cpu())] for gid in super_nid]
            sim_nid = []
            for i, xs in enumerate(subset_nids):
                print(xs)
                # check_node_maps(xs, super_xs, self.sub_to_sup_mappings[i], self.sub_to_sup_mappings[-1])
                gnids = []
                feats = []
                feats.append(graphs[i].x[xs].detach().cpu())
                for k, sub_idx in enumerate(xs):
                    # print(sub_idx)
                    gid = sub_to_sup_mappings[i][int(sub_idx.detach().cpu())]
                    gnids.append(gid)

                    # super_nid_from_j = self.sup_to_sub_mapping[-1][gid]
                    # sim_mapped_ids.append(super_nid_from_j in super_nid)
                    # sim_mapped_to_sup_to_sup.append(super_nid_from_j in super_nid_from_super)
                    # super_x_from_j = self.graphs[-1].x[super_nid_from_j]
                    # x_with_j = self.graphs[i].x[int(sub_idx.detach().cpu())]
                    # feat_sim.append(super_x_from_j == x_with_j)
                sim_nid.append(gnids)
                feat_sim.append(feats)

            print("simnid", sim_nid)
            id_eq = []
            for x, y in zip(sim_nid[0], sim_nid[1]):
                id_eq.append(x == y)  # (x == sim_nid[0] for x in sim_nid))
            print("ids and index equal", id_eq)
            all_nids = sim_nid[0]
            for s in sim_nid[1:]:
                all_nids.extend(s)

            print(" set compare")
            print("number NOT matching ids: all id collection length - set all id collection: ",
                  len(all_nids) - len(set(all_nids)))
            print("set all:")
            print(set(all_nids))
            print("number common: ", len(set(all_nids)))
            print("length compare")
            for l, s in enumerate(sim_nid):
                print("graph level ", l, " number nids: ", len(s))
            # print("feats equal: ", np.all(np.array(feat_sim)==feat_sim[0]))
            # print("feat set equal ", len(set(feat_sim[0] + [f[1:] for f in feat_sim])) == 1)
            # print("feasim", feat_sim)

            # pout(( "similarity of node feaurtes from maps",feat_sim,"mapped ids correct ",sim_mapped_ids))
    def average_node_filtration_value(self, graph):
        return graph
class GraphFiltrationFamilyWrapper():
    def __init__(self, data, thresholds, num_neighbors, num_layers, batch_size, filter_function=None,input_dict=None):
        ###############################################################
        # if a filtration function hasn't been applied to the data,
        # e.g. edge inference for filter function value assignment
        # (logistic prediction of an edge as homopholous for pers-
        # sistence hierarchy.
        ##############################################################
        #edge mlp to assign predicted weights to use as real values for edge filtration
        if filter_function:
            _, _, _, data = filter_function(input_dict, assign_edge_weights=True)
        # compute persistence filtration for graph hierarchy
        pout(("Performing Filtration,"
              "On",
              "Inference Dataset"))
        self.filtration_graph_hierarchy = FiltrationGraphHierarchy(graph=data,
                                                              persistence=thresholds,
                                                              filtration=None)
        pout(("Filtration on",
              "Inference",
              "Done"))
        self.graphs, self.sub_to_sup_mappings, self.sup_to_sub_mapping = (self.filtration_graph_hierarchy.graphs,
                                                           self.filtration_graph_hierarchy.sub_to_sup_mappings,
                                                           self.filtration_graph_hierarchy.supergraph_to_subgraph_mappings)

        pout(("%%%%%%%" "FINISHED CREATED INFERENCE GRAPH HIERARCHY", "Total graphs: ", len(self.graphs), "%%%%%%%"))
        graph_levels = np.flip(np.arange(len(self.graphs) + 1))
        # get neighborhood loaders for each sublevel graph
        self.sublevel_graph_loaders = [get_graph_dataloader(graph,
                                                                 shuffle=False,
                                                                 num_neighbors=num_neighbors,
                                                            num_layers=num_layers,
                                                            batch_size=batch_size)
                                  for graph in self.graphs]
        super_graph = self.graphs[-1]
        # hierarchical graph neighborhood sampler
        self.hierarchical_graph_sampler = MultiScaleGraphFiltrationSampler(super_data=super_graph,
                                                                           subset_samplers=self.sublevel_graph_loaders,
                                                                           subset_to_super_mapping=self.sub_to_sup_mappings,
                                                                           super_to_subset_mapping=self.sup_to_sub_mapping,
                                                                           batch_size=batch_size,
                                                                           shuffle=False)
    def get_graphs(self):
        return self.graphs
    def get_node_maps(self):
        return self.sub_to_sup_mappings, self.sup_to_sub_mapping

    def get_graph_loaders(self):
        return self.sublevel_graph_loaders

    def get_graph_sampler(self):
        return self.hierarchical_graph_sampler



def get_graph_dataloader(graph, batch_size, num_layers, shuffle=False, num_neighbors=[-1]):

    if graph.num_nodes < 1200:
        num_workers = 0
    else:
        num_workers = 8

    if shuffle:
        neighborloader = NeighborLoader(data=graph,
                              batch_size=batch_size,
                              num_neighbors=num_neighbors[: num_layers],
                                        subgraph_type='induced', #for undirected graph
                              # directed=False,#True,
                              shuffle=shuffle,
                            num_workers=num_workers
                                        ) #for graph in self.graphs]
        neighborsampler = NeighborSampler(
            graph.edge_index,
            # node_idx=train_idx,
            # directed=False,
            sizes=num_neighbors[: num_layers],
            batch_size=batch_size,
            # subgraph_type='induced',  # for undirected graph
            # directed=True,
            shuffle=shuffle,
            num_workers=num_workers
        )
    else:
        neighborloader = NeighborLoader(data=graph,
                              batch_size=batch_size,
                              num_neighbors=num_neighbors,
                                        subgraph_type='induced', #for undirected graph
                              # directed=False,#True,
                              shuffle=shuffle
                                        # num_workers=8
                                        ) #for graph in self.graphs]

        neighborsampler = NeighborSampler(
            graph.edge_index,
            # node_idx=train_idx,
            # directed=False,
            sizes=num_neighbors,
            batch_size=batch_size,
            # subgraph_type='induced',  # for undirected graph
            # directed=True,
            shuffle=shuffle,
            # num_workers=8,
        )
    return neighborsampler

class UniformativeDummyEmbedding(nn.Module):
    def __init__(self, dim_out, dim_in=1):
        super().__init__()
        b = torch.ones(dim_in, dim_out, dtype=torch.float)
        self.register_buffer('ones', b)

    def forward(self, batch):
        assert batch.dtype == torch.long
        return self.ones.expand(batch.size(0), -1)

    @property
    def dim(self):
        return self.ones.size(1)

