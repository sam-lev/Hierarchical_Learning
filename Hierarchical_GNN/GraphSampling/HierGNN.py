# import os.path as osp
# import os
# from typing import List, Optional, Tuple, Union
# import json
# import time
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
# from torch_geometric.nn.conv import MessagePassing
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
#
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from .utils import pout, homophily_edge_labels,  edge_index_from_adjacency, node_degree_statistics
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

# def
class SubgraphFilterGNN(nn.Module):
    def __init__(self, in_dim,
                 dim_hidden,
                 out_dim,
                 num_layers,
                 dropout=0,
                 use_batch_norm = True):

        super().__init__()

        self.num_feats = in_dim
        self.dim_hidden = dim_hidden
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.model_nbr_msg_aggr = torch.nn.ModuleList()
        # GraphSage, ego- neighborhood embedding seperation performs better
        self.model_nbr_msg_aggr.append(SAGEConv(in_channels=self.num_feats,
                                          out_channels=self.dim_hidden))
        # self.bns.append(nn.BatchNorm1d(n_2)) self.bns = nn.ModuleList()
        self.batch_norms = []
        batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        for _ in range(self.num_layers - 1):
            self.model_nbr_msg_aggr.append(SAGEConv(in_channels=self.dim_hidden,
                                              out_channels=self.dim_hidden))
            # batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)

        self.model_out_embedding = nn.Linear(self.dim_hidden,  # * self.num_layers,
                                       self.out_dim)

        batch_norm_layer = nn.LayerNorm(self.out_dim) if self.use_batch_norm else nn.Identity(self.out_dim)
        self.batch_norms.append(batch_norm_layer)
        self.dropout = nn.Dropout(self.dropout)
        self.act = nn.GELU()
        self.jump = JumpingKnowledge(mode='cat')
        # self.probability = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)  # nn.Softmax(dim=1) #nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        for embedding_layer in self.model_nbr_msg_aggr:
            embedding_layer.reset_parameters()
        self.model_out_embedding.reset_parameters()

    def set_device(self, device):
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]
        self.to(device)

    def forward(self, x, adjs, filtration_function_in, filtration_function_hidden, filtration_function_out, single_sample=False):

        xs = []
        num_targets = 0
        edge_index = adjs[0][0]
        x_source = x
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]
            x_source = x[size[1]:]
            num_targets = size[1]
            edge_index = edge_index
            #
            # pout(("in gnnmoodule",f"x in size {x.size()}", f"in x target size{x_target.size()}"))
            if i == 0:
                sub_filtration_value = filtration_function_in(x=x,
                                                              x_target=x_target,
                                                              edge_index=edge_index,
                                                              degree=None,
                                                              single_sample=single_sample,
                                                              seperate=False)
            else:
                sub_filtration_value = filtration_function_hidden(x=x,
                                                                  x_target=x_target,
                                                                  edge_index=edge_index,
                                                                  degree=None,
                                                                  single_sample=single_sample,
                                                                  seperate=False)
            x = self.model_nbr_msg_aggr[i](x, edge_index)# (x,x_target), edge_index)#(x, x_target), edge_index)
            x = self.dropout(x)
            # if not single_sample:
            x = self.batch_norms[i](x)
            if i!=self.num_layers-1:
                x = self.act(x)

            # x = sub_filtration_value_hidden[:size[1]] * x  # multiply new target node embeddings with filter coefficient
            # if i!=0:
            #     x = x[:size[1]] + x_target

            #filtration coefficient scaling
            x = sub_filtration_value * x

            x_target = x[:size[1]]
            xs.append(x)

        # x = self.jump(xs)
        # x = self.model_out_embedding(x)
        # pout(("x out size",x.size(), "out x target size" , x_target.size()))

        # sub_filtration_value = filtration_function_hidden(x=x, #(x_source, x[: num_targets]),
        #                                                   x_target=x[: num_targets],
        #                                                   edge_index=edge_index,
        #                                                   degree=None,
        #                                                   single_sample=single_sample)
        # scale new embeddings by filtration coefficient
        # x = sub_filtration_value * x[:num_targets]
        x_target = x[:num_targets]
        return x,x_target #.squeeze(1)


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

        pout(("%%%%%%%" "FINISHED CREATED GRAPH HIERARCHY", "Total graphs: ", len(self.graphs), "%%%%%%%"))
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
            pout(("map length ", len(m)))
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
class SubGraphFiltration(torch.nn.Module):
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
                 eps: float = 0.5, # we initialize epsilon based on
                 # number of graph levels so all initially contribute equally
                 cat_seperate=False,
                 use_node_degree=None,
                 set_node_degree_uninformative=None,
                 use_node_feat=None,
                 gin_number=None,
                 gin_dimension=None,
                 gin_mlp_type=None,
                 **kwargs
                 ):
        super().__init__()

        dim = gin_dimension

        max_node_deg = max_node_deg
        num_node_feat = dim_in

        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        if use_node_degree:
            self.embed_deg = nn.Embedding(max_node_deg + 1, dim)
        else:
            self.embed_deg = None

        # self.embed_feat = nn.Embedding(num_node_feat, dim, dtype=torch.float) if use_node_feat else None

        # self.embed_feat= nn.Embedding.from_pretrained(data.x,
        #                                                     freeze=False).requires_grad_(True) if use_node_feat else None
        # self.edge_embeddings.weight.data.copy_(train_data.edge_attr)
        # self.edge_embeddings.weight.requires_grad = True
        # self.edge_embeddings#.to(device)

        #dim_input = dim * ((self.embed_deg is not None) + (self.embed_feat is not None))
        self.cat = 2 if cat_seperate else 1
        dims = [self.cat * dim_in] + (gin_number) * [dim] # [dim_input] + (gin_number) * [dim]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.GELU() # torch.nn.functional.leaky_relu

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            # pout(("n1 ", n_1, " n2 ",n_2))
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, eps=eps,train_eps=True))
            batch_norm_layer = nn.LayerNorm(n_2)# nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.bns.append(batch_norm_layer)

        self.fc = nn.Sequential(
            nn.Linear(sum(dims), dim),
            nn.LayerNorm(dim),#nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim_in),
            nn.LayerNorm(dim_in),#nn.BatchNorm1d(dim_in),
            nn.GELU(),
            nn.Linear(dim_in, 1),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        self.fc.reset_parameters()

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
        z=[x]
        for conv, bn in zip(self.convs, self.bns):
            # pout((f"in filtration x size {x.size()} and x_target size {x_target.size()}"))
            x = conv(x,edge_index)#z[-1], edge_index)
            # if not single_sample:
            x = bn(x)
            x = self.act(x)
            z.append(x)
        # x = z[-1]
        filt_feat_val = z[-1]
        # x = self.global_pool_fn(x, batch.batch)
        # z=z[1:]
        # z = [z[0][1]]+z[1:]
        x = torch.cat(z, dim=1)
        ret = self.fc(x)
        # ret = ret.squeeze()
        return ret.view(-1,1)


class FiltrationHierarchyGraphSampler:
    def __init__(self,
                 super_data,
                 subset_samplers,
                 subset_to_super_mapping,
                 super_to_subset_mapping,
                 batch_size,
                 shuffle=False):
        self.super_data = super_data
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
        # pout(("end idx ", self.current_idx))

        # # Get valid nodes for each subset graph
        # valid_subset_nodes = []
        # for i,mapping in enumerate(self.subset_to_super_mapping):
        #     valid_nodes = [n for n in sampled_indices.tolist() if n in mapping.values()]
        #     valid_nodes_sub_idx = [self.super_to_subset_mapping[i][n] for n in valid_nodes]
        #     valid_subset_nodes.append(torch.tensor(valid_nodes_sub_idx))
        #     """ Should I resample subgraph nodes to ensure same sized batches, or add dummy nodes?"""

        # # Get valid nodes for each subset graph
        valid_subset_nodes = []
        for i,mapping in enumerate(self.super_to_subset_mapping):
            valid_nodes = [n for n in sampled_indices.tolist() if n in mapping.keys()]
            valid_nodes_sub_idx = [mapping[n] for n in valid_nodes]
            valid_subset_nodes.append(torch.tensor(valid_nodes_sub_idx))
        # Get neighborhood information from each subset graph
        subset_samples = []
        # for i, sampler in enumerate(self.subset_samplers):
        #     subset_samples.append(sampler.sample(valid_subset_nodes[i]))
        for i, sampler in enumerate(self.subset_samplers):
            if len(valid_subset_nodes[i]) > 0:  # Ensure there are valid nodes to sample .sample_from_nodes()
                subgraph_samples = sampler.collate_fn(valid_subset_nodes[i])
                # pout(("subsamples size ", subgraph_samples[0], " len valid subnodes ", len(valid_subset_nodes[i])))
                # subgraph_samples = sampler.filter_fn(subgraph_samples)
                subset_samples.append(subgraph_samples) # .sample(valid_subset_nodes[i]))
                # sample_from_nodes collate_fn
            else:
                subset_samples.append((0, torch.tensor([]), []))  # Handle empty batches

        return sampled_indices, subset_samples


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
                 ):
        super().__init__()

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
        self.out_dim = out_dim

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

        hierarchicalgraphloader = FiltrationGraphHierarchy(graph=train_data,
                                                           persistence=self.thresholds,
                                                           filtration=None)

        self.graphs, self.sub_to_sup_mappings, self.supergraph_to_subgraph_mappings = (hierarchicalgraphloader.graphs,
                                                                          hierarchicalgraphloader.sub_to_sup_mappings,
                                                                          hierarchicalgraphloader.supergraph_to_subgraph_mappings)
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
            num_workers = 1
        else:
            num_workers = 4

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
    def train_net(self, input_dict):
        return self.hierarchical_successive_train_net(input_dict)


    def aggregate_edge_attr(self, edge_attr, edge_index):
        edge_attr_target = edge_attr[edge_index]
        # pout(("target edge attr", edge_attr_target, "target edge shape", edge_attr_target.shape))
        # torch.mean(torch.max(a, -1, True)[0], 1, True)
        return torch.max(edge_attr_target, -1, True)[0]

    # @profile
    @torch.no_grad()
    def inference(self, input_dict):
        return self.node_inference(input_dict)
    
    def hierarchical_successive_train_net(self, input_dict,
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

            del adjs, batch_size, n_id, loss, out, x, y
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
            self.graph_level += 1
            # Save embeddings for each node in the last graph
            subgraph_embeddings = {}
            with torch.no_grad():
                for batch_size, n_id, adjs in self.graphlevelloader:#batch in self.graphlevelloader:
                    # batch.to(device)
                    # n_id = batch.n_id
                    for nid in n_id.detach().cpu().numpy():
                        n_embedding = data.x[nid].detach().cpu().numpy()
                        subgraph_embeddings[nid] = n_embedding
                    # for i, node in zip(n_id.detach().cpu().numpy(), data.x[n_id].detach().cpu().numpy()):#zip(e_id.detach().cpu().numpy()
                    #     subgraph_embeddings[i] = node#node] = out[i]#.detach().cpu().numpy()

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
            return total_loss/total_training_points , approx_acc, total_val_loss
        else:
            return total_loss / total_training_points, approx_acc, 666

    @torch.no_grad()
    def node_inference(self, input_dict):
        # input dict input_dict = {"data": self.data, "y": self.y, "device": self.device, "dataset": self.dataset}
        self.eval()
        self.training = False
        device = input_dict["device"]
        # x = input_dict["x"].to(device)
        data = input_dict["data"]

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
def check_node_maps( n_i, n_j, sub_to_super_i, sub_to_super_j):
    pout(("similarity of global maps"))
    n_i_super = []
    n_j_super = []
    for i in n_i:
        n_i_super.append(sub_to_super_i[int(i.detach().cpu())])
    for j in n_j:
        n_j_super.append(sub_to_super_j[int(j.detach().cpu())])
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
                 filtration_function=None,
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
        self.thresholds = args.persistence
        self.val_data = val_data
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
        self.hierarchical_graph_sampler = FiltrationHierarchyGraphSampler(super_data=self.super_graph,
                                                                          subset_samplers=self.sublevel_graph_loaders,
                                                                          subset_to_super_mapping=self.sub_to_sup_mappings,
                                                                          super_to_subset_mapping=self.sup_to_sub_mapping,
                                                                          batch_size=self.batch_size,
                                                                          shuffle=True)
        #
        # validation graph hierarchy
        #
        # if self.filtration_function:
        #     _, _, _, data = self.graph_hierarchy_filter_function(input_dict,
        #                                                          assign_edge_weights=True)
        # compute persistence filtration for graph hierarchy
        pout(("Performing Filtration,"
              "On",
              "Inference Dataset"))
        self.val_filtration_graph_hierarchy = FiltrationGraphHierarchy(graph=self.val_data,
                                                              persistence=self.thresholds,
                                                              filtration=None)
        pout(("Filtration on",
              "Inference",
              "Done"))
        self.val_graphs, self.val_sub_to_sup_mappings, self.val_sup_to_sub_mapping = (self.val_filtration_graph_hierarchy.graphs,
                                                           self.val_filtration_graph_hierarchy.sub_to_sup_mappings,
                                                           self.val_filtration_graph_hierarchy.supergraph_to_subgraph_mappings)

        pout(("%%%%%%%" "FINISHED CREATED INFERENCE GRAPH HIERARCHY", "Total graphs: ", len(self.val_graphs), "%%%%%%%"))
        self.val_graph_levels = np.flip(np.arange(len(self.val_graphs) + 1))
        # get neighborhood loaders for each sublevel graph
        self.val_sublevel_graph_loaders = [self.get_graph_dataloader(graph,
                                                                     shuffle=False,
                                                                     num_neighbors=[-1],
                                                                     batch_size=1,
                                                                     num_workers=0)
                                  for graph in self.val_graphs]
        self.val_super_graph = self.val_graphs[-1]
        # hierarchical graph neighborhood sampler
        self.val_hierarchical_graph_sampler = FiltrationHierarchyGraphSampler(super_data=self.val_super_graph,
                                                                     subset_samplers=self.val_sublevel_graph_loaders,
                                                                     subset_to_super_mapping=self.val_sub_to_sup_mappings,
                                                                     super_to_subset_mapping=self.val_sup_to_sub_mapping,
                                                                     batch_size=self.batch_size,
                                                                     shuffle=False)
        # Graph_Filtration_Family_Val = GraphFiltrationFamilyWrapper( data=val_data,
        #                                                             thresholds=self.thresholds,
        #                                                             num_neighbors=[-1],
        #                                                             num_layers=self.num_layers,
        #                                                             batch_size=self.batch_size)
        #
        # self.val_graphs = Graph_Filtration_Family_Val.get_graphs()
        # self.val_sub_to_sup_mappings, self.val_sup_to_sub_mapping = Graph_Filtration_Family_Val.get_node_maps()
        # self.val_sublevel_graph_loaders = Graph_Filtration_Family_Val.get_graph_loaders()
        # self.val_hierarchical_graph_sampler = Graph_Filtration_Family_Val.get_graph_sampler

        ######################################################################################
        #           Define message passing / neighborhood aggration scheme for each
        #             graph levelset for learned node embeddings per-graph level
        ######################################################################################
        self.levelset_modules = [SubgraphFilterGNN(in_dim=self.num_feats,
                                                   dim_hidden=self.dim_hidden,
                                                   out_dim=self.dim_hidden,  #outputs node_embedding of dimension dim hidden
                                                   num_layers=self.num_layers,
                                                   dropout=self.dropout,
                                                   use_batch_norm=self.use_batch_norm) \
                                 for graph in self.graphs]

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
        self.levelset_graph_filtration_functions_in = [SubGraphFiltration(max_node_deg=int(max_degree),
                                                                          dim_in=self.num_feats,  #input nodes
                                                                          eps=epsilon,
                                                                          use_node_degree=False,
                                                                          set_node_degree_uninformative=False,
                                                                          use_node_feat=True,
                                                                          gin_number=1,
                                                                          gin_dimension=self.num_feats * 2,  # as factor
                                                                          gin_mlp_type ='lin_gelu_lin')
                                                       for graph in self.graphs]
        self.levelset_graph_filtration_functions_hidden = [SubGraphFiltration(max_node_deg=int(max_degree),
                                                                              dim_in=self.dim_hidden,
                                                                              eps=epsilon,
                                                                              use_node_degree=False,
                                                                              set_node_degree_uninformative=False,
                                                                              use_node_feat=True,
                                                                              gin_number=1,
                                                                              gin_dimension=self.dim_hidden * 2,
                                                                              gin_mlp_type ='lin_gelu_lin')
                                                           for graph in self.graphs]
        # self.levelset_graph_filtration_functions_out = [SubLevelGraphFiltration(max_node_deg=int(max_degree),
        #                                                                            dim_in=self.dim_hidden,
        #                                                                            eps=epsilon,
        #                                                                            use_node_degree=False,
        #                                                                            set_node_degree_uninformative=False,
        #                                                                            use_node_feat=True,
        #                                                                            gin_number=1,
        #                                                                            gin_dimension=self.dim_hidden * 2,
        #                                                                            gin_mlp_type ='lin_gelu_lin') for graph in self.graphs]

        # MLP out layer combining (concatenating) each nodes',
        # at each graph level in the hierarchy's,
        # learned embedding representation
        # self.hypergraph_node_embedding = torch.nn.Linear(self.dim_hidden * len(self.graphs) ,
        #                                                    self.out_dim)
        multiscale_node_embedding = []

        # multiscale_node_embedding.append(
        #     torch.nn.Linear(self.dim_hidden * len(self.graphs), self.dim_hidden) # aggregation accross graph neighborhoods
        # )
        # multiscale_node_embedding.append(
        #     nn.LayerNorm(self.dim_hidden))
        # multiscale_node_embedding.append(
        #     nn.GELU())
        # multiscale_node_embedding.append(
        #     torch.nn.Linear(self.dim_hidden, self.out_dim)       # out layer
        # )

        # different combonations of nodes are shared accross graphs
        # to concat those that span different numbers of graphs
        # requires all len(graphs) embeddings as input to only
        # supergraph embeddings (if no nodes shared in subgraphs)
        self.multiscale_nbr_aggrs = []
        for l in range(len(self.graphs)):
            # add classifiers in increasing order for number of graphs with common nodes
            num_shared_levels = len(self.graphs)-np.flip(np.arange(len(self.graphs)))[l]
            # pout(("graph levels ", self.graph_levels))
            self.multiscale_nbr_aggrs.append(nn.Linear(self.dim_hidden * (self.graph_levels[l]+1),
                                                       self.out_dim))




        # GraphSage, ego- neighborhood embedding seperation performs better
        self.multiscale_nbr_msg_aggr = SAGEConv(in_channels=self.dim_hidden * len(self.graphs),
                                                out_channels=self.dim_hidden)
        # # self.bns.append(nn.BatchNorm1d(n_2)) self.bns = nn.ModuleList()
        # self.batch_norms = []
        # batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        # self.batch_norms.append(batch_norm_layer)
        #
        # for _ in range(self.num_layers - 2):
        #     self.multiscale_nbr_msg_aggr.append(SAGEConv(in_channels=self.dim_hidden ,
        #                                       out_channels=self.dim_hidden))
        #     # batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        #     batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        #     self.batch_norms.append(batch_norm_layer)
        self.lin_out = nn.Linear(self.dim_hidden , self.out_dim)


        # self.multiscale_nbr_aggr = nn.Linear(self.dim_hidden * len(self.graphs), self.out_dim) #nn.Sequential( *multiscale_node_embedding )
        # self.multiscale_nbr_aggr = nn.Linear(self.dim_hidden , self.out_dim)

        self.probability = nn.Sigmoid() if self.out_dim == 1 else nn.Softmax(dim=1) #nn.Softmax(dim=1) #nn.Sigmoid()

        # for record keeping
        # self.print_class_attr_and_methods()

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

    def get_graph_dataloader(self, graph, shuffle=True, num_neighbors=[-1], num_workers=None, batch_size=1):
        if not num_workers:
            if graph.num_nodes < 1200:
                num_workers = 0
            else:
                num_workers = 4
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
        """ DONT FORGET TO CHANGE ALL CALLS TO INCVOUDE BATCH_SIZE, NUM_WORKERS< SHUFFLE"""



    def forward(self, subset_xs, subset_adjs, subset_bs, subset_ys=None, num_empty=0):
        # Super-graph convolution
        # super_x = self.levelset_modules[-1](super_x, super_adjs)#(super_x, super_x[super_adjs[0].src_node]), super_adjs[0].edge_index)
        # super_filtration_value = self.levelset_graph_filtration_functions[-1](super_x)
        # super_x = super_filtration_value * super_x
        # pout((f"length total num adjs {len(subset_adjs)}"))
        # Subset-graph convolutions
        subset_outs = []
        empty_count = len(self.levelset_modules) - len(subset_xs)
        for i, (subset_x, subset_adj) in enumerate(zip(subset_xs, subset_adjs)):
            # pout((f"Starting forward i={i}",f"x size {subset_x.size()}",f"adj sizes (edge_index,size) {subset_adj[0][0].size()},{subset_adj[0][2][1]}"))
            single_sample = False
            if subset_x.size(0) <= 1: # invalid if empty (0) and can't use batch norm if one element
                if subset_x.size(0) == 0:
                    padding = torch.ones(subset_x.size(0),#subset_xs[-1].size(0), #supergraph_emb_size - out.size(0),
                                         self.dim_hidden).to(subset_x.device)
                    subset_outs.append(padding)
                    # torch.ones(subset_x.size(0),
                    #            self.dim_hidden, dtype=torch.float).to(subset_x.device))
                    continue  # Skip if no valid nodes
                else:
                    single_sample = True

            """   %%%%%%%%%%%%%%%%%%%%%     hard set gnn module to be shared      !!!!!     %%%%%%%%%%%%%%%%%%%%%%"""
            subset_x, subset_x_target = self.levelset_modules[i](subset_x, subset_adj,
                                                filtration_function_in=self.levelset_graph_filtration_functions_in[i],         #filtration function of input node reps
                                                filtration_function_hidden=self.levelset_graph_filtration_functions_hidden[i],
                                                filtration_function_out=None,#self.levelset_graph_filtration_functions_out[i],
                                                single_sample=single_sample) #filtration function of hidden reps
            # pout((f"subset x in forward after out {subset_x.size()}"))
            if i != len(subset_xs)-1:
                subset_outs.append(subset_x_target)
            else:
                subset_outs.append(subset_x)



        for i in range(num_empty):
            subset_outs.append(subset_outs[-1])

        # Pad subset_outs to the same size as super_x
        supergraph_emb_size = subset_outs[-1].size(0)
        padded_subset_outs = []
        for i, out in enumerate(subset_outs[:-1]): # subset_outs[:-1]):
            # pout(("sub out size", out.size()))
            if out.size(0) < supergraph_emb_size:
                # padding = UniformativeDummyEmbedding(dim_in=supergraph_emb_size - out.size(0),
                #                                      dim_out=out.size(1)).to(out.device)
                padding_size = supergraph_emb_size - out.size(0)
                padding = torch.ones(padding_size, out.size(1)).to(out.device)
                out = torch.cat([out, padding], dim=0)
            padded_subset_outs.append(out)

        # Concatenate super-graph and subset-graph embeddings
        combined = torch.cat([subset_outs[-1]] + padded_subset_outs, dim=1)



        last_edge_index_super = subset_adjs[-1][-1][0]
        last_size_super = subset_adjs[-1][-1][2][1]
        #multiply each embedding
        # combined = subset_outs[-1] # start wit super out
        # for sub_out in padded_subset_outs[:-1]:
        #     combined = combined * sub_out

        # out = self.multiscale_nbr_aggr(combined)

        out = self.multiscale_nbr_msg_aggr((combined, combined[:last_size_super]),last_edge_index_super)
        out = self.lin_out(out)
        # sorted_multilevel_shared_out = sorted(subset_outs, key=lambda sub_out: sub_out.size(0))
        # # Print the sizes of the sorted items
        #
        # sorted_num_multilevel_nodes = [sub_out.size(0) for sub_out in sorted_multilevel_shared_out]
        # # pout(("sorted number shared nodes ", sorted_num_multilevel_nodes))
        # super_out = subset_outs[-1]
        # combined_nodes = []
        # for level, sub_out in enumerate(sorted_multilevel_shared_out):#[:-1]):
        #     if level != len(sorted_multilevel_shared_out)-1:
        #         graph_level_out = sub_out
        #     else:
        #         graph_level_out = super_out[sorted_num_multilevel_nodes[level-1]:super_out.size(0)]
        #     sublevel_collection = []
        #     sublevel_collection.append(graph_level_out)
        #     if level==0:
        #         num_shared = sorted_num_multilevel_nodes[level]
        #         last_level = 0
        #     else:
        #         num_shared = sorted_num_multilevel_nodes[level] - sorted_num_multilevel_nodes[level-1]
        #         last_level = sorted_num_multilevel_nodes[level-1]
        #     for sup_out_shared in sorted_multilevel_shared_out[level:]:
        #         sup_n = sup_out_shared[last_level:last_level+num_shared]
        #         sublevel_collection.append(sup_n)
        #     combined = torch.cat(sublevel_collection,dim=1).to(super_out.device)
        #     combined_nodes.append(combined)
        #
        # comb_outs = []
        # for multiscale_nbr_aggr, combined_embeddings in zip(self.multiscale_nbr_aggrs,combined_nodes):
        #     # pout(("number nodes across graphs", combined_embeddings.size()))
        #     combined_embeddings = combined_embeddings.to(super_out.device)
        #     multilevel_out = multiscale_nbr_aggr(combined_embeddings)
        #     comb_outs.append(multilevel_out)
        #
        # out = torch.cat(comb_outs,dim=0)
        return self.probability(out).squeeze(1)#F.log_softmax(out, dim=1)

    def train_net(self, input_dict):
        return self.hierarchical_joint_train_net(input_dict)
    def hierarchical_joint_train_net(self, input_dict):
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
                          "loss_op": loss_op}

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
        for filtration_function in self.levelset_graph_filtration_functions_in:
            filtration_function.set_device(device)
        for filtration_function in self.levelset_graph_filtration_functions_hidden:
            filtration_function.set_device(device)
        # for filtration_function in self.levelset_graph_filtration_functions_out:
        #     filtration_function.set_device(device)
        self.multiscale_nbr_aggrs = [mixed_scale_aggr.to(device) for mixed_scale_aggr in self.multiscale_nbr_aggrs]
        # while True:
        for node_indices, subset_samples in self.hierarchical_graph_sampler:
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
                subset_nids.append(n_id)
                subset_xs.append(self.graphs[i].x[n_id].to(device))
                subset_ys.extend(self.graphs[i].y[n_id[:batch_size]].to(device))
                subset_bs.append(batch_size)
                total_training_points += batch_size

            # super_xs = subset_nids[-1]
            # feat_sim = []
            # sim_mapped_ids = []
            # for i, xs in enumerate(subset_nids):
            #     check_node_maps(xs, super_xs, self.sub_to_sup_mappings[i], self.sub_to_sup_mappings[-1])
            #     for sub_idx in xs:
            #         gid = self.sub_to_sup_mappings[i][int(sub_idx.detach().cpu())]
            #         super_nid_from_j = self.sup_to_sub_mapping[-1][gid]
            #         sim_mapped_ids.append(super_nid_from_j in super_xs)
            #         super_x_from_j = self.graphs[-1].x[super_nid_from_j]
            #         x_with_j = self.graphs[i].x[int(sub_idx.detach().cpu())]
            #         feat_sim.append(super_x_from_j == x_with_j)
            # pout(( "similarity of node feaurtes from maps",feat_sim,"mapped ids correct ",sim_mapped_ids))


            with autocast():
                out = self(subset_xs=subset_xs,
                           subset_adjs=subset_adjs,
                           subset_bs=subset_bs,
                           subset_ys=None,
                           num_empty=skip_count)

            # y = self.super_graph.y[node_indices].to(device)#n_id[:batch_size]].to(device)
            # out_size = out.size(0)
            # num_sampled = np.min([out_size, self.batch_size])
            supergraph_bs = subset_bs[-1]
            y = torch.tensor(subset_ys).to(device)
            y=y[:supergraph_bs]
            out=out[:supergraph_bs]
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
                val_pred, val_loss, val_ground_truth = self.inference(val_input_dict, validation=True)
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


        torch.cuda.empty_cache()

        if scheduler is not None:
            scheduler.step(total_loss / total_training_points)

        if epoch % eval_steps == 0 and epoch != 0:
            return total_loss/total_training_points , approx_acc, val_loss
        else:
            return total_loss / total_training_points, approx_acc, 666
    def inference(self, input_dict, validation=False):
        r"""
        Given input data, applies Hierarchical Filtration Function to infer
        filtration alues to edge, performes topological filtration for persistence graph hierarchy,
        and infers with trained HJT model
        :param input_dict:
        :return:
        """
        device = input_dict["device"]
        loss_op = input_dict["loss_op"]

        # total_epochs = input_dict["total_epochs"]
        # epoch = input_dict['epoch']
        # eval_steps = input_dict['eval_steps']
        # input_steps = input_dict["steps"]
        #
        # if not validation:
        #     data = input_dict["data"]
        #     Multilevel_Graph_Wrapper = GraphFiltrationFamilyWrapper(data,self.thresholds,self.num_neighbors,self.batch_size)
        # if False:#validation:
        #     data = self.val_data
        #     graphs, sub_to_sup_mappings, sup_to_sub_mapping = self.val_graphs, self.val_sub_to_sup_mappings, self.val_sup_to_sub_mapping
        #     sublevel_graph_loaders = self.val_sublevel_graph_loaders
        #     hierarchical_graph_sampler = self.val_hierarchical_graph_sampler
        # else:
        data = input_dict["data"]
        ###############################################################
        # if a filtration function hasn't been applied to the data,
        # e.g. edge inference for filter function value assignment
        # (logistic prediction of an edge as homopholous for pers-
        # sistence hierarchy.
        ##############################################################
        if self.filtration_function:
            _, _, _, data = self.graph_hierarchy_filter_function(input_dict,
                                                                 assign_edge_weights=True)
        # compute persistence filtration for graph hierarchy
        pout(("Performing Filtration,"
              "On",
              "Inference Dataset"))
        filtration_graph_hierarchy = FiltrationGraphHierarchy(graph=data,
                                                              persistence=self.thresholds,
                                                              filtration=None)
        pout(("Filtration on",
              "Inference",
              "Done"))
        graphs, sub_to_sup_mappings, sup_to_sub_mapping = (filtration_graph_hierarchy.graphs,
                                                           filtration_graph_hierarchy.sub_to_sup_mappings,
                                                           filtration_graph_hierarchy.supergraph_to_subgraph_mappings)

        pout(("%%%%%%%" "FINISHED CREATED INFERENCE GRAPH HIERARCHY", "Total graphs: ", len(graphs), "%%%%%%%"))
        graph_levels = np.flip(np.arange(len(graphs) + 1))
        # get neighborhood loaders for each sublevel graph
        sublevel_graph_loaders = [self.get_graph_dataloader(graph,
                                                                 shuffle=False,
                                                                 num_neighbors=[-1],
                                                            num_workers=0,
                                                            batch_size=1)
                                  for graph in graphs]
        super_graph = graphs[-1]
        # hierarchical graph neighborhood sampler
        hierarchical_graph_sampler = FiltrationHierarchyGraphSampler(super_data=super_graph,
                                                                     subset_samplers=sublevel_graph_loaders,
                                                                     subset_to_super_mapping=sub_to_sup_mappings,
                                                                     super_to_subset_mapping=sup_to_sub_mapping,
                                                                     batch_size=self.batch_size,
                                                                     shuffle=False)

        self.eval()
        self.training = False

        approx_thresholds = np.arange(0.0, 1.0, 0.1)

        total_loss, total_correct, total_training_points= 0, 0, 0
        predictions, labels = [], []

        # self.batch_norms = [bn.to(device) for bn in self.batch_norms]
        with torch.no_grad():
            for node_indices, subset_samples in hierarchical_graph_sampler:
                loss = 0

                subset_bs = []
                subset_adjs = []
                subset_xs = []
                subset_ys = []
                skip_count = 0
                for i, (batch_size, n_id, adjs) in enumerate(subset_samples):
                    if len(n_id) == 0:
                        skip_count += 1
                        continue  # Skip empty batches
                    # adjs = [adj.to(device) for adj in adjs]
                    adjs = [adjs.to(device)]
                    subset_adjs.append(adjs)
                    subset_xs.append(graphs[i].x[n_id].to(device))
                    subset_ys.extend(graphs[i].y[n_id[:batch_size]].to(device))
                    subset_bs.append(batch_size)
                    """
                    
                    batching might be wrong
                    """

                with autocast():
                    out = self(subset_xs=subset_xs,
                               subset_adjs=subset_adjs,
                               subset_bs=subset_bs,
                               subset_ys=None,
                               num_empty=skip_count)

                # y = self.super_graph.y[node_indices].to(device)#n_id[:batch_size]].to(device)
                # out_size = out.size(0)
                # num_sampled = np.min([out_size, self.batch_size])

                supergraph_bs = subset_bs[-1]
                total_training_points += supergraph_bs
                y = torch.tensor(subset_ys).to(device)
                y = y[:supergraph_bs]
                out = out[:supergraph_bs]

                # pout(("ys in inference hjt ", y))

                loss = loss_op(out, y)
                total_loss += loss.item()
                # for metrics
                if self.num_classes > 2:
                    out = out.argmax(axis=1)
                predictions.extend(out.detach().cpu().numpy())
                labels.extend(y.detach().cpu().numpy())

                del adjs, batch_size, n_id, loss, out,\
                    subset_xs, subset_adjs, y,\
                    node_indices, subset_samples
                torch.cuda.empty_cache()

        total_loss = total_loss / float(total_training_points)
        # pout(("PREDS ", predictions, "LABELS:", labels))
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)
        return predictions, total_loss, labels

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
        self.hierarchical_graph_sampler = FiltrationHierarchyGraphSampler(super_data=super_graph,
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



def get_graph_dataloader(graph, batch_size, num_layers, shuffle=True, num_neighbors=[-1]):

    if graph.num_nodes < 1200:
        num_workers = 1
    else:
        num_workers = 4

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