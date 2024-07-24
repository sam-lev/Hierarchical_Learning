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
from torch_geometric.utils import homophily

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
    else:
        raise ValueError("Unknown gin_mlp_type!")

# def
class LevelSetMessageAggregator(nn.Module):
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
        self.batch_norms = []
        batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        for _ in range(self.num_layers - 1):
            self.edge_emb_mlp.append(SAGEConv(in_channels=self.dim_hidden,
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

    def forward(self, x, adjs):
        xs = []
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.model_nbr_msg_aggr[i]((x,x_target), edge_index)#(x, x_target), edge_index)
            x = self.dropout(x)
            x = self.batch_norms[i](x)
            x = self.act(x)
            xs.append(x)
        # x = self.jump(xs)
        x = self.model_out_embedding(x)#[batch_size])#[batch])
        # x = x_source + x
        x = self.batch_norms[-1](x)
        return x #.squeeze(1)


class FiltrationHierarchyGraphLoader():
    def __init__(self,
                 graph,
                 persistence: Optional[Union[float, List[float]]],
                 num_neighbors: List[int],
                 num_classes,
                 filtration = None,
                 batch_size = [-1],
                 num_layers = 2):

        self.graph = graph
        self.num_neighbors = num_neighbors
        self.thresholds = persistence

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.filtration = filtration

        pout((" %%%%% %%%% %%%% %%%% %%% %%%% ", "PERFORMING FILTRATION OF GRAPH"))
        self.graphs, self.node_mappings, self.sub_to_sup_mappings = self.process_graph_filtrations(data=self.graph,
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
        # data = copy.copy(data)#.clone()
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
        # data = data.clone()
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



        node_mappings = []
        graphs = []

        for threshold in thresholds:
            G = nx.Graph()
            data_clone = data.clone()
            edge_emb = data_clone.edge_attr.cpu().numpy()
            y = data_clone.y.cpu().numpy()
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
            graphs.append(G)

            # if nid_mapping is None:
            node_mapping = {node: i for i, node in enumerate(G.nodes())}
            # else:
            #     node_mapping = {node: nid_mapping[node] for i, node in enumerate(G.nodes())}
            node_mappings.append(node_mapping)
        return graphs, node_mappings

    def pyg_to_networkx(self, data, clone_data=False):
        # if clone_data:
        # data = copy.copy(data)#.clone()
        # Initialize a directed or undirected graph based on your need
        G = nx.Graph()# nx.DiGraph() if data.is_directed() else nx.Graph()

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

    def nx_to_pyg(self, graph, node_mapping=True, graph_level=None):
        # graph = copy.copy(graph)

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
        # full_graph, node_mapping = self.filtration_to_networkx(filtration=filtration,
        #                                                        data=data,
        #                                                        clone_data=True,
        #                                                        node_mapping=True)

        # sort in decending order to high homopholous graphs first
        if 0.0 not in thresholds:
            thresholds.append(0.0)
        thresholds = np.flip(np.sort(thresholds))
        # Create filtered graphs
        filtered_graphs, nodeidx_mappings = self.create_filtered_graphs(filtration=filtration,
                                                                     thresholds=thresholds,
                                                                     data=data,
                                                                     clone_data=True)
        # nid_mapping=node_mapping)

        # filtered_graphs.append(full_graph)
        # node_mappings.append(node_mapping)

        for m in nodeidx_mappings:
            pout(("map length ", len(m)))
        # Convert back to PyG data objects
        pyg_graphs = [self.nx_to_pyg(graph) for i, graph in enumerate(filtered_graphs)]
        sub_to_sup_mappings = [{sub_id: global_id for global_id,sub_id\
                                             in subidx_map.items()}\
                                            for subidx_map in nodeidx_mappings]
        return pyg_graphs, nodeidx_mappings, sub_to_sup_mappings



####################################################
#
#
class SubLevelGraphFiltration(torch.nn.Module):
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
                 num_targets,
                 eps: float = 0.5, # we initialize epsilon based on
                 # number of graph levels so all initially contribute equally
                 use_node_degree=None,
                 set_node_degree_uninformative=None,
                 use_node_label=None,
                 gin_number=None,
                 gin_dimension=None,
                 gin_mlp_type=None,
                 **kwargs
                 ):
        super().__init__()

        dim = gin_dimension

        max_node_deg = max_node_deg
        num_node_lab = num_targets

        # if set_node_degree_uninformative and use_node_degree:
        #     self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        if use_node_degree:
            self.embed_deg = nn.Embedding(max_node_deg + 1, dim)
        else:
            self.embed_deg = None

        self.embed_lab = nn.Embedding(num_node_lab, dim) if use_node_label else None

        dim_input = dim * ((self.embed_deg is not None) + (self.embed_lab is not None))

        dims = [dim_input] + (gin_number) * [dim]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, eps=eps,train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))

        self.fc = nn.Sequential(
            nn.Linear(sum(dims), dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):

        node_deg = batch.node_deg
        node_lab = batch.node_lab

        edge_index = batch.edge_index

        tmp = [e(x) for e, x in
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None]

        tmp = torch.cat(tmp, dim=1)

        z = [tmp]

        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)

        x = torch.cat(z, dim=1)
        ret = self.fc(x).squeeze()
        return ret


class FiltrationHierarchyGraphSampler:
    def __init__(self,
                 super_data,
                 subset_samplers,
                 subset_to_super_mapping,
                 batch_size,
                 shuffle=False):
        self.super_data = super_data
        self.subset_samplers = subset_samplers
        self.subset_to_super_mapping = subset_to_super_mapping
        self.batch_size = batch_size
        self.num_nodes = super_data.num_nodes
        self.current_idx = 0
        self.shuffle = shuffle

    def sample(self):
        if self.current_idx >= self.num_nodes:
            self.current_idx = 0  # Reset for the next epoch

        if self.shuffle:
            # Sample nodes from the super-graph
            node_indices = torch.randint(0, self.super_data.num_nodes, (self.batch_size,))
        else:
            # Sample nodes from the super-graph without shuffling
            end_idx = min(self.current_idx + self.batch_size, self.num_nodes)
            node_indices = torch.arange(self.current_idx, end_idx)
            self.current_idx = end_idx

        # Get valid nodes for each subset graph
        valid_subset_nodes = []
        for mapping in self.subset_to_super_mapping:
            valid_nodes = [n for n in node_indices.tolist() if n in mapping.values()]
            valid_subset_nodes.append(torch.tensor(valid_nodes))

        # Get neighborhood information from each subset graph
        subset_samples = []
        # for i, sampler in enumerate(self.subset_samplers):
        #     subset_samples.append(sampler.sample(valid_subset_nodes[i]))
        for i, sampler in enumerate(self.subset_samplers):
            if len(valid_subset_nodes[i]) > 0:  # Ensure there are valid nodes to sample
                subset_samples.append(sampler.sample(valid_subset_nodes[i]))
            else:
                subset_samples.append((0, torch.tensor([]), []))  # Handle empty batches

        return node_indices, subset_samples

class HierGNN(torch.nn.Module):
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

        hierarchicalgraphloader = FiltrationHierarchyGraphLoader(graph=train_data,
                                                                 persistence=self.thresholds,
                                                                 num_neighbors=self.num_neighbors,
                                                                 num_classes=self.num_classes,
                                                                 filtration=None,
                                                                 batch_size=self.batch_size,
                                                                 num_layers=self.num_layers)

        self.graphs, self.node_mappings = hierarchicalgraphloader.graphs, hierarchicalgraphloader.node_mappings
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

            pout(("Batch Size in Edge Inference ", self.batch_size))

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
                node_mappings=[self.node_mappings[self.graph_level-1],
                               self.node_mappings[self.graph_level]])



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

    def initialize_from_subgraph(self, subgraph, supergraph, graph_level, node_mappings):
        pout(("graph level ", graph_level, "graph levelS ", self.graph_levels, " node mappings length ",
              len(node_mappings)))
        subgraph_mapping = node_mappings[0]
        supgraph_mapping = node_mappings[1]
        subsup_mapping = {sub_id: global_id for global_id,sub_id in subgraph_mapping.items()}
        # supsub_mapping = {global_id: sub_id for sub_id, global_id in supgraph_mapping.items()}


        # for node, i in subgraph_mapping.items():

        new_node_features = supergraph.x.clone().detach().cpu().numpy()
        for node, embedding in subgraph.items():
            global_id = subsup_mapping[node]
            new_node_features[supgraph_mapping[global_id]] = embedding#supsub_mapping[global_id]] = embedding
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


class HierJGNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 args,
                 data,
                 # split_masks,
                 processed_dir,
                 graph_filtration,
                 out_dim = 1,
                 train_data=None,
                 test_data = None,
                 # in_channels,
                 # hidden_channels,
                 # out_channels
                 ):
        super(HierJGNN, self).__init__()
        # base params
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
        self.thresholds = args.persistence
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

        ###################################################################################
        #         Hierarchical Joint Training requires the topological persistence graph hierarchy
        #         at inference time in order to employ the neighborhood message aggregating models
        #                 trained for each levelset graph in the graph sequence.
        ###################################################################################
        self.graph_hierarchy_filter_function = graph_filtration

        ####################################################################################
        #         Compute persistence filtration hierarchy for leve-set graphs
        #             Collect Neighborhood samplers for each graphlevel
        ####################################################################################
        # compute persistence filtration for graph hierarchy
        filtration_graph_hierarchy = FiltrationHierarchyGraphLoader(graph=train_data,
                                                                 persistence=self.thresholds,
                                                                 num_neighbors=self.num_neighbors,
                                                                 num_classes=self.num_classes,
                                                                 filtration=None,
                                                                 batch_size=self.batch_size,
                                                                 num_layers=self.num_layers)
        self.graphs, self.sub_to_sup_mappings, self.sup_to_sub_mapping = (filtration_graph_hierarchy.graphs,
                                                                          filtration_graph_hierarchy.sub_to_sup_mappings,
                                                                          filtration_graph_hierarchy.node_mappings)
        pout(("%%%%%%%" "FINISHED CREATED GRAPH HIERARCHY", "Total graphs: ", len(self.graphs), "%%%%%%%"))
        self.graph_levels = np.flip(np.arange(len(self.graphs) + 1))
        # get neighborhood loaders for each sublevel graph
        self.sublevel_graph_loaders = [self.get_graph_dataloader(graph,
                                                          shuffle=True,
                                                          num_neighbors=self.num_neighbors[:self.num_layers])\
                                       for graph in self.graphs]
        self.super_graph = self.graphs[-1]
        #hierarchical graph neighborhood sampler
        self.hierarchical_graph_sampler = FiltrationHierarchyGraphSampler(self.super_graph,
                                                                          self.sublevel_graph_loaders,
                                                                          self.sub_to_sup_mappings,
                                                                          self.batch_size,
                                                                          shuffle=True)

        ######################################################################################
        #           Define message passing / neighborhood aggration scheme for each
        #             graph levelset for learned node embeddings per-graph level
        ######################################################################################
        self.levelset_modules = [LevelSetMessageAggregator(in_dim=self.num_feats,
                                                          dim_hidden=self.dim_hidden,
                                                          out_dim=self.dim_hidden,    #outputs node_embedding of dimension dim hidden
                                                          num_layers=self.num_layers,
                                                          dropout=self.dropout,
                                                          use_batch_norm=self.use_batch_norm)\
                                for graph in self.graphs]

        #####################################################################################
        #     per sublevel graph filtration function, learns attention factor for learned
        #     node embeddings from respective sublevel neighborhoods before combining for
        #           all embeddings for hierarchical representation of node
        #####################################################################################
        # define learnable epsilon of GIN s.t. each graphs node embedding contributes equally
        epsilon = float(1.0 / len(self.graph))
        self.levelset_graph_filtration_functions = [SubLevelGraphFiltration(max_node_deg=max_degree,
                                                                            num_targets=num_targets,
                                                                            eps=epsilon,
                                                                            use_node_degree=None,
                                                                            use_node_label=None,
                                                                            gin_number=1,
                                                                            gin_dimension=self.dim_hidden,
                                                                            gin_mlp_type ='lin_bn_lrelu_lin')\
                                                    for graph in self.graphs]
        # MLP out layer combining (concatenating) each nodes',
        # at each graph level in the hierarchy's,
        # learned embedding representation
        self.hypergraph_node_embedding = torch.nn.Linear(self.dim_hidden * (len(self.graphs) + 1),
                                                           out_channels)

    def get_graph_dataloader(self, graph, shuffle=True, num_neighbors=[-1]):

        if graph.num_nodes < 1200:
            num_workers = 1
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
                num_workers=num_workers
            )
        else:
            neighborloader = NeighborLoader(data=graph,
                                  batch_size=self.batch_size,
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
                sizes=self.num_neighbors[: self.num_layers],
                batch_size=self.batch_size,
                # subgraph_type='induced',  # for undirected graph
                # directed=False,#True,
                shuffle=shuffle,
                # num_workers=8,
            )
        return neighborsampler


    def forward(self, super_x, super_adjs, subset_xs, subset_adjs):
        # Super-graph convolution
        super_x = self.levelset_modules[-1](super_x, super_adjs)#(super_x, super_x[super_adjs[0].src_node]), super_adjs[0].edge_index)
        super_filtration_value = self.levelset_graph_filtration_functions[-1](super_x)
        super_x = super_filtration_value * super_x

        # Subset-graph convolutions
        subset_outs = []
        for i, (subset_x, subset_adj) in enumerate(zip(subset_xs, subset_adjs)):
            if subset_x.size(0) == 0:
                subset_outs.append(torch.zeros(subset_x.size(0), self.dim_hidden).to(subset_x.device))
                continue  # Skip if no valid nodes
            """ need to put each modules batch norms to a device"""
            subset_x = self.levelset_modules[i](subset_x, subset_adj)
            sub_filtration_value = self.levelset_graph_filtration_functions[i](subset_x)
            subset_x = sub_filtration_value * subset_x
            subset_outs.append(subset_x)

        # Concatenate super-graph and subset-graph embeddings
        combined = torch.cat([super_x] + subset_outs, dim=1)
        out = self.hypergraph_node_embedding(combined)
        return F.log_softmax(out, dim=1)


    def train(self, input_dict):
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

        self.batch_norms = [bn.to(device) for bn in self.batch_norms]

        while True:

            node_indices, subset_samples = self.hierarchical_graph_sampler.sample()

            if node_indices.numel() == 0:
                break  # End of epoch

            optimizer.zero_grad()
            loss = 0

            super_adjs = [adj.to(device) for adj in subset_samples[-1][2]]  # Last subset sample is the supergraph
            super_x = self.super_graph.x[node_indices].to(device)
            subset_adjs = []
            subset_xs = []

            for i, (batch_size, n_id, adjs) in enumerate(subset_samples):
                if len(n_id) == 0:
                    continue  # Skip empty batches
                adjs = [adj.to(device) for adj in adjs]
                subset_adjs.append(adjs)
                subset_xs.append(self.graphs[i].x[n_id].to(device))
                total_training_points += batch_size

            with autocast():
                out = self(super_x,
                           super_adjs,
                           subset_xs,
                           subset_adjs)

            y = self.super_graph.y[node_indices].to(device)#n_id[:batch_size]].to(device)
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
                super_x, subset_xs, subset_adjs, super_adjs, y, \
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

                # if scheduler is not None:
                #     scheduler.step(val_loss)

                del val_loss, val_pred, val_ground_truth


        torch.cuda.empty_cache()

        if scheduler is not None:
            scheduler.step(total_loss / total_training_points)

        if epoch % eval_steps == 0 and epoch != 0:
            return total_loss/total_training_points , approx_acc, val_loss
        else:
            return total_loss / total_training_points, approx_acc, 666
    def inference(self, input_dict):
        r"""
        Given input data, applies Hierarchical Filtration Function to infer
        filtration alues to edge, performes topological filtration for persistence graph hierarchy,
        and infers with trained HJT model
        :param input_dict:
        :return:
        """
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]

        # total_epochs = input_dict["total_epochs"]
        # epoch = input_dict['epoch']
        # eval_steps = input_dict['eval_steps']
        # input_steps = input_dict["steps"]
        #
        # data = input_dict["data"]
        #
        _, _, _, data = self.graph_hierarchy_filter_function(input_dict,
                                                             assign_edge_weights=True)
        # compute persistence filtration for graph hierarchy
        filtration_graph_hierarchy = FiltrationHierarchyGraphLoader(graph=data,
                                                                    persistence=self.thresholds,
                                                                    num_neighbors=self.num_neighbors,
                                                                    num_classes=self.num_classes,
                                                                    filtration=None,
                                                                    batch_size=self.batch_size,
                                                                    num_layers=self.num_layers)
        graphs, sub_to_sup_mappings, sup_to_sub_mapping = (filtration_graph_hierarchy.graphs,
                                                           filtration_graph_hierarchy.sub_to_sup_mappings,
                                                           filtration_graph_hierarchy.node_mappings)
        pout(("%%%%%%%" "FINISHED CREATED INFERENCE GRAPH HIERARCHY", "Total graphs: ", len(graphs), "%%%%%%%"))
        graph_levels = np.flip(np.arange(len(graphs) + 1))
        # get neighborhood loaders for each sublevel graph
        sublevel_graph_loaders = [self.get_graph_dataloader(graph,
                                                                 shuffle=False,
                                                                 num_neighbors=[-1])
                                  for graph in graphs]
        super_graph = graphs[-1]
        # hierarchical graph neighborhood sampler
        hierarchical_graph_sampler = FiltrationHierarchyGraphSampler(super_graph,
                                                                     sublevel_graph_loaders,
                                                                     sub_to_sup_mappings,
                                                                     self.batch_size,
                                                                     shuffle=False)



        self.eval()
        self.training = False

        approx_thresholds = np.arange(0.0, 1.0, 0.1)

        total_loss, total_correct, total_samples= 0, 0, 0
        predictions, labels = [], []

        # self.batch_norms = [bn.to(device) for bn in self.batch_norms]
        with torch.no_grad():
            while True:

                node_indices, subset_samples = self.hierarchical_graph_sampler.sample()

                if node_indices.numel() == 0:
                    break  # End of epoch

                # optimizer.zero_grad()
                loss = 0

                super_adjs = [adj.to(device) for adj in subset_samples[-1][2]]  # Last subset sample is the supergraph
                super_x = self.super_graph.x[node_indices].to(device)
                subset_adjs = []
                subset_xs = []

                for i, (batch_size, n_id, adjs) in enumerate(subset_samples):
                    if len(n_id) == 0:
                        continue  # Skip empty batches
                    adjs = [adj.to(device) for adj in adjs]
                    subset_adjs.append(adjs)
                    subset_xs.append(self.graphs[i].x[n_id].to(device))
                    total_samples += batch_size

                with autocast():
                    out = self(super_x,
                               super_adjs,
                               subset_xs,
                               subset_adjs)

                y = self.super_graph.y[node_indices].to(device)  # n_id[:batch_size]].to(device)
                loss = loss_op(out, y)
                total_loss += loss.item()

                predictions.extend(out)
                labels.extend(y)

                del adjs, batch_size, n_id, loss, out,\
                    super_x, subset_xs, subset_adjs, super_adjs, y,\
                    node_indices, subset_samples
                torch.cuda.empty_cache()


        total_loss = total_loss / total_samples

        return torch.tensor(predictions), total_loss, torch.tensor(labels)
