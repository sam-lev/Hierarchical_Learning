# import os.path as osp
# import os
# from typing import List, Optional, Tuple, Union
# import json
# import time
import numpy as np
from copy import copy
from copy import deepcopy
from sklearn.metrics import f1_score
from typing import Callable, List, Optional
import torch
import dionysus as dion
import networkx as nx
from torch import Tensor
from torch_geometric.nn import GINEConv, GATConv, GCNConv, NNConv, EdgeConv, SAGEConv
from torch.nn import Embedding
import torch.nn as nn
import torch.nn.functional as F
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
from .utils import pout, homophily_edge_labels,  edge_index_from_adjacency
#profiling tools
from guppy import hpy
# from memory_profiler import profile
# from memory_profiler import memory_usage
from typing import Union, List,Optional
from sklearn.metrics import accuracy_score

from .conv import SAgeConv
from .experiments.metrics import optimal_metric_threshold




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

        self.graphs, self.node_mappings = self.process_graph_filtrations(data=self.graph,
                                                                         thresholds=self.thresholds,
                                                                         )

        pout(("%%%%%%%" "FINISHED CREATED GRAPH HIERARCHY", "Total graphs: ", len(self.graphs), "%%%%%%%"))
        self.graph_levels = np.flip(np.arange(len(self.graphs) + 1))
        graph = self.graphs[0]
        self.graphlevelloader = self.get_graph_dataloader(graph,
                                                          shuffle=True,
                                                          num_neighbors=self.num_neighbors[:self.num_layers])
        self.graph_level = 0


    def expand_labels(self, labels):
        neg_labels = ~labels
        labels = labels.type(torch.FloatTensor)  # labels.type(torch.FloatTensor)
        neg_labels = neg_labels.type(torch.FloatTensor)
        labels = [neg_labels, labels]
        # if not as_logit:
        return torch.stack(labels, dim=1)

    def get_graph_dataloader(self, graph, shuffle=True, num_neighbors=[-1]):
        # pout(("graph data edge index",graph.edge_index))
        neighborloader = NeighborLoader(data=graph,
                                        batch_size=self.batch_size,
                                        num_neighbors=num_neighbors,
                                        # directed=True,
                                        shuffle=shuffle,
                                        # num_workers=8
                                        )  # for graph in self.graphs]

        neighborsampler = NeighborSampler(
            graph.edge_index,
            # node_idx=train_idx,
            sizes=self.num_neighbors[: self.num_layers],
            batch_size=self.batch_size,
            shuffle=shuffle,
            # num_workers=8,
        )
        return neighborsampler

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
        supergraph.x = new_node_features
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
        if clone_data:
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
        if clone_data:
            data = data.clone()

        edge_emb = data.edge_attr.cpu().numpy()
        y = data.y.cpu().numpy()

        node_mappings = []
        graphs = []

        for threshold in thresholds:
            G = nx.Graph()

            for simplex in filtration:
                if simplex.data >= threshold:
                    u, v = simplex
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
        if clone_data:
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

    def nx_to_pyg(self, graph, node_mapping=True, graph_level=None):

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
                    num_nodes=graph.number_of_nodes())
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
        filtered_graphs, node_mappings = self.create_filtered_graphs(filtration=filtration,
                                                                     thresholds=thresholds,
                                                                     data=data,
                                                                     clone_data=True)
        # nid_mapping=node_mapping)

        # filtered_graphs.append(full_graph)
        # node_mappings.append(node_mapping)

        for m in node_mappings:
            pout(("map length ", len(m)))
        # Convert back to PyG data objects
        pyg_graphs = [self.nx_to_pyg(graph) for i, graph in enumerate(filtered_graphs)]
        return pyg_graphs, node_mappings




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
        self.edge_emb_mlp.append(SAgeConv(in_channels=self.num_feats,
                                          out_channels=self.dim_hidden,
                                          dropout=args.dropout,
                                          hidden_dim_multiplier=1))

        batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)

        # construct MLP classifier
        for _ in range(self.num_layers - 1):
            self.edge_emb_mlp.append(SAgeConv(in_channels=self.dim_hidden,
                                              out_channels=self.dim_hidden,
                                                dropout=args.dropout,
                                              hidden_dim_multiplier=1))
            # batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)

        # self.edge_emb_mlp.append(SAgeConv(self.dim_hidden,
        #                                   self.out_dim,
        #                                   dropout=args.dropout,
        #                                   hidden_dim_multiplier=1))

        batch_norm_layer = nn.LayerNorm(self.out_dim) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)
        self.edge_pred_mlp = nn.Linear(self.dim_hidden, self.out_dim)

        self.act = nn.GELU()

        self.probability = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1) #nn.Softmax(dim=1) #nn.Sigmoid()

        self.dropout = nn.Dropout(self.dropout)


        self.num_neighbors = [25, 25, 10, 5, 5, 5, 5, 5, 5, 5]

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

        hierarchicalgraphloader = FiltrationHierarchyGraphLoader(graph=data,
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
        # pout(("graph data edge index",graph.edge_index))
        neighborloader = NeighborLoader(data=graph,
                              batch_size=self.batch_size,
                              num_neighbors=num_neighbors,
                              # directed=True,
                              shuffle=shuffle,
                                        # num_workers=8
                                        ) #for graph in self.graphs]

        neighborsampler = NeighborSampler(
            graph.edge_index,
            # node_idx=train_idx,
            sizes=self.num_neighbors[: self.num_layers],
            batch_size=self.batch_size,
            shuffle=shuffle,
            # num_workers=8,
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
        x_source = x
        for i, (edge_index, _, size) in enumerate(adjs):#embedding_layer in enumerate(self.edge_emb_mlp):#(edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            if i != self.num_layers - 1:
                x = self.dropout(x)
            x = self.edge_emb_mlp[i]((x, x_target), edge_index)


            x = self.batch_norms[i](x)

            if i != self.num_layers - 1:
                x = self.act(x)

        x = self.edge_pred_mlp(x)
        # x = x_source + x
        x = self.batch_norms[-1](x)

        return self.probability(x.squeeze(1))

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

        #
        # from crit
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        grad_scalar = input_dict["grad_scalar"]
        scheduler = None#input_dict["scheduler"]
        #
        #
        """ NOTE ON WHAT NEEDS TO BE DONE:
                create a hierarchical set of neioghborhood loaders for each each leve
                of the graph hierarchy based on persistent filtration BUT ALSO for each
                collection of nodes comprising the training set, validations set, and 
                graph in it's entirety """

        data = input_dict["train_data"]
        data = data.to(device)

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
        last_loss, last_acc = 0, 0
        total_val_loss = 0


        val_data = input_dict["val_data"]
        val_input_dict = {"data": val_data,
                            "device": device,
                          "dataset": input_dict["dataset"],
                          "loss_op":loss_op}


        total_epochs = input_dict["total_epochs"]
        epoch = input_dict['epoch']
        eval_steps = input_dict['eval_steps']
        input_steps = input_dict["steps"]






        self.train()
        self.training = True

        sanity = 0

        length_training_batches = data.y.size()[0]
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]

        total_training_points = 0
        predictions = []
        all_labels = []
        # for graph_level, trainloader in enumerate(self.graphLoaders):#_size, n_id, adjs in self.train_loader:

        #for batch in self.graphlevelloader:
        for batch_size, n_id, adjs in self.graphlevelloader:
            # batch=batch.to(device)
            # batch_size = batch.batch_size




            optimizer.zero_grad()

            adjs = [adj.to(device) for adj in adjs]
            # n_id = batch.n_id
            x = data.x[n_id].to(device)
            x_target = x[:batch_size]#
            y =  data.y[n_id[:batch_size]].to(device)#.float()
            # edge_index = adjs.edge_index

            out = self(x, adjs)#edge_index)
            # out = self(batch)
            # y = batch.y#.squeeze()
            # y = batch.y.unsqueeze(-1)
            # y = y.type_as(out)

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

            if input_steps != -1:
                self.steps += 1  # self.model.steps
                counter = self.steps
            else:
                self.steps = -1
                counter = epoch



            if self.num_classes > 2:  # (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                #total_correct += out.argmax(dim=-1).eq(y.argmax(dim=-1)).float().item()
                total_correct += int(out.argmax(axis=1).eq(y).sum())
                all_labels.extend(y)  # ground_truth)
                predictions.extend(out.argmax(axis=1))
            else:

                # if False:  # because too slow
                preds = out.detach().cpu().numpy()
                ground_truth = y.detach().cpu().numpy()
                optimal_threshold, optimal_score = optimal_metric_threshold(y_probs=preds,
                                                                            y_true=ground_truth,
                                                                            metric=accuracy_score,
                                                                            metric_name='accuracy')
                # optimal_threshold = 0.0
                thresh_out = out >= optimal_threshold
                predictions.extend(thresh_out)
                all_labels.extend(y)
                # total_correct += (out.long() == thresh_out).float().sum().item()#int(out.eq(y).sum())
                total_correct += (y == thresh_out).float().sum().item()  # int(out.eq(y).sum())
                # approx_acc = (y == thresh_out).float().mean().item()

        predictions = torch.tensor(predictions)
        all_labels = torch.tensor(all_labels)

        # if epoch % eval_steps == 0 and epoch != 0:
        with torch.no_grad():
            self.eval()
            val_pred, val_loss, val_ground_truth = self.inference(val_input_dict)
        self.training = True
        self.train()
        # val_out, val_loss, val_labels = self.inference(val_input_dict)
        # if scheduler is not None:
        #     scheduler.step(val_loss)
        num_classes = len(all_labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets != 1:
            predictions = predictions
            approx_acc = (predictions == all_labels).float().mean().item()

            val_pred = val_pred
            val_acc = (val_pred == val_ground_truth).float().mean().item()
            # val_acc = (val_pred.argmax(axis=-1) == val_ground_truth).float().mean().item()
            print(">>> epoch: ", epoch, " validation acc: ", val_acc)
        else:
            # if False:  # because too slow
            val_pred = val_pred.detach().cpu().numpy()
            val_ground_truth = val_ground_truth.detach().cpu().numpy()
            # val_acc = accuracy_score(val_ground_truth, val_pred)
            val_optimal_threshold, val_acc = optimal_metric_threshold(y_probs=val_pred,
                                                                                y_true=val_ground_truth,
                                                                                metric=accuracy_score,
                                                                                metric_name='accuracy',
                                                                      num_targets=num_targets)


            all_labels = all_labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            train_opt_thresh, approx_acc = optimal_metric_threshold(y_probs=predictions,
                                                                                y_true=all_labels,
                                                                                metric=accuracy_score,
                                                                                metric_name='accuracy',
                                                                      num_targets=num_targets)

            print(">>> epoch: ", epoch, " validation acc: ", val_acc)
            # thresh_out = out >= optimal_threshold
            # # total_correct += (out.long() == thresh_out).float().sum().item()#int(out.eq(y).sum())
            # val_total_correct += (val_ground_truth == thresh_out)  # int(out.eq(y).sum())

        if scheduler is not None:
            scheduler.step(val_loss)

        total_val_loss += val_loss  # .item(


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
        #len(self.graphlevelloader)
        return total_loss/total_training_points , approx_acc, total_val_loss

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

        inference_loader = self.get_graph_dataloader(data, shuffle=False, num_neighbors=[-1])
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

                # optimizer.zero_grad()

                adjs = [adj.to(device) for adj in adjs]

                x = data.x[n_id].to(device)
                y = data.y[n_id[:batch_size]].to(device)#.float()
                # y = batch.y

                out = self(x, adjs)


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
        supsubsub_mapping = {global_id: sub_id for sub_id,global_id in subgraph_mapping.items()}
        supsub_mapping = {global_id: sub_id for sub_id, global_id in supgraph_mapping.items()}


        # for node, i in subgraph_mapping.items():

        new_node_features = supergraph.x.clone().detach().cpu().numpy()
        for node, embedding in subgraph.items():
            global_id = supsubsub_mapping[node]
            new_node_features[supgraph_mapping[global_id]] = embedding#supsub_mapping[global_id]] = embedding
        supergraph.x = new_node_features
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

    def filtration_to_networkx(self,filtration, data, clone_data=False, node_mapping=True):
        if clone_data:
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
    def create_filtered_graphs(self,filtration, thresholds, data, clone_data=False, nid_mapping=None):
        if clone_data:
            data = data.clone()

        edge_emb = data.edge_attr.cpu().numpy()
        y = data.y.cpu().numpy()


        node_mappings = []
        graphs = []

        for threshold in thresholds:
            G = nx.Graph()

            for simplex in filtration:
                if simplex.data >= threshold:
                    u, v = simplex
                    G.add_edge(u, v, weight=simplex.data, embedding=edge_emb[simplex])
                    G.nodes[u]['y'] = y[u]#, features=data.x[u])#.tolist() if data.x is not None else {})
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

    def pyg_to_networkx(self,data, clone_data=False):
        if clone_data:
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
    def nx_to_pyg(self, graph, node_mapping = True, graph_level = None):

        target_type = torch.long if self.num_classes > 1 else torch.float
        # Mapping nodes to contiguous integers
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        int_mapping = {v:u for u,v in node_mapping.items()}
        # Convert edges to tensor format
        edge_list = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges], dtype=torch.float)
        #edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.nodes(data=True)], dtype=torch.float)

        num_nodes = graph.number_of_nodes()
        node_features = torch.tensor([attr['features'] for node, attr in graph.nodes(data=True)], dtype=torch.float)
        y = torch.tensor([attr['y'] for node, attr in graph.nodes(data=True)], dtype=target_type)
        # node_embeddings = torch.tensor([graph.nodes[i]['embeddings'] for i in range(num_nodes)], dtype=torch.float)

        #x = torch.tensor([graph[u]['features'] for u in graph.nodes], dtype=torch.float)
        #y = torch.tensor([graph[u]['y'] for u in graph.nodes], dtype=torch.float)
        # edge_emb = torch.tensor([graph[u]['embedding'] for u in graph.nodes], dtype=torch.float)

        data = Data(x=node_features,
                    edge_index=edge_index,
                    y=y,
                    edge_attr=edge_attr,
                    # edge_embedding=edge_emb,
                    num_nodes=graph.number_of_nodes())
        return data

