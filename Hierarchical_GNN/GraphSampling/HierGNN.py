# import os.path as osp
# import os
# from typing import List, Optional, Tuple, Union
# import json
# import time
import numpy as np
import copy
from sklearn.metrics import f1_score
from typing import Callable, List, Optional
import torch
import dionysus as dion
import networkx as nx
from torch import Tensor
from torch_geometric.nn import GINEConv, GATConv, GCNConv, NNConv, EdgeConv
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
from .utils import pout, homophily_edge_labels, add_edge_attributes
#profiling tools
from guppy import hpy
# from memory_profiler import profile
# from memory_profiler import memory_usage
class HierarchicalGraphData(InMemoryDataset):
    def __init__(self, root : str, name : str,
                 split_percents=[0.5, 0.2, 1.0],
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self.name = name
        self.split_masks = {}
        self.split_percents = split_percents
        self.split_masks["split_percents"] = self.split_percents
        self.train_percent = self.split_percents[0]
        self.val_percent = self.split_percents[1]
        self.test_percent = self.split_percents[2]

        super().__init__(root, transform, pre_transform, pre_filter)
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        url = "123"
        download_url(url, self.raw_dir)
        ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

class HierGNN(torch.nn.Module):
    #fp = open('./run_logs/memory_profiler.log', 'w+')
    # @profile#stream=fp)
    def __init__(self,
                 args,
                 data,
                 split_masks,
                 processed_dir,
                 train_data=None,
                 test_data = None,
                 # in_channels,
                 # hidden_channels,
                 # out_channels
                 ):
        super().__init__()

        train_idx = split_masks["train"]
        # train_idx = split_masks["train"]
        self.split_masks = split_masks
        self.save_dir = processed_dir

        self.type_model = args.type_model
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = data.x.shape[-1]#args.num_feats
        self.batch_size = args.batch_size
        self.dropout = args.dropout

        self.device = args.device

        self.data = data

        self.edge_index = data.edge_index

        heterophily_num_class = 2
        self.cat = 2

        self.c1 = 0
        self.c2 = 0

        self.inf_threshold = args.inf_threshold
        self.threshold_pred = False

        self.weight_decay = args.weight_decay

        self.edge_emb_mlp = torch.nn.ModuleList()
        self.edge_emb_mlp.append(GCNConv(self.cat * self.num_feats , self.dim_hidden))
        # inherent class imbalance / overfitting
        # self.edge_emb_mlp.append(nn.BatchNorm1d(self.dim_hidden))
        for _ in range(self.num_layers - 1):
            self.edge_emb_mlp.append(GCNConv(self.dim_hidden, self.dim_hidden))
            # self.edge_emb_mlp.append(nn.BatchNorm1d(self.dim_hidden))
        self.edge_emb_mlp.append(GCNConv(self.dim_hidden, self.num_classes))

        self.batchnorm = nn.BatchNorm1d(self.dim_hidden)
        self.activation = nn.Sigmoid() #nn.Softmax(dim=1) #nn.Sigmoid()

        self.thresholds = args.persistence
        self.graphs = self.process_graph_filtrations(data=self.data,
                                                     thresholds=self.thresholds,
                                                     filtration=None)

        graph = self.graphs[0]
        self.graphlevelloader = self.get_graph_dataloader(graph)
        self.graph_level = 0



    def get_graph_dataloader(self, graph):
        return DataLoader(graph,
                                        batch_size=self.batch_size,
                                               # num_neighbors=[-1],
                                        shuffle=True) #for graph in self.graphs]


    def l2_loss(self, weight, factor):
        return factor * torch.square(weight).sum()

    def forward(self, data):
        # for i, (edge_index, _, size) in enumerate(adjs):
        #     x_target = x[: size[1]]  # Target nodes are always placed first.
        #     x = self.convs[i]((x, x_target), edge_index)
        x, edge_attr = data.x, data.edge_index
        for i,embedding_layer in enumerate(self.edge_emb_mlp):
            x = embedding_layer(x, edge_attr)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.batchnorm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # x = self.activation( x )

        return F.log_softmax( x , dim=1 )

    #fp = open('./run_logs/training_memory_profiler.log', 'w+')
    # @profile#stream=fp)
    def train_net(self, input_dict):
        a = """
        device = input_dict["device"]

        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        scheduler = input_dict["scheduler"]

        total_loss = total_correct = 0
        last_loss, last_acc = 0, 0


        val_data = input_dict["val_data"]
        val_input_dict = {"data": val_data,
                            "device": device,
                          "dataset": input_dict["dataset"],
                          "loss_op":loss_op}

        # y_full = self.edge_labels(labels=self.train_dataset.data.y.to(device),
        #                      edge_index=self.train_dataset.data.edge_index.to(device),
        #                           device=device)
        # hetero_sz = torch.sum(y_full, 0)[0]
        # homoph_sz = torch.sum(y_full, 0)[1]
        # max_class = torch.max(torch.tensor((hetero_sz,
        #                                     homoph_sz)))
        # self.c1 = max_class/hetero_sz
        # self.c2 = max_class/homoph_sz

        self.train()
        self.training = True
        # self.edge_embeddings.weight.requires_grad = True
        # self.edge_embeddings.to(device)

        first_idx = torch.tensor([0,0]).to(device)
        # sanity_check_embedding = self.edge_embeddings.weight.clone()

        sanity = 0

        train_sample_size = total_batches = 0

        for batch in self.train_loader:#_size, n_id, adjs in self.train_loader:
            batch=batch.to(device)
            batch_size, n_id = batch.batch_size, batch.n_id.to(device)#batch_size, n_id.to(device), adjs
            # edge_index, e_id, size = adjs
            # _, _, size = adjs

            # batch.n_id # global node index of each node
            e_id = batch.e_id.to(device)#adjs.e_id.to(device) # global edge index of each edge
            edge_index  = batch.edge_index.to(device)#adjs.edge_index.to(device)
            # batch.input_id # global index of input_nodes
            # batch_data = self.train_dataset.data.x[n_id].to(device)

            train_sample_size += edge_index.shape[1]
            total_batches += 1

            row, col = edge_index

            # x = all_nodes.x[n_id].to(device)  # subgraph ref
            # y = labels[n_id].to(device)  # subgraph ref
            y = self.edge_labels(labels=batch.y.to(device),
                                 edge_index=edge_index, device=device)
            # y = y[:,1].to(device)
            y = y.to(device)
            # pout(("shape batch", y.shape))

            #torch.tensor(data.edge_attr[e_id], requires_grad=True).to(device)
            # edge_attr = data.edge_attr[e_id]

            optimizer.zero_grad()
            # out = self(x[n_id], adjs)
            out, edge_embedding = self(edge_index = e_id)





            # update edge attribute with new embedding
            # self.dataset.data.edge_attr[e_id] = edge_embedding

            if isinstance(loss_op, torch.nn.NLLLoss):
                pout(("performing softmax"))
                out = F.log_softmax(out, dim=-1)

            # hetero_sz = torch.sum(y, 0)[0]
            # homoph_sz = torch.sum(y, 0)[1]
            # self.c1 = self.c1 + hetero_sz
            # self.c2 = self.c2 + homoph_sz
            # max_class = torch.max(torch.tensor((self.c1,
            #                                     self.c2)))
            # sum_class_count = torch.sum(torch.tensor((self.c1,self.c2)))

            # loss = self.c1 * loss_op(out[:,0], y[:,0]) \
            #        + self.c2 * loss_op(out[:,1], y[:,1])

            loss = loss_op(out, y)

            # pout(("c1", self.c1, "c2", self.c2,"sum", sum_class_count,
            #       "max", max_class, "wc1", max_class/self.c1))
            # compute l2 loss of weights in mlps used for
            # leaarning embeddings and prediction
            # Compute l2 loss component
            l2_factor = self.weight_decay
            l2_loss = 0.0
            for parameter in self.edge_emb_mlp.parameters():
                l2_loss += self.l2_loss(parameter, l2_factor)#.view(-1))
            for parameter in self.edge_pred_mlp.parameters():
                l2_loss += self.l2_loss(parameter, l2_factor)#.view(-1))

            # add L2 normalization of weights
            loss += l2_loss

            # back prop
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            if self.threshold_pred:
                threshold_out = torch.zeros_like(out)
                mask = out[:] >= 0.5
                threshold_out[mask] = 1.0
                mask = out[:] < 0.5
                threshold_out[mask] = 0.0
                # threshold_out[~mask] = 0.0
                out = threshold_out

            if (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCELoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                total_correct += int(out[:,1].eq(y[:,1]).sum()) #y[:,1]
                total_correct += int(out.argmax(dim=-1).eq(y[:,1]).sum())
            else:
                total_correct += int(out.eq(y).sum())

            # last_acc = total_correct# float(total_correct)
            # pout(("edge embedding weight:", self.edge_embeddings.weight))
            # pout(("edge embedding weight data:", self.edge_embeddings.weight.data))
            # pout(("edge embedding grad", self.edge_embeddings.weight.grad))
            # pout(("embedding updated:", ~torch.eq(self.edge_embeddings(e_id), edge_embedding )))
            # pout(("embedding sanity check, updated:", ~torch.eq(self.edge_embeddings.weight,
            #                                                    sanity_check_embedding)))
            # pout(("edge embedding", edge_attr))

            torch.cuda.empty_cache()
            del y, batch_size, n_id, loss, out, l2_loss, l2_factor, edge_embedding, batch


            # x = scatter(data.x, data.batch, dim=0, reduce='mean')
        val_out, val_loss, val_labels = self.inference(val_input_dict)
        val_labels = val_labels.to(device)
        val_out = val_out.to(device)
        threshold_out = torch.zeros_like(val_out)
        mask = val_out[:] >= self.inf_threshold
        threshold_out[mask] = 1.0
        # threshold_out[~mask] = 0.0
        val_out = threshold_out

        # train_pred = train_out.to("cpu")  # .argmax(dim=-1).to("cpu")
        # y_true = self.y
        val_correct = val_out.eq(val_labels)
        val_acc = val_correct.sum().item() / float(val_labels.size()[0])
        val_loss = val_acc
        # scheduler.step(val_loss)

        # for multilabel
        # train_size_edges = y.size(0)
        # train_size = (
        #     train_size_edges
        #     if isinstance(loss_op, torch.nn.NLLLoss) or isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss)
        #     else train_size_edges * self.num_classes
        # )
        '''val_acc = self.test()
        '''
        #len(self.train_dataset.data)
        # return total_loss/ self.train_size_edges, total_correct / train_size#total_loss, total_correct / self.train_size[0]  # / len(self.train_loader)
        # return total_loss / float(len(self.train_loader)), total_correct / float(train_size)
        return total_loss / float(total_batches) , total_correct / float(train_sample_size), val_loss
        """
        return self.hierarchical_successive_train_net(input_dict)


    def aggregate_edge_attr(self, edge_attr, edge_index):
        edge_attr_target = edge_attr[edge_index]
        # pout(("target edge attr", edge_attr_target, "target edge shape", edge_attr_target.shape))
        # torch.mean(torch.max(a, -1, True)[0], 1, True)
        return torch.max(edge_attr_target, -1, True)[0]

    # @profile
    @torch.no_grad()
    def inference(self, input_dict):
        # input dict input_dict = {"data": self.data, "y": self.y, "device": self.device, "dataset": self.dataset}
        self.eval()
        self.training = False
        device = input_dict["device"]
        # x = input_dict["x"].to(device)
        data = input_dict["data"]#.to(device)
        data = add_edge_attributes(data)

        all_nodes = data.x
        dataset = input_dict["dataset"]
        all_data = dataset.data.to(device)
        all_node_labels = all_data.y.to(device)
        node_labels = data.y.to(device)

        # for validation testing
        loss_op  = input_dict["loss_op"]

        labels = data.y.to(device)
        # all_edge_idx = data.edge_index.to(device)
        labels = self.edge_labels(labels=labels, edge_index=data.edge_index)#torch.eq(labels[all_edge_idx[0]], labels[all_edge_idx[1]])
        # pout(("edge labels", labels, "edge labels size", labels.size(0),
        #       "shape", labels.shape))
        # pout(("sum of edge labels", torch.sum(labels, 0)))
        graph_sz = labels.size(0)
        hetero_sz = torch.sum(labels, 0)[0]
        homoph_sz  = torch.sum(labels, 0)[1]
        pout(("heterophily, homophily percents", hetero_sz/graph_sz, " ", homoph_sz/graph_sz ))
        edge_pred = torch.zeros(data.edge_attr.shape[0]).type(torch.FloatTensor)
        # edge_pred = torch.zeros(data.edge_index.shape).type(torch.FloatTensor)
        # edge_pred.to(device)
        node_pred = torch.zeros(data.y.shape)
        # edge_index = data.edge_index.to(device)
        # edge_attr = data.edge_attr.to(device)


        row, col = data.edge_index
        # pout(("row col", row, col, "row_shape col_shape", row.shape, col.shape))
        # dim edfe feature shoould be num_edges , 2 * dim_node_features
        data.edge_attr = torch.cat([all_nodes[row], all_nodes[col]], dim=-1)

        # pout(("edge attr shape", data.edge_attr.shape))

        # create edge embeddings
        # pred_edge_embeddings = nn.Embedding.from_pretrained(data.edge_index.shape).requires_grad_(False)
        # self.edge_embeddings.requires_grad_(False)
        # self.edge_embeddings.weight.requires_grad = False
        # self.edge_embeddings.to(device)

        # pout(("labels shape in inf", labels.shape))

        x = data.x
        row, col = data.edge_index
        # data.edge_attr = torch.cat([x[row], x[col]], dim=-1)
        data.edge_attr.to(device)

        data.edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        data.edge_embeddings.to(device)

        data.edge_weights = torch.zeros(data.edge_index.shape)

        global_edge_index = data.edge_index.t().cpu().numpy()

        # create learnable edge embeddings
        # train_edge_embeddings = nn.Embedding(row.shape[0],self.cat * self.num_feats)
        edge_embeddings = nn.Embedding.from_pretrained(data.edge_emb,
                                                            freeze=True).requires_grad_(False).to(device)

        # edge_embeddings = nn.Embedding.from_pretrained(self.edge_embeddings.weight.data.detach().clone(),
        #                                                freeze=True).requires_grad_(False)

        inference_loader = NeighborLoader(
            data,#copy.copy(data),#.edge_index,
            input_nodes=None,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],
            batch_size=data.edge_index.shape[1],
            shuffle=False,
        )

        train_sample_size = 0
        #     # [test_split[0]:test_split[1]],#.edge_index,  # split_masks["test"],  # .edge_index,
        #     # node_idx=self.split_masks["test"],
        #     sizes=[-1],
        #     batch_size=data.edge_index.shape[1],#self.batch_size,
        #     shuffle=False,
        # )
        # Hence, an item returned by :class:`NeighborSampler` holds the current
        # :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the computation,
        # and a list of bipartite graph objects via the tupl :obj:`(edge_index, e_id, size)`,
        # where :obj:`edge_index` represents the bipartite edges between source
        # and target nodes, :obj:`e_id` denotes the IDs of original edges in
        # the full graph, and :obj:`size` holds the shape of the bipartite graph.
        # For each bipartite graph, target nodes are also included at the beginning
        # of the list of source nodes so that one can easily apply skip-connections
        # or add self-loops.
        # kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
        # subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
        #                                  num_neighbors=[-1], shuffle=False, )
        edges = []
        edge_weights = []
        x_pred = []
        preds = []
        ground_truths = []
        total_loss = 0
        for batch in inference_loader:#_size, n_id, adjs  in inference_loader:
            with torch.no_grad():
                batch_size, n_id = batch.batch_size, batch.n_id.to(device)  # batch_size, n_id.to(device), adjs
                # edge_index, e_id, size = adjs
                # _, _, size = adjs

                # batch.n_id # global node index of each node
                e_id = batch.e_id.to(device)  # adjs.e_id.to(device) # global edge index of each edge
                edge_index = batch.edge_index.to(device)
                train_sample_size += edge_index.shape[1]
                # batch_size, n_id, adjs = batch_size.to(device), n_id.to(device), adjs
                # n_id.to(device)
                # edge_index, e_id, size = adjs
                # edge_index, e_id, size = edge_index.to(device), e_id.to(device), size
                # x = all_nodes[n_id].to(device) # subgraph ref
                # pout(("shape batch", x.shape))
                # x_target = x[:size[1]]
                edge_attr = edge_embeddings(e_id).to(device)#all_edge_attr[e_id]
                # edge_emb = self.edge_emb_mlp(edge_attr)
                # out = self.edge_pred_mlp(edge_emb)

                for i, embedding_layer in enumerate(self.edge_emb_mlp):
                    edge_attr = embedding_layer(edge_attr)
                    if i != self.num_layers - 1:
                        edge_attr = F.relu(edge_attr)
                        edge_attr = self.batchnorm(edge_attr)
                    # data.edge_attr[edge_index] = edge_emb
                data.edge_embeddings[e_id] = edge_attr
                out = self.edge_pred_mlp(edge_attr)
                out = self.activation(out)#[:,1]#[:,1]

                # return just homophily prediction
                preds.append(out[:,1])
                # add predicted edge value as weight for each edge
                for edge, p in zip(e_id.t().cpu().numpy(), out[:,1].cpu().numpy()):
                    source, target = global_edge_index[edge]
                    edges.append((source, target))
                    data.edge_weights[edge] = p#.append(p)

                    #self.filtration.append(dion.Simplex([source, target], p))




                # pout(("edge_index", edge_index, "e_id", e_id))

                batch_labels = self.edge_labels(labels=batch.y.to(device),#data.y[n_id].to(device),
                                                edge_index=edge_index, dtype="int")

                total_loss += loss_op(out, batch_labels.to(device))
                ground_truths.append(batch_labels[:,1])

                # if isinstance(loss_op, torch.nn.NLLLoss):
                #     edge_logit = F.log_softmax(out, dim=-1)
                #     edge_logit = edge_logit.argmax(dim=-1)
                #     edge_logit = F.softmax(out, dim=-1)

                # edge_logit = edge_logit.cpu()

                # pout(("logit shape", edge_logit.shape, "edge pred shape", edge_pred.shape, "edge id shape", e_id.shape))

                # x_pred.append(edge_logit)
                # edge_pred[e_id] = edge_logit.cpu()

        # self.filtration.sort()

        pred = torch.cat(preds, dim=0)#.cpu()#.numpy()
        ground_truth = torch.cat(ground_truths, dim=0)#.cpu()#.numpy()
        # pout(("xpred ", x_pred))
        # pout(("edge pred ", pred))

        return pred, total_loss/train_sample_size, ground_truth#torch.from_numpy(np.cat(x_pred,axis=0))##torch.stack(torch.tensor(x_pred).tolist(),dim=0)  # _all

    def hierarchical_successive_train_net(self, input_dict,
                                          filtration=None, thresholds=[.5, 1.0]):

        device = input_dict["device"]

        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        scheduler = input_dict["scheduler"]

        data = input_dict["data"]

        total_loss = total_correct = 0
        last_loss, last_acc = 0, 0


        val_data = input_dict["val_data"]
        val_input_dict = {"data": val_data,
                            "device": device,
                          "dataset": input_dict["dataset"],
                          "loss_op":loss_op}

        epoch = input_dict["epoch"]
        total_epochs = input_dict["total_epochs"]






        self.train()
        self.training = True

        sanity = 0

        train_sample_size = total_batches = 0

        subgraph_embeddings = {}
        # for graph_level, trainloader in enumerate(self.graphLoaders):#_size, n_id, adjs in self.train_loader:

        for batch_idx, batch in enumerate(self.graphlevelloader):
            batch=batch.to(device)



            optimizer.zero_grad()
            # out = self(x[n_id], adjs)
            out = self(batch)


            y = batch.y
            loss = loss_op(out, y)

            # Compute l2 loss component
            # l2_factor = self.weight_decay
            # l2_loss = 0.0
            # for parameter in self.edge_emb_mlp.parameters():
            #     l2_loss += self.l2_loss(parameter, l2_factor)#.view(-1))
            # for parameter in self.edge_pred_mlp.parameters():
            #     l2_loss += self.l2_loss(parameter, l2_factor)#.view(-1))
            # # add L2 normalization of weights
            # loss += l2_loss

            # back prop
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            if self.threshold_pred:
                threshold_out = torch.zeros_like(out)
                mask = out[:] >= 0.5
                threshold_out[mask] = 1.0
                mask = out[:] < 0.5
                threshold_out[mask] = 0.0
                # threshold_out[~mask] = 0.0
                out = threshold_out

            if (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCELoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                total_correct += int(out[:,1].eq(y[:,1]).sum()) #y[:,1]
                total_correct += int(out.argmax(dim=-1).eq(y[:,1]).sum())
            else:
                total_correct += int(out.eq(y).sum())

            torch.cuda.empty_cache()
            #del y, batch_size, n_id, loss, out, l2_loss, l2_factor, edge_embedding, batch

        if epoch % (total_epochs // len(self.graphLoaders)) == 0 and epoch != 0:
            pout(("%%%%%%"))
            pout(("Moving up graph level hierarchy for successive training"))
            pout(("%%%%%%"))
            self.graph_level += 1
            # Save embeddings for each node in the last graph
            subgraph_embeddings = {}
            with torch.no_grad():
                for batch in self.graphlevelloader:
                    out = self(batch)
                    for i, node in enumerate(batch.x):
                        subgraph_embeddings[node] = out[i].detach().numpy()
            self.graphs[self.graph_level] = self.initialize_from_subgraph(subgraph_embeddings,
                                                                          self.graphs[self.graph_level])
            self.graphlevelloader = self.get_graph_dataloader(self.graphs[self.graph_level])
            # x = scatter(data.x, data.batch, dim=0, reduce='mean')
        val_out, val_loss, val_labels = self.inference(val_input_dict)
        val_labels = val_labels.to(device)
        val_out = val_out.to(device)
        threshold_out = torch.zeros_like(val_out)
        mask = val_out[:] >= self.inf_threshold
        threshold_out[mask] = 1.0
        # threshold_out[~mask] = 0.0
        val_out = threshold_out
        val_correct = val_out.eq(val_labels)
        val_acc = val_correct.sum().item() / float(val_labels.size()[0])
        val_loss = val_acc
        return total_loss / float(total_batches) , total_correct / float(train_sample_size), val_loss

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

        labels = data.y.to(device)

        graph_sz = labels.size(0)
        hetero_sz = torch.sum(labels, 0)[0]
        homoph_sz = torch.sum(labels, 0)[1]
        pout(("heterophily, homophily percents", hetero_sz / graph_sz, " ", homoph_sz / graph_sz))



        inference_loader = NeighborLoader(
            data,  # copy.copy(data),#.edge_index,
            input_nodes=None,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],
            batch_size=data.edge_index.shape[1],
            shuffle=False,
        )

        train_sample_size = 0
        #     # [test_split[0]:test_split[1]],#.edge_index,  # split_masks["test"],  # .edge_index,
        #     # node_idx=self.split_masks["test"],
        #     sizes=[-1],
        #     batch_size=data.edge_index.shape[1],#self.batch_size,
        #     shuffle=False,
        # )
        # Hence, an item returned by :class:`NeighborSampler` holds the current
        # :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the computation,
        # and a list of bipartite graph objects via the tupl :obj:`(edge_index, e_id, size)`,
        # where :obj:`edge_index` represents the bipartite edges between source
        # and target nodes, :obj:`e_id` denotes the IDs of original edges in
        # the full graph, and :obj:`size` holds the shape of the bipartite graph.
        # For each bipartite graph, target nodes are also included at the beginning
        # of the list of source nodes so that one can easily apply skip-connections
        # or add self-loops.
        # kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
        # subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
        #                                  num_neighbors=[-1], shuffle=False, )
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
            for batch in inference_loader:  # _size, n_id, adjs  in inference_loader:

                # batch_size, n_id = batch.batch_size, batch.n_id.to(device)
                # x = data.x[n_id].to(device)
                # y = batch.y.to(device)
                # for i, embedding_layer in enumerate(self.edge_emb_mlp):
                #     x = embedding_layer(x)
                #     if i != self.num_layers - 1:
                #         x = F.relu(x)
                #         x = self.batchnorm(x)
                # out = F.log_softmax( x , dim=1 )

                out = self(batch)
                pred = out.argmax(dim=1)

                for nid, p in zip(batch.n_id.detach().cpu().numpy(), out[:,1].detach().cpu().numpy()):
                    node_ids.append(batch.n_id.cpu().numpy())
                    node_pred_dict[nid] = p#.append

                train_sample_size += batch.batch_size.cpu().float()
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())




                # total_loss += loss_op(out, batch.y.cpu().numpy())


        # pred = torch.cat(preds, dim=0)  # .cpu()#.numpy()
        # ground_truth = torch.cat(ground_truths, dim=0)  #
        data.node_preds = torch.tensor([node_pred_dict[i] for i in range(len(node_pred_dict))],
                                       dtype=torch.float)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute F1 score
        f1 = f1_score(all_labels, all_preds, average='micro')

        return all_preds, f1, all_labels  # torch.from_numpy(np.cat(x_pred,axis=0))##torch.stack(torch.tensor(x_pred).tolist(),dim=0)  # _all

    def initialize_from_subgraph(self, subgraph, supergraph):

        new_node_features = supergraph.x.clone()
        for node, embedding in subgraph.items():
            new_node_features[node] = embedding
        supergraph.x = new_node_features
        return supergraph

    def pyg_to_dionysus(self, data):
        # Extract edge list and weights
        # data = data.copy()
        edge_index = data.edge_index.t().cpu().numpy()
        edge_attr = data.edge_weights.cpu().numpy()
        filtration = dion.Filtration()
        for i, (u, v) in enumerate(edge_index):
            filtration.append(dion.Simplex([u, v], edge_attr[i]))
        filtration.sort()
        return filtration


    def create_filtered_graphs(self,filtration, thresholds, data):
        # data = data.copy()
        graphs = []
        edge_emb = data.edge_embeddings.cpu().numpy()
        y = data.y.cpu().numpy()
        for threshold in thresholds:
            G = nx.Graph()
            for simplex in filtration:
                if simplex.data <= threshold:
                    u, v = simplex
                    G.add_edge(u, v, weight=simplex.data, embedding=edge_emb[simplex])
                    G.nodes[u]['y'] = y[u]#, features=data.x[u])#.tolist() if data.x is not None else {})
                    G.nodes[v]['y'] = y[v]
                    G.nodes[u]['features'] = data.x[u].tolist() if data.x is not None else {}
                    G.nodes[v]['features'] = data.x[v].tolist() if data.x is not None else {}

            graphs.append(G)
        return graphs

    def nx_to_pyg(self, graph):
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges], dtype=torch.float)

        num_nodes = graph.number_of_nodes()
        node_features = torch.tensor([graph.nodes[i]['features'] for i in range(num_nodes)], dtype=torch.float)
        y = torch.tensor([graph.nodes[i]['y'] for i in range(num_nodes)], dtype=torch.float)
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

    def process_graph_filtrations(self, thresholds, data=None, filtration=None):
        # Convert PyG data to Dionysus filtration
        if data is not None and filtration is None:
            filtration = self.pyg_to_dionysus(data)
        filtration = self.filtration if filtration is None else filtration

        # Create filtered graphs
        filtered_graphs = self.create_filtered_graphs(filtration=filtration, thresholds=thresholds, data=data)

        # Convert back to PyG data objects
        pyg_graphs = [self.nx_to_pyg(graph) for graph in filtered_graphs]
        return pyg_graphs