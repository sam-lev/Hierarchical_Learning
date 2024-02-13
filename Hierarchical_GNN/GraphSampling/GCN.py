# import os.path as osp
# import os
# from typing import List, Optional, Tuple, Union
# import json
# import time
import numpy as np
import copy

import torch
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
# from torch_geometric.nn.conv import MessagePassing
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
#
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from .utils import pout, homophily_edge_labels, add_edge_attributes

class GCN(torch.nn.Module):
    def __init__(self,
                 args,
                 data,
                 split_masks,
                 processed_dir,
                 train_data=None,
                 test_data = None
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
        self.num_classes = 2# data.y.shape[-1]#args.num_classes
        self.num_feats = data.x.shape[-1]#args.num_feats
        self.batch_size = args.batch_size
        self.dropout = args.dropout

        self.device = args.device

        self.data = data

        self.edge_index = data.edge_index

        heterophily_num_class = 2
        self.cat = 2

        self.inf_threshold = args.inf_threshold

        self.weight_decay = 5e-8#args.weight_decay

        self.edge_emb_mlp = torch.nn.ModuleList()
        self.edge_emb_mlp.append(nn.Linear(self.cat * self.num_feats , self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.edge_emb_mlp.append(nn.Linear(self.dim_hidden, self.dim_hidden))
        self.edge_emb_mlp.append(nn.Linear(self.dim_hidden, self.cat * self.num_feats))

        self.edge_pred_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.cat * self.num_feats, heterophily_num_class)
        )

        self.train_dataset = args.dataset
        self.dataset = args.dataset


        train_split = self.split_masks["split_idx"][0]
        test_split = self.split_masks["split_idx"][2]
        self.train_idx = torch.nonzero(self.train_dataset.data.train_mask, as_tuple=False).squeeze()
        self.edge_train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

        self.train_size = self.train_dataset.data.train_mask.size(0)

        # n_id global node index
        # e_id global edge index
        # input_id global id of the input_nodes
        # num_sampled_edges / num_sampled_nodes
        # self.train_data = add_edge_attributes(train_data)  # .to(device)
        if train_data is None:
            self.train_size_edges = train_split[1]
        else:
            self.train_dataset.data = train_data
            self.train_size_edges = train_data.edge_index.size(0)

        self.train_loader = NeighborLoader(#Sampler(
            data=copy.copy(train_data),
            input_nodes=None,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            #sizes=[-1],
            num_neighbors=[-1],
            batch_size=self.batch_size,
            shuffle=True,
            directed=False,
        )

        self.train_size = train_data.edge_index.size(0)

        x = self.dataset.data.x
        row, col = self.dataset.data.edge_index
        # pout(("row col", row, col, "row_shape col_shape", row.shape, col.shape))
        # dim edfe feature shoould be num_edges , 2 * dim_node_features
        # initialize edge embeddings with adjacent node features
        # self.train_dataset.data.edge_attr = torch.cat([x[row], x[col]], dim=-1)
        #
        # pout(("edge attr shape", self.train_dataset.data.edge_attr.shape))
        #
        # # create learnable edge embeddings
        # # train_edge_embeddings = nn.Embedding(row.shape[0],self.cat * self.num_feats)
        # self.train_edge_embeddings = nn.Embedding.from_pretrained(self.train_dataset.data.edge_attr,
        #                                                      freeze=False).requires_grad_(True)
        # self.train_edge_embeddings.weight.requires_grad = True
        self.dataset.data.edge_attr = torch.cat([x[row], x[col]], dim=-1)
        # create learnable edge embeddings
        # train_edge_embeddings = nn.Embedding(row.shape[0],self.cat * self.num_feats)
        self.edge_embeddings = nn.Embedding.from_pretrained(self.dataset.data.edge_attr,
                                                                  freeze=False).requires_grad_(True)
        self.edge_embeddings.weight.data.copy_(self.dataset.edge_attr)
        self.edge_embeddings.weight.requires_grad = True



    def edge_embed_idx(selfself, row_idx, col_idx, num_col):
        return row_idx * num_col + col_idx

    def edge_labels(self, labels, edge_index, dtype="float", device=None):
        labels = torch.eq(labels[edge_index[0]], labels[edge_index[1]])
        neg_labels = ~labels
        # if device is not None:
        #     neg_labels.to(device)
        # #if dtype == "float":
        labels = labels.type(torch.FloatTensor)#labels.type(torch.FloatTensor)
        neg_labels = neg_labels.type(torch.FloatTensor)
        # labels = torch.cat((neg_labels,labels),1)
        # pout(("labels shape after cat", labels.shape))

        # X = torch.rand((100, 10))
        # new = torch.zeros((labels.size()[0],2))
        # # mask = X[:, 2] <= threshold
        # # new[mask] = X[mask]
        # new[:, 0] = neg_labels[:]#X[~mask, 2]
        # new[:,1] = labels
        labels = [neg_labels, labels]
        return torch.stack(labels, dim=1)

    def l2_loss(self, weight, factor):
        return factor * torch.square(weight).sum()

    def forward(self, edge_index):

        edge_attr = self.edge_embeddings(edge_index)
        for i,embedding_layer in enumerate(self.edge_emb_mlp):
            edge_attr = embedding_layer(edge_attr)
            if i != self.num_layers - 1:
                edge_attr = F.relu(edge_attr)
                edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
        # data.edge_attr[edge_index] = edge_emb
        edge_logit = self.edge_pred_mlp(edge_attr)
        # edge_logit = F.log_softmax( edge_logit , dim=-1 )

        return edge_logit, edge_attr#F.log_softmax( edge_logit , dim=-1 )

    def train_net(self, input_dict):

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


        all_nodes = self.train_dataset.data.to(device)

        self.train()
        self.training = True
        self.edge_embeddings.weight.requires_grad = True
        # self.edge_embeddings.to(device)

        first_idx = torch.tensor([0,0]).to(device)
        sanity_check_embedding = self.edge_embeddings.weight.clone()

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
            batch_data = self.train_dataset.data.x[n_id].to(device)

            train_sample_size += edge_index.shape[1]
            total_batches += 1

            row, col = edge_index

            # x = all_nodes.x[n_id].to(device)  # subgraph ref
            # y = labels[n_id].to(device)  # subgraph ref
            y = self.edge_labels(labels=batch.y.to(device), edge_index=edge_index, device=device)
            # y = y[:,1].to(device)
            y = y.to(device)
            # pout(("shape batch", y.shape))

            #torch.tensor(data.edge_attr[e_id], requires_grad=True).to(device)
            # edge_attr = data.edge_attr[e_id]

            optimizer.zero_grad()

            out, edge_embedding = self(edge_index = e_id)

            out = out#[:,1]





            # update edge attribute with new embedding
            self.dataset.data.edge_attr[e_id] = edge_embedding

            if isinstance(loss_op, torch.nn.NLLLoss):
                pout(("performing softmax"))
                out = F.log_softmax(out, dim=-1)

            loss = loss_op(out, y)

            # # compute l2 loss of weights in mlps used for
            # # leaarning embeddings and prediction
            # # Compute l2 loss component
            # l2_factor = self.weight_decay
            # l2_loss = 0.0
            # for parameter in self.edge_emb_mlp.parameters():
            #     l2_loss += self.l2_loss(parameter, l2_factor)#.view(-1))
            # for parameter in self.edge_pred_mlp.parameters():
            #     l2_loss += self.l2_loss(parameter, l2_factor)#.view(-1))
            #
            # # add L2 normalization of weights
            # loss += l2_loss

            # back prop
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            threshold_out = torch.zeros_like(out)
            mask = out[:] >= 0.5
            threshold_out[mask] = 1.0
            # threshold_out[~mask] = 0.0
            out = threshold_out

            if (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                total_correct += int(out[:,1].eq(y[:,1]).sum()) #y[:,1]
            else:
                total_correct += int(out.eq(y).sum())

            last_acc = total_correct# float(total_correct)
            # pout(("edge embedding weight:", self.edge_embeddings.weight))
            # pout(("edge embedding weight data:", self.edge_embeddings.weight.data))
            # pout(("edge embedding grad", self.edge_embeddings.weight.grad))
            # pout(("embedding updated:", ~torch.eq(self.edge_embeddings(e_id), edge_embedding )))
            # pout(("embedding sanity check, updated:", ~torch.eq(self.edge_embeddings.weight,
            #                                                    sanity_check_embedding)))
            # pout(("edge embedding", edge_attr))




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

        train_size_edges = y.size(0)

        train_size = (
            train_size_edges
            if isinstance(loss_op, torch.nn.NLLLoss) or isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss)
            else train_size_edges * self.num_classes
        )
        '''val_acc = self.test()
        '''
        #len(self.train_dataset.data)
        # return total_loss/ self.train_size_edges, total_correct / train_size#total_loss, total_correct / self.train_size[0]  # / len(self.train_loader)
        # return total_loss / float(len(self.train_loader)), total_correct / float(train_size)
        return total_loss / float(total_batches) , total_correct / float(train_sample_size), val_loss


    def aggregate_edge_attr(self, edge_attr, edge_index):
        edge_attr_target = edge_attr[edge_index]
        # pout(("target edge attr", edge_attr_target, "target edge shape", edge_attr_target.shape))
        # torch.mean(torch.max(a, -1, True)[0], 1, True)
        return torch.max(edge_attr_target, -1, True)[0]


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
        data.edge_attr = torch.cat([x[row], x[col]], dim=-1)
        data.edge_attr.to(device)
        # create learnable edge embeddings
        # train_edge_embeddings = nn.Embedding(row.shape[0],self.cat * self.num_feats)
        edge_embeddings = nn.Embedding.from_pretrained(data.edge_attr,
                                                            freeze=True).requires_grad_(False).to(device)

        # edge_embeddings = nn.Embedding.from_pretrained(self.edge_embeddings.weight.data.detach().clone(),
        #                                                freeze=True).requires_grad_(False)

        inference_loader = NeighborLoader(
            copy.copy(data),#.edge_index,
            input_nodes=None,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],
            batch_size=data.edge_index.shape[1],
            shuffle=False,
        )
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
        count = 0
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
                    # data.edge_attr[edge_index] = edge_emb
                out = self.edge_pred_mlp(edge_attr)
                edge_logit = out#[:,1]#[:,1]

                # return just homophily prediction
                preds.append(out[:,1])

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
        pred = torch.cat(preds, dim=0)#.cpu()#.numpy()
        ground_truth = torch.cat(ground_truths, dim=0)#.cpu()#.numpy()
        # pout(("xpred ", x_pred))
        # pout(("edge pred ", pred))

        return pred, total_loss/len(inference_loader), ground_truth#torch.from_numpy(np.cat(x_pred,axis=0))##torch.stack(torch.tensor(x_pred).tolist(),dim=0)  # _all