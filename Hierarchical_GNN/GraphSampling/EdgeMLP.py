# import os.path as osp
# import os
# from typing import List, Optional, Tuple, Union
# import json
# import time
from builtins import input

import numpy as np
import copy

import torch
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
# from torch_geometric.nn.conv import MessagePassing
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
#
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from .utils import pout, homophily_edge_labels, init_edge_embedding
#profiling tools
from guppy import hpy
# from memory_profiler import profile
# from memory_profiler import memory_usage

class EdgeMLP(torch.nn.Module):
    #fp = open('./run_logs/memory_profiler.log', 'w+')
    # @profile#stream=fp)
    def __init__(self,
                 args,
                 data,
                 # split_masks,
                 processed_dir,
                 train_data=None,
                 test_data = None
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
        self.num_classes = 2# data.y.shape[-1]#args.num_classes
        self.num_feats = data.x.shape[-1]#args.num_feats
        self.batch_size = args.batch_size
        self.steps = 0
        self.dropout = args.dropout

        self.device = args.device

        self.data = data

        self.edge_index = data.edge_index

        self.out_dim = self.num_classes if self.num_classes != 2 else 1

        self.cat = 2

        self.c1 = 0
        self.c2 = 0

        self.inf_threshold = args.inf_threshold
        self.threshold_pred = False

        self.weight_decay = args.weight_decay

        self.use_batch_norm = args.use_batch_norm

        self.edge_emb_mlp = torch.nn.ModuleList()
        self.batch_norms = []


        # or identity op if use_batch_norm false
        # inherent class imbalance / overfitting
        batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        # batch_norm_layer = nn.BatchNorm1d if self.use_batch_norm else nn.Identity

        self.edge_emb_mlp.append(nn.Linear(self.cat * self.num_feats, self.dim_hidden))
        # batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)
        # construct MLP classifier
        for _ in range(self.num_layers - 2):
            self.edge_emb_mlp.append(nn.Linear(self.dim_hidden, self.dim_hidden))
            """batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            # batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)"""
        batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        # batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)
        # self.edge_emb_mlp.append(nn.Linear(self.dim_hidden,
        #                                    self.dim_hidden))#self.cat * self.num_feats))
        self.edge_pred_mlp = nn.Linear(self.dim_hidden, self.out_dim)

        self.num_neighbors = [6]*46#[-1, 25, 25, 10, 5, 5, 5, 5, 5, 5, 5]



        self.dropout = nn.Dropout(p=self.dropout)
        self.act = nn.GELU()
        self.probability = nn.Sigmoid() #nn.Softmax(dim=1) #nn.Sigmoid()

        self.train_dataset = args.dataset
        self.dataset = args.dataset

        self.reset_parameters()

        a = """
        row, col = self.train_dataset.data.edge_index
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
        self.train_dataset.data.edge_attr = torch.cat([x[row], x[col]], dim=-1)
        # create learnable edge embeddings
        # train_edge_embeddings = nn.Embedding(row.shape[0],self.cat * self.num_feats)
        self.edge_embeddings = nn.Embedding.from_pretrained(self.train_dataset.data.edge_attr,
                                                                  freeze=False).requires_grad_(True)
        self.edge_embeddings.weight.data.copy_(self.train_dataset.edge_attr)
        self.edge_embeddings.weight.requires_grad = True
        """

    def reset_parameters(self):
        for embedding_layer in self.edge_emb_mlp:
            embedding_layer.reset_parameters()
        self.edge_pred_mlp.reset_parameters()
    def edge_embed_idx(selfself, row_idx, col_idx, num_col):
        return row_idx * num_col + col_idx

    def edge_labels(self, labels, edge_index, as_logit=False,
                    dtype="float", device=None):
        # adj_n1 = edge_index[:,0]
        # adj_n2 = edge_index[:,1]
        adj_n1 , adj_n2 = edge_index
        labels = torch.eq(labels[adj_n1], labels[adj_n2])
        neg_labels = ~labels
        labels = labels.type(torch.FloatTensor)#labels.type(torch.FloatTensor)
        neg_labels = neg_labels.type(torch.FloatTensor)
        # labels = [neg_labels, labels]
        # if not as_logit:
        return labels# labels.unsqueeze(-1)#torch.unsqueeze(labels,1)#)#torch.cat(labels#)stack(labels, dim=1)

    def l2_loss(self, weight, factor):
        return factor * torch.square(weight).sum()

    def forward(self, edge_index, edge_attr=None):
        # for i, (edge_index, _, size) in enumerate(adjs):
        #     x_target = x[: size[1]]  # Target nodes are always placed first.
        #     x = self.convs[i]((x, x_target), edge_index)
        x = self.edge_embeddings(edge_index) if edge_attr is None else edge_attr
        # x = self.edge_emb_mlp[0](x)
        # x = self.act(x)
        # x = self.dropout(x)
        # x_res = x
        for i,embedding_layer in enumerate(self.edge_emb_mlp):
            # if i == 0:
            #     x_res = x
            #     continue
            # x_hidden = x_res

            x = embedding_layer(x)
            if i == 0:
                x = self.batch_norms[0](x)
            # x_res = x_hidden + x_res
            x = self.dropout(x)
            if i != self.num_layers - 1:


                x = self.act(x)

        # x = x + x_res  # torch.cat([x,x_res],axis=1)
        x = self.batch_norms[-1](x)
        edge_logit = self.edge_pred_mlp(x).squeeze(1)#_jump)

        edge_logit = self.probability( edge_logit )

        return edge_logit, x#F.log_softmax( edge_logit , dim=-1 )


    #fp = open('./run_logs/training_memory_profiler.log', 'w+')
    # @profile#stream=fp)
    def train_net(self, input_dict):
        ###############################
        train_data = input_dict["train_data"]
        x = train_data.x
        row, col = train_data.edge_index
        device = input_dict["device"]

        epoch = input_dict['epoch']
        eval_steps = input_dict['eval_steps']
        input_steps = input_dict["steps"]

        train_data = init_edge_embedding(train_data)

        # self.edge_embeddings = nn.Embedding.from_pretrained(train_data.edge_attr,
        #                                                     freeze=False).requires_grad_(True)
        # self.edge_embeddings.weight.data.copy_(train_data.edge_attr)
        # self.edge_embeddings.weight.requires_grad = True
        # self.edge_embeddings#.to(device)

        ################################

        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        grad_scalar = input_dict["grad_scalar"]
        scheduler = input_dict["scheduler"]


        val_data = input_dict["val_data"]
        val_input_dict = {"data": val_data,
                            "device": device,
                          "dataset": input_dict["dataset"],
                          "loss_op":loss_op}


        train_loader = NeighborLoader(  # Sampler(
            train_data,  # copy.copy(train_data),
            input_nodes=None,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],#self.num_neighbors[: self.num_layers],
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=8
            # directed=True,
        )

        self.train()
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]

        train_sample_size, total_training_points, number_batches = 0, 0, 0.0
        val_sample_size = 0
        batch_sizes = []
        train_edge_embeddings = {}
        total_acc = 0.
        total_loss = total_correct = total_val_loss = 0
        for batch in train_loader:
            number_batches += 1.0
            batch=batch.to(device)


            optimizer.zero_grad()

            # e_id = batch.e_id.to(device)#adjs.e_id.to(device) # global edge index of each edge
            # edge_index  = batch.edge_index.to(device)

            train_sample_size += batch.edge_index.size(0)#shape[1]

            y = self.edge_labels(labels=batch.y,#.to(device),
                                 edge_index=batch.edge_index,
                                 device=device)
            # y = y[:,1].to(device)
            y = y.to(device)

            batch_sizes.append(y.size()[0])
            # pout(("shape batch", y.shape))


            # out = self(x[n_id], adjs)
            # optimizer.zero_grad()


            # out, edge_embedding = self(edge_index = batch.e_id,
            #                            edge_attr=None)#edge_attr)
            out, edge_embedding = self(edge_index=batch.edge_index,
                                       edge_attr=batch.edge_attr)

            # if self.num_classes == 2:
            #     _, out = out.max(dim=1)

            loss = loss_op(out, y)

            #
            #
            #
            """
                   
                      !!!!!!!! added mean to get loss per sample
                   
                   
            """
            grad_scalar.scale(loss.mean()).backward()
            grad_scalar.step(optimizer)
            grad_scalar.update()
            #
            #
            #
            # loss.backward()
            # optimizer.step()
            """
            
                              !!!!!!!!!!!!!  added sum to take loss per sample
            
            """
            total_loss += loss.sum().item()

            if input_steps != -1:
                self.steps += 1#self.model.steps
                counter = self.steps
            else:
                self.steps = -1
                counter = epoch

            if counter % eval_steps == 0 and counter != 0:
                with torch.no_grad():
                    val_pred, val_loss, val_ground_truth  = self.inference(val_input_dict)
                    # val_out, val_loss, val_labels = self.inference(val_input_dict)
                    if scheduler is not None:
                        scheduler.step(val_loss)

                    total_val_loss += val_loss#.item()





            # update edge attribute with new embedding
            # edge_attr = self.edge_embeddings(e_id).to(device)
            # for e, emb in zip(e_id.detach().cpu().numpy(), edge_embedding.detach().cpu().numpy()):
            #     train_edge_embeddings[e] = emb
            #     train_data.edge_attr[e] = emb
            # for edge, emb_e in zip(e_id.detach().cpu().numpy(), edge_attr.detach().cpu().numpy()):
            #     source, target = global_edge_index[edge]
            #     # # train_edge_embeddings[(source, target)] = emb_e
            #     # # train_data.edge_attr[edge] = torch.tensor(emb_e,dtype=torch.float)#edge_embedding




            if self.num_classes > 2:#(isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                total_correct += out.argmax(dim=-1).eq(y.argmax(dim=-1)).float().mean().item()#.sum() #y[:,1]
                total_acc = total_correct/np.sum(batch_sizes).item()
            else:
                probs = out
                probs = probs > 0.5
                total_correct = (probs.long() == y).float().mean().item()#int(out.eq(y).sum())



            torch.cuda.empty_cache()
            # del y, batch_size, n_id, loss, out, edge_embedding

        # train_data.edge_attr = torch.tensor([train_edge_embeddings[i] for i in range(len(train_edge_embeddings))],
        #                                dtype=torch.float)
        # train_data.edge_attr = torch.from_numpy(
        #     np.array([train_edge_embeddings[(train_data.edge_index[0, i].item(),
        #                            train_data.edge_index[1, i].item())] for i in
        #      range(train_data.edge_index.size(1))]))#, dtype=torch.float)
        self.train_data = train_data
            # x = scatter(data.x, data.batch, dim=0, reduce='mean')


        train_size_edges = train_data.y.size(0)

        train_size = (
            train_size_edges
            if isinstance(loss_op, torch.nn.NLLLoss) or isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss)
            else train_size_edges * self.num_classes
        )
        '''val_acc = self.test()
        '''
        total_loss = total_loss/train_sample_size#number_batches
        total_val_loss = total_val_loss#/number_batches

        pout(("Step:", self.steps))

        avg_total_per_batch = np.sum(batch_sizes) / np.mean(batch_sizes)

        return total_loss , total_correct, total_val_loss#float(train_sample_size), val_loss

    def get_train_data(self):
        return self.train_data
    def aggregate_edge_attr(self, edge_attr, edge_index):
        edge_attr_target = edge_attr[edge_index]
        # pout(("target edge attr", edge_attr_target, "target edge shape", edge_attr_target.shape))
        # torch.mean(torch.max(a, -1, True)[0], 1, True)
        return torch.max(edge_attr_target, -1, True)[0]

    # @profile
    @torch.no_grad()
    def inference_adapted_but_old(self, input_dict):
        # input dict input_dict = {"data": self.data, "y": self.y, "device": self.device, "dataset": self.dataset}
        self.eval()
        self.training = False
        device = input_dict["device"]
        # x = input_dict["x"].to(device)
        data = input_dict["data"]#.to(device)

        data = init_edge_embedding(data)

        # for validation testing
        loss_op  = input_dict["loss_op"]

        labels = self.edge_labels(labels=data.y, edge_index=data.edge_index)

        self.batch_norms = [bn.to(device) for bn in self.batch_norms]




        data.edge_weights = torch.zeros(data.edge_index.shape)

        global_edge_index = data.edge_index.t().cpu().numpy()

        # # create learnable edge embeddings
        # # train_edge_embeddings = nn.Embedding(row.shape[0],self.cat * self.num_feats)
        # edge_embeddings = nn.Embedding.from_pretrained(data.edge_attr,
        #                                                     freeze=True).requires_grad_(False).to(device)


        inference_loader = NeighborLoader(
            data=data,#copy.copy(data),#.edge_index,
            input_nodes=None,
            # directed=True,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],
            batch_size=self.batch_size, #data.edge_index.shape[1],
            shuffle=False,
            # num_workers=8
        )

        train_sample_size = 0
        edges = []
        edge_weights = []
        x_pred = []
        preds = []
        ground_truths = []
        total_loss = 0
        edge_embeddings_dict = {}
        edge_weights_dict = {}
        batch_sizes = []
        number_batches = 0.0
        with torch.no_grad():
            for batch in inference_loader:#_size, n_id, adjs  in inference_loader:
                number_batches += 1.0
                #batch_size, n_id = batch.batch_size, batch.n_id.to(device)  # batch_size, n_id.to(device), adjs
                # edge_index, e_id, size = adjs
                # _, _, size = adjs

                # batch.n_id # global node index of each node
                batch =batch.to(device)
                e_id = batch.e_id # adjs.e_id.to(device) # global edge index of each edge
                edge_index = batch.edge_index
                train_sample_size += edge_index.shape[1]
                edge_attr = batch.edge_attr#edge_embeddings(e_id).to(device)

                # for i, embedding_layer in enumerate(self.edge_emb_mlp):
                #     edge_attr = embedding_layer(edge_attr)
                #     if i != self.num_layers - 1:
                #         edge_attr = F.relu(edge_attr)
                #         edge_attr = self.batch_norms[i](edge_attr)
                #     # data.edge_attr[edge_index] = edge_emb
                # out = self.edge_pred_mlp(edge_attr)
                out, emb = self(edge_index=batch.edge_index,
                                       edge_attr=batch.edge_attr)#self(edge_attr)


                # if self.num_classes ==2:
                #     _, out = out.max(dim=1)

                # return just homophily prediction
                # _,prediction = out.max(dim=1)
                # add predicted edge value as weight for each edge
                for edge, p in zip(e_id.detach().cpu().numpy(), out.detach().cpu().numpy()):
                    source, target = global_edge_index[edge]
                    edges.append((source, target))
                    edge_weights_dict[(source,target)] = p#[0]#[p]# edge_weights_dict[edge] = [p]#.append
                for edge, emb_e in zip(e_id.detach().cpu().numpy(), edge_attr.detach().cpu().numpy()):
                    source, target = global_edge_index[edge]
                    edge_embeddings_dict[(source, target)] = emb_e
                    #self.filtration.append(dion.Simplex([source, target], p))



                # pout(("edge_index", edge_index, "e_id", e_id))

                batch_labels = self.edge_labels(labels=batch.y.to(device),#data.y[n_id].to(device),
                                                edge_index=edge_index,
                                                )
                # if self.num_classes > 2:  # (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                #     total_correct += out.argmax(dim=-1).eq(batch_labels.argmax(dim=-1)).float().mean().item()  # .sum() #y[:,1]
                #     total_acc = total_correct / np.sum(batch_sizes).item()
                # else:
                #     probs = out > 0.5
                #     total_correct = (probs.long() == batch_labels).float().mean().item()  #
                preds.extend(out)  # .argmax(dim=1))#[:,1])
                # batch_labels.to(device)
                batch_sizes.append(batch_labels.size()[0])

                loss = loss_op(out, batch_labels.to(device))

                total_loss += loss.item()#/batch_sizes[-1]
                # _, l = batch_labels.max(dim=1)
                ground_truths.extend(batch_labels)#extend(batch_labels)#.argmax(dim=1))#[:,1])

                # if isinstance(loss_op, torch.nn.NLLLoss):
                #     edge_logit = F.log_softmax(out, dim=-1)
                #     edge_logit = edge_logit.argmax(dim=-1)
                #     edge_logit = F.softmax(out, dim=-1)

                # edge_logit = edge_logit.cpu()

                # pout(("logit shape", edge_logit.shape, "edge pred shape", edge_pred.shape, "edge id shape", e_id.shape))

                # x_pred.append(edge_logit)
                # edge_pred[e_id] = edge_logit.cpu()
        # pout((edge_weights_dict))
        data.edge_weights = torch.from_numpy(np.array(
            [edge_weights_dict[(data.edge_index[0, i].item(),
                                data.edge_index[1, i].item())] for i in range(data.edge_index.size(1))]
        ))

        data.edge_attr = torch.from_numpy(np.array(
            [edge_embeddings_dict[(data.edge_index[0, i].item(),
                                   data.edge_index[1, i].item())] for i in range(data.edge_index.size(1))]
        ))


        # preds = torch.cat(preds, dim=1)#.cpu()#.numpy()
        # ground_truths = torch.cat(ground_truths, dim=1)#.cpu()#.numpy()

        return torch.tensor(preds), total_loss/number_batches, torch.tensor(ground_truths)

    @torch.no_grad()
    def inference(self,input_dict):
        pred, loss, ground_truth, _ = self.edge_inference(input_dict)
        return pred, loss, ground_truth

    @torch.no_grad()
    def edge_inference(self, input_dict):
        # input dict input_dict = {"data": self.data, "y": self.y, "device": self.device, "dataset": self.dataset}
        self.eval()
        self.training = False
        device = input_dict["device"]
        # x = input_dict["x"].to(device)
        data = input_dict["data"]#.to(device)

        data = init_edge_embedding(data)

        # for validation testing
        loss_op  = input_dict["loss_op"]

        # labels = data.y.to(device)
        # labels = self.edge_labels(labels=data.y, edge_index=data.edge_index)#torch.eq(labels[all_edge_idx[0]], labels[all_edge_idx[1]])

        data.edge_weights = torch.zeros(data.edge_index.shape)

        global_edge_index = data.edge_index.t().cpu().numpy()

        # # create learnable edge embeddings
        # # train_edge_embeddings = nn.Embedding(row.shape[0],self.cat * self.num_feats)
        # edge_embeddings = nn.Embedding.from_pretrained(data.edge_embeddings,
        #                                                     freeze=True).requires_grad_(False).to(device)


        inference_loader = NeighborLoader(
            data=data,#copy.copy(data),#.edge_index,
            input_nodes=None,
            # directed=True,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],
            batch_size=self.batch_size, #data.edge_index.shape[1],
            shuffle=False,
            # num_workers=8
        )

        self.batch_norms = [bn.to(device) for bn in self.batch_norms]


        train_sample_size = 0
        edges = []
        edge_weights = []
        x_pred = []
        preds = []
        ground_truths = []
        total_loss = 0
        edge_embeddings_dict = {}
        edge_weights_dict = {}
        batch_sizes = []
        number_batches = 0.0
        with torch.no_grad():
            for batch in inference_loader:#_size, n_id, adjs  in inference_loader:
                number_batches += 1.0
                #batch_size, n_id = batch.batch_size, batch.n_id.to(device)  # batch_size, n_id.to(device), adjs
                # edge_index, e_id, size = adjs
                # _, _, size = adjs

                # batch.n_id # global node index of each node
                batch = batch.to(device)
                e_id = batch.e_id
                train_sample_size += batch.edge_index.size(0)#shape[1]
                edge_attr = batch.edge_attr#edge_embeddings(e_id).to(device)#

                # for i, embedding_layer in enumerate(self.edge_emb_mlp):
                #     edge_attr = embedding_layer(edge_attr)
                #     if i != self.num_layers - 1:
                #         edge_attr = F.relu(edge_attr)
                #         edge_attr = self.batch_norms[i](edge_attr)

                out, emb = self(edge_index=batch.edge_index,
                                       edge_attr=batch.edge_attr)#self.edge_pred_mlp(edge_attr)
                # out = self.probabil(out)#[:,1]#[:,1]

                # if self.num_classes ==2:
                #     _, out = out.max(dim=1)

                # add predicted edge value as weight for each edge
                for edge, p in zip(e_id.detach().cpu().numpy(), out.detach().cpu().numpy()):
                    source, target = global_edge_index[edge]
                    edges.append((source, target))
                    edge_weights_dict[(source,target)] = p#[0]#[p]# edge_weights_dict[edge] = [p]#.append
                for edge, emb_e in zip(e_id.detach().cpu().numpy(), edge_attr.detach().cpu().numpy()):
                    source, target = global_edge_index[edge]
                    edge_embeddings_dict[(source, target)] = emb_e
                    #self.filtration.append(dion.Simplex([source, target], p))




                # pout(("edge_index", edge_index, "e_id", e_id))

                batch_labels = self.edge_labels(labels=batch.y,  #data.y[n_id].to(device),
                                                edge_index=batch.edge_index,
                                                )
                preds.extend(out)
                batch_sizes.append(batch_labels.size()[0])
                loss = loss_op(out, batch_labels.to(device))
                """
                Added 
                
                """
                total_loss += loss.item()# / batch_sizes[-1]
                ground_truths.extend(batch_labels)#append(batch_labels)#.argmax(dim=1))#[:,1])

                # if isinstance(loss_op, torch.nn.NLLLoss):
                #     edge_logit = F.log_softmax(out, dim=-1)
                #     edge_logit = edge_logit.argmax(dim=-1)
                #     edge_logit = F.softmax(out, dim=-1)

                # edge_logit = edge_logit.cpu()

                # pout(("logit shape", edge_logit.shape, "edge pred shape", edge_pred.shape, "edge id shape", e_id.shape))

                # x_pred.append(edge_logit)
                # edge_pred[e_id] = edge_logit.cpu()
        data.edge_weights = torch.from_numpy(np.array(
            [edge_weights_dict[(data.edge_index[0, i].item(),
                                data.edge_index[1, i].item())] for i in range(data.edge_index.size(1))]))

        data.edge_attr = torch.from_numpy(np.array(
            [edge_embeddings_dict[(data.edge_index[0, i].item(),
                                   data.edge_index[1, i].item())] for i in range(data.edge_index.size(1))]
                                      ))

        #

        # pred = torch.cat(preds, dim=0)#.cpu()#.numpy()
        # ground_truth = torch.cat(ground_truths, dim=0)#.cpu()#.numpy()

        avg_total_per_batch = np.sum(batch_sizes)/np.mean(batch_sizes)
        return torch.tensor(preds), total_loss/train_sample_size, torch.tensor(ground_truths), data.to("cpu")
