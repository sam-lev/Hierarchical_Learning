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
from .utils import pout, homophily_edge_labels, init_edge_embedding, node_degree_statistics
#profiling tools
# from guppy import hpy
# from memory_profiler import profile
# from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score
from sklearn import metrics
from .experiments.metrics import optimal_metric_threshold
from torch.cuda.amp import autocast

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
            batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            # batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
            self.batch_norms.append(batch_norm_layer)
        batch_norm_layer = nn.BatchNorm1d(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        # batch_norm_layer = nn.LayerNorm(self.dim_hidden) if self.use_batch_norm else nn.Identity(self.dim_hidden)
        self.batch_norms.append(batch_norm_layer)
        # self.edge_emb_mlp.append(nn.Linear(self.dim_hidden,
        #                                    self.dim_hidden))#self.cat * self.num_feats))
        self.edge_pred_mlp = nn.Linear(self.dim_hidden, self.out_dim)

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

        pout((" NOW USING NEIGHBOR SAMPLING PER HOP:"))
        pout((self.num_neighbors))

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
            if i != self.num_layers - 1:
                x = self.dropout(x)

            x = embedding_layer(x)

            x = self.batch_norms[i](x)
            # if i == 0:
            # x_res = x_hidden + x_res
            if i != self.num_layers - 1:
                x = self.act(x)

        # x = x + x_res  # torch.cat([x,x_res],axis=1)
        edge_logit = self.edge_pred_mlp(x)#.squeeze(1)#_jump)
        x = self.batch_norms[-1](x)

        edge_logit = self.probability( edge_logit ).squeeze(1)

        return edge_logit #F.log_softmax( edge_logit , dim=-1 )


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

        # Compute the degree of each node
        if epoch == 0:
            max_degree, avg_degree, degrees = node_degree_statistics(train_data)
            pout((" MAX DEGREE ", max_degree," AVERAGE DEGREE ", avg_degree))
        #     if avg_degree > 25:
        #         self.num_neighbors = [25]
        #         self.val_batch_size = self.batch_size
        #     else:
        #         self.num_neighbors = [-1]
        #         self.val_batch_size = self.batch_size
        #     pout(("NOW USING NEW VALIDATION BATCH SIZE AND NUMBER NEIGHBORS"))
        #     pout(("NUMBER NEIGHBORS", self.num_neighbors))
        #     pout(("VAL BATCH_SIZE", self.val_batch_size))
        #     pout(("Number Nodes Training ", train_data.num_nodes))

        if train_data.num_nodes < 1200:
            num_workers=1
        else:
            num_workers=8

        train_loader = NeighborLoader(  # Sampler(
            train_data,  # copy.copy(train_data),
            input_nodes=None,
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],#self.num_neighbors[: self.num_layers],
            batch_size=self.batch_size,
            shuffle=True,
            # drop_last=True,
            num_workers=num_workers
            # directed=False
            # subgraph_type='induced'
        )

        self.train()
        self.training = True
        self.batch_norms = [bn.to(device) for bn in self.batch_norms]


        approx_thresholds = np.arange(0.0, 1.0, 0.1)

        train_sample_size, total_training_points, number_batches = 0, 0, 0.0
        val_sample_size = 0
        batch_sizes = []
        train_edge_embeddings = {}
        total_acc = 0.
        total_loss = total_correct = total_val_loss = 0.
        predictions = []
        all_labels = []
        for batch in train_loader:
            optimizer.zero_grad()
            number_batches += 1.0
            batch=batch.to(device)
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
            with autocast():
                out = self(edge_index=batch.edge_index,
                                           edge_attr=batch.edge_attr)

            # if self.num_classes == 2:
            #     _, out = out.max(dim=1)
            out=out[:batch.batch_size]
            y=y[:batch.batch_size]
            loss = loss_op(out, y)
            #
            #
            #
            """
                   
                      !!!!!!!! added mean to get loss per sample
                   
                   
            """
            grad_scalar.scale(loss).backward()
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
            total_loss += loss.item()

            if input_steps != -1:
                self.steps += 1#self.model.steps
                counter = self.steps
            else:
                self.steps = -1
                counter = epoch



            if self.num_classes > 2:#(isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                #total_correct += out.argmax(dim=-1).eq(y.argmax(dim=-1)).float().item()#.sum() #y[:,1]
                #total_acc = total_correct/np.sum(batch_sizes).item()
                total_correct += int(out.argmax(dim=-1).eq(y).sum())
                # preds = out.detach().cpu().numpy()
                # ground_truth = y.detach().cpu().numpy()
                all_labels.extend(y)#ground_truth)
                predictions.extend(out)#.argmax(axis=1))

            else:

                # if False: # because too slow
                # preds = out.detach().cpu().numpy()
                # ground_truth = y.detach().cpu().numpy()
                # optimal_threshold, optimal_score = optimal_metric_threshold(y_probs=preds,
                #                                                                     y_true=ground_truth,
                #                                                             metric=accuracy_score,
                #                                                                     metric_name='accuracy',
                #                                                             thresholds=approx_thresholds)
                # # optimal_threshold = 0.0
                # thresh_out = out >= optimal_threshold#).float()
                predictions.extend(out)#thresh_out)
                all_labels.extend(y)
                # # total_correct += (out.long() == thresh_out).float().sum().item()#int(out.eq(y).sum())
                # total_correct += (y == thresh_out).float().sum()#.item()  # int(out.eq(y).sum())
                # # approx_acc = (y == thresh_out).float().mean().item()
                # # print(">>> epoch: ", epoch, " Approx Train ACC: ", optimal_score)


            del y, batch, loss, out
            torch.cuda.empty_cache()


        predictions = torch.tensor(predictions)
        all_labels = torch.tensor(all_labels)
        if self.num_classes > 2:  # (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
            # total_correct += out.argmax(dim=-1).eq(y.argmax(dim=-1)).float().item()#.sum() #y[:,1]
            # total_acc = total_correct/np.sum(batch_sizes).item()

            approx_acc = (predictions == all_labels).float().mean().item()

            del predictions, all_labels
        else:
            predictions = (predictions > 0).float()
            total_correct = (predictions == all_labels).float().sum()
            approx_acc = total_correct/all_labels.numel()
            # total_correct = float(predictions.eq(all_labels).sum()) #chanmged
            all_labels = all_labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            # approx_acc = total_correct/all_labels.shape[0] # changed


            # approx_acc = 0
            # train_opt_thresh, approx_acc = optimal_metric_threshold(y_probs=predictions,
            #                                                                     y_true=all_labels,
            #                                                                     metric=accuracy_score,
            #                                                                     metric_name='accuracy',
            #                                                         thresholds=approx_thresholds)

        if epoch % eval_steps == 0 and epoch != 0:
            with torch.no_grad():
                self.eval()
                val_pred, val_loss, val_ground_truth = self.inference(val_input_dict)
            self.training = True
            self.train()
            # val_out, val_loss, val_labels = self.inference(val_input_dict)

            if self.num_classes > 2:  # (isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(loss_op, torch.nn.NLLLoss)):
                # total_correct += out.argmax(dim=-1).eq(y.argmax(dim=-1)).float().item()#.sum() #y[:,1]
                # total_acc = total_correct/np.sum(batch_sizes).item()
                # approx_acc = (predictions == all_labels).float().mean().item()

                val_acc = (val_pred == val_ground_truth).float().mean().item()
                print(">>> epoch: ", epoch, " validation acc: ", val_acc)
                del val_ground_truth, val_pred

            else:
                # if False:  # because too slow
                val_preds = val_pred.detach().cpu().numpy()
                val_ground_truth = val_ground_truth.detach().cpu().numpy()
                val_optimal_threshold, val_optimal_score = optimal_metric_threshold(y_probs=val_preds,
                                                                                    y_true=val_ground_truth,
                                                                                    metric=accuracy_score,
                                                                                    metric_name='accuracy',
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
                                                                        thresholds=approx_thresholds)
                print("Epoch: ", epoch, f" Val ACC: {val_optimal_score:.4f}",
                      f" Val ROC: {val_roc:.4f}")
                # thresh_out = out >= optimal_threshold
                # # total_correct += (out.long() == thresh_out).float().sum().item()#int(out.eq(y).sum())
                # val_total_correct += (val_ground_truth == thresh_out)  # int(out.eq(y).sum())

                del predictions, all_labels

            if scheduler is not None:
                scheduler.step(val_loss)

            total_val_loss += val_loss

            del val_loss

        # train_data.edge_attr = torch.tensor([train_edge_embeddings[i] for i in range(len(train_edge_embeddings))],
        #                                dtype=torch.float)
        # train_data.edge_attr = torch.from_numpy(
        #     np.array([train_edge_embeddings[(train_data.edge_index[0, i].item(),
        #                            train_data.edge_index[1, i].item())] for i in
        #      range(train_data.edge_index.size(1))]))#, dtype=torch.float)
        # self.train_data = train_data
        # x = scatter(data.x, data.batch, dim=0, reduce='mean')


        train_size_edges = train_data.y.size(0)

        train_size = (
            train_size_edges
            if isinstance(loss_op, torch.nn.NLLLoss) or isinstance(loss_op, torch.nn.CrossEntropyLoss) or isinstance(loss_op, torch.nn.BCEWithLogitsLoss)
            else train_size_edges * self.num_classes
        )
        '''val_acc = self.test()
        '''
        total_loss = total_loss/number_batches#train_sample_size#number_batches

        # pout(("Step:", self.steps))

        # predictions = torch.tensor(predictions)
        # all_labels = torch.tensor(all_labels)
        # t_loss = total_correct / train_sample_size
        # if self.num_classes > 2:
        #     predictions = predictions.argmax(axis=1)
        #     approx_acc = (predictions == all_labels).float().mean().item()
        # else:
        #     approx_acc = accuracy_score(all_labels.cpu().numpy(),
        #                                 predictions.cpu().numpy())
        # del val_loss
        torch.cuda.empty_cache()

        if epoch % eval_steps == 0 and epoch != 0:
            return total_loss , approx_acc, total_val_loss#float(train_sample_size), val_loss
        else:
            return total_loss, approx_acc, 666
    def get_train_data(self):
        return self.train_data
    def aggregate_edge_attr(self, edge_attr, edge_index):
        edge_attr_target = edge_attr[edge_index]
        # pout(("target edge attr", edge_attr_target, "target edge shape", edge_attr_target.shape))
        # torch.mean(torch.max(a, -1, True)[0], 1, True)
        return torch.max(edge_attr_target, -1, True)[0]

    @torch.no_grad()
    def inference(self,input_dict):
        pred, loss, ground_truth, _ = self.edge_inference(input_dict)
        return pred, loss, ground_truth



    @torch.no_grad()
    def edge_inference(self, input_dict, assign_edge_weights = False):
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

        pout(("Batch Size in Edge Inference ", self.batch_size))

        data.edge_weights = torch.zeros(data.edge_index.shape)

        global_edge_index = data.edge_index.t().cpu().numpy()

        inference_loader = NeighborLoader(
            data=data,#copy.copy(data),#.edge_index,
            input_nodes=None,
                                        # subgraph_type='induced',
            # edge_index = train_data.edge_index,
            # input_nodes=self.train_dataset.data.train_mask,
            # sizes=[-1],
            num_neighbors=[-1],
            batch_size=self.batch_size, #data.edge_index.shape[1],
            shuffle=False,
            # drop_last=True,
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

        # pout(("BEFORE INFER"))

        with torch.no_grad():
            for batch in inference_loader:#_size, n_id, adjs  in inference_loader:
                self.eval()
                self.training = False
                number_batches += 1.0




                # target_nodes = batch.n_id[:batch.batch_size]
                # target_edge_index = target_nodes[batch.edge_index]
                # target_attr = batch.edge_attr[]
                batch = batch.to(device)
                batch_size = batch.batch_size

                train_sample_size += batch.edge_index.size(0)

                e_id = batch.e_id

                with autocast():
                    out = self(edge_index=batch.edge_index,
                                           edge_attr=batch.edge_attr)

                batch_labels = self.edge_labels(labels=batch.y,  #data.y[n_id].to(device),
                                                edge_index=batch.edge_index
                                                )
                # pout(("AFTER OUT INFER AND LABELS"))
                # pout((batch_labels.size()))
                # batch_labels = batch_labels

                batch_labels = batch_labels.to(device)
                batch_labels_target = batch_labels[:batch_size]
                # pout(("after label cut"))

                out_target = out[:batch_size]

                # loss = loss_op(out_target, batch_labels_target)
                loss = loss_op(out_target, batch_labels_target)

                if assign_edge_weights:
                    for edge, p in zip(e_id.detach().cpu().numpy(), out.detach().cpu().numpy()):
                        source, target = global_edge_index[edge]
                        edges.append((source, target))
                        edge_weights_dict[(source, target)] = p  # [0]#[p]# edge_weights_dict[edge] = [p]#.append

                # pout(("AFTER LOSS INFER"))

                preds.extend(out)
                total_loss += loss#.item()# / batch_sizes[-1]
                ground_truths.extend(batch_labels)

                # del batch, loss, out, out_target, batch_labels, e_id, batch_size, batch_labels_target
                del batch, loss, out, batch_labels, e_id,batch_size, batch_labels_target, out_target

                torch.cuda.empty_cache()

        # pout(("after inference loop"))
        del global_edge_index

        if assign_edge_weights:
            # pout(("%%%%%% ", "ASSIGNING INFERED FILTRATION VALUES TO EDGES"))
            data.edge_weights = torch.from_numpy(np.array(
                [edge_weights_dict[(data.edge_index[0, i].item(),
                                    data.edge_index[1, i].item())] for i in range(data.edge_index.size(1))]))
            # pout(("Done."))

        # del edge_weights_dict
        # data.edge_attr = torch.from_numpy(np.array(
        #     [edge_embeddings_dict[(data.edge_index[0, i].item(),
        #                            data.edge_index[1, i].item())] for i in range(data.edge_index.size(1))]
        #                               ))

        #

        # pred = torch.cat(preds, dim=0)#.cpu()#.numpy()
        # ground_truth = torch.cat(ground_truths, dim=0)#.cpu()#.numpy()
        # ground_truths = [item for sublist in ground_truths for item in sublist]
        # preds = [item for sublist in preds for item in sublist]
        # avg_total_per_batch = np.sum(batch_sizes)/np.mean(batch_sizes)

        torch.cuda.empty_cache()

        # pout(("returning inference"))

        if assign_edge_weights:
            return torch.tensor(preds), total_loss/train_sample_size, torch.tensor(ground_truths), data.to("cpu")
        else:
            return torch.tensor(preds), total_loss / train_sample_size, torch.tensor(ground_truths), None



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
                                        subgraph_type='induced',
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
