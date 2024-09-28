import os.path as osp
import os
from typing import List, Optional, Tuple, Union
import json
import time
import numpy as np

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import negative_sampling
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import NeighborSampler, NodeLoader, DataLoader
from torch_geometric.nn.conv import MessagePassing

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.aggr import Aggregation, MultiAggregation

from ._GraphSampling import _GraphSampling

from utils import GB, MB, compute_tensor_bytes, get_memory_usage
from .utils import pout

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform = T.Compose([
#     T.NormalizeFeatures(),
#     T.ToDevice(device),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                       add_negative_train_samples=False),
# ])
# data_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'Planetoid')
# dataset = Planetoid(data_path, name='Cora')
# print(">>>> Data Stats")
# print(">>>> Number Features ", dataset.num_features)
# print(">>>> Number Classes  ", dataset.num_classes)
#
# dataset = Planetoid(data_path, name='Cora', transform=transform)
# # After applying the `RandomLinkSplit` transform, the data is transformed from
# # a data object to a list of tuples (train_data, val_data, test_data), with
# # each element representing the corresponding split.
# train_data, val_data, test_data = dataset[0]




class MLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        # num_classes: int = None,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        model=None,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        super(MLP, self).__init__()#**kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)


        if self.project:
            pout(("Using Project for mlp instantiation"))
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        # if self.aggr is None:
        #     self.fuse = False  # No "fused" message_and_aggregate.
        #     self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)
        # if False:
        #     a = 1
        # self.__user_args__ = self.inspector.keys(
        #     ['message', 'aggregate', 'update']).difference(self.special_args)
        # self.kwargs = kwargs

        # if 'aggr' in kwargs.keys() and  isinstance(kwargs['aggr'], (tuple, list)):
        #
        #     aggr_out_channels = in_channels[0]
        #     if self.out_channels is not None:
        #         aggr_out_channels = self.out_channels
        #     if kwargs['mode'] == 'cat':
        #         aggr_out_channels =  in_channels[0] * len(kwargs['aggrs'])
        # else:
        aggr_out_channels = in_channels[0]

        self.in_channels = in_channels
        self.out_channels = out_channels
        # GCNConv(in_channels, hidden_channels)  # torch.nn.

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)

        if self.root_weight:
            pout(("Using root weight in mlp instantiation"))
            self.lin_r = Linear(in_channels[1], out_channels,bias=False)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])#1])

        x_l = x[0]
        out = self.lin_l(x_l)

        x_r = x[0]#1]                                        #
        if self.root_weight and x_r is not None:          #   !!  We shouldn't add the
            out = out + self.lin_r(x_r)                   #   !!  target node embedding
                                                          #   !!  for MLP
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out






# mlp_init = True
#
# # instantiate model
# num_features = 1433
# num_classes  = 7
# dim_hidden   = 64*4
# gnn_model = GCN(num_features, dim_hidden, num_classes).to(device)
# mlp_model = MLP(num_features, dim_hidden, num_classes).to(device)
#
# data = dataset[0]

# optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
# optimizer_mlp = torch.optim.Adam(mlp_model.parameters())
# loss_func = torch.nn.CrossEntropyLoss()
#
# epochs_mlp = 40
# epochs_gnn = 40

# f_gnn: graph neural network model
# f_mlp: PeerMLP of f_gnn
# Train PeerMLP for N epochs
# for epoch in range(epochs_mlp):
#     # for X, Y in dataloader_mlp:
#     mlp_model.train()
#     pred = mlp_model(train_data.x)
#     loss = loss_func(pred[train_data.train_mask], train_data.y[train_data.train_mask])
#     optimizer_mlp.zero_grad()
#     loss.backward()
#     optimizer_mlp.step()

#
# print("Model's state_dict:")
# for param_tensor in mlp_model.state_dict():
#     print(param_tensor, "\t", mlp_model.state_dict()[param_tensor].size())
#
# # Initialize GNN with MLPInit
# weight_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'Weights')
# # device = torch.device("cuda")
# torch.save(mlp_model.state_dict(), osp.join(weight_path,"w_mlp.pt"))
#
# if mlp_init:
#     gnn_model.load_state_dict(torch.load(osp.join(weight_path,"w_mlp.pt"),
#                                          map_location=device))
#     gnn_model.to(device)

#


# Train GNN for n epochs

# best_val_acc = final_test_acc = 0
# for epoch in range(epochs_gnn):
#     # for X, A, Y in dataloader_gnn:
#     gnn_model.train()
#     pred = gnn_model(train_data.x, train_data.edge_index)
#     loss = loss_func(pred[train_data.train_mask], train_data.y[train_data.train_mask])
#
#     train_acc, val_acc, tmp_test_acc = test()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#         print(">>>> acc ", best_val_acc)
#
#     optimizer_gnn.zero_grad()
#     loss.backward()
#     optimizer_gnn.step()


class MLPInit(torch.nn.Module):#_GraphSampling):
    def __init__(self,
                 args,
                 data,
                 processed_dir,
                 # split_masks,
                 input_dict,
                 evaluator,
                 dataset=None,
                 gnn_model=None):

        # train_idx = split_masks["train"]
        # self.split_masks = split_masks
        self.save_dir = processed_dir

        super(MLPInit, self).__init__()#args, data, train_idx, processed_dir)

        self.eval_steps = args.eval_steps
        self.input_dict = input_dict

        self.evaluator = evaluator

        device = input_dict["device"]
        self.device = device
        self.x = input_dict["x"].to(device)
        self.y = input_dict["y"].to(device)

        self.data = data

        self.epochs = args.epochs
        self.convs = torch.nn.ModuleList()

        self.type_model = args.type_model
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.train_size = train_idx.size(0)
        self.dropout = args.dropout
        self.train_idx = train_idx

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.convs.append(MLP(self.num_feats,
                              self.dim_hidden,
                              # num_classes=self.num_classes,
                              model=gnn_model))
        for _ in range(self.num_layers - 2):
            self.convs.append(MLP(self.dim_hidden,
                                  self.dim_hidden,
                                  # num_classes=self.num_classes,
                                  model=gnn_model))
        self.convs.append(MLP(self.dim_hidden,
                              self.num_classes,
                              num_classes=self.num_classes,
                              model=gnn_model))

        # self.optimizer = input_dict["optimizer"]
        # self.loss_op = input_dict["loss_op"]
        self.optimizer = torch.optim.Adam(self.convs.parameters(), lr=args.lr)
        self.loss_op = torch.nn.CrossEntropyLoss()


        # num_neighbors = [25, 10, 5, 5, 5, 5, 5, 5, 5]#[0, 0, 0, 0, 0, 0, 0, 0, 0]#
        # self.train_loader = NeighborSampler(
        #     data.edge_index,
        #     node_idx=train_idx,
        #     sizes=num_neighbors[: self.num_layers],
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=8,
        # )
        # self.test_loader = NeighborSampler(
        #     data.edge_index, sizes=[-1], batch_size=1024, shuffle=False
        # )

        self.train_loader = DataLoader(#odeLoader(  # NeighborSampler(
            data.train_mask,  # .edge_index,
            # node_sampler=DataLoader,
            # input_nodes=train_idx,
            # sizes=num_neighbors[: self.num_layers],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )
        self.test_loader = DataLoader(  # NeighborSampler(
            data,#split_masks["test"],  # .edge_index,
            # node_sampler=DataLoader,
            # input_nodes=None,#train_idx,
            # sizes=num_neighbors[: self.num_layers],
            batch_size=1024,#self.batch_size,
            shuffle=False,
            num_workers=8,
        )


        # self.gnn_model = gnn_model

    def forward(self, x, adjs):
        for i, conv in enumerate(self.convs):#adjs):
            # x_target = x#[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x), adjs)#edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def train_net(self, input_dict):

        device = input_dict["device"]
        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        # optimizer = input_dict["optimizer"]
        # loss_op = input_dict["loss_op"]

        total_loss = total_correct = 0
        self.train()
        #for data in self.train_loader:#enumerate(self.train_loader):#_size, n_id, adjs in self.train_loader:

        nodes, labels = self.data.x.to(device), self.data.y.to(device)
        # nodes = nodes[self.split_masks["train"]]
        # labels = labels[self.split_masks["train"]]
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        # adjs = [adj.to(device) for adj in adjs]

        out = self(nodes, labels)#x[n_id], adjs)
        # if isinstance(self.loss_op, torch.nn.NLLLoss):
        # out = F.log_softmax(out, dim=-1)


        loss = self.loss_op(out, labels)#y[n_id[:batch_size]])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += float(loss.item())
        # if isinstance(self.loss_op, torch.nn.NLLLoss):
        total_correct += int(out.argmax(dim=-1).eq(labels).sum())#.eq(y[n_id[:batch_size]]).sum())
        # else:
        #     total_correct += int(out.eq(labels).sum())#y[n_id[:batch_size]]).sum())

        train_size = (
            self.train_size
            if isinstance(self.loss_op, torch.nn.NLLLoss)
            else self.train_size * self.num_classes
        )
        val_acc = self.test()
        print("%%%% val acc", val_acc)
        return total_loss , total_correct / self.train_size #/ len(self.train_loader)

    def train_mlp_model(self):
        self.train()
        input_dict = self.input_dict
        train_loss, train_acc = self.train_net(input_dict)
        return train_loss, train_acc



    def save_mlp_weights(self):
        # Initialize GNN with MLPInit
        self.weight_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'Weights')
        # device = torch.device("cuda")
        torch.save(self.state_dict(), osp.join(self.weight_path, "w_mlp.pt"))

    def init_model_weights(self, gnn_model):
        gnn_model.load_state_dict(torch.load(osp.join(self.weight_path, "w_mlp.pt"),
                                             map_location=self.device))
        gnn_model.to(self.device)
        return gnn_model

    def mem_speed_bench(self, input_dict):
        torch.cuda.empty_cache()
        device = input_dict["device"]

        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        model_opt_usage = get_memory_usage(0, False)
        usage_dict = {
            "model_opt_usage": model_opt_usage,
            "data_mem": [],
            "act_mem": [],
            "peak_mem": [],
            "duration": [],
        }
        print(
            "model + optimizer only, mem: %.2f MB"
            % (usage_dict["model_opt_usage"] / MB)
        )

        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        init_mem = get_memory_usage(0, False)
        data_mem = init_mem - usage_dict["model_opt_usage"]
        usage_dict["data_mem"].append(data_mem)
        print("data mem: %.2f MB" % (data_mem / MB))

        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        for i, (batch_size, n_id, adjs) in enumerate(self.train_loader):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            iter_start_time = time.time()
            torch.cuda.synchronize()
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = self(x[n_id], adjs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            print(f'num_sampled_nodes: {out.shape[0]}')
            loss = loss_op(out, y[n_id[:batch_size]])
            before_backward = get_memory_usage(0, False)
            act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
            usage_dict["act_mem"].append(act_mem)
            print("act mem: %.2f MB" % (act_mem / MB))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            iter_end_time = time.time()
            duration = iter_end_time - iter_start_time
            print("duration: %.4f sec" % duration)
            usage_dict["duration"].append(duration)
            peak_usage = torch.cuda.max_memory_allocated(0)
            usage_dict["peak_mem"].append(peak_usage)
            print(f"peak mem usage: {peak_usage / MB}")
            torch.cuda.empty_cache()
            del adjs, batch_size, n_id, loss, out

        with open(
            "./%s_mlpinit_mem_speed_log.json" % (self.saved_args["dataset"]), "w"
        ) as fp:
            info_dict = {**self.saved_args, **usage_dict}
            del info_dict["device"]
            json.dump(info_dict, fp)
        exit()

    @torch.no_grad()
    def test(self):
        self.eval()
        nodes, labels = self.data.x.to(self.device), self.data.y.to(self.device)
        # nodes = nodes[self.split_masks["valid"]]
        # labels = labels[self.split_masks["valid"]]
        pred = self(nodes, labels).argmax(dim=-1)

        accs = []
        accs.append(int((pred== labels).sum()) / int(len(labels)))
        return accs[0]

    @torch.no_grad()
    def inference(self, input_dict):
        self.eval()
        device = input_dict["device"]
        data = input_dict["data"]
        x_all = []# input_dict["x"]
        x, labels = data.x.to(device), data.y.to(device)
        for i, conv in enumerate(self.convs):
            xs = []
            #for batch, (nodes, labels) in enumerate(self.test_loader):#_, n_id, adj in self.test_loader:
            # nodes, labels = self.data.x.to(device), self.data.y.to(device)#nodes.to(device), labels.to(device)
            #edge_index, _, size = adj.to(device)
            #x = x_all[n_id].to(device)
            # x_target = nodes# x[: size[1]]
            x = conv((x, x), labels)# edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
            x_all.append(x.cpu())
        # x_all = torch.cat(x_all, dim=0)
        return x, input_dict["dataset"], labels

    # @torch.no_grad()
    # def test_net(self):
    #     self.eval()
    #     input_dict = {"x": self.x, "y": self.y, "device": self.device}
    #     out = self.inference(input_dict)
    #
    #     if self.evaluator is not None:
    #         y_true = self.y.unsqueeze(-1)
    #         y_pred = out.argmax(dim=-1, keepdim=True)
    #
    #         train_acc = self.evaluator.eval(
    #             {
    #                 "y_true": y_true[self.split_masks["train"]],
    #                 "y_pred": y_pred[self.split_masks["train"]],
    #             }
    #         )["acc"]
    #         valid_acc = self.evaluator.eval(
    #             {
    #                 "y_true": y_true[self.split_masks["valid"]],
    #                 "y_pred": y_pred[self.split_masks["valid"]],
    #             }
    #         )["acc"]
    #         test_acc = self.evaluator.eval(
    #             {
    #                 "y_true": y_true[self.split_masks["test"]],
    #                 "y_pred": y_pred[self.split_masks["test"]],
    #             }
    #         )["acc"]
    #     else:
    #
    #         if not self.multi_label:
    #             pred = out.argmax(dim=-1).to("cpu")
    #             y_true = self.y
    #             correct = pred.eq(y_true)
    #             train_acc = (
    #                     correct[self.split_masks["train"]].sum().item()
    #                     / self.split_masks["train"].sum().item()
    #             )
    #             valid_acc = (
    #                     correct[self.split_masks["valid"]].sum().item()
    #                     / self.split_masks["valid"].sum().item()
    #             )
    #             test_acc = (
    #                     correct[self.split_masks["test"]].sum().item()
    #                     / self.split_masks["test"].sum().item()
    #             )
    #
    #         else:
    #             pred = (out > 0).float().numpy()
    #             y_true = self.y.numpy()
    #             # calculating F1 scores
    #             train_acc = (
    #                 f1_score(
    #                     y_true[self.split_masks["train"]],
    #                     pred[self.split_masks["train"]],
    #                     average="micro",
    #                 )
    #                 if pred[self.split_masks["train"]].sum() > 0
    #                 else 0
    #             )
    #
    #             valid_acc = (
    #                 f1_score(
    #                     y_true[self.split_masks["valid"]],
    #                     pred[self.split_masks["valid"]],
    #                     average="micro",
    #                 )
    #                 if pred[self.split_masks["valid"]].sum() > 0
    #                 else 0
    #             )
    #
    #             test_acc = (
    #                 f1_score(
    #                     y_true[self.split_masks["test"]],
    #                     pred[self.split_masks["test"]],
    #                     average="micro",
    #                 )
    #                 if pred[self.split_masks["test"]].sum() > 0
    #                 else 0
    #             )
    #
    #     return out, (train_acc, valid_acc, test_acc)

    def train_and_test_mlp_model(self, seed):
        results = []
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_mlp_model()#epoch)  # -wz-run
            print(
                f"Seed: {seed:02d}, "
                f"Epoch: {epoch:02d}, "
                f"Loss: {train_loss:.4f}, "
                f"Approx Train Acc: {train_acc:.4f}"
            )

            if epoch % self.eval_steps == 0 and epoch != 0:
                out, result = self.test_net()
                results.append(result)
                train_acc, valid_acc, test_acc = result
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {train_loss:.4f}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Valid: {100 * valid_acc:.2f}% "
                    f"Test: {100 * test_acc:.2f}%"
                )

        results = 100 * np.array(results)
        best_idx = np.argmax(results[:, 1])
        best_train = results[best_idx, 0]
        best_valid = results[best_idx, 1]
        best_test = results[best_idx, 2]
        print(
            f"Best train: {best_train:.2f}%, "
            f"Best valid: {best_valid:.2f}% "
            f"Best test: {best_test:.2f}%"
        )

        self.save_mlp_weights()

        return best_train, best_valid, best_test