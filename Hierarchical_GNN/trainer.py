import os
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset


import numpy as np
import copy

import torch
import torch_geometric.datasets
from sklearn.metrics import f1_score
from torch.profiler import ProfilerActivity, profile
from torch_geometric.transforms import ToSparseTensor, ToUndirected, RandomLinkSplit
import torch_geometric.transforms as Transform
from torch.optim.lr_scheduler import ReduceLROnPlateau

from GraphSampling import *
# from LP.LP_Adj import LabelPropagation_Adj
# from Precomputing import *
from GraphSampling.MLPInit import *
from utils import pout, pout_model
from GraphSampling.utils import add_edge_attributes

from TopologicalPriorsDataset import *

def edge_label(data, node_idx=None):
    edge_index = data.edge_index
    labels = data.y
    labels = torch.eq(labels[edge_index[0]], labels[edge_index[1]])
    labels = labels.type(torch.FloatTensor)

    return labels

def get_splits(data, split_percents = [0.3, .3, 1.0]):
    x = data.x
    y = data.y

    labels = y  # torch.eq(y[data.edge_index[0]], y[data.edge_index[1]])

    num_nodes = x.size(0)
    num_edges = data.edge_index.size(1)


    train_percent = split_percents[0]
    val_percent = split_percents[1]
    test_percent = split_percents[2]

    train_split = [0, int(train_percent * num_nodes)]
    train_mask = torch.zeros(num_nodes)  # , dtype=torch.bool)
    train_mask[:int(train_percent * num_nodes)] = 1.
    data.train_mask = train_mask.to(torch.bool)
    # val mask

    val_split = [int(train_percent * num_nodes),
                 int((train_percent + val_percent) * num_nodes)]
    val_mask = torch.zeros(num_nodes)  # , dtype=torch.bool)
    val_mask[int(train_percent * num_nodes):int((train_percent + val_percent) * num_nodes)] = 1.
    data.val_mask = val_mask.to(torch.bool)

    # test mask
    if test_percent == 1.0:
        test_split = [0, int(num_nodes)]
        data.test_mask = torch.ones(num_nodes, dtype=torch.bool)
    else:
        test_split = [int((train_percent + val_percent) * num_nodes), int(num_nodes)]
        test_mask = torch.zeros(num_nodes)
        test_mask[int((train_percent + val_percent) * num_nodes):] = 1.
        data.test_mask = test_mask.to(torch.bool)

    #############
    edge_labels = torch.eq(y[data.edge_index[0]], y[data.edge_index[1]])
    edge_labels = edge_labels.type(torch.FloatTensor)  # .to(torch.float)
    # train mask
    edge_train_mask = torch.zeros(num_edges)  # , dtype=torch.bool)
    edge_train_mask[:int(train_percent * num_edges)] = 1.
    data.edge_train_mask = edge_train_mask.to(torch.bool)
    # val mask
    edge_val_mask = torch.zeros(edge_labels.shape[0])  # , dtype=torch.bool)
    edge_val_mask[int(train_percent * num_edges):int((train_percent + val_percent) * num_edges)] = 1.
    data.edge_val_mask = edge_val_mask.to(torch.bool)
    # test mask
    if test_percent == 1.0:
        test_split_edge = [0, int(num_edges)]
        data.edge_test_mask = torch.ones(num_edges, dtype=torch.bool)
    else:
        test_split_edge = [int((train_percent + val_percent) * num_edges), int(num_edges)]
        edge_test_mask = torch.zeros(num_edges)
        edge_test_mask[int((train_percent + val_percent) * num_edges):] = 1.
        data.edge_test_mask = edge_test_mask.to(torch.bool)

    split_masks = {}

    # split ratios
    split_masks["split_idx"] = torch.tensor(np.array([train_split, val_split, test_split]))
    split_masks["train"] = data.train_mask
    split_masks["valid"] = data.val_mask
    split_masks["test"] = data.test_mask
    # edge masks
    split_masks["edge_train"] = data.edge_train_mask
    split_masks["edge_valid"] = data.edge_val_mask
    split_masks["edge_test"] = data.edge_test_mask
    return data, split_masks

# def get_splits(x, split_frac=[0.3,0.2,0.5]):
#     N_tr = int(x.size(0) * split_frac[0])
#     N_val = int(x.size(0) * split_frac[0])
#     N_test = N_tr+N_val if split_frac[2] != 1.0 else 0
#     split_idx = [[0,N_tr],[N_tr,N_tr+N_val],[N_tr,x.size(0)]]
#     train_mask = torch.zeros(x.size(0), dtype=torch.bool)
#     train_mask[:N_tr] = True
#     val_mask = torch.zeros(x.size(0), dtype=torch.bool)
#     val_mask[N_tr:N_tr+N_val] = True
#     test_mask = torch.zeros(x.size(0), dtype=torch.bool)
#     test_mask[N_test:] = True
#     split_masks = {}
#     split_masks['split_idx'] = split_idx
#     split_masks["train"] = train_mask
#     split_masks["valid"] = val_mask
#     split_masks["test"] = test_mask
#     return split_masks


def load_data(dataset_name, to_sparse=True, homophily=0.1):
    if dataset_name in ["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]:
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        T = ToSparseTensor() if to_sparse else lambda x: x
        if to_sparse and dataset_name == "ogbn-arxiv":
            T = lambda x: ToSparseTensor()(ToUndirected()(x))
        dataset = PygNodePropPredDataset(name=dataset_name, root=root, transform=T)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=dataset_name)
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]

        x = data.x
        y = data.y = data.y.squeeze()

    elif dataset_name in ["Reddit", "Flickr", "AmazonProducts", "Yelp"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        T = ToSparseTensor() if to_sparse else lambda x: x
        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset = dataset_class(path, transform=T)
        processed_dir = dataset.processed_dir
        data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y
        # E = data.edge_index.shape[1]
        # N = data.train_mask.shape[0]
        # data.edge_idx = torch.arange(0, E)
        # data.node_idx = torch.arange(0, N)
    elif dataset_name in ["Planetoid"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        transform = ToSparseTensor() if to_sparse else lambda x: x
        # Transform.Compose([
        #     Transform.NormalizeFeatures(),
        #     # Transform.ToDevice(device),
        #     Transform.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                       add_negative_train_samples=False),
        # ])
        # data_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'Planetoid')
        # dataset = Planetoid(data_path, name='Cora')
        # print(">>>> Data Stats")
        # print(">>>> Number Features ", dataset.num_features)
        # print(">>>> Number Classes  ", dataset.num_classes)

        # dataset = Planetoid(path, name='Cora', transform=transform)

        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset = dataset_class(path,name='Cora', transform=transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.
        processed_dir = dataset.processed_dir
        data = dataset[0]
        # train_data, val_data, test_data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y
    elif dataset_name in ["WebKB"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        transform = ToSparseTensor() if to_sparse else lambda x: x
        # Transform.Compose([
        #     Transform.NormalizeFeatures(),
        #     # Transform.ToDevice(device),
        #     Transform.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                       add_negative_train_samples=False),
        # ])
        # data_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'Planetoid')
        # dataset = Planetoid(data_path, name='Cora')
        # print(">>>> Data Stats")
        # print(">>>> Number Features ", dataset.num_features)
        # print(">>>> Number Classes  ", dataset.num_classes)

        # dataset = Planetoid(path, name='Cora', transform=transform)

        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset = dataset_class(path,name='wisconsin', transform=transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.
        processed_dir = dataset.processed_dir
        data = dataset[0]
        # train_data, val_data, test_data = dataset[0]
        evaluator = None

        split_masks = get_splits(data)
        data.train_mask = split_masks['train']
        data.val_mask = split_masks['valid']
        data.test_mask = split_masks['test']
        dataset.data = data
        x = data.x
        y = data.y
    elif dataset_name in ["MixHopSyntheticDataset"]:
        pout(("homophily of graph", homophily))
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        transform = lambda x: x #ToSparseTensor() if to_sparse else lambda x: x
        # Transform.Compose([
        #     Transform.NormalizeFeatures(),
        #     # Transform.ToDevice(device),
        #     Transform.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                       add_negative_train_samples=False),
        # ])


        # dataset = Planetoid(path, name='Cora', transform=transform)

        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset =  dataset_class(path,
                                 homophily=homophily,
                                 transform = transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.
        processed_dir = dataset.processed_dir
        data = dataset[0]
        # train_data, val_data, test_data = dataset[0]
        evaluator = None




        data, split_masks = get_splits(data)
        data.train_mask = split_masks['train']
        data.val_mask = split_masks['valid']
        data.test_mask = split_masks['test']
        dataset.data = data
        x = data.x
        y = data.y


        pout(("dimension of training data", x.shape, "dimension edge index", data.edge_index.shape))


    elif "TopologicalPriorsDataset" in dataset_name:
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )

        split_percents = [0.3,0.2,0.5]
        dataset = TopologicalPriorsDataset(name=dataset_name, root=root, split_percents=split_percents)
        data = dataset.data
        x = data.x
        y = data.y

        # pout((y))

        edge_index  = data.edge_index
        processed_dir = dataset.processed_dir

        evaluator = None





        data, split_masks = get_splits(data)
        data.train_mask = split_masks['train']
        data.val_mask = split_masks['valid']
        data.test_mask = split_masks['test']
        dataset.data = data
        x = data.x
        y = data.y

        # pout(("Geometric points of nodes", dataset.node_points))


    else:
        raise Exception(f"the dataset of {dataset_name} has not been implemented")
    return data, x, y, split_masks, evaluator, processed_dir, dataset



def idx2mask(idx, N_nodes):
    mask = torch.tensor([False] * N_nodes, device=idx.device)
    mask[idx] = True
    return mask


class trainer(object):
    def __init__(self, args, seed):

        self.dataset = args.dataset
        self.device = torch.device(f"cuda:{args.cuda_num}" if args.cuda else "cpu")
        self.args = args
        self.args.device = self.device

        # self.seed = seed

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.eval_steps = args.eval_steps

        # used to indicate multi-label classification.
        # If it is, using BCE and micro-f1 performance metric
        self.multi_label = args.multi_label
        if self.multi_label:
            self.loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_op = torch.nn.BCEWithLogitsLoss()#torch.nn.NLLLoss() #torch.nn.CrossEntropyLoss()#

        # threshold for class assignment from inferred logit value
        self.inf_threshold = 0.5
        args.inf_threshold = self.inf_threshold
        (
            self.data,
            self.x,
            self.y,
            self.split_masks,
            self.evaluator,
            self.processed_dir,
            self.dataset,
        ) = load_data(args.dataset, args.tosparse, args.homophily)

        #
        #            NOTE GLOBAL CHANGE TO DATA FOR EDGE FEAT
        #
        # self.data = add_edge_attributes(self.data)
        labels = self.y
        labels = torch.eq(labels[self.data.edge_index[0]], labels[self.data.edge_index[1]])

        #
        #  To Float
        #
        # self.labels = labels.type(torch.FloatTensor)

        args.dataset = self.dataset


        # Performs an edge-level random split into training, validation and test sets of a Data
        # or a HeteroData object (functional name: random_link_split).
        # The split is performed such that the training split does not include edges
        # in validation and test splits; and the validation split does not include edges in the test split.
        edge_train_transform = RandomLinkSplit(is_undirected=True, num_val=0.3, num_test=0.1)
        self.train_data, self.val_data, self.test_data = edge_train_transform(self.data)

        # pout(("dataset nodes points", self.dataset.node_points))

        if self.type_model in ["GCN", "GCN_MLPInit"]:# and self.type_model in ["MLPInit"]:
            self.model = GCN(
                args, self.data, self.split_masks, self.processed_dir,
                train_data=self.train_data, test_data=self.test_data
            )
        else:
            raise NotImplementedError("please specify `type_model`")
        self.model.to(self.device)

        #if len(list(self.model.parameters())) != 0:
        self.optimizer =  torch.optim.SGD(#)
                self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=0.9
            )
        #else:
        #    self.optimizer = None
        # Define the learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                           factor=0.1, patience=3, threshold=1e-2)

        if "MLPInit" in self.type_model:# in ["GraphSAGE_MLPInit"]:

            gnn_model = self.model
            pout("gnn model param")
            pout(self.model)

            input_dict = self.get_input_dict(0)
            mlp_init_model = MLPInit(
                args,
                self.data,
                self.processed_dir,
                split_masks = self.split_masks,
                input_dict=input_dict,
                evaluator= self.evaluator,
                dataset=self.dataset,
                gnn_model=self.model
            )
            self.model = mlp_init_model.to(self.device)
            self.train_and_test(seed)
            self.model.save_mlp_weights()
            pout("mlp_init model param")
            pout_model(self.model)
            pout(" initializing gnn model with trained mlp weights ")
            self.model = mlp_init_model.init_model_weights(gnn_model=gnn_model)
            pout(" mlp initialized ")

    def mem_speed_bench(self):
        input_dict = self.get_input_dict(0)
        self.model.mem_speed_bench(input_dict)

    def train_ensembling(self, seed):
        # assert isinstance(self.model, (SAdaGCN, AdaGCN, GBGCN))
        input_dict = self.get_input_dict(0)
        acc = self.model.train_and_test(input_dict)
        return acc

    def test_cpu_mem(self, seed):
        input_dict = self.get_input_dict(0)
        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
        ) as prof:
            acc = self.model.train_and_test(input_dict)

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        return acc

    def train_and_test(self, seed):
        results = []
        results_train = []
        results_test = []
        for epoch in range(self.epochs):
            train_loss, train_acc, val_loss = self.train_net(epoch)  # -wz-run
            seed = int(seed)
            print(
                f"Seed: {seed:02d}, "
                f"Epoch: {epoch:02d}, "
                f"Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Approx Train Acc: {train_acc:.4f}"
            )

            if epoch % self.eval_steps == 0 and epoch != 0:
                out, result, f1_scores = self.test_net()

                results.append(result[2])
                results_test.append(result[1])
                results_train.append(result[0])
                train_acc, valid_acc, test_acc = result
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {train_loss:.4f}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Valid: {100 * valid_acc:.2f}% "
                    f"Test: {100 * test_acc:.2f}%"
                )

        train_f1, test_f1, all_f1 = f1_scores
        results = 100 * np.array(results)
        results_train = 100 * np.array(results_train)
        results_test = 100 * np.array(results_test)
        best_idx = np.argmax(results)
        best_idx_test = np.argmax(results_test)
        best_idx_train = np.argmax(results_train)
        best_train = results_train[best_idx_train]
        best_valid = results[best_idx]
        best_test = results_test[best_idx_test]
        print(
            f"Best train: {best_train:.2f}%, "
            f"Best valid (all): {best_valid:.2f}% "
            f"Best test: {best_test:.2f}% "
            f" F1 Train: {train_f1:.2f} "
            f"F1 Test: {test_f1:.2f} "
            f"F1 All: {all_f1:.2f} "
        )

        return best_train, best_valid, best_test

    def train_net(self, epoch):
        self.model.train()
        input_dict = self.get_input_dict(epoch)
        train_loss, train_acc, val_loss = self.model.train_net(input_dict)
        return train_loss, train_acc, val_loss

    def get_input_dict(self, epoch):
        if self.type_model in [
            "GraphSAGE",
            "GraphSAGE_MLPInit",
            "GCN",
            "GCN_MLPInit",
            "GraphSAINT",
            "ClusterGCN",
            "FastGCN",
            "LADIES",
        ]:
            input_dict = {
                "x": self.x,
                "y": self.y,
                "split_masks": self.split_masks,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "loss_op": self.loss_op,
                "device": self.device,
                "data": self.data,
                "train_data":self.train_data,
                "val_data":copy.copy(self.val_data),
                "dataset": self.dataset,
            }
        elif self.type_model in ["DST-GCN", "_GraphSAINT", "GradientSampling"]:
            input_dict = {
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
                "epoch": epoch,
                "split_masks": self.split_masks,
            }
        elif self.type_model in ["LP_Adj"]:
            input_dict = {
                "split_masks": self.split_masks,
                "data": self.data,
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        elif self.type_model in [
            "SIGN",
            "SGC",
            "SAGN",
            "GAMLP",
            "GPRGNN",
            "PPRGo",
            "Bagging",
            "SAdaGCN",
            "AdaGCN",
            "AdaGCN_CandS",
            "AdaGCN_SLE",
            "EnGCN",
            "GBGCN",
        ]:
            input_dict = {
                "split_masks": self.split_masks,
                "data": self.data,
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        else:
            Exception(f"the model of {self.type_model} has not been implemented")
        return input_dict

    @torch.no_grad()
    def test_net(self):
        self.model.eval()
        train_input_dict = {"data": self.train_data, "y": self.y,"loss_op":self.loss_op,
                            "device": self.device, "dataset": self.dataset}
        test_input_dict = {"data": self.test_data, "y": self.y,"loss_op":self.loss_op,
                            "device": self.device, "dataset": self.dataset}
        all_input_dict = {"data": self.data, "y": self.y,"loss_op":self.loss_op,
                           "device": self.device, "dataset": self.dataset}

        # infer over training set
        pout(("train set"))
        train_out, loss, train_labels = self.model.inference(train_input_dict)
        # train_out.to("cpu")

        edge_index = self.train_data.edge_index
        # train_labels = self.train_data.y
        # train_labels = torch.eq(train_labels[edge_index[0]], train_labels[edge_index[1]])
        y_true_train = train_labels#.type(torch.FloatTensor)

        # infer over test set
        pout(("test set"))
        test_out, loss, test_labels = self.model.inference(test_input_dict)
        # test_out.to("cpu")
        edge_index = self.test_data.edge_index
        # test_labels = self.test_data.y
        # test_labels = torch.eq(test_labels[edge_index[0]], test_labels[edge_index[1]])
        y_true_test = test_labels#.type(torch.FloatTensor)

        # infer over entire graph
        pout(("all set"))
        all_out, loss, all_labels = self.model.inference(all_input_dict)
        # all_out.to("cpu")
        edge_index = self.dataset.data.edge_index
        # all_labels = self.dataset.data.y
        # all_labels = torch.eq(all_labels[edge_index[0]], all_labels[edge_index[1]])
        y_true_all = all_labels#.type(torch.FloatTensor)

        pout(("y true train shape", y_true_train.size))
        pout(("y true test shape", y_true_test.size))
        pout(("y true all shape", y_true_all.size))

        if self.evaluator is not None:
            y_true = self.y.unsqueeze(-1)
            y_pred = out.argmax(dim=0, keepdim=True)

            train_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["train"]],
                    "y_pred": y_pred[self.split_masks["train"]],
                }
            )["acc"]
            valid_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["valid"]],
                    "y_pred": y_pred[self.split_masks["valid"]],
                }
            )["acc"]
            test_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["test"]],
                    "y_pred": y_pred[self.split_masks["test"]],
                }
            )["acc"]
        else:

            if not self.multi_label:
                # train results threshold prediction
                threshold_out = torch.zeros_like(train_out)
                mask = train_out[:] >= self.inf_threshold
                threshold_out[mask] = 1.0
                # threshold_out[~mask] = 0.0
                train_out = threshold_out

                train_pred = train_out.to("cpu")#.argmax(dim=-1).to("cpu")
                # y_true = self.y
                train_correct = train_pred.eq(y_true_train)
                train_acc = train_correct.sum().item() / float(train_labels.size()[0])#self.split_masks["train"].sum().item()

                # test results threshold inf pred
                threshold_out = torch.zeros_like(test_out)
                mask = test_out[:] >= self.inf_threshold
                threshold_out[mask] = 1.0
                # threshold_out[~mask] = 0.0
                test_out = threshold_out

                test_pred = test_out.to("cpu") #.argmax(dim=-1).to("cpu")
                # y_true = self.y
                test_correct = test_pred.eq(y_true_test)
                test_acc = test_correct.sum().item() / float(test_labels.size()[0]) #self.split_masks["valid"].sum().item()

                # all results thresholded
                threshold_out = torch.zeros_like(all_out)
                mask = all_out[:] >= self.inf_threshold
                threshold_out[mask] = 1.0
                # threshold_out[~mask] = 0.0
                all_out = threshold_out

                all_pred = all_out.to("cpu")
                # y_true = self.y
                all_correct = all_pred.eq(y_true_all)
                all_acc = all_correct.sum().item() / float(all_labels.size()[0]) #self.split_masks["test"].sum().item()

                # Compute F1 scores
                train_f1 = (
                    f1_score(
                        y_true_train,
                        train_pred,
                        average="micro",
                    )
                    if train_pred.sum() > 0
                    else 0
                )

                test_f1 = (
                    f1_score(
                        y_true_test,
                        test_pred,
                        average="micro",
                    )
                    if test_pred.sum() > 0
                    else 0
                )

                all_f1 = (
                    f1_score(
                        y_true_all,
                        all_pred,
                        average="micro",
                    )
                    if all_pred.sum() > 0
                    else 0
                )

            # # pred = out.argmax(dim=-1).to("cpu")
                # pred = out.to("cpu")
                #
                # if isinstance(self.loss_op, torch.nn.CrossEntropyLoss) or isinstance(self.loss_op, torch.nn.BCEWithLogitsLoss) or isinstance(self.loss_op, torch.nn.NLLLoss):
                #     correct = pred.argmax(dim=-1).eq(y_true)
                # else:
                #     correct = pred.eq(y_true)
                #
                # # if isinstance(self.loss_op, torch.nn.CrossEntropyLoss):
                # #     correct = int(pred.argmax(dim=0).eq(y_true).sum())
                # # else:
                # #     correct = int(pred.eq(y_true).sum())
                #
                # # correct = pred.eq(y_true)
                # if False:
                #     train_acc = (
                #         correct[self.split_masks["edge_train"]].sum().item()
                #         / self.split_masks["edge_train"].sum().item()
                #     )
                #     valid_acc = (
                #         correct[self.split_masks["edge_valid"]].sum().item()
                #         / self.split_masks["edge_valid"].sum().item()
                #     )
                #     test_acc = (
                #         correct[self.split_masks["edge_test"]].sum().item()
                #         / self.split_masks["edge_test"].sum().item()
                #     )
                # train_acc = (
                #         correct.sum().item()
                #         / y_true.sum().item()
                # )
                # valid_acc = (
                #         correct.sum().item()
                #         / y_true.sum().item()
                # )
                # test_acc = (
                #         correct.sum().item()
                #         / y_true.sum().item()
                # )
            else:

                pred = (out > 0).float().numpy()
                # y_true = self.labels.numpy()
                # calculating F1 scores
                y_true = y_true.numpy()
                train_acc = (
                    f1_score(
                        y_true,#[self.split_masks["edge_train"]],
                        pred,#[self.split_masks["edge_train"]],
                        average="micro",
                    )
                    if pred.sum() > 0#[self.split_masks["edge_train"]].sum() > 0
                    else 0
                )

                valid_acc = (
                    f1_score(
                        y_true,#[self.split_masks["edge_valid"]],
                        pred,#[self.split_masks["edge_valid"]],
                        average="micro",
                    )
                    if pred.sum() > 0#[self.split_masks["edge_valid"]].sum() > 0
                    else 0
                )

                test_acc = (
                    f1_score(
                        y_true,
                        pred,
                        average="micro",
                    )
                    if pred.sum() > 0
                    else 0
                )

        return all_out, (train_acc, test_acc, all_acc), (train_f1, test_f1, all_f1)
