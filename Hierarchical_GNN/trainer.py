import sys

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

import torch_geometric.datasets
from torch.cuda.amp import GradScaler

# from torch_geometric.datasets import HeterophilousGraphDataset
# import sklearn.metrics.f1_score as f1
# import sklearn.metrics.roc_auc_score as roc_auc
from sklearn import metrics
import random
import numpy as np

from torch.profiler import ProfilerActivity, profile
from torch_geometric.transforms import Compose, ToSparseTensor, ToUndirected, RandomLinkSplit, AddSelfLoops

from GraphSampling import *
from GraphSampling.HierGNN import HierJGNN, HierSGNN
# from LP.LP_Adj import LabelPropagation_Adj
# from Precomputing import *
from GraphSampling.MLPInit import *
from utils import pout_model
# from GraphSampling.utils import init_edge_embedding

from TopologicalPriorsDataset import *

from torch.optim.lr_scheduler import ReduceLROnPlateau
#memory profiling tools
# from memory_profiler import profile
# from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score

from GraphSampling.experiments.metrics import optimal_metric_threshold


def get_splits(data, split_percents = [0.3, .3, 1.0]):
    """
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
    split_masks["edge_test"] = data.edge_test_mask"""
    return data, []

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


def load_data(dataset_name, datasubset_name,
              to_sparse=True, homophily=0.1):

    pout(("Loading Data: " + dataset_name, " with datasubset: "+datasubset_name ))

    if dataset_name in ["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]:
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        T = ToSparseTensor() if to_sparse else lambda x: x
        if to_sparse and dataset_name == "ogbn-arxiv":
            T = lambda x: ToSparseTensor()(ToUndirected()(x))
        dataset = PygNodePropPredDataset(name=datasubset_name, root=root, transform=T)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=dataset_name)
        data = dataset[0]
        # split_masks = {}
        # for split in ["train", "valid", "test"]:
        #     mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        #     mask[split_idx[split]] = True
        #     data[f"{split}_mask"] = mask
        #     split_masks[f"{split}"] = data[f"{split}_mask"]

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
        # split_masks = {}
        # split_masks["train"] = data.train_mask
        # split_masks["valid"] = data.val_mask
        # split_masks["test"] = data.test_mask
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
        transform_sparse = ToSparseTensor() if to_sparse else lambda x: x

        transform = Compose([ AddSelfLoops(), ToUndirected(), transform_sparse ])
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

        dataset = dataset_class(path,
                                name=datasubset_name,
                                transform=transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.
        processed_dir = dataset.processed_dir
        data = dataset[0]
        # train_data, val_data, test_data = dataset[0]
        evaluator = None
        # split_masks = {}
        # split_masks["train"] = data.train_mask
        # split_masks["valid"] = data.val_mask
        # split_masks["test"] = data.test_mask
        x = data.x
        y = data.y
    elif dataset_name in ["WebKB"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        transform_sparse = ToSparseTensor() if to_sparse else lambda x: x

        transform = Compose([ AddSelfLoops(), ToUndirected(), transform_sparse ])
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
        dataset = dataset_class(path,name=datasubset_name, transform=transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.
        processed_dir = dataset.processed_dir
        data = dataset[0]
        # train_data, val_data, test_data = dataset[0]
        evaluator = None

        # split_masks = get_splits(data)
        # data.train_mask = split_masks['train']
        # data.val_mask = split_masks['valid']
        # data.test_mask = split_masks['test']
        # dataset.data = data
        x = data.x
        y = data.y
    elif dataset_name in ["WikipediaNetwork"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        transform_sparse = ToSparseTensor() if to_sparse else lambda x: x

        transform = Compose([ AddSelfLoops(),  ToUndirected(), transform_sparse ])
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
        dataset = dataset_class(path,name=datasubset_name, transform=transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.
        processed_dir = dataset.processed_dir
        data = dataset[0]
        # train_data, val_data, test_data = dataset[0]
        evaluator = None

        # split_masks = get_splits(data)
        # data.train_mask = split_masks['train']
        # data.val_mask = split_masks['valid']
        # data.test_mask = split_masks['test']
        # dataset.data = data
        x = data.x
        y = data.y
    elif dataset_name in ["MixHopSyntheticDataset"]:
        # pout(("homophily of graph", homophily))
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        transform_sparse = ToSparseTensor() if to_sparse else lambda x: x

        transform = Compose([ AddSelfLoops(), ToUndirected(), transform_sparse ])
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




        # data, split_masks = get_splits(data)
        # data.train_mask = split_masks['train']
        # data.val_mask = split_masks['valid']
        # data.test_mask = split_masks['test']
        # dataset.data = data
        x = data.x
        y = data.y


        # pout(("dimension of training data", x.shape, "dimension edge index", data.edge_index.shape))

    elif dataset_name in ["HeterophilousGraphDataset"]:
        pout(("USING HETEROPHILOUSGRAPHDATASET"))
        # pout(("homophily of graph", homophily))
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        transform_sparse = ToSparseTensor() if to_sparse else lambda x: x

        transform = Compose([ AddSelfLoops(), ToUndirected(), transform_sparse ])
        # Transform.Compose([
        #     Transform.NormalizeFeatures(),
        #     # Transform.ToDevice(device),
        #     Transform.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                       add_negative_train_samples=False),
        # ])



        # self.name in [
        #     'roman_empire',
        #     'amazon_ratings',
        #     'minesweeper',
        #     'tolokers',
        #     'questions',
        # ]
        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset =  dataset_class(path,
                                 name=datasubset_name, # questions',
                                 transform = transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.
        processed_dir = dataset.processed_dir
        data = dataset[0]
        # train_data, val_data, test_data = dataset[0]
        evaluator = None




        # data, split_masks = get_splits(data)
        # data.train_mask = split_masks['train']
        # data.val_mask = split_masks['valid']
        # data.test_mask = split_masks['test']
        # dataset.data = data
        x = data.x
        y = data.y

        edge_weights = [(edge[0], edge[1], 0) for edge in data.edge_index]
        data.edge_weights = edge_weights

        # pout(("dimension of training data", x.shape, "dimension edge index", data.edge_index.shape))



    elif "TopologicalPriorsDataset" in dataset_name:
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )

        split_percents = [0.3,0.2,0.5]
        dataset = TopologicalPriorsDataset(name=dataset_name, root=root, split_percents=split_percents)
        data = dataset(0)
        x = data.x
        y = data.y

        # pout((y))

        edge_index  = data.edge_index
        processed_dir = dataset.processed_dir

        evaluator = None





        # data, split_masks = get_splits(data)
        # data.train_mask = split_masks['train']
        # data.val_mask = split_masks['valid']
        # data.test_mask = split_masks['test']
        # dataset.data = data
        x = data.x
        y = data.y

        # pout(("Geometric points of nodes", dataset.node_points))


    else:
        raise Exception(f"the dataset of {dataset_name} has not been implemented")

    # all_nodes = data.x
    # row, col = data.edge_index
    # data.edge_attr = torch.cat([all_nodes[row], all_nodes[col]], dim=-1)
    # # dataset.data = data

    return data, x, y, evaluator, processed_dir, dataset

def update_dataset(data, dataset):
    # data, split_masks = get_splits(data)
    # data.train_mask = split_masks['train']
    # data.val_mask = split_masks['valid']
    # data.test_mask = split_masks['test']
    x = data.x
    y = data.y
    # dataset.data = data
    return data, x, y, None, None, dataset
def idx2mask(idx, N_nodes):
    mask = torch.tensor([False] * N_nodes, device=idx.device)
    mask[idx] = True
    return mask


class trainer(object):
    def __init__(self, args, seed):


        print(" MANUALLY SETTING SEED AT START OF TRAINER"," NEW SEED :")

        args.random_seed = seed
        print(f"seed (which_run) = <{seed}>")

        self.set_seed(args)
        pout(("ARGS MULTILABEL", args.multi_label))
        self.dataset = args.dataset
        self.device = torch.device(f"cuda:{args.cuda_num}" if args.cuda else "cpu")
        self.args = args
        self.args.device = self.device

        # self.seed = seed

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.eval_steps = args.eval_steps
        self.train_by_steps = args.train_by_steps
        self.steps = -1


        """# used to indicate multi-label classification.
        # If it is, using BCE and micro-f1 performance metric
        self.multi_label = args.multi_label
        if self.multi_label:
            self.loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_op = torch.nn.BCEWithLogitsLoss()#torch.nn.NLLLoss() #torch.nn.CrossEntropyLoss()#
        """

        # threshold for class assignment from inferred logit value
        self.inf_threshold = 0.5
        args.inf_threshold = self.inf_threshold
        (
            self.data,
            self.x,
            self.y,
            # self.split_masks,
            self.evaluator,
            self.processed_dir,
            self.dataset,
        ) = load_data(args.dataset, datasubset_name=args.data_subset,
                      to_sparse=args.tosparse, homophily=args.homophily)

        #
        #            NOTE GLOBAL CHANGE TO DATA FOR EDGE FEAT
        #
        # self.data = add_edge_attributes(self.data)
        labels = self.y
        labels = torch.eq(labels[self.data.edge_index[0]], labels[self.data.edge_index[1]])





        # Performs an edge-level random split into training, validation and test sets of a Data
        # or a HeteroData object (functional name: random_link_split).
        # The split is performed such that the training split does not include edges
        # in validation and test splits; and the validation split does not include edges in the test split.
        edge_train_transform = RandomLinkSplit(is_undirected=True,
                                               num_val=0.2,
                                               num_test=0.2,
                                               add_negative_train_samples=False)
        self.train_data, self.val_data, self.test_data = edge_train_transform(self.data)

        # pout(("dataset nodes points", self.dataset.node_points))

        if self.type_model in ["EdgeMLP", "GCN_MLPInit"]:# and self.type_model in ["MLPInit"]:
            #
            #  check if multicloss to adjust metrics and loss
            self.multi_label = False
            self.num_targets = 1
            if self.num_targets == 1:
                labels = labels.float()
                self.y = self.y.float()
                self.data.y = self.data.y.float()
                # self.dataset.data.y = self.dataset.data.y.float()
            args.dataset = self.dataset

            self.train_data, self.val_data, self.test_data = edge_train_transform(self.data)

            self.loss_op = F.binary_cross_entropy_with_logits #if self.num_targets == 1 else F.cross_entropy

            self.metric = 'ROC AUC' if self.num_targets == 1 else 'accuracy'

            self.model = EdgeMLP(
                args, self.data, self.processed_dir,
                train_data=self.train_data, test_data=self.test_data
            )
        # initialize and learn edge embeddings and predicted weights for
        # topological filtration and hierarchical learning
        elif self.type_model in ["HierGNN"]:# and self.type_model in ["MLPInit"]:


            #
            # learn and infer edge embeddings and update dataset
            # binary classification for edge inference
            #
            ( args,
              self.data,
              self.x,
              self.y,
              # self.split_masks,
              self.dataset
              ) = self.learn_edge_embeddings(args)

            pout((" multilabel before checking for node inference", self.multi_label,
                  " number of targets before ", self.num_targets))
            #
            #  check if multicloss to adjust metrics and loss
            #  potentially no longer binaryt classification for node inference
            #
            self.multi_label = len(self.y.unique())
            args.multi_label = self.multi_label
            self.num_targets = 1 if self.multi_label == 2 else self.multi_label
            self.num_classes = self.num_targets
            args.num_classes = self.num_targets
            args.multi_label = False if self.num_targets == 1 else True
            self.multi_label = args.multi_label
            if self.num_targets == 1:
                self.y = self.y.float()
                self.data.y = self.data.y.float()
                # self.dataset.data.y = self.dataset.data.y.float()
            else:
                self.y = self.y.long()#type(torch.LongTensor)
                self.data.y = self.data.y.long()#type(torch.LongTensor)
                # self.dataset.data.y = self.dataset.data.y.long()#type(torch.LongTensor)
            args.dataset = self.dataset

            self.train_data, self.val_data, self.test_data = edge_train_transform(self.data)

            pout((" multilabel after checking for node inference", self.multi_label,
                  " number of targets after ", self.num_targets))

            self.loss_op = F.binary_cross_entropy_with_logits if self.num_targets == 1 else F.cross_entropy

            self.metric = 'ROC AUC' if self.num_targets == 1 else 'accuracy'
            # then perform hierarchical learning
            #
            pout(("%%%%%%%%%%", "BEGINNING HIERARCHICAL GNN FOR NODE CLASSIFICATION","%%%%%%%%%%"))

            self.type_model = self.type_model + "_" + args.hier_model
            args.type_model = self.type_model

            pout(("USING MODEL:"))
            pout((self.type_model))

            #
            # Instantiate the Hierarchical GNN Model
            #
            if args.hier_model == "HST":
                self.model = HierSGNN(
                    args, self.data, self.processed_dir,out_dim = self.num_targets,
                    train_data=self.train_data, test_data=self.test_data
                )
            elif args.hier_model == "HJT":
                self.model = HierJGNN(
                    args=args, data=self.data, processed_dir=self.processed_dir,
                    train_data=self.train_data, filtration_function=None
                    #, out_dim, dim_hidden, in_channels taken in model
                )
            else:
                raise NotImplementedError(f"{args.hier_model} not implemented. Please specify valid hierarchical gnn `hier_model`")
        else:
            raise NotImplementedError("please specify `type_model`")


        self.model.to(self.device)
        old_implementation = """
        self.optimizer =  torch.optim.SGD(#)
                self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=0.9
            )

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                           factor=0.1, patience=3, threshold=1e-4)
        """
        #
        # used by gnn critical eval
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
        # self.scheduler = get_lr_scheduler_with_warmup(optimizer=self.optimizer, num_warmup_steps=args.num_warmup_steps,
        #                                          num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)
        self.scheduler = None #ReduceLROnPlateau(self.optimizer, mode='min',factor=0.1, patience=3, threshold=1e-4)
        self.grad_scalar = GradScaler(enabled=args.amp)
        #
        #
        if "MLPInit" in self.type_model:# in ["GraphSAGE_MLPInit"]:
            self.mlp_embedding_initialization(args, seed)




        # pout(("WHY DOES MULTILABEL CHANGE", "self", self.multi_label))

    def learn_edge_embeddings(self, args):
        # first train and infer over edge embeddings to update graph
        # for filtration
        seed = args.random_seed

        self.multi_label = False
        self.num_targets = 1
        if self.num_targets == 1:
            self.y = self.y.float()
            self.data.y = self.data.y.float()
            # self.dataset.data.y = self.dataset.data.y.float()

        self.loss_op = F.binary_cross_entropy_with_logits

        pout(("%%%", "LEARNING AND CLASSIFYING EDGE EMBEDDINGS", "..."))
        self.model = EdgeMLP(
            args, self.data,self.processed_dir,
            train_data=self.train_data, test_data=self.test_data
        )

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=args.lr2,
                                           weight_decay=args.weight_decay)
        # self.scheduler = get_lr_scheduler_with_warmup(optimizer=self.optimizer, num_warmup_steps=args.num_warmup_steps,
        #                                               num_steps=args.num_steps,
        #                                               warmup_proportion=args.warmup_proportion)
        self.grad_scalar = GradScaler(enabled=args.amp)
        # self.optimizer = torch.optim.SGD(  # )
        #     self.model.parameters(), lr=args.lr,
        #     weight_decay=args.weight_decay, momentum=0.9
        # )
        self.scheduler = None #ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4)

        self.train_and_test(seed)

        input_dict = self.get_input_dict(0)
        self.test_net()

        pred, loss, ground_truth, self.data = self.model.edge_inference(input_dict,
                                                                        assign_edge_weights=True)

        pout(("%%%%%%%", "FINSISHED INFERING OVER EDGES", "%%%%%%%"))
        pout(("updating dataset"))
        (
            self.data,
            self.x,
            self.y,
            # self.split_masks,
            self.evaluator,
            self.processed_dir,
            self.dataset,
        ) = update_dataset(self.data, self.dataset)
        args.dataset = self.dataset
        args.data = self.data
        # From a crit look at evaluating gnns benchmark paper:
        # ese datasets, we use the same splits with duplicates removed.
        # For eac ts, we fix 10 random 50%/25%/25% train/validation/test splits. We tr
        # edge_train_transform = RandomLinkSplit(is_undirected=False,
        #                                        num_val=0.2,
        #                                        num_test=0.2,
        #                                        add_negative_train_samples=False)
        # self.train_data, self.val_data, self.test_data = edge_train_transform(self.data)

        torch.cuda.empty_cache()
        del self.model

        return args, self.data, self.x, self.y,  self.dataset

    def mlp_embedding_initialization(self,args, seed):
        gnn_model = self.model
        pout("gnn model param")
        pout(self.model)

        input_dict = self.get_input_dict(0)
        mlp_init_model = MLPInit(
            args,
            self.data,
            self.processed_dir,
            # split_masks=self.split_masks,
            input_dict=input_dict,
            evaluator=self.evaluator,
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
        f1s_test = []
        f1s_train = []
        f1s_all = []
        roc_train = []
        roc_test = []
        # f1_scores = []
        for epoch in range(self.epochs):

            if self.train_by_steps:
                if epoch == 0:
                    self.model.steps = 0
                    self.steps = 0
            else:
                self.steps = -1

            train_loss, train_acc, val_loss = self.train_net(epoch)  # -wz-run

            if self.train_by_steps:

                self.steps += self.model.steps
                counter = self.steps
            else:
                counter = epoch
                self.steps = -1
            # if self.type_model  in ["HierGNN"]:
            #     self.trian_data = self.model.get_train_data()
            seed = int(seed)
            if val_loss != 666:
                print(
                    f"Seed: {seed:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Approx Train Acc: {train_acc:.4f}"
                )
            else:
                print(
                    f"Seed: {seed:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {train_loss:.4f}, "
                    #f"Val Loss: {val_loss:.4f}, "
                    f"Approx Train Acc: {train_acc:.4f}"
                )
            #
            # need to adapt test during training eval to be
            # only evaluation set and entire graph opnly for near last epocjs
            #

            # if self.train_by_steps:
            #     perform_eval = counter % self.eval_steps < self.eval_steps
            # else:
            #     perform_eval = counter % self.eval_steps == 0

            if epoch % self.eval_steps == 0 and epoch != 0:
                out, result, f1s, roc = self.test_net()
                # tr_f1, te_f1, a_f1 = f1s
                f1s_train.append(f1s[0])
                f1s_test.append(f1s[1])
                f1s_all.append(f1s[2])
                results.append(result[2])
                results_test.append(result[1])
                results_train.append(result[0])
                roc_train.append(roc[0])
                roc_test.append(roc[1])
                train_acc, valid_acc, test_acc = result
                if self.multi_label:
                    print(
                        f"Epoch: {epoch:02d}, "
                        # f"Loss: {train_loss:.4f}, "
                        # f"Train: {100 * train_acc:.2f}%, "
                        # f"Valid: {100 * valid_acc:.2f}% "
                        f"Test Acc: {100 * result[1]:.2f}% "
                        # f"Train RoC: {roc[0]:.2f}"
                        # f"Test RoC: {roc[1]:.2f}"
                    )
                else:
                    print(
                        f"Epoch: {epoch:02d}, "
                        # f"Loss: {train_loss:.4f}, "
                        # f"Train: {100 * train_acc:.2f}%, "
                        # f"Valid: {100 * valid_acc:.2f}% "
                        f"Test Acc: {100 * result[1]:.2f}% "
                        # f"Train RoC: {roc[0]:.2f}"
                        f"Test RoC: {roc[1]:.2f}"
                    )
        train_f1, test_f1, all_f1 = (f1s_train[np.argmax(f1s_train)],
                                     f1s_test[np.argmax(f1s_test)],
                                     f1s_all[np.argmax(f1s_all)])
        results = 100 * np.array(results)
        results_train = 100 * np.array(results_train)
        results_test = 100 * np.array(results_test)
        best_idx = np.argmax(results)
        best_idx_test = np.argmax(results_test)
        best_idx_train = np.argmax(results_train)
        best_train = results_train[best_idx_train]
        best_valid = results[best_idx]
        best_test = results_test[best_idx_test]
        best_idx_train_roc = np.argmax(roc_train)
        best_roc_train = roc_train[best_idx_train_roc]
        best_idx_test_roc = np.argmax(roc_test)
        best_roc_test = roc_train[best_idx_test_roc]

        print(
            # f"Best train: {best_train:.2f}%, "
            #f"Best valid (all): {best_valid:.2f}% "
            f"Best test: {best_test:.2f}% "
            #f"F1 Train: {train_f1:.2f} "
            f"F1 Test: {test_f1:.2f} "
            #f"F1 All: {all_f1:.2f} "
            #f"Best ROC Train: {best_roc_train:.2f} "
            f"Best ROC Test: {best_roc_test:.2f} "
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
            "EdgeMLP",
            "GCN_MLPInit",
            "HierGNN",
            "HierGNN_HST",
            "HierGNN_HJT",
            "GraphSAINT",
            "ClusterGCN",
            "FastGCN",
            "LADIES",
        ]:
            input_dict = {
                "x": self.x,
                "y": self.y,
                # "split_masks": self.split_masks,
                "optimizer": self.optimizer,
                "grad_scalar": self.grad_scalar,
                "scheduler": self.scheduler,
                "loss_op": self.loss_op,
                "device": self.device,
                "data": self.data,
                "train_data":self.train_data,
                "val_data":self.val_data,
                "dataset": self.dataset,
                "epoch": epoch,
                "eval_steps":self.eval_steps,
                "total_epochs":self.epochs,
                "steps": self.steps
            }
        elif self.type_model in ["DST-GCN", "_GraphSAINT", "GradientSampling"]:
            input_dict = {
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
                "epoch": epoch,
                # "split_masks": self.split_masks,
            }
        elif self.type_model in ["LP_Adj"]:
            input_dict = {
                # "split_masks": self.split_masks,
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
                # "split_masks": self.split_masks,
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
        # pout(("In test net", "multilabel?", self.multi_label))
        self.model.eval()
        """train_input_dict = {"data": self.train_data, "y": self.y,"loss_op":self.loss_op,
                            "device": self.device, "dataset": self.dataset}"""
        test_input_dict = {"data": self.test_data, "y": self.y,"loss_op":self.loss_op,
                            "device": self.device, "dataset": self.dataset}
        all_input_dict = {"data": self.data, "y": self.y,"loss_op":self.loss_op,
                           "device": self.device, "dataset": self.dataset}

        # infer over training set
        # pout(("train set"))
        """train_out, loss, y_train = self.model.inference(train_input_dict)"""
        test_out, loss, y_test = self.model.inference(test_input_dict)
        """all_out, loss, y_all = self.model.inference(all_input_dict)"""
        if self.evaluator is not None:

            """pout(( "%%%%%%%%%%%%%%", "Testing with Evaluator ", "%%%%%%%%%%%"))
            y_true = self.y.unsqueeze(-1)
            y_pred = all_out.argmax(dim=0, keepdim=True)

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
            )["acc"]"""
        else:



            if self.multi_label == True:

                """train_thresh, train_acc = optimal_metric_threshold(train_out,
                                                                   y_train,
                                                                   metric=metrics.accuracy_score,
                                                               metric_name='accuracy',
                                                               num_targets=self.num_targets)"""
                # test_pred= test_out.argmax(axis=1)
                # test_acc = (test_out == y_test).float().mean().item()
                test_thresh, test_acc = optimal_metric_threshold(test_out,
                                                                 y_test,
                                                                   metric=metrics.accuracy_score,
                                                               metric_name='accuracy',
                                                               num_targets=self.num_targets)

                """all_thresh, all_acc = optimal_metric_threshold(all_out,
                                                               y_all,
                                                                   metric=metrics.accuracy_score,
                                                               metric_name='accuracy',
                                                               num_targets=self.num_targets)
                
                
                """
                test_f1 = 0
                test_roc = 0


                # total_correct = 0
                # correct = pred.eq(data.y).sum().item()
                # accuracy = correct / (num_nodes * num_classes)
                # total_correct = train_out.argmax(dim=-1).eq(y_train).sum().item()
                # train_acc = total_correct/float(y_train.size(0)*self.num_targets)
                #
                #
                # total_correct = 0
                # total_correct = test_out.argmax(dim=-1).eq(y_test).sum().item()
                # test_acc = total_correct / float(y_test.size(0) * self.num_targets)
                #
                # total_correct = 0
                # # for y_t, y_s in zip(y_all, all_out):
                # total_correct = all_out.argmax(dim=-1).eq(y_all).sum().item()
                # all_acc = total_correct / float(y_all.size(0) * self.num_targets)

                # print(" ALL ACCURACY thresh ", all_acc)
                # all_acc_mean = (all_out == y_all).float().mean().item()
                # pout((" ALL ACCURACY Mean ", all_acc_mean))

                # train_thresh, train_acc = optimal_metric_threshold(y_score_train,
                #                                                    y_true_train,
                #                                                    metric='accuracy')
                # test_thresh, test_acc = optimal_metric_threshold(y_score_test,
                #                                                  y_true_test,
                #                                                  metric='accuracy')
                # all_thresh, all_acc = optimal_metric_threshold(y_score_all,
                #                                                y_true_all,
                #                                                metric='accuracy')

                # train_thresh, train_f1 = optimal_metric_threshold((train_out > 0).float().numpy(),
                #                                                   y_true_train,
                #                                                   metric=metrics.f1_score)
                # test_thresh, test_f1 = optimal_metric_threshold((test_out > 0).float().numpy(),
                #                                                 y_true_test,
                #                                                 metric=metrics.f1_score)
                # all_thresh, all_f1 = optimal_metric_threshold((all_out > 0).float().numpy(),
                #                                               y_true_all,
                #                                               metric=metrics.f1_score)

                """
                            cCHANGER THIAS AHCKL HACL HACL
                """



            if self.multi_label == False:

                """y_true_train = y_train.cpu().numpy()
                y_score_train = train_out.cpu().numpy()"""

                y_true_test = y_test.cpu().numpy()
                y_score_test = test_out.cpu().numpy()

                """y_true_all = y_all.cpu().numpy()
                y_score_all = all_out.cpu().numpy()"""

                """train_thresh, train_acc = optimal_metric_threshold(y_score_train,
                                                                   y_true_train,
                                                               metric = accuracy_score,
                                                                   metric_name='accuracy')"""
                test_thresh, test_acc = optimal_metric_threshold(y_score_test,
                                                                 y_true_test,
                                                               metric = accuracy_score,
                                                                 metric_name='accuracy')
                """all_thresh, all_acc = optimal_metric_threshold(y_score_all,
                                                               y_true_all,
                                                               metric = accuracy_score,
                                                               metric_name='accuracy')
                """
                """train_thresh, train_f1 = optimal_metric_threshold(y_score_train,
                                                                        y_true_train,
                                                                        metric=metrics.f1_score)"""
                test_thresh, test_f1 = optimal_metric_threshold(y_score_test,
                                                                       y_true_test,
                                                                       metric=metrics.f1_score)
                """all_thresh, all_f1 = optimal_metric_threshold(y_score_all,
                                                                       y_true_all,
                                                                       metric=metrics.f1_score)
                
                all_thresh, all_roc = optimal_metric_threshold(y_score_all,
                                                                   y_true_all,
                                                                   metric=metrics.roc_auc_score,
                                                                      metric_name='ROC AUC')
                """
                all_thresh, test_roc = optimal_metric_threshold(y_score_test,
                                                                   y_true_test,
                                                                   metric=metrics.roc_auc_score,
                                                                      metric_name='ROC AUC')
                """all_thresh, train_roc = optimal_metric_threshold(y_score_train,
                                                                   y_true_train,
                                                                   metric=metrics.roc_auc_score,
                                                                      metric_name='ROC AUC')"""
                #print("ALL ACC Thresh: ",all_acc, " ALL ROC: ", all_roc )


            """
            # train results threshold prediction
            train_pred = train_out > 0.5
            train_acc = (train_pred.long() == y_train).float().mean().item()
            print((train_acc))
            # threshold_out = train_out.argmax(dim=-1).to("cpu")
            # train_pred = threshold_out.to("cpu")#.argmax(dim=-1).to("cpu")
            # y_true = self.y
            # train_correct = train_pred.eq(y_true_train)#.argmax(dim=-1)
            # train_acc = train_correct.sum().item() / float(train_labels.size()[0])#self.split_masks["train"].sum().item()

            # test_pred = threshold_out.to("cpu") #.argmax(dim=-1).to("cpu")
            # # y_true = self.y
            # test_correct = test_pred.eq(y_true_test.argmax(dim=-1).to("cpu"))
            # test_acc = test_correct.sum().item() / float(test_labels.size()[0]) #self.split_masks["valid"].sum().item()
            test_pred = test_out > 0.5
            test_acc = (test_pred.long() == y_test).float().mean().item()

            # threshold_out = all_out.argmax(dim=-1).to("cpu")
            # all_pred = threshold_out.to("cpu")
            # # y_true = self.y
            # all_correct = all_pred.eq(y_true_all.argmax(dim=-1).to("cpu"))
            # all_acc = all_correct.sum().item() / float(all_labels.size()[0]) #self.split_masks["test"].sum().item()

            all_pred = all_out > 0.5
            all_acc = (all_pred.long() == y_all).float().mean().item()

            pout((y_train, train_out))
            # Compute F1 scores
            train_f1 = (
                metrics.f1_score(
                    y_train.to("cpu"),
                    train_pred.to("cpu"),#.to("cpu"),
                    average="micro",
                )
                # if train_pred.sum() > 0
                # else 0
            )

            test_f1 = (
                metrics.f1_score(
                    y_test.to("cpu"),
                    test_pred.to("cpu"),
                    average="micro",
                )
                # if test_pred.sum() > 0
                # else 0
            )

            all_f1 = (
                metrics.f1_score(
                    y_all.to("cpu"),
                    all_pred.to("cpu"),
                    average="micro",
                )
                # if all_pred.sum() > 0
                # else 0
            )
            """


            # all_roc = metrics.roc_auc_score(y_true_all.argmax(dim=1).cpu().numpy(),
            #         all_out.argmax(dim=1).cpu().numpy())#.argmax(dim=-1).to("cpu"),)

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
            # else:
            #     pout(("%%%%%%%%%%%", "Testing inference results",
            #           "multi_Label: ", self.multi_label, "%%%%%%%%%%"))
            #     pred = (all_out > 0).float().numpy()
            #     # y_true = self.labels.numpy()
            #     # calculating F1 scores
            #     y_true = y_all.numpy()
            #     train_acc = (
            #         f1_score(
            #             y_true,#[self.split_masks["edge_train"]],
            #             pred,#[self.split_masks["edge_train"]],
            #             average="micro",
            #         )
            #         if pred.sum() > 0#[self.split_masks["edge_train"]].sum() > 0
            #         else 0
            #     )
            #
            #     valid_acc = (
            #         f1_score(
            #             y_true,#[self.split_masks["edge_valid"]],
            #             pred,#[self.split_masks["edge_valid"]],
            #             average="micro",
            #         )
            #         if pred.sum() > 0#[self.split_masks["edge_valid"]].sum() > 0
            #         else 0
            #     )
            #
            #     test_acc = (
            #         f1_score(
            #             y_true,
            #             pred,
            #             average="micro",
            #         )
            #         if pred.sum() > 0
            #         else 0
            #     )
        train_roc = 0
        all_roc = 0
        train_f1 = 0
        all_f1 = 0
        all_out, loss, y_all = 0,0,0
        all_acc = 0
        train_acc = 0
        return (all_out, (train_acc, test_acc, all_acc),
                (train_f1, test_f1, all_f1), (train_roc,test_roc,all_roc))
    def compute_metrics(self, logits):
        self.model.eval()
        train_input_dict = {"data": self.train_data, "y": self.y, "loss_op": self.loss_op,
                            "device": self.device, "dataset": self.dataset}
        test_input_dict = {"data": self.test_data, "y": self.y, "loss_op": self.loss_op,
                           "device": self.device, "dataset": self.dataset}
        all_input_dict = {"data": self.data, "y": self.y, "loss_op": self.loss_op,
                          "device": self.device, "dataset": self.dataset}

        # infer over training set
        # pout(("train set"))
        train_out, loss, y_train = self.model.inference(train_input_dict)
        test_out, loss, y_test = self.model.inference(test_input_dict)
        all_out, loss, y_all = self.model.inference(all_input_dict)
        logits = train_out
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    def set_seed(self, args):


        #if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
        if args.cuda:
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
            # torch.cuda.memory.set_per_process_memory_fraction(0.99, device=0)
            # torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)