from texttable import Texttable
# from torch_sparse import SparseTensor
import torch
from torch import Tensor

from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import coalesce, degree
import torch.nn.functional as F
import numpy as np

MB = 1024 ** 2
GB = 1024 ** 3

def print_args(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        # if k in ['lr', 'dst_sample_rate', 'dst_walk_length', 'dst_update_interval', 'dst_update_rate']:
        t.add_row([k, _dict[k]])
    print(t.draw())

def init_edge_embedding(data):

    x = data.x#.detach().cpu().numpy()
    adj_n1, adj_n2 = data.edge_index
    x1 = x[adj_n1]
    x2 = x[adj_n2]
    # for heterophily meodels that concatenate ego- and neighborhood-
    # embeddings perform better
    # dim edfe feature shoould be num_edges , 2 * dim_node_features
    concat_adj_features = torch.cat([x1,x2], dim=-1)
    data.edge_attr = concat_adj_features
    # data.edge_attr = torch.pow(x[adj_n1] - x[adj_n2], exponent=2)#

    # cossim = F.cosine_similarity(x1,x2,dim=1)
    # # pout(("cossim shape", cossim.shape, "cossim unsqueeze ", cossim.unsqueeze(1).shape,
    # #       "x shape ", x.shape))
    # duplicated_cosim = cossim.unsqueeze(1)#.repeat(x1.shape[-1])
    # combined_node_features = torch.add(x1,x2)
    # scale_factor = 10
    # edge_features = torch.add(combined_node_features,duplicated_cosim,alpha=scale_factor)
    # data.edge_attr = edge_features

    # data.edge_attr = torch.cdist(x[adj_n1], x[adj_n2], p=1.0,
    #                              compute_mode='donot_use_mm_for_euclid_dist')
    """row, col = data.edge_index
    data.edge_attr = torch.cat([data.x[row], data.x[col], data.edge_attr], dim=-1)"""

    return data

def homophily_edge_labels(data):
    labels = data.y
    edge_index = data.edge_index
    y_edge = torch.eq(labels[edge_index[0]], labels[edge_index[1]])
    return y_edge

def node_degree_statistics(data):
    # Compute the degree of each node
    node_degrees = degree(data.edge_index[0], data.num_nodes)

    # Get the maximum node degree
    max_degree = node_degrees.max().item()
    avg_degree = node_degrees.mean().item()

    return max_degree, avg_degree, node_degrees

def get_memory_usage(gpu, print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.int64, torch.long]:
            ret += np.prod(x.size()) * 8
        if x.dtype in [torch.float32, torch.int, torch.int32]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size())
        else:
            print(x.dtype)
            raise ValueError()
    return ret

def pout(show=None):
    if isinstance(show, list) or isinstance(show, tuple):
        print("    *")
        for elm in show:
            if isinstance(elm, str):
                print("    * ",elm)
            else:
                print("    * ", str(elm))
        print("    *")
    else:
        print("    *")
        if isinstance(show, str):
            print("    * ", show)
        else:
            print("    * ", str(show))
        print("    *")


def edge_index_from_adjacency(adj):
    # adj_t = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    edge_index = adj.nonzero().t().contiguous()
    return edge_index