from texttable import Texttable
from torch_sparse import SparseTensor
import torch
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

def add_edge_attributes(data):
    x = data.x
    row, col = data.edge_index

    # dim edfe feature shoould be num_edges , 2 * dim_node_features
    data.edge_attr = torch.cat([x[row], x[col]], dim=-1)# np.concatenate([x[row], x[col]], axis=-1)#self.mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))

    pout(("add edge attr shape", np.shape(data.edge_index)))

    """row, col = data.edge_index
    data.edge_attr = torch.cat([data.x[row], data.x[col], data.edge_attr], dim=-1)"""

    return data

def homophily_edge_labels(data):
    labels = data.y
    edge_index = data.edge_index
    y_edge = torch.eq(labels[edge_index[0]], labels[edge_index[1]])
    return y_edge


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