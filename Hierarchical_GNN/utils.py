import os
from texttable import Texttable
from torch_sparse import SparseTensor
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

MB = 1024 ** 2
GB = 1024 ** 3

def add_edge_attributes(data):
    x = data.x
    row, col = data.edge_index

    # dim edfe feature shoould be num_edges , 2 * dim_node_features
    data.edge_attr = torch.cat([x[row], x[col]], dim=-1)# np.concatenate([x[row], x[col]], axis=-1)#self.mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
    """row, col = data.edge_index
    data.edge_attr = torch.cat([data.x[row], data.x[col], data.edge_attr], dim=-1)"""

    return data

def group_labels( lst):
    for i in range(0, len(lst), 2):
        yield list(lst[i : i + 2])

def read_label_file(label_file, raw_dir=None, x=None):

    if x is not None:
        num_nodes = x.shape[0]
    else:
        msc_feats_file = os.path.join(raw_dir, "feats.txt")
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        num_nodes = len(feat_lines)


    #
    # Read labels
    #
    label_file = os.path.join(raw_dir, "labels.txt")
    label_file = open(label_file, "r")
    label_lines = label_file.readlines()
    label_file.close()
    labels = group_labels([1.,0.]*num_nodes)
    labels = [ list(neg_class) for neg_class in labels]

    for gid, l in enumerate(label_lines):
        tmplist = l.split(' ')
        label = [0., 1.] if int(tmplist[0]) == 1 else [1., 0.]
        labels[gid] = label  # [1])
    labels = np.array(labels).astype(dtype=np.float64)
    # labels = np.stack(labels, axis = 0)
    y = torch.from_numpy(labels).to(torch.float)  # .argmax(dim=-1)
    return y
def print_args(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        # if k in ['lr', 'dst_sample_rate', 'dst_walk_length', 'dst_update_interval', 'dst_update_rate']:
        t.add_row([k, _dict[k]])
    print(t.draw())

def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1, num_reset=None):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        # resets =
        # if num_steps
        # intervals = np.flip(np.arange(len(num_reset) + 1))
        # epoch == int(total_epochs / self.graph_levels[self.graph_level])
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler

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


def pout_model(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def get_memory_usage(gpu, print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated

def edge_index_from_adjacency(adj):
    # adj_t = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    edge_index = adj.nonzero().t().contiguous()
    return edge_index

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

def draw_segmentation(self, dirpath, image, data, ridge=True, valley=True, draw_sublevel_set = False,invert=False):
    X = self.X #if not invert else self.Y
    Y = self.Y #if not invert else self.X
    original_image = image
    # black_box = np.zeros((X, Y)) if not invert else np.zeros(
    #     (Y, X))
    cmap = cm.get_cmap('bwr')
    # cmap.set_under('black')
    # cmap.set_bad('black')
    plt.set_cmap(cmap)
    # fig = plt.imshow(black_box, cmap=cmap, alpha=None, vmin=0)
    # plt.axis('off')
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)

    original_image = np.stack((original_image.astype(np.float32),) * 3, axis=-1)

    if original_image.shape[0] == 3:
        mapped_image = np.transpose(original_image, (2, 1, 0))
    elif original_image.shape[1] == 3:
        mapped_image = np.transpose(original_image, (0, 2, 1))
    else:
        mapped_image = original_image

    label_map_image = copy.deepcopy(mapped_image)

    c = 0

    # Hence, an item returned by :class:`NeighborSampler` holds the current
    # :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the computation,
    # and a list of bipartite graph objects via the tupl :obj:`(edge_index, e_id, size)`,
    # where :obj:`edge_index` represents the bipartite edges between source
    # and target nodes, :obj:`e_id` denotes the IDs of original edges in
    # the full graph, and :obj:`size` holds the shape of the bipartite graph.
    # For each bipartite graph, target nodes are also included at the beginning
    # of the list of source nodes so that one can easily apply skip-connections
    # or add self-loops.
    kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
    subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                     num_neighbors=[-1], shuffle=False,)

    # for data in self.test_loader:
    #     # data.edge_attr = self.edge_attr
    #     data = data.to(device)
    #     nodes, labels, edge_index, edge_attr = data.x, data.y, data.edge_index, data.edge_attr
    for gnode in self.gid_gnode_dict.values():
        gid = gnode.gid
        if not draw_sublevel_set:
            prediction = self.node_gid_to_prediction[gid]
            partition = self.node_gid_to_partition[gid]
            label =self.node_gid_to_label[gid]
            gnode = self.gid_gnode_dict[gid]
        else:
            prediction = [0.,1.]#self.node_gid_to_prediction[gid]
            partition = 'test'#self.node_gid_to_partition[gid]
            label = [1.,0.]
        if isinstance(prediction,
                      (int, np.integer)) or isinstance(prediction, (float, np.float)):
            label_color = cmap(float(prediction))
        else:
            if len(prediction) == 3:
                label_color = cmap(0.56) if float(prediction[2]) > 0.5 else cmap(float(prediction[1]))
            else:
                if type(prediction) == list:
                    if prediction == []:
                        #print('perd ' , prediction)
                        #print(gnode.gid)
                        #print(partition)
                        #print(self.node_gid_to_feature[gnode.gid])
                        continue
                    label_color = cmap(float(prediction[len(prediction) - 1]))
                else:
                    label_color = cmap(prediction)

        if original_image is not None:
            x = 1#1#1#1#1#  if invert else 0
            y = 0#0#0#0#0#0  if invert else 1
            scale = 255.
            if partition == 'train':
                label_color = [255, 51, 255]
                scale = 1
            if partition == 'val':
                label_color = [51, 255, 51]
                scale = 1
            if draw_sublevel_set:
                label_color = [190,0, 0]  if gnode.sublevel_set else [0,0, 255]
                scale=1
            if len(mapped_image.shape) == 2:
                for p in np.array(gnode.points):
                    if len(p) == 1:
                        continue
                    mapped_image[int(p[x]), int(p[y])] = int(label_color[0] * scale)
                    mapped_image[int(p[x]), int(p[y])] = int(label_color[1] * scale)
                    mapped_image[int(p[x]), int(p[y])] = int(label_color[2] * scale)

                    msc_ground_seg_color = cmap(float(label[1]))
                    label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[0] * 255)
                    label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[1] * 255)
                    label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[2] * 255)
            else:
                for p in np.array(gnode.points):
                    if len(p) == 1:
                        continue
                    mapped_image[int(p[x]), int(p[y]), 0] = int(label_color[0] * scale)
                    mapped_image[int(p[x]), int(p[y]), 1] = int(label_color[1] * scale)
                    mapped_image[int(p[x]), int(p[y]), 2] = int(label_color[2] * scale)

                    msc_ground_seg_color = cmap(float(label[1]))
                    label_map_image[int(p[x]), int(p[y]), 0] = int(msc_ground_seg_color[0] * 255)
                    label_map_image[int(p[x]), int(p[y]), 1] = int(msc_ground_seg_color[1] * 255)
                    label_map_image[int(p[x]), int(p[y]), 2] = int(msc_ground_seg_color[2] * 255)


    if original_image is not None:
        if len(mapped_image.shape) != 2:
            map_im = np.transpose(mapped_image, (0, 1, 2))
            lab_map_im = np.transpose(label_map_image, (0, 1, 2))
        else:
            map_im = mapped_image
            lab_map_im = label_map_image
        plt.figure()
        plt.title("Input Image")
        import matplotlib as mplt
        max_val = np.max(mapped_image)
        min_val = np.min(mapped_image)
        mapped_image = (mapped_image.astype(np.float32) - min_val) / (max_val - min_val)
        plt.imsave(os.path.join(dirpath, 'inference.png'),mapped_image.astype(np.float32))

        max_val = np.max(label_map_image)
        min_val = np.min(label_map_image)
        label_map_image = (label_map_image.astype(np.float32) - min_val) / (max_val - min_val)
        plt.imsave(os.path.join(dirpath , 'groundseg.png'),label_map_image.astype(np.float32))




        # Img = Image.fromarray(map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
        # Img.save(os.path.join(dirpath, 'inference.png'))
        #
        # Img = Image.fromarray(
        #     lab_map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
        # Img.save(os.path.join(dirpath , 'groundseg.png'))

    plt.close()