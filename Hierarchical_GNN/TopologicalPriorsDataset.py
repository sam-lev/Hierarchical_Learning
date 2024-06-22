import logging
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch

import numpy as np

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import remove_self_loops

from utils import pout



class TopologicalPriorsDataset(InMemoryDataset):




    def __init__(self, root: str, name: str, split: str = "train",
                 split_percents = [0.5,0.2,1.0],
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        self.names = ['Retinal']  # , 'NEURON', 'FOAM']
        self.name = self.names[0]
        self.root = root

        self.edge_arr = []
        self.edge_y = []
        self.node_points = []
        self.edge_points = []

        # self.split_masks = {}
        # self.split_percents = split_percents
        # self.split_masks["split_percents"] = self.split_percents
        # self.train_percent = self.split_percents[0]
        # self.val_percent = self.split_percents[1]
        # self.test_percent = self.split_percents[2]

        super().__init__(root, transform, pre_transform, pre_filter)

        path = self.processed_paths[0]



        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['feats.txt','labels.txt', 'im0236_la2_700_605.raw.mlg_nodes.txt',
                'im0236_la2_700_605.raw.mlg_geom.txt', 'im0236_la2_700_605.raw.mlg_edges.txt']#[f'{self.name}.pt']

    @property
    def processed_file_names(self):
        return [f'{name}.pt' for name in self.names]

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    def process(self):
        # if 'mlg' in self.raw_paths[0]:
        data_list = [self.process_GEOM()]
        torch.save(self.collate(data_list), self.processed_paths[0])

    def group_labels(self, lst):
        for i in range(0, len(lst), 2):
            yield list(lst[i: i + 2])

    def process_GEOM(self) -> List[Data]:

        # edge_index, points = self.read_from_geo_file()
        edge_index = self.read_from_geo_file()



        #load features
        msc_feats_file = os.path.join(self.raw_dir, "feats.txt")
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        features = []
        for v in feat_lines:
            gid_feats = v.split(' ')
            gid = int(gid_feats[0])
            feats = np.array(gid_feats[1:])
            features.append(np.array(gid_feats[1:]))
        features = np.stack(features).astype(dtype=np.float32)
        x = torch.from_numpy(features).to(torch.float)

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        #
        # Read labels
        #
        label_file = os.path.join(self.raw_dir, "labels.txt")
        label_file = open(label_file, "r")
        label_lines = label_file.readlines()
        label_file.close()
        # labels = []
        labels = self.group_labels([1., 0.] * num_nodes)
        labels = []#[list(neg_class) for neg_class in labels]



        for gid, l in enumerate(label_lines):
            tmplist = l.split(' ')
            label = [0., 1.] if int(tmplist[0]) == 1 else [1., 0.]
            labels.append(label)  # [1])
        if len(label_lines) < num_nodes:
            for i in range( num_nodes-len(label_lines)):
                labels.append([1., 0.])

        labels = np.array(labels).astype(dtype=np.float64)

        # pout(("labels shape in topo dataset gen",labels.shape, "num nodes", num_nodes, "num edges", num_edges))
        # labels = np.stack(labels, axis = 0)
        y = torch.from_numpy(labels).to(torch.float)  # .argmax(dim=-1)

        self.edge_y = np.zeros(num_edges)
        self.edge_y = torch.from_numpy(self.edge_y).to(torch.float)

        train_split = [0, int(self.train_percent * num_nodes)]
        train_mask = torch.zeros(num_nodes)  # , dtype=torch.bool)
        train_mask[:int(self.train_percent * num_nodes)] = 1.
        train_mask = train_mask.to(torch.bool)

        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        # val mask

        val_split = [int(self.train_percent * num_nodes),
                     int((self.train_percent+self.val_percent) * num_nodes)]
        val_mask = torch.zeros(num_nodes)  # , dtype=torch.bool)
        val_mask[int(self.train_percent * num_nodes):int((self.train_percent+self.val_percent) * num_nodes)] = 1.
        val_mask = val_mask.to(torch.bool)

        # test mask
        if self.test_percent == 1.0:
            test_split = [0, int(num_nodes)]
            test_mask = torch.ones(num_nodes)
            test_mask = test_mask.to(torch.bool)
        else:
            test_split = [int((self.train_percent+self.val_percent) * num_nodes), int(num_nodes)]
            test_mask = torch.zeros(num_nodes)
            test_mask[int((self.train_percent+self.val_percent) * num_nodes):] = 1.
            test_mask = test_mask.to(torch.bool)

        # self.split_masks["split_idx"] = [train_split,val_split,test_split]

        data = Data(x=x, edge_index=edge_index, y=y, train_idx=train_idx,
                    train_mask=train_mask,test_mask=test_mask,val_mask=val_mask)#points=points, y=y)

        data = data if self.pre_transform is None else self.pre_transform(data)
        # data = data if self.pre_filter is None else self.pre_filter(data)


        return data

    def _group_xy(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def _read_mlg_line(self, line):
        tmplist = line.split(" ")
        gid = int(tmplist[0])
        dim = int(tmplist[1])
        points = [
                i for i in self._group_xy([float(i) for i in tmplist[2:]])
            ] #read the rest of the the points in the arc as xy tuples
        return gid, dim, points
        # if self.dim == 1:
        #     self.centroid = get_centroid(self.points)
        #     self.vec = translate_points_by_centroid([self], self.centroid)


        #
        #
        #     NEED TO READ IN FEATURES AS WELL AND INCLUDE GID< DIM< AND POINTS INTO
        #     DATA.ATTR (E.G. FEATURES) OR AS SEP. ATTRIBUTE OF DATA OBJECT
        #
        # NEED TO READ IN GEO FILE , MAKE ADJACENCY MATRIX FROM EDGE AND NODE, READ IN POINTS
        # FROM GEOM (USE AS FEAT? OR SAVE AS DATA ATTRIBUTE (dATA OBJECT! DUH)
        # SAVE GID
        # SO HAVE DATA.GID_EDGE AND DATA.GID_NODE 
        #

    # will need minibatch for adj, getognn gor graph nxidx to feat and label
    # getofeaturegraph for loading feat
    def read_from_geo_file(self):
        fname_base = self.raw_paths[0].split('/')[-1].split('.')[0]

        pout(("Processed pathasss", self.raw_paths))

        nodesname = [f for f in self.raw_paths if "mlg_nodes" in f][0]# fname_base + ".mlg_nodes.txt"
        arcsname = [f for f in self.raw_paths if "mlg_edges" in f][0]# fname_base + ".mlg_edges.txt"
        geoname = [f for f in self.raw_paths if "mlg_geom" in f][0]#fname_base + ".mlg_geom.txt"

        geo_file = open(geoname, "r")
        geo_lines = geo_file.readlines()
        geo_file.close()

        edge_file = open(arcsname, "r")
        edge_lines = edge_file.readlines()
        edge_file.close()

        node_file = open(nodesname, "r")
        node_lines = node_file.readlines()
        node_file.close()
        self.edge_count = 0
        self.vertex_count = 0
        # getoelm_idx = 0


        row = []
        col = []

        for l in edge_lines:
            self.edge_count += 1
            tmplist = l.split(' ')
            gid_v1 = int(tmplist[1])
            gid_v2 = int(tmplist[2])
            gid_edge = int(tmplist[0])

            row.append(gid_v1)
            col.append(gid_v2)
            row.append(gid_v2)
            col.append(gid_v1)

        row = torch.from_numpy(np.array(row)).to(torch.long)
        col = torch.from_numpy(np.array(col)).to(torch.long)
        edge_index = torch.stack([row,
                                  col], dim=0)
        node_points = []
        edge_points = []
        edge_gid = []
        node_gid = []

        for l in geo_lines:
            gid, dim, pnts = self._read_mlg_line(l)
            if dim == 0:
                edge_gid.append(gid)
                edge_points.append(pnts)
            if dim == 1:
                node_gid.append(gid)
                node_points.append(pnts)

        self.node_points = node_points
        self.edge_points = edge_points

        return edge_index#, points

