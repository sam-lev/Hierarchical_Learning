# filtration for nested sequence of subgraphs with
# user specified persistence values for edge filtration function
# based on edge weights
import dionysus as d
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from .utils import networkx_to_pyg_data, pyg_batch_to_networkx,pyg_to_networkx
def filtration_graph_sequence(graph, threshold):
    edges = [(0, 1, 0.5), (1, 2, 0.7), (2, 0, 0.2), (0, 3, 0.9)]
    f = d.Filtration()

    for u, v, weight in edges:
        f.append(d.Simplex([u, v], weight))
    f.sort()

    def create_subgraph(f, threshold):
        G = nx.Graph()
        for simplex in f:
            if len(simplex) == 2 and simplex.data <= threshold:
                G.add_edge(simplex[0], simplex[1], weight=simplex.data)
        return G

    thresholds = [0.2, 0.5, 0.7, 0.9]
    for t in thresholds:
        g = create_subgraph(f, t)
        nx.draw(g, with_labels=True, edge_color=[g[u][v]['weight'] for u,v in g.edges()])
        plt.show()

#
# construct PyTorch Geometric dataset from
# sequence of multiple graphs
#
class GraphSequenceDataset(Dataset):
    def __init__(self, graph_list):
        super(GraphSequenceDataset, self).__init__()
        self.graphs = graph_list

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    # dataset = GraphSequenceDataset(pyg_graph_sequence)

    def networkx_to_pyg_data(self, graph):
        # Mapping nodes to contiguous integers
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        # Convert edges to tensor format
        edge_list = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Adding edge weights as features
        edge_weight = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges()], dtype=torch.float)

        # Create PyG Data object
        data = Data(edge_index=edge_index, edge_attr=edge_weight)

        return data

    #pyg_graph_sequence = [networkx_to_pyg_data(g) for g in graph_sequence]

    # # example use of dataset
    #
    # loader = DataLoader(dataset, batch_size=10, shuffle=True)
    #
    # for data in loader:
    #     print(data)
    #     # Example: Pass 'data' to your graph neural network model

# examples

# # Create dataset from a list of NetworkX graphs (assuming you have multiple subgraphs)
# graph_sequence = [create_subgraph(f, t) for t in [0.2, 0.5, 0.7, 0.9]]
# pyg_graph_sequence = [networkx_to_pyg_data(g) for g in graph_sequence]
#
# dataset = GraphSequenceDataset(pyg_graph_sequence)


