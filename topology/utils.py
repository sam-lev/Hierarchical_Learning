#
# move between networkx and PyGeom dataset graphs
#
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset


# Example function to convert a NetworkX graph to PyTorch Geometric Data
def networkx_to_pyg_data(graph):
    # Get node mapping
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    # Convert edges to appropriate tensor format
    edge_list = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Optional: Add edge weights as features
    edge_weight = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges()], dtype=torch.float)

    # Create PyTorch Geometric Data
    data = Data(edge_index=edge_index, edge_attr=edge_weight)

    return data


# Example usage with one of the subgraphs
# subgraph = nx.Graph()
# subgraph.add_edge(0, 1, weight=0.5)
# subgraph.add_edge(1, 2, weight=0.7)
# subgraph.add_edge(2, 0, weight=0.2)
#
# pyg_data = networkx_to_pyg_data(subgraph)
# print(pyg_data)

# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Edge list format
# edge_attr = torch.tensor([0.5, 0.5, 0.7, 0.7], dtype=torch.float)  # Edge weights
#
# data = Data(edge_index=edge_index, edge_attr=edge_attr)
# loader = DataLoader([data], batch_size=1)


# Function to convert PyTorch Geometric data to a NetworkX graph
def pyg_batch_to_networkx(data):
    G = nx.Graph()

    # Add edges and optionally edge attributes
    edge_index = data.edge_index.cpu().numpy()
    if data.edge_attr is not None:
        edge_attr = data.edge_attr.cpu().numpy()
        for i, (u, v) in enumerate(zip(edge_index[0], edge_index[1])):
            G.add_edge(u, v, weight=edge_attr[i])
    else:
        for u, v in zip(edge_index[0], edge_index[1]):
            G.add_edge(u, v)

    return G


# Convert each batch to NetworkX and print
# for batch in loader:
#     nx_graph = pyg_to_networkx(batch)
#     print(nx_graph.edges(data=True))

def pyg_to_networkx(data):
    # Initialize a directed or undirected graph based on your need
    G = nx.DiGraph() if data.is_directed() else nx.Graph()

    # Add nodes along with node features if available
    for i in range(data.num_nodes):
        node_features = data.x[i].tolist() if data.x is not None else {}
        G.add_node(i, features=node_features)

    # Add edges along with edge attributes if available
    edge_index = data.edge_index.t().cpu().numpy()
    if data.edge_attr is not None:
        edge_attributes = data.edge_attr.cpu().numpy()
        for idx, (source, target) in enumerate(edge_index):
            G.add_edge(source, target, weight=edge_attributes[idx])
    else:
        for source, target in edge_index:
            G.add_edge(source, target)

    return G


# Example usage
# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Edges
# edge_attr = torch.tensor([0.5, 0.5, 0.7, 0.7], dtype=torch.float)  # Edge weights
# data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
#
# nx_graph = pyg_to_networkx(data)
# print(nx_graph.edges(data=True))  # Print edges with attributes
# print(nx_graph.nodes(data=True))  # Print nodes with attributes