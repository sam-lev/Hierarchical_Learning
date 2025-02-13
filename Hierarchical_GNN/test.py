import torch
from torch_geometric.data import Data
import numpy as np
# Sample graph data
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# Sample node features
x = torch.tensor([[1], [2], [3]], dtype=torch.float)

# Creating a Data object
data = Data(x=x, edge_index=edge_index)
print(data)
print("after first")
# Dictionary with new node attributes_
node_attr_dict = {
    0: [0.1, 0.2],
    1: [0.3, 0.4],
    2: [0.5, 0.6]
}

# Convert node attribute dictionary to tensor
node_attr_tensor = torch.tensor([node_attr_dict[i] for i in range(len(node_attr_dict))], dtype=torch.float)#torch.tensor([node_attr for node, node_attr in node_attr_dict.items()], dtype=torch.float)#

# Add new node attributes to the Data object
data.node_attr = node_attr_tensor

# Dictionary with new edge attributes
edge_attr_dict = {
    (0, 1): [0.1],
    (1, 0): [0.2],
    (1, 2): [0.3],
    (2, 1): [0.4]
}

#edge_attr_dict = {
#    (0, 1): [0.1],
#    (1, 0): [0.2],
#    (1, 2): [0.3],
#    (2, 1): [0.4]
#}
edge_attr_dict2 = {
    (0, 1): [0.1, 1.0, 10.0],
    (1, 0): [0.2, 2.0, 20.0],
    (1, 2): [0.3, 3.0, 30.0],
    (2, 1): [0.4, 4.0, 40.0]
}
# Convert edge attribute dictionary to tensor
edge_attr_tensor = torch.tensor([edge_attr_dict[(edge_index[0, i].item(), edge_index[1, i].item())] for i in range(edge_index.size(1))], dtype=torch.float)

# Add new edge attributes to the Data object
data.edge_attr = edge_attr_tensor

print(data)
print("edge attr:")
print(data.edge_attr)
print("node attr:")
print(data.node_attr)


print("iter dict test")

dict_test = {
    0: [0.1, 0.2],
    1: [0.3, 0.4],
    2: [0.5, 0.6]
}

#for a, b in dict_test.items():
#    print("a", a)
#    print("b", b)
print(np.array(list(dict_test.keys())))
