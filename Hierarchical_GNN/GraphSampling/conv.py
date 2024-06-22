from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm, degree, add_self_loops

# from dgl import ops
# from dgl.nn.functional import edge_softmax

import torch
from torch import nn

class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res

        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x

    def reset_parameters(self):
        self.linear_1.reset_parameters()
        self.linear_2.reset_parameters()
class SAgeConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation (as described in Eq. (3) of the paper).
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels,#: Union[int, Tuple[int, int]],
        out_channels: int,
        hidden_dim_multiplier,
        dropout,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        # if self.root_weight:
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.feed_forward_module = FeedForwardModule(dim=in_channels[0],
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x,#: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        """
        
        # else:
        #     x_res = x
        # if self.project and hasattr(self, 'lin'):
        #     x_res = (self.lin(x_res[0]).relu(), x_res[1])

        # propagate_type: (x: OptPairTensor)
        message = self.propagate(edge_index, x=x)#, size=size)
        # out = self.lin_l(out)

        # x_r = x_res[1]
        x_res = torch.cat((x, message), axis=1)

        # if self.root_weight and x_r is not None:
        #     out = out + self.lin_r(x_r)

        out = self.feed_forward_module(x_res)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out"""
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x[0].size(0))

        #is desired perform transformation on target nodes here
        # Start propagating messages.
        # calls message , aggregate then update
        x = (self.lin_r(x[0]), x[1])
        return self.propagate(edge_index, x=x)

    # def message(self, x_j):
    #     return x_j



    def message(self, x_j: Tensor) -> Tensor:
        r"""
         would perform linear or op on neighbors here
         eg functional transform of neighbors to be passed to sourceConstructs messages from node :math:`j` to node :math:`i`
                in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
                :obj:`edge_index`.
                This function can take any argument as input which was initially
                passed to :meth:`propagate`.
                Furthermore, tensors passed to :meth:`propagate` can be mapped to the
                respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
                :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
                """
        message = self.lin(x_j)
        return message

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)

    def update(self, aggr_out, x) -> Tensor:
        # performs combination function psi of source and neighbor node
        # message represntations
        # Concat target node features with aggregated messages
        r"""Updates node embeddings in analogy to
                :math:`\gamma_{\mathbf{\Theta}}` for each node
                :math:`i \in \mathcal{V}`.
                Takes in the output of aggregation as first argument and any argument
                which was initially passed to :meth:`propagate`.
                """
        # x = kwargs['x']
        # is desired perform transformation on target nodes here
        if isinstance(x, Tensor):
            x = (x, x)
        x_target = x[0]#self.lin_l(x[0])
        out = torch.cat([x_target, aggr_out], dim=1)
        out = self.feed_forward_module(out)
        # Apply the final linear transformation
        return out

    # def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
    #     if isinstance(adj_t, SparseTensor):
    #         adj_t = adj_t.set_value(None, layout=None)
    #     return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
