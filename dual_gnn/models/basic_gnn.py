import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from collections import Counter
from torch_scatter import scatter_add
from torch_geometric.typing import Adj
from typing import Optional, Callable, List
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout, Parameter
from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv, GATConv, MessagePassing, \
    SGConv, GATv2Conv, ARMAConv, FiLMConv, SuperGATConv, TransformerConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.conv import APPNP as APPNPConv


class BasicGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last', base_model=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        self.convs = ModuleList()

        self.norms = None
        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if out_channels is not None:
            self.out_channels = out_channels
            if jk == 'cat':
                self.lin = Linear(num_layers * hidden_channels, out_channels)
            else:
                self.lin = Linear(hidden_channels, out_channels)
        else:
            if jk == 'cat':
                self.out_channels = num_layers * hidden_channels
            else:
                self.out_channels = hidden_channels

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    def output(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        assert hasattr(self, 'lin')
        
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')



class GCN(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(GCNConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, **kwargs))


class GraphSAGE(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(SAGEConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, **kwargs))


class GIN(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(
            GINConv(GIN.MLP(in_channels, hidden_channels), **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GINConv(GIN.MLP(hidden_channels, hidden_channels), **kwargs))

    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )


class GAT(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            GATConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(GATConv(hidden_channels, out_channels, **kwargs))



class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        layers = []
        layers.append(Linear(in_channels, hidden_channels))
        layers.append(self.act)
        layers.append(torch.nn.Dropout(dropout))
        for i in range(1, num_layers):
            layers.append(Linear(hidden_channels, hidden_channels))
            layers.append(self.act)
            layers.append(torch.nn.Dropout(dropout))

        if out_channels is not None:
            layers.append(Linear(hidden_channels, out_channels))

        self.model = Sequential(*layers)

    def reset_parameters(self):
        for module in self.model:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, batch: Tensor, _: Adj, *args, **kwargs) -> Tensor:
        # TODO(palowitch): Does our scaffolding ever invoke the else clause?
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')



class APPNP(torch.nn.Module):
    def __init__(self, iterations: int, alpha: float,
                 in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 cached=False):
        super(APPNP, self).__init__()

        self.mlp = MLP(in_channels, hidden_channels, num_layers, out_channels, dropout, act)
        self.appnp = APPNPConv(iterations, alpha, cached=cached)

        print(self.appnp)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        x = self.mlp(x, edge_index)
        x = self.appnp(x, edge_index)
        #return F.log_softmax(x, dim=1)   # don't think we need this here
        return x



class SGC(torch.nn.Module):
    def __init__(self, iterations: int,
                 in_channels: int, hidden_channels: int,
                 out_channels: Optional[int] = None,
                 cached=False, dropout:float =0):
        super(SGC, self).__init__()

        if out_channels is None:
            out_channels = hidden_channels
        self.sgc = SGConv(in_channels=in_channels, out_channels=out_channels, K=iterations, cached=cached)
        self.dropout = dropout

        print(self.sgc)

    def reset_parameters(self):
        self.sgc.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        x = self.sgc(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #return F.log_softmax(x, dim=1) # don't think we need this here
        return x



class GATv2(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            GATv2Conv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(GATv2Conv(hidden_channels, out_channels, **kwargs))



class ARMA(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(ARMAConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                ARMAConv(hidden_channels, hidden_channels, **kwargs))



class FiLM(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(FiLMConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                FiLMConv(hidden_channels, hidden_channels, **kwargs))



class Transformer(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            TransformerConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(TransformerConv(hidden_channels, out_channels, **kwargs))



class SuperGAT(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
        out_channels: Optional[int] = None, dropout: float = 0.0,
        act: Optional[Callable] = ReLU(inplace=True),
        norm: Optional[torch.nn.Module] = None, jk: str = 'last',
        **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            SuperGATConv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(SuperGATConv(hidden_channels, out_channels, **kwargs))


class CachedGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels,  weight=None, bias=None, path_len=0, improved=False,
                 use_bias=True, device=None, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.device = device
        self.cache_dict = {}

        if weight is None:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(torch.float32))
            glorot(self.weight)
        else:
            self.weight = weight
            # print("use shared weight")

        if bias is None:
            if use_bias:
                self.bias = Parameter(torch.Tensor(out_channels).to(torch.float32))
            else:
                self.register_parameter('bias', None)
            zeros(self.bias)
        else:
            self.bias = bias
            # print("use shared bias")

    def reset_parameters(self):
        zeros(self.bias)
        zeros(self.weight)
        # self.cache_dict = {}


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name="default_cache", edge_weight=None):
        x = torch.matmul(x, self.weight)
        if not cache_name in self.cache_dict:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cache_dict[cache_name] = edge_index, norm
        else:
            edge_index, norm = self.cache_dict[cache_name]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class PPMIConv(CachedGCNConv):

    def __init__(self, in_channels, out_channels, weight=None, bias=None, improved=False, use_bias=True,
                 path_len=5, device=None, **kwargs):
        super().__init__(in_channels, out_channels, weight, bias, improved, use_bias, device, **kwargs)
        self.path_len = path_len
        self.device = device

    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):

        adj_dict = {}

        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        for a, b in edge_index.t().cpu().numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]

        walk_counters = {}

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter

        for _ in range(40):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1

                    current_a = b

        normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

        prob_sums = Counter()

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / self.path_len)
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)

        edge_index = torch.tensor(new_edge_index).t().to(self.device)
        edge_weight = torch.tensor(edge_weight).to(self.device)


        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index.to(self.device), (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32).to(self.device)




class UDAGCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                            dropout=0.0, base_model=None, type="gcn", path_len=0, device=None):
        
        super(UDAGCN_Encoder, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight.to(device) for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias.to(device) for conv_layer in base_model.conv_layers]

        self.dropout_layers = [torch.nn.Dropout(dropout).to(device) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv
        self.conv_layers = torch.nn.ModuleList([
            model_cls(in_channels, hidden_channels,
                     weight=weights[0],
                     bias=biases[0],
                     path_len=path_len,
                     device=device).to(device),
            model_cls(hidden_channels, out_channels,
                     weight=weights[1],
                     bias=biases[1],
                     path_len=path_len,
                     device=device).to(device)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()


class Attention(torch.nn.Module):
    def __init__(self, in_channels, dropout=0.1, device=None):
        super().__init__()
        self.dense_weight = torch.nn.Linear(in_channels, 1).to(device)
        self.dropout = torch.nn.Dropout(dropout).to(device)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs
    
    def reset_parameters(self):
        self.dense_weight.reset_parameters()



class UDAGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, path_len: int = 10, device=None, **kwargs):
        super().__init__()
        self.encoder = UDAGCN_Encoder(in_channels, hidden_channels, hidden_channels, dropout=dropout, type="gcn", device=device).to(device)
        self.ppmi_encoder = UDAGCN_Encoder(in_channels, hidden_channels, hidden_channels, dropout=dropout, base_model=self.encoder, type="ppmi", path_len=path_len, device=device)
        self.att_model = Attention(hidden_channels, device=device)

    def forward(self, data, cache_name, mask=None):
        gcn_output = self.encoder(data.x, data.edge_index, cache_name)
        if mask is not None:
            gcn_output = gcn_output[mask]
        ppmi_output = self.ppmi_encoder(data.x, data.edge_index, cache_name)
        if mask is not None:
            ppmi_output = ppmi_output[mask]
        outputs =  self.att_model([gcn_output, ppmi_output])
        return outputs
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.ppmi_encoder.reset_parameters()
        self.att_model.reset_parameters()




        




