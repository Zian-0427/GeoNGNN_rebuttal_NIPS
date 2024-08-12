import torch.nn as nn
import torch
from utils.initializers import he_orthogonal_init
from torch_geometric.nn import HeteroLinear
from torch_geometric.nn import Sequential, GraphNorm

class Dense(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        activation_fn: torch.nn.Module = None,
        use_layer_norm = False,
        hetero: bool = False,
        num_types: int = 4
    ):
        super().__init__()
        if hetero:
            self.linear = HeteroLinear(
                in_channels=in_features,
                out_channels=out_features,
                num_types=num_types,
                is_sorted=False,
                bias=bias
            )
        else:
            self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        self._activation = torch.nn.Identity() if activation_fn is None else activation_fn
        self.layer_norm = GraphNorm(out_features) if use_layer_norm else torch.nn.Identity()
        
        
        

    def reset_parameters(self):
        if not self.in_features == 1:
            he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, x_type: torch.Tensor = None):
        
        # linear layer
        if x_type is not None:
            if x.dim() == 3:
                x = self.linear(x.flatten(0, 1), 
                                x_type.repeat(1, x.shape[1]).flatten(0, 1)
                                ).view(x.shape[0], x.shape[1], -1)
            else:
                x = self.linear(x, x_type)
        else:
            x = self.linear(x)
            
        
        x = self.layer_norm(x)
        x = self._activation(x)
        return x


class Residual(nn.Module):
    def __init__(
        self,
        mlp_num: int,
        hidden_dim: int,
        activation_fn: torch.nn.Module = None,
        bias: bool = True,
        add_end_activation: bool = True,
        use_layer_norm = False,
        hetero: bool = False,
        num_types: int = 4
    ):
        super().__init__()
        assert mlp_num > 0
        end_activation_fn = activation_fn if add_end_activation else None
        
        self.mlps = Sequential('x, x_type',
            [
                (
                    Dense(hidden_dim, hidden_dim, bias=bias, activation_fn=activation_fn, use_layer_norm=use_layer_norm, hetero=hetero, num_types=num_types), 
                    'x, x_type -> x'
                    )
                if i != mlp_num - 1 else
                (
                    Dense(hidden_dim, hidden_dim, bias=bias, activation_fn=end_activation_fn, use_layer_norm=use_layer_norm, hetero=hetero, num_types=num_types), 
                    'x, x_type -> x'
                    )
                for i in range(mlp_num)
            ]
        )
        
            
    def forward(self, x: torch.Tensor, x_type: torch.Tensor = None):
        return self.mlps(x, x_type) + x
    
    



class Envelope(nn.Module):
    '''
    Gasteiger, Johannes, Janek Groß, and Stephan Günnemann. "Directional message passing for molecular graphs." arXiv preprint arXiv:2003.03123 (2020).
    '''
    def __init__(self, r_max, p):
        super().__init__()
        self.r_max = r_max
        self.p = p
    def forward(self, x):
        smooth_coef = (
                    1.0
                    - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                    + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                    - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
            )
        smooth_coef[x >= self.r_max] = 0.0
        return smooth_coef
