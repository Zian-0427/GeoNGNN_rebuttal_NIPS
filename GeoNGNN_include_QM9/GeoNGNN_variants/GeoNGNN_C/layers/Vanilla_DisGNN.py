import torch
import torch.nn as nn
from layers.basic_layers import Residual, Dense
from layers.basis_layers import rbf_class_mapping
from torch_scatter import scatter
from torch.nn import Sequential
from layers.basic_layers import Envelope
EPS = 1e-10


class VD_Conv(nn.Module):
    def __init__(self, hidden_dim, activation_fn, ef_dim):
        super().__init__()

        self.emb_mlp = nn.ModuleList(
                    [
                        Sequential(
                                Dense(
                                    in_features=hidden_dim,
                                    out_features=hidden_dim,
                                    activation_fn=activation_fn,
                                ),
                                # nn.Dropout(0.3),
                                Dense(
                                    in_features=hidden_dim,
                                    out_features=hidden_dim,
                                    activation_fn=activation_fn,
                            )
                            ) for _ in range(3)
                        ] 
                    )

        self.output_mlp = Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,  
                    )
        self.e_linear = Dense(
                    in_features=ef_dim,
                    out_features=hidden_dim,
                    bias=False,
                )
        self.conv_mlp = Sequential(
                Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,
                    ), 
                Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,
                    ), 
                )
        



    def forward(self, scalar, ef, edge_index, C):
            row = edge_index[:, 0]
            col = edge_index[:, 1]
            s2s, scalar_dst = [self.emb_mlp[j](scalar) for j in range(2)] 
            ef_proj = self.e_linear(ef)
            scalar_edge = torch.stack([scalar_dst[i][col[i]] for i in range(scalar_dst.shape[0])]) * ef_proj
        
            # Aggregate edge to scalar
            conv = torch.stack([scatter(scalar_edge[i], row[i], dim=0, reduce='sum', dim_size=s2s.shape[1]) for i in range(scalar_edge.shape[0])]) * C
                
            conv = self.conv_mlp(conv)
            scalar = s2s * conv # (N, hidden_dim)

            scalar = self.output_mlp(scalar) 

            return scalar
        
class Vanilla_DisGNN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 ef_dim,
                 is_inner,
                 activation_fn,
                 layer_num,
                 rbf,
                 is_pre=False,
                 use_smooth=False,
                 extend_r=None,
                 ablation_PE=False,
                 ):
        super().__init__()

        # Anchor distance encoding
        if is_inner:
            self.init_layer = Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn,
                        )
            self.DE_layer = nn.Sequential(
                    Dense(
                    in_features=ef_dim,
                    out_features=hidden_dim,
                    activation_fn=activation_fn
                ), 
                    # nn.Dropout(0.3),
                    Dense(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    activation_fn=activation_fn
                ), 
                    )
            self.rbf_fn = rbf_class_mapping[rbf](
                num_rbf=ef_dim, 
                rbound_upper=7.0,
                rbf_trainable=False,
            )
            
        self.VD_Convs = nn.ModuleList()

        for _ in range(layer_num):
            self.VD_Convs.append(
                VD_Conv(
                    hidden_dim=hidden_dim,
                    activation_fn=activation_fn,
                    ef_dim=ef_dim,
                )
            )


        if not is_inner:
            self.pooling_MLP = nn.Sequential(
                    *nn.ModuleList([
                        Residual(
                            mlp_num=2,
                            hidden_dim=hidden_dim,
                            activation_fn=activation_fn,
                        )
                        for i in range(3)
                    ])
            )

        if is_inner or is_pre:
            self.output_scalar = Residual(
                            mlp_num=2,
                            hidden_dim=hidden_dim,
                            activation_fn=activation_fn,
                        )
        
        if is_inner:
            self.pooling_MLP = nn.Sequential(
                    *nn.ModuleList([
                        Residual(
                            mlp_num=2,
                            hidden_dim=hidden_dim,
                            activation_fn=activation_fn,
                        )
                        for _ in range(3)
                    ])
            )
    
        self.p = 6
        self.envelope = Envelope(r_max=extend_r, p=self.p)
        self.layer_num = layer_num
        self.is_inner = is_inner
        self.ablation_PE = ablation_PE
        self.extend_r = extend_r
        self.use_smooth = use_smooth
        self.is_pre = is_pre
        
    def forward(
        self, scalar, ef,  edge_index, C=None,
        dist=None, batch_index=None, subg_batch_index=None
    ):
        # Distance Incoding for inner GNN
        if self.is_pre:
            scalar /= 1000
        if self.is_inner and not self.ablation_PE:
            dist_rbf = self.rbf_fn(dist.unsqueeze(-1)) 
            scalar = self.DE_layer(dist_rbf) * self.init_layer(scalar)
            if self.use_smooth:
                smooth_coef = self.envelope(dist.unsqueeze(-1))
                scalar = scalar * smooth_coef
                
        for i in range(self.layer_num):
            scalar = self.VD_Convs[i](
                scalar=scalar,
                ef=ef,
                edge_index=edge_index,
                C=C,
            ) + scalar
        
        # Pooling
        if self.is_inner:
            scalar = self.output_scalar(scalar)
            subg_scalar = torch.stack([scatter(scalar[i], subg_batch_index[i], dim=0, reduce='sum') for i in range(scalar.shape[0])])
            subg_scalar = self.pooling_MLP(subg_scalar)
    
            return scalar, subg_scalar
        elif self.is_pre:
            scalar = self.output_scalar(scalar)
            
            return scalar
        else:
            graph_sum = torch.stack([scatter(scalar[i], batch_index[i], dim=0, reduce='sum') for i in range(scalar.shape[0])])
            graph = self.pooling_MLP(graph_sum)
            
            return graph