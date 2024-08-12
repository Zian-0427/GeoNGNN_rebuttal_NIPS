import torch.nn as nn
import torch
from layers.Mol2Graph import SparseMol2Graph
from layers.basic_layers import Dense
from layers.Vanilla_DisGNN import Vanilla_DisGNN
EPS = 1e-10

        
class GeoNGNN(nn.Module):
    def __init__(self, 
        z_hidden_dim,
        hidden_dim,
        ef_dim,
        rbf,
        max_z,
        outer_rbound_upper,
        inner_rbound_upper,
        activation_fn,
        inner_layer_num,
        outer_layer_num,
        inner_cutoff,
        outer_cutoff,
        predict_force,
        ablation_innerGNN,
        global_y_std,
        global_y_mean,
        C,
        subg_C,
        ablation_PE,
        extend_r,
        pre_outer=False,
        out_channel=1,
    ):
        super().__init__()
        
        # manage inner GNN
        if pre_outer:
            self.pre_M2G = SparseMol2Graph(
                z_hidden_dim=z_hidden_dim,
                ef_dim=ef_dim,
                rbf=rbf,
                max_z=max_z,
                rbound_upper=outer_rbound_upper,
                r_max=outer_cutoff,
                smooth=True
            )
            
            self.pre_GNN = Vanilla_DisGNN(
                hidden_dim=hidden_dim,
                ef_dim=ef_dim,
                activation_fn=activation_fn,
                layer_num=3,
                is_inner=False,
                is_pre=True,
                rbf=rbf
            )
            
            
        if not ablation_innerGNN:
            self.inner_M2G = SparseMol2Graph(
                z_hidden_dim=z_hidden_dim,
                ef_dim=ef_dim,
                rbf=rbf,
                max_z=max_z,
                rbound_upper=inner_rbound_upper,
                r_max=inner_cutoff,
                smooth=True,
                activation_fn=activation_fn,
            )
            
            self.label_embedding = nn.Embedding(3, z_hidden_dim, padding_idx=0)
            
            self.label_proj = Dense(
                in_features=z_hidden_dim,
                out_features=hidden_dim,
                activation_fn=activation_fn,
                
            )
            self.inner_proj = Dense(
                in_features=z_hidden_dim,
                out_features=hidden_dim,
                activation_fn=activation_fn,
                
            )
            self.inner_GNN = Vanilla_DisGNN(
                hidden_dim=hidden_dim,
                ef_dim=ef_dim,
                activation_fn=activation_fn,
                layer_num=inner_layer_num,
                rbf=rbf,
                is_inner=True,
                ablation_PE=ablation_PE,
                use_smooth=True,
                extend_r=extend_r
            )
            self.outer_fuse = nn.Sequential(
                Dense(
                    in_features=hidden_dim*2,
                    out_features=hidden_dim,
                    activation_fn=activation_fn,
                ),
                Dense(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    activation_fn=activation_fn,
                ),
                Dense(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    activation_fn=activation_fn,
                )
            )

        
        # manage outer GNN
        self.outer_M2G = SparseMol2Graph(
            z_hidden_dim=z_hidden_dim,
            ef_dim=ef_dim,
            rbf=rbf,
            max_z=max_z,
            rbound_upper=outer_rbound_upper,
            r_max=outer_cutoff,
            smooth=True
        )
        
        self.outer_GNN = Vanilla_DisGNN(
            hidden_dim=hidden_dim,
            ef_dim=ef_dim,
            activation_fn=activation_fn,
            layer_num=outer_layer_num,
            is_inner=False,
            rbf=rbf
        )
        

        self.output_linear = Dense(
            in_features=hidden_dim,
            out_features=out_channel,
            bias=False
        )
        
        self.outer_proj = nn.Sequential(
            Dense(
                in_features=z_hidden_dim,
                out_features=hidden_dim,
                activation_fn=activation_fn
            ),
            # nn.Dropout(0.3),
            Dense(
                in_features=hidden_dim,
                out_features=hidden_dim,
                activation_fn=activation_fn
            )
        )
        
        
        
        self.C = C
        self.subg_C = subg_C
        self.predict_force = predict_force
        self.ablation_innerGNN = ablation_innerGNN
        self.global_y_std = global_y_std
        self.global_y_mean = global_y_mean
        self.out_bn = nn.BatchNorm1d(hidden_dim)
        self.ablation_PE = ablation_PE
        self.pre_outer = pre_outer


    def forward(self, batch_data):
        
        C, subg_C = self.C, self.subg_C
        
        # Original graph info: z, pos, indices
        outer_pos = batch_data.pos # (bs, graph_size, 3)
        outer_pos.requires_grad_(True)
        outer_z = batch_data.z
        edge_index, batch_index, edge_features = batch_data.edge_index, batch_data.batch_index, batch_data.edge_features
        
        # Subgraph info: subg_indices, subg_labels
        subg_node_index, subg_node_center_index, subg_edge_index, subg_batch_index, subg_edge_features = (
            batch_data.subg_node_index, 
            batch_data.subg_node_center_index, 
            batch_data.subg_edge_index,
            batch_data.subg_batch_index,
            batch_data.subg_edge_features
        ) # (NM, 1), (NM, 1), (2, EM), (NM, 1)
        subg_node_label = batch_data.subg_node_label
        
        # outer_scalar GNN
        
        if self.pre_outer:
            pre_scalar, pre_ef = self.pre_M2G(outer_z, outer_pos, edge_index=edge_index, edge_features=edge_features, center_pos=None)
            pre_scalar_out = self.pre_GNN(
                scalar=pre_scalar,
                ef=pre_ef,
                edge_index=edge_index,
                C=C,
                batch_index=batch_index,
            )
            
        if not self.ablation_innerGNN:
            # recalculate inner_pos, inner_dist
            inner_z = torch.stack([outer_z[i][subg_node_index[i]] for i in range(outer_z.shape[0])], dim=0)
            inner_pos = torch.stack([outer_pos[i][subg_node_index[i]] for i in range(outer_pos.shape[0])], dim=0)
            center_pos = torch.stack([outer_pos[i][subg_node_center_index[i]] for i in range(outer_pos.shape[0])], dim=0)

            inner_dist = torch.norm(inner_pos - center_pos, dim=-1, keepdim=True).squeeze() # (NM, 1)
            
            # transform inner info
            inner_scalar, inner_ef = self.inner_M2G(inner_z, inner_pos, edge_index=subg_edge_index, edge_features=subg_edge_features, center_pos=center_pos) 
            
            if not self.ablation_PE:
                inner_label_emb = self.label_embedding(subg_node_label) # (NM, label_dim)
                inner_scalar = self.label_proj(inner_label_emb) * self.inner_proj(inner_scalar) # (NM, hidden_dim)
            else:
                inner_scalar = self.inner_proj(inner_scalar) # (NM, hidden_dim)
            
            if self.pre_outer:
                subg_pre_scalar_out = torch.stack([pre_scalar_out[i][subg_node_index[i]] for i in range(pre_scalar_out.shape[0])], dim=0)
                inner_scalar = inner_scalar + subg_pre_scalar_out
            
            # inner GNN
            _, outer_subg_scalar_env = self.inner_GNN(
                scalar=inner_scalar,
                ef=inner_ef,
                dist=inner_dist,
                edge_index=subg_edge_index,
                C=subg_C,
                subg_batch_index=subg_batch_index,
            ) # (NM, hidden_dim), (NM, hidden_dim)

        # Environment Fusion
        
        outer_scalar, outer_ef = self.outer_M2G(outer_z, outer_pos, edge_index=edge_index, edge_features=edge_features, center_pos=None)
        if not self.ablation_innerGNN:
            outer_scalar = self.outer_proj(outer_scalar)
            tmp = torch.cat([outer_subg_scalar_env, outer_scalar], dim=-1)
            outer_scalar = self.outer_fuse(tmp) + outer_scalar

        else:
            outer_scalar = self.outer_proj(outer_scalar) # (bs * graph_size, hidden_dim)
            
        
        outer_graph = self.outer_GNN(
            scalar=outer_scalar,
            ef=outer_ef,
            edge_index=edge_index,
            C=C,
            batch_index=batch_index,
        )
        
        # output
        output = self.output_linear(outer_graph)
        
        if self.global_y_mean is not None:
            pred_energy = output * self.global_y_std + self.global_y_mean
        else:
            pred_energy = output
        
        # calculate force
        if self.predict_force:
            
            pred_force = -torch.autograd.grad(
                    [torch.sum(pred_energy)], 
                    [outer_pos],
                    retain_graph=True,
                    create_graph=True
                    )[0]
            
            outer_pos.requires_grad_(False)
        
            return pred_energy, pred_force
        else:
            return pred_energy
