import torch.nn as nn
from torch import Tensor
import torch
from layers.basis_layers import rbf_class_mapping
from layers.basic_layers import Envelope
from layers.basic_layers import Dense

z_embedding_dim = [13, 4, 8, 6, 6, 8, 2, 4]
edge_features_embedding_dim = [6, 3, 3, 8]

class SparseMol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbound_upper: float,
                 max_z: int,
                 smooth: bool = False,
                 r_max: float = 10,
                 activation_fn=None,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=False,
                    **kwargs
                )
        self.z_emb = nn.ModuleList([nn.Embedding(z_embedding_dim[_], z_hidden_dim, padding_idx=None) for _ in range(8)])
        self.edge_features_emb = nn.ModuleList([nn.Embedding(edge_features_embedding_dim[_], ef_dim, padding_idx=None) for _ in range(4)])
        self.dV_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=ef_dim,
            padding_idx=None
        )
        self.dV_rbf = rbf_class_mapping[rbf](
                    num_rbf=ef_dim,
                    rbound_lower=-rbound_upper,
                    rbound_upper=rbound_upper, 
                    rbf_trainable=False,
                    **kwargs
                )
        self.smooth = smooth
        self.p = 6
        self.envelope = Envelope(r_max=r_max, p=self.p)
        
        self.e_fuse = nn.Sequential(
                Dense(
                    in_features=ef_dim*2,
                    out_features=ef_dim,
                    activation_fn=activation_fn,
                ),
                # nn.Dropout(0.3),
                Dense(
                    in_features=ef_dim,
                    out_features=ef_dim,
                    activation_fn=activation_fn,
                ),
                # nn.Dropout(0.3),
                Dense(
                    in_features=ef_dim,
                    out_features=ef_dim,
                    activation_fn=activation_fn,
                )
            )
        

    def forward(self, z: Tensor, pos: Tensor, edge_index: Tensor, edge_features: Tensor, center_pos = None):
        '''
            z (N, ): atomic number
            pos (N, 3): atomic position
            edge_index (2, E): edge indices
        '''
        # print(edge_index, edge_index.max(), '\n', '\n')
        emb1 = self.z_emb[0](z[...,0])
        for i in range(2, 7):
            emb1 = emb1 + self.z_emb[i](z[...,i])
        # add chiral tag

        posu = torch.stack([pos[i][edge_index[i][0]] for i in range(pos.shape[0])], dim=0) # (E, 3)
        posv = torch.stack([pos[i][edge_index[i][1]] for i in range(pos.shape[0])], dim=0)
        
        ev = posu - posv
        el = torch.norm(ev, dim=-1, keepdim=True) # (E, 1)
        ef = self.rbf_fn(el) # (E, ef_dim)

        for i in range(4):
            ef = ef + self.edge_features_emb[i](edge_features[...,i])
        
        if center_pos is not None:
            
            center_pad = torch.stack([center_pos[i][edge_index[i][1]] for i in range(pos.shape[0])], dim=0)
            ecross = torch.cross(posu, posv, dim=-1)
            
            # sign
            dV = torch.sum(ecross * center_pad, dim=-1)
            dV_sign = torch.sign(dV).int() + 1
            dV_emb = self.dV_embedding(dV_sign)
            ef = ef * dV_emb
            
            # rbf
            # dV = torch.sum(ecross * center_pad, dim=-1, keepdim=True)
            # dV_emb = self.dV_rbf(dV)
            # tmp = torch.cat([ef, dV_emb], dim=-1)
            # ef = self.e_fuse(tmp) + ef
        
        if self.smooth:
            smooth_coef = self.envelope(el) # (E, 1)
            ef = ef * smooth_coef
        
        return emb1, ef




class Mol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbf_trainable: bool,
                 rbound_upper: float,
                 max_z: int,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=rbf_trainable,
                    **kwargs
                )
        self.z_emb = nn.Embedding(max_z + 1, z_hidden_dim, padding_idx=0)
        

    def forward(self, z: Tensor, pos: Tensor, **kwargs):
        '''
            z (B, N)
            pos (B, N, 3)
            emb1 (B, N, z_hidden_dim)
            ef (B, N, N, ef_dim)
            ev (B, N, N, 3)
        '''
        emb1 = self.z_emb(z)
        
        if kwargs.get("edge_index", None) is not None:
            edge_index = kwargs["edge_index"]
            ev = pos[edge_index[0]] - pos[edge_index[1]] # (E, 3)
            el = torch.norm(ev, dim=-1, keepdim=True) # (E, 1)
            ef = self.rbf_fn(el) # (E, ef_dim)
        else:
            B, N = z.shape[0], pos.shape[1]
            ev = pos.unsqueeze(2) - pos.unsqueeze(1)
            el = torch.norm(ev, dim=-1, keepdim=True)
            ef = self.rbf_fn(el.reshape(-1, 1)).reshape(B, N, N, -1)
        
        return emb1, ef





if __name__ == "__main__":
    rbf_fn = rbf_class_mapping["nexpnorm"](
            num_rbf=32, 
            rbound_upper=10, 
            rbf_trainable=False,
        )
    print(rbf_fn(torch.tensor(12, dtype=torch.float32)))


