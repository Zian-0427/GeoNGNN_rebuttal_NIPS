import setproctitle
import pytorch_lightning as pl
import sys
import time
import os
from argparse import ArgumentParser
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")
from scripts.script_utils import trainer_setup, test, train, get_cfgs
from utils.select_free_gpu import select_free_gpu
from datasets.QM9 import qm9_datawork
from scripts.script_utils import model_dict, transform_dict

'''
    get args
'''

parser = ArgumentParser()
parser.add_argument("--model", choices=["GeoNGNN"], default="GeoNGNN")
parser.add_argument("--ds", choices=["qm9"], default="qm9")
parser.add_argument("--dname", default="4")
parser.add_argument("--devices", nargs="+", type=int, default=None)
parser.add_argument("--data_dir", default="~/datasets/QM9")
parser.add_argument("--version", default="NO_VERSION")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--ckpt_homo", default=None)
parser.add_argument("--ckpt_lumo", default=None)
parser.add_argument("--merge", nargs="+", type=str, default=None)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--proj_name", default=None)

args = parser.parse_args()
# print log
print(f"{'-'*10}ARGS{'-'*10}")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

from lightningModule.RG_module import RG_module


model_name = args.model
dataset_name = args.ds
data_name = args.dname
data_dir = args.data_dir
resume = args.resume
checkpoint_path_homo = args.ckpt_homo
checkpoint_path_lumo = args.ckpt_lumo
skip_train = args.skip_train
version = args.version
devices = args.devices
merge_list = args.merge

accelerator = "gpu"
if devices is None:
    devices = [select_free_gpu()]
elif devices == [-1]:
    devices = None
    accelerator = "cpu"


'''
    get hparams
'''
config_path = "~/GeoNGNN/GeoNGNN_github/hparams/{}_{}.yaml".format(model_name, dataset_name)
specific_config_path = "~/GeoNGNN/GeoNGNN_github/hparams/specific/{}_{}_specific.yaml".format(model_name, dataset_name)
if not os.path.exists(specific_config_path):
    specific_config_path = None
config = get_cfgs(config_path, merge_list, specific_config_path, data_name)

scheduler_config = config.scheduler_config
optimizer_config = config.optimizer_config
model_config = config.model_config

# trainer_config
trainer_config = config.trainer_config
validation_interval = trainer_config.validation_interval
log_every_n_steps = 100
early_stopping_patience = trainer_config.early_stopping_patience
max_epochs = trainer_config.max_epochs

# global_config
global_config = config.global_config
seed = config.global_config.seed

# data_config
data_config = config.data_config
train_batch_size = data_config.train_batch_size
val_batch_size = data_config.val_batch_size
test_batch_size = data_config.test_batch_size



print(f"{'-'*10}MODEL CONFIG{'-'*10}")
print(model_config)
print(f"{'-'*10}OPTIMIZER CONFIG{'-'*10}")
print(optimizer_config)
print(f"{'-'*10}SCHEDULER CONFIG{'-'*10}")
print(scheduler_config)
print(f"{'-'*10}DATA CONFIG{'-'*10}")
print(data_config)
print(f"{'-'*20}")

pl.seed_everything(seed)

'''
    prepare data
'''
predict_force = False
datawork = qm9_datawork
    
'''
    prepare model and transform
'''

model = model_dict[model_name]
transform = transform_dict[model_name]

setproctitle.setproctitle("GeoNGNN@{}-{}-{}-{}".format(model_name, dataset_name, data_name, version))


train_dl, val_dl, test_dl, global_y_mean, global_y_std = datawork(
    root=data_dir,
    name=data_name,
    batch_size=[train_batch_size, val_batch_size, test_batch_size],
    subgraph_cutoff=model_config.subg_cutoff,
    cutoff=model_config.cutoff,
    extend_r=model_config.extend_r,
    transform=transform,
    max_neighbor=model_config.max_neighbor,
)




import torch
from torch_scatter import scatter
from tqdm import tqdm

# neighbor_num = torch.empty([0], dtype=torch.long, device=f"cuda:{devices[0]}")
# subg_neighbor_num = torch.empty([0], dtype=torch.long, device=f"cuda:{devices[0]}")

# for data in tqdm(iter(train_dl)):

#     row, col = data.edge_index.to(f"cuda:{devices[0]}")
#     neighbor_num_add = scatter(torch.ones_like(row).float(), row, dim=0, reduce='sum', dim_size=row.max()+1)
#     neighbor_num = torch.cat([neighbor_num, neighbor_num_add])

#     row, col = data.subg_edge_index.to(f"cuda:{devices[0]}")
#     try:
#         subg_neighbor_num_add = scatter(torch.ones_like(row).float(), row, dim=0, reduce='sum', dim_size=row.max()+1)
#         subg_neighbor_num = torch.cat([subg_neighbor_num, subg_neighbor_num_add])
#     except:
#         print("No subgraph")

# C = neighbor_num.mean().item() ** - model_config.C_power
# try:
#     subg_C = subg_neighbor_num.mean().item() ** - model_config.C_power
# except:
#     subg_C = 1.
# if subg_C > 1.0:
#     subg_C = 1.0

# print(f"{'-'*10}C={C}{'-'*10}")
# print(f"{'-'*10}subg_C={subg_C}{'-'*10}")

output_layer_type = "dip" if int(data_name) == 0 else "elc" if int(data_name) == 5 else "linear"







module_instance_homo = RG_module.load_from_checkpoint(
    checkpoint_path=checkpoint_path_homo,
    strict=True
    ).to("cuda")



module_instance_lumo = RG_module.load_from_checkpoint(
    checkpoint_path=checkpoint_path_lumo,
    strict=True
    ).to("cuda")



from utils.loss_fns import loss_fn_map
import tqdm 
loss_fn  = loss_fn_map["l1"]
loss_sum = 0.
batch_sum = 0
with torch.no_grad():
    for data_batch in tqdm.tqdm(test_dl):
        data_batch.to("cuda")
        batch_size = data_batch.y.shape[0]
        
        homo = module_instance_homo.ema_model(data_batch)
        lumo = module_instance_lumo.ema_model(data_batch)
        
        pred_y = (homo - lumo).abs().squeeze().to("cuda")

        y = data_batch.y.squeeze().to("cuda")
        assert torch.all(y > 0)
        
        
        loss_sum += loss_fn(pred_y, y) * batch_size
        batch_sum += batch_size

    print(loss_sum/batch_sum)