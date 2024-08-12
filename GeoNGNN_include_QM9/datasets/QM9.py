from torch_geometric.datasets import qm9
from torch.utils.data import random_split
from typing import Optional, List
from torch.utils.data import DataLoader
import torch
from datasets.transform_collate import collate_
import os
import pickle
import tqdm
    
def get_indices(qm9_dataset, root, type_="train_large_test_small"):

    assert type_ in ["train_small_test_large", "train_large_test_small"]
    assert isinstance(qm9_dataset, qm9.QM9)
    max_size = 21 if type_ == "train_small_test_large" else 14
    indices_dir = os.path.join(root, "indices")
    save_name = f"{type_}_{max_size}_indices.pkl"
    if os.path.exists(os.path.join(indices_dir, save_name)):
        with open(os.path.join(indices_dir, save_name), "rb") as f:
            indices = pickle.load(f)
        return indices
    else:
        get_meta_info = False
        if get_meta_info:
            # run across the dataset to find the basic statistics
            size_dict = {}
            for data in tqdm.tqdm(qm9_dataset):
                size = data.z.size(0)
                if size not in size_dict:
                    size_dict[size] = 1
                else:
                    size_dict[size] += 1
            print(size_dict)
            exit(0)
        train_indices, val_indices, test_indices = [], [], []
        
        if type_ == "train_small_test_large":
            for i, data in tqdm.tqdm(enumerate(qm9_dataset)):
                size = data.z.size(0)
                if size <= max_size:
                    train_indices.append(i)
                else:
                    test_indices.append(i)
        else:
            for i, data in tqdm.tqdm(enumerate(qm9_dataset)):
                size = data.z.size(0)
                if size <= max_size:
                    test_indices.append(i)
                else:
                    train_indices.append(i)
        train_indices_num = int(len(train_indices) / 12 * 11)
        valid_indices_num = len(train_indices) - train_indices_num
        train_indices, val_indices = random_split(train_indices, [train_indices_num, valid_indices_num])
        # save the indices
        if os.path.exists(indices_dir) is False:
            os.makedirs(indices_dir)
        indices = [train_indices, val_indices, test_indices]
        with open(os.path.join(indices_dir, save_name), "wb") as f:
            pickle.dump(indices, f)
        
        return [train_indices, val_indices, test_indices]

    
def qm9_datawork(
    root: str, 
    name: str,
    transform,
    max_neighbor,
    batch_size: List[int],
    subgraph_cutoff: float = -1.0,
    cutoff: float = -1.0,
    extend_r=1.0,
    **kwargs,
    ):
    name = name if type(name) == int else int(name)
    
    if transform is not None:
        transform_ = transform(
            dname=name,
            subgraph_cutoff=subgraph_cutoff,
            cutoff=cutoff,
            extend_r=extend_r,
            max_neighbor=max_neighbor
            )
    else:
        transform_ = None
    
    qm9_dataset = qm9.QM9(root=root, transform=transform_)
    
    # dist_list = []
    # for i, data in enumerate(qm9_dataset):
    #     pos = data.pos
    #     dist_matrix = torch.norm(pos[:, None, :] - pos[None, :, :], p=2, dim=-1)
    #     dist_list.extend(dist_matrix.flatten().tolist())
    #     # remove all zero distance
    # dist_list = [x for x in dist_list if x > 1e-6]
        
    # print(f"Max distance: {max(dist_list)}, Min distance: {min(dist_list)}")
    # print(f"Mean distance: {sum(dist_list) / len(dist_list)}")


    
    
    
    
    train_batch_size, val_batch_size, test_batch_size = batch_size

    # train_indices, val_indices, test_indices = get_indices(qm9_dataset=qm9_dataset, type_="train_large_test_small", root=root) # train_large_test_small train_small_test_large
    # train_dataset, val_dataset, test_dataset = (
    #     torch.utils.data.Subset(qm9_dataset, train_indices),
    #     torch.utils.data.Subset(qm9_dataset, val_indices),
    #     torch.utils.data.Subset(qm9_dataset, test_indices),
    # )
    train_data_num, val_data_num, test_data_num = [
        110000, 10000, len(qm9_dataset) - 110000 - 10000
    ]
    train_dataset, val_dataset, test_dataset = random_split(
        qm9_dataset, 
        [train_data_num, 
        val_data_num, 
        test_data_num]
        )
    
    # This is mainly for resuming. Since the model may have high variance during training, NAN may be met and when resuming, reshuffling the data may help.
    from torch.utils.data import RandomSampler
    import time
    from torch import Generator
    g = Generator()
    g.manual_seed(int(time.time() * 1000))
    sampler = RandomSampler(train_dataset, generator=g)
    
    import functools
    collate = functools.partial(collate_, name=name)
    # get dataloaders, note that batch size must be 1 because batch is already divided in dataset.
    train_dataloader, val_dataloader, test_dataloader = (
        DataLoader(train_dataset, num_workers=16, batch_size=train_batch_size, persistent_workers=False, collate_fn=collate, sampler=sampler), 
        DataLoader(val_dataset, num_workers=16, batch_size=val_batch_size, persistent_workers=False, shuffle=False, collate_fn=collate), 
        DataLoader(test_dataset, num_workers=16, batch_size=test_batch_size, persistent_workers=False, shuffle=False, collate_fn=collate),
    )
    
    

    
    global_y_mean = qm9_dataset.data.y[:, int(name)][train_dataset.indices].mean()
    global_y_std = qm9_dataset.data.y[:, int(name)][train_dataset.indices].std()
    
    return train_dataloader, val_dataloader, test_dataloader, global_y_mean, global_y_std


