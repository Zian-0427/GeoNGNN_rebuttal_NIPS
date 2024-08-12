from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch
from torch.utils.data import DataLoader
from typing import List
import random
from tqdm import tqdm

def group_same_size(
    dataset: Data,
):
    data_list = list(dataset)
    data_list.sort(key=lambda data: data.z.shape[0])
    # grouped dataset by size
    grouped_dataset = []
    for i in range(len(data_list)):
        data = data_list[i]
        if i == 0:
            group = [data]
        else:
            last_data = data_list[i-1]
            if data.z.shape[0] == last_data.z.shape[0]:
                group.append(data)
            else:
                grouped_dataset.append((last_data.z.shape[0], group))
                group = [data]
    grouped_dataset.append((last_data.z.shape[0], group))
    return grouped_dataset

def batch_same_size(
    grouped_dataset: Data,
    batch_size: int,
):
            
    # batched dataset, according to the batch size. 
    batched_dataset = []
    for size, group in grouped_dataset:
        batch_num_in_group = (len(group) // batch_size) + 1 if len(group) % batch_size != 0 else len(group) // batch_size

        for i in range(batch_num_in_group):
            lower_bound = i * batch_size
            upper_bound = min((i+1) * batch_size, len(group))
            
            batch = group[lower_bound:upper_bound]

            y = torch.cat([batch[i].y.unsqueeze(0) for i in range(len(batch))], dim=0)
            z = torch.cat([batch[i].z.unsqueeze(0) for i in range(len(batch))], dim=0)
            pos = torch.cat([batch[i].pos.unsqueeze(0) for i in range(len(batch))], dim=0)
            batch_index = torch.cat([batch[i].batch_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            subg_node_index = torch.cat([batch[i].subg_node_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            subg_node_center_index = torch.cat([batch[i].subg_node_center_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            subg_batch_index = torch.cat([batch[i].subg_batch_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            subg_size = torch.cat([batch[i].subg_size.unsqueeze(0) for i in range(len(batch))], dim=0)
            subg_edge_index = torch.cat([batch[i].subg_edge_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            subg_node_label = torch.cat([batch[i].subg_node_label.unsqueeze(0) for i in range(len(batch))], dim=0)
            edge_index = torch.cat([batch[i].edge_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            mass = torch.cat([batch[i].mass.unsqueeze(0) for i in range(len(batch))], dim=0)
            edge_features = torch.cat([batch[i].edge_features.unsqueeze(0) for i in range(len(batch))], dim=0)
            subg_edge_features = torch.cat([batch[i].subg_edge_features.unsqueeze(0) for i in range(len(batch))], dim=0)
            name = [batch[i].name for i in range(len(batch))]
            batched_dataset.append(
                Data(
                    y=y, 
                    z=z, 
                    pos=pos,
                    batch_size=z.shape[0],
                    graph_size=z.shape[1],
                    name = name,
                    batch_index = batch_index,
                    subg_node_index = subg_node_index,
                    subg_node_center_index = subg_node_center_index,
                    subg_batch_index = subg_batch_index,
                    subg_size = subg_size,
                    subg_edge_index = subg_edge_index,
                    subg_node_label = subg_node_label,
                    edge_index = edge_index,
                    mass = mass,
                    edge_features = edge_features,
                    subg_edge_features = subg_edge_features,
                )
            )
    return batched_dataset



class batched_chiral(Dataset):
    def __init__(self, root, data, transform=None, pre_transform=None, batch_size: int = None, indices: List[int] = None, shuffle=False):
        super(batched_chiral, self).__init__(root, transform, pre_transform)
        self.data = data
        self.size = len(indices)
        self.size = len(indices)
        self.id = indices[0]
        self.batch_size = batch_size
        self.flag = False
        self.fulldataset = self[indices]
        self.shuffle = shuffle
        if shuffle:
            self.reshuffle()
        tmp = group_same_size(self.fulldataset)
        self.batched_data = batch_same_size(tmp, self.batch_size)
        self.flag = True

        self.data_num = self.size
        self.size = len(self.batched_data) # batched version, a datapoint is a batch.
        self.data = None
    
    def __getitem__(self, index):
        if not self.flag:
            return [self.transform(self.data[x]) for x in tqdm(index, "Transforming...")]
        return self.batched_data[index]
    
    def __len__(self):
        return self.size
    
    def __repr__(self) -> str:
        return f"Batched_Chiral(batch_size={self.batch_size}, size={self.size})"
    
    @property
    def raw_file_names(self):
        return ["rs/rs_raw.pt"] # A fake name. The raw data and process function are defined outside. See README.
    
    @property
    def processed_file_names(self):
        return ["rs/rs_full.pt"] # Also not used. The data is loaded outside.
    
    
    def download(self):
        pass
    
    def process(self):
        pass
        
    def len(self):
        return self.size
    
    
    def get(self, idx):
        return self.batched_data[idx]
    
    def reshuffle(self):
        tmp_dataset = self.fulldataset
        random.shuffle(tmp_dataset)
        
        tmp_dataset = sorted(tmp_dataset, key=lambda x: x.name[0])
        tmp_dataset = sorted(tmp_dataset, key=lambda x: x.name[1])

        for i in range(len(tmp_dataset)):
            if i == 0 or tmp_dataset[i].name[0] != tmp_dataset[i-1].name[0]:
                tmp_dataset[i].name[2] = 0
            else:
                tmp_dataset[i].name[2] = tmp_dataset[i-1].name[2] + 1
        tmp_dataset = sorted(tmp_dataset, key=lambda x: (x.name[1], x.name[2]))
        self.fulldataset = []
        nn = len(tmp_dataset)
        i = 0
        cnt=0
        while i+1 < nn:
            if tmp_dataset[i].name[1] == tmp_dataset[i+1].name[1] and tmp_dataset[i].name[0] != tmp_dataset[i+1].name[0]:
                self.fulldataset.append(tmp_dataset[i])
                self.fulldataset.append(tmp_dataset[i+1])
                i += 2
            else:
                cnt+=1
                i += 1


    def reshuffle_grouped_dataset(self):
        # return
        if self.shuffle:
            self.reshuffle()
            tmp = group_same_size(self.fulldataset)
            self.batched_data = None
            self.batched_data = batch_same_size(tmp, self.batch_size)



def collate(batch):
    return batch[0]


def RS_datawork(
    name: str,
    root: str,
    batch_size: List[int],
    transform,
    max_neighbor,
    device,
    subgraph_cutoff: float = -1.0,
    cutoff: float = -1.0,
    extend_r=1.0,
    **kwargs,
    ):

    from functools import partial
    print("Loading dataset...")
    data = torch.load(f"{root}/processed/rs/rs_full.pt")
    print("Finished.")
    chiral_dataset_group_partial = partial(batched_chiral, root=root, data=data)
    
    if transform is not None:
        transform_ = transform(
            dname=name,
            subgraph_cutoff=subgraph_cutoff,
            cutoff=cutoff,
            extend_r=extend_r,
            max_neighbor=max_neighbor,
            chiral=True
            )
    else:
        transform_ = None

    train_num, val_num, test_num = 326865, 70099, 69719

    # the full dataset is concatenated by train, val and test dataset, therefore sequential indices are used.
    train_indices = [x for x in range(train_num)]
    val_indices = [x for x in range(train_num, train_num + val_num)]
    test_indices = [x for x in range(train_num + val_num, train_num + val_num + test_num)]
    
    
    train_batch_size, val_batch_size, test_batch_size = batch_size
    
    train_dataset, val_dataset, test_dataset = (
        chiral_dataset_group_partial(indices=train_indices, batch_size=train_batch_size, transform=transform_, shuffle=True),
        chiral_dataset_group_partial(indices=val_indices, batch_size=val_batch_size, transform=transform_, shuffle=True),
        chiral_dataset_group_partial(indices=test_indices, batch_size=test_batch_size, transform=transform_, shuffle=True),
    )

    # get dataloaders, note that batch size must be 1 because batch is already divided in dataset.
    train_dataloader, val_dataloader, test_dataloader = (
        DataLoader(train_dataset, num_workers=12, batch_size=1, persistent_workers=False, shuffle=True, collate_fn=collate), 
        DataLoader(val_dataset, num_workers=12, batch_size=1, persistent_workers=False, shuffle=False, collate_fn=collate), 
        DataLoader(test_dataset, num_workers=12, batch_size=1, persistent_workers=False, shuffle=False, collate_fn=collate),
    )

    # calculate global mean and std
    global_y_mean, global_y_std = None, None

    # TODO: Split and get val dataset
    return train_dataloader, val_dataloader, test_dataloader, global_y_mean, global_y_std

