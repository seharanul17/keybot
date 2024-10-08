import os
import numpy as np
import json
import torch
from dataset.dataset import Dataset, collate_fn

def dataloader(config, split:str, data:list, off_train_aug=False):
    # split : 'train' / 'val' / 'test'
    dataset = Dataset(config=config, split=split, data=data, off_train_aug=off_train_aug)

    # loader
    if split == 'train':
        if off_train_aug:
            # print('off_train_aug is True, shuffle (x), drop_last (x), train_aug (x)')
            shuffle = False
            drop_last = False
        else:
            shuffle = True
            drop_last = True
    else:
        shuffle = False
        drop_last = False

    def _init_fn(worker_id):
        np.random.seed(config.seed + worker_id)

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, worker_init_fn=_init_fn,
                              batch_size=config.Train.batch_size, num_workers=config.MISC.num_workers, collate_fn=collate_fn, drop_last=drop_last)
    return data_loader

def get_split_data(config, split=None):
    if split == 'train':
        with open(os.path.join(config.PATH.DATA.TABLE, 'train.json'), 'r') as f:
            train_data = json.load(f)
        return train_data
    elif split =='val':
        with open(os.path.join(config.PATH.DATA.TABLE, 'val.json'), 'r') as f:
            val_data = json.load(f)
        return val_data

    elif split =='test':
        with open(os.path.join(config.PATH.DATA.TABLE, 'test.json'), 'r') as f:
            test_data = json.load(f)
        return test_data

def get_dataloader(config, split, off_train_aug=False):
    data = get_split_data(config, split)
    loader = dataloader(config, split, data, off_train_aug)
    return loader
