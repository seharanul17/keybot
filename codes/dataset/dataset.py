import os
import copy
from munch import Munch
import numpy as np

import torch

import dataset.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split:str, data:list, off_train_aug):

        config.DICT_KEY = Munch.fromDict({})
        config.DICT_KEY.IMAGE = 'image'
        config.DICT_KEY.BBOX = 'bbox_{}'.format(config.Dataset.image_size[0])
        config.DICT_KEY.POINTS = 'points_{}'.format(config.Dataset.image_size[0])
        config.DICT_KEY.RAW_SIZE = 'raw_size_row_col'
        config.DICT_KEY.PSPACE = 'pixelSpacing'
        # init
        self.config = config
        self.split = split
        self.data = data

        # for debugging
        if self.config.debug:
            self.data = data[:10]

        if config.Model.bbox_predictor:
            import albumentations
            bbox_params = albumentations.BboxParams(format='coco', label_fields=['category_ids'])
        else:
            bbox_params = None

        if self.split == 'train':
            if off_train_aug:
                self.transformer = transforms.fake
            else:
                if self.config.Dataset.aug.type == 'aug1':
                    self.transformer = transforms.aug1(self.config.Dataset.aug.p, bbox_params)
                elif self.config.Dataset.aug.type == 'aug2':
                    self.transformer = transforms.aug2(self.config.Dataset.aug.p, bbox_params)
                elif self.config.Dataset.aug.type == 'subpixel_aug':
                    self.transformer = transforms.subpixel_aug(self.config.Dataset.image_size, bbox_params)
                elif self.config.Dataset.aug.type == 'aug_isbi_cnn19':
                    self.transformer = transforms.aug_isbi_cnn19(self.config.Dataset.image_size, bbox_params)
                elif self.config.Dataset.aug.type == 'aug_isbi_cnn19_flip':
                    self.transformer = transforms.aug_isbi_cnn19_flip(self.config.Dataset.image_size, bbox_params)
                elif self.config.Dataset.aug.type == 'aug_new1':
                    self.transformer = transforms.aug_new1(self.config.Dataset.image_size, bbox_params)
                elif self.config.Dataset.aug.type == 'fake':
                    self.transformer = transforms.fake
                else:
                    print('ERROR ::: no such data transformation')
                    raise
        else:
            self.transformer = transforms.fake

        self.loadimage = self.load_npy

        if self.config.Dataset.NAME.lower() == 'isbi':
            for item in self.data:
                item[self.config.DICT_KEY.PSPACE] = [0.1, 0.1]
                item[self.config.DICT_KEY.RAW_SIZE] = [2400, 1935] #row, col
                item[self.config.DICT_KEY.IMAGE] = item[self.config.DICT_KEY.IMAGE] + '.npy'
        else:
            for item in self.data:
                item[self.config.DICT_KEY.IMAGE] = item[self.config.DICT_KEY.IMAGE].replace('.png', '.npy')

        if self.config.test_pixelspacing_one:
            for item in self.data:
                item[self.config.DICT_KEY.PSPACE] = [1.0, 1.0]

        if self.split == 'test' and self.config.split_test_dataset is not None:

            split_num = self.config.split_test_dataset # 0~9
            split_data_len = len(self.data)//10

            if split_num != 9:
                print('Data range: {}~{}'.format(split_data_len*split_num,split_data_len*(split_num+1)))
                split_data = self.data[split_data_len*split_num:split_data_len*(split_num+1)]
            else:
                # split_num == 9
                print('Data range: {}~{}'.format(split_data_len*split_num,len(self.data)))
                split_data = self.data[split_data_len*split_num:]
            self.data = split_data


        if self.config.Model.bbox_predictor:
            self.config.DICT_KEY.BBOX = 'bbox_{}'.format(config.Dataset.image_size[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # indexing
        item = self.data[index]

        # image load
        img_path = os.path.join(self.config.PATH.DATA.IMAGE, item[self.config.DICT_KEY.IMAGE])
        img, row, column = self.loadimage(img_path, item[self.config.DICT_KEY.RAW_SIZE])

        # pixel spacing
        pspace_list = item[self.config.DICT_KEY.PSPACE] # row, column
        raw_size_and_pspace = torch.tensor([row, column] + pspace_list)

        # points load (13,2) (column, row)==(xy)
        coords = copy.deepcopy(item[self.config.DICT_KEY.POINTS])
        coords.append([1.0,1.0])

        if self.config.Model.bbox_predictor:
            bbox = item[self.config.DICT_KEY.BBOX]
            transformed = self.transformer(image=img, keypoints=coords, bboxes=[bbox], category_ids=[1])
            img, coords = transformed["image"], transformed["keypoints"]
            bbox = torch.tensor(transformed['bboxes'])

            # [minx, miny, w, h] -> [x1, y1, x2, y2] (x: column, y: row)
            bbox = torch.tensor(bbox, dtype=torch.float)
            bbox[:, 2] += bbox[:, 0]
            bbox[:, 3] += bbox[:, 1]
            # bbox normalize (->0~1)
            bbox[:, 0] = bbox[:, 0] / self.config.Dataset.image_size[0] # x = col
            bbox[:, 1] = bbox[:, 1] / self.config.Dataset.image_size[1] # y = row
            bbox[:, 2] = bbox[:, 2] / self.config.Dataset.image_size[0] # x = col
            bbox[:, 3] = bbox[:, 3] / self.config.Dataset.image_size[1] # y = row
            additional = bbox
        else:
            transformed = self.transformer(image=img, keypoints=coords)
            img, coords = transformed["image"], transformed["keypoints"]
            additional = torch.tensor([]) # None

        coords = np.array(coords)

        # np array to tensor (800, 640)=(row, col)
        img = torch.tensor(img, dtype=torch.float)
        img = img.permute(2, 0, 1)
        img /= 255.0 # 0~255 to 0~1
        img = img * 2 - 1 # 0~1 to -1~1


        # 13, 2 (1024 x 1024 unit) (column, row) > (row, column)
        coords = torch.tensor(copy.deepcopy(coords[:, ::-1]), dtype=torch.float)
        if self.config.Dataset.NAME == 'Cephalo':
            # flip -> True
            morph_loss_mask = (coords[-1] == torch.tensor([1.0, 1.0], dtype=torch.float)).all() or (coords[-1] == torch.tensor([1.0, img.shape[2]-1.0], dtype=torch.float)).all()
        else:
            # flip -> False
            morph_loss_mask = (coords[-1] == torch.tensor([1.0, 1.0], dtype=torch.float)).all()

        coords = coords[:-1]

        # hint
        if self.split == 'train':
            # random hint
            num_hint = np.random.choice(range(self.config.Dataset.num_keypoint), size=None, p=self.config.Hint.num_dist)
            hint_indices = np.random.choice(range(self.config.Dataset.num_keypoint), size=num_hint, replace=False) #[1,2,3]
        else:
            hint_indices = None

        return img_path, img, raw_size_and_pspace, hint_indices, coords, additional, index, morph_loss_mask

    def load_npy(self, img_path, size=None):
        img = np.load(img_path)
        if size is not None:
            row, column = size
        else:
            row, column = img.shape[:2]
        return img, row, column

def collate_fn(batch):
    batch = list(zip(*batch))
    batch_dict = {
        'input_image_path':batch[0], # list
        'input_image':torch.stack(batch[1]),
        'label':{'morph_offset':torch.stack(batch[2]),
                'coord': torch.stack(batch[4]),
                'morph_loss_mask':torch.stack(batch[7])
                 },
        'pspace':torch.stack(batch[2]),
        'hint':{'index': list(batch[3])},
        'additional': torch.stack(batch[5]),
        'index':list(batch[6])
    }
    return Munch.fromDict(batch_dict)