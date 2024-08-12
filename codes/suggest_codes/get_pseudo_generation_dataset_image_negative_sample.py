import numpy as np
import torch
import copy
import pickle
import torchvision
import json

with open('error_list.json','r') as f:
    remove_train_list,remove_val_list,_ = json.load(f)

class RefineDataset(torch.utils.data.Dataset):
    def __init__(self, keypoint_loader,
                 split='train',
                 aug_p=[0.3, 0.3, 0.2], inference_mode=False
                 ):
        batch_list = []
        assert keypoint_loader.batch_size == 1
        for idx, batch in enumerate(keypoint_loader):
            if split == 'train':
                image_name = batch.input_image_path[0].split('/')[-1]
                if image_name in remove_train_list:
                    # print('Train Del: ', image_name)
                    continue
            elif split == 'val':
                image_name = batch.input_image_path[0].split('/')[-1]
                if idx in remove_val_list:
                    # print('Validation Del: ', idx, image_name)
                    continue
            coord = batch.label.coord[0]
            image = batch.input_image[0]

            batch_list.append([batch.input_image_path[0], image, coord])
        self.batch_list = batch_list

        self.image_size = (256,128)
        self.resize_image = torchvision.transforms.Resize(self.image_size)


        self.num_bones = 17
        self.inference_mode = inference_mode
        self.aug_p = aug_p

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, idx):
        image = self.batch_list[idx][1]
        keypoints = self.batch_list[idx][2]

        image, label, keypoints_neg, keypoints_pos = self.transform(image, keypoints)

        label = torch.tensor(label).float()
        keypoints_neg = torch.tensor(keypoints_neg).float()
        keypoints_pos = torch.tensor(keypoints_pos).float()
        return image, label, keypoints_neg, keypoints_pos

    def transform(self, image, keypoints):
        if self.inference_mode:
            keypoints_neg, label, keypoints_pos = keypoints, np.zeros(68), keypoints
        else:
            p1, p2, p3= self.aug_p
            p = np.random.rand()
            if p > 1-p1:
                keypoints_neg, label, keypoints_pos = self.make_negative_sample_localize(keypoints)
            elif p > 1-p1-p2:
                keypoints_neg, label, keypoints_pos = self.make_negative_sample(keypoints)
            elif p > 1-p1-p2-p3:
                keypoints_neg, label, keypoints_pos = self.make_negative_sample_flip(keypoints)
            else:
                keypoints_neg, label, keypoints_pos = keypoints, np.zeros(68), keypoints

        image, keypoints_neg, keypoints_pos = self.resize_image_keypoitns(image, keypoints_neg, keypoints_pos)
        return image, label, keypoints_neg, keypoints_pos


    def resize_image_keypoitns(self, image, keypoints_neg, keypoints_pos=None):
        c, h, w = image.shape

        image = self.resize_image(image)

        new_keypoints_neg = torch.zeros_like(keypoints_neg)
        new_keypoints_neg[:, 0] = keypoints_neg[:, 0] / h * self.image_size[0]
        new_keypoints_neg[:, 1] = keypoints_neg[:, 1] / w * self.image_size[1]

        new_keypoints_pos = torch.zeros_like(keypoints_pos)
        new_keypoints_pos[:, 0] = keypoints_pos[:, 0] / h * self.image_size[0]
        new_keypoints_pos[:, 1] = keypoints_pos[:, 1] / w * self.image_size[1]

        return image, new_keypoints_neg, new_keypoints_pos

    def make_negative_sample_localize(self, keypoints):
        keypoints_neg = copy.deepcopy(keypoints)
        label = np.zeros(68)

        p = np.random.rand()
        if p > 2/3:
            p2 = np.random.rand()
            if p2 > 2/3:
                start_idx = np.random.randint(0,68)
                end_idx = 68
            elif p2 > 1/3:
                start_idx = 0
                end_idx = np.random.randint(start_idx + 1, 68 + 1)
            else:
                start_idx = np.random.randint(0,68)
                end_idx = np.random.randint(start_idx + 1, 68 + 1)

            if end_idx > 4:
                start_over_4 = max(4, start_idx)
                for replace_idx in range(start_over_4, end_idx):
                    keypoint_idx = replace_idx
                    replace_target_idx = keypoint_idx - 4
                    keypoints_neg[replace_idx] = keypoints[replace_target_idx]
                    label[replace_idx] = 1

            if start_idx < 4:
                end_under_4 = min(end_idx,4)
                for replace_idx in range(start_idx, end_under_4):
                    keypoints_neg[replace_idx] = keypoints[replace_idx] + (keypoints[replace_idx] - keypoints[replace_idx+4])
                    label[replace_idx] = 1
            return keypoints_neg, label.astype(np.float32), keypoints

        elif p > 1/3:
            p2 = np.random.rand()
            if p2 > 3 / 4:
                start_idx = np.random.randint(0, 68)
                end_idx = 68
            elif p2 > 2 / 4:
                start_idx = 0
                end_idx = np.random.randint(start_idx + 1, 68 + 1)
            elif p2 > 1 / 4:
                start_idx = np.random.randint(0, 68)
                end_idx = np.random.randint(start_idx + 1, 68 + 1)
            else:
                start_idx = 0
                end_idx = 68

            if start_idx < 64:
                end_under_64 = min(end_idx, 64)
                for replace_idx in range(start_idx, end_under_64):
                    keypoint_idx = replace_idx
                    replace_target_idx = keypoint_idx + 4
                    keypoints_neg[replace_idx] = keypoints[replace_target_idx]
                    label[replace_idx] = 1

            if end_idx > 64:
                start_over_64 = max(start_idx, 64)
                for replace_idx in range(start_over_64, end_idx):
                    keypoints_neg[replace_idx] = keypoints[replace_idx] + (
                                keypoints[replace_idx] - keypoints[replace_idx - 4])
                    label[replace_idx] = 1

            return keypoints_neg, label.astype(np.float32), keypoints
        else:
            return keypoints_neg, label.astype(np.float32), keypoints

    def make_negative_sample(self, keypoints):
        keypoints_neg = copy.deepcopy(keypoints)
        label = np.zeros(68)
        n_negative_sample = np.random.choice(10, size=None, p=[0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

        if n_negative_sample > 0:
            replace_idx_list = np.random.choice(68, size=n_negative_sample, replace=False)
            for replace_idx in replace_idx_list:
                keypoint_idx = replace_idx
                replace_target_idx = keypoint_idx + np.random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if replace_target_idx < 0:
                    replace_target_idx = replace_target_idx + 68
                elif replace_target_idx >= 68:
                    replace_target_idx = replace_target_idx - 68

                keypoints_neg[replace_idx] = keypoints[replace_target_idx]
                label[replace_idx] = 1
        return keypoints_neg, label.astype(np.float32), keypoints

    def make_negative_sample_flip(self, keypoints, p=0.9):
        keypoints_pos = keypoints
        keypoints_neg = copy.deepcopy(keypoints)
        label = np.zeros(68)

        for idx in range(0, 68, 2):
            if np.random.rand() < p:
                keypoints_neg[idx] = keypoints[idx + 1]
                keypoints_neg[idx + 1] = keypoints[idx]
                label[idx] = 1
                label[idx+1] = 1

        return keypoints_neg, label, keypoints_pos
