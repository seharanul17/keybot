import numpy as np
import torch
import torchvision
import json


with open('error_list.json','r') as f:
    remove_train_list,remove_val_list,_ = json.load(f)

class SuggestionDataset(torch.utils.data.Dataset):
    def __init__(self, keypoint_loader, inference_mode=False,
                 split='train', negative_sample_p=[0.2, 0.4, 0.2, 0.1, 0.1]):
        batch_list = []
        assert keypoint_loader.batch_size == 1
        for idx, batch in enumerate(keypoint_loader):
            image_name = batch.input_image_path[0].split('/')[-1]

            if split == 'train':
                if image_name in remove_train_list:
                    # print('Train Del: ', image_name)
                    continue
            elif split == 'val':
                if idx in remove_val_list:
                    # print('Validation Del: ', idx, image_name)
                    continue

            batch_list.append([batch.input_image_path[0], batch.input_image[0], batch.label.coord[0]])
        self.batch_list = batch_list

        self.num_bones = 17
        self.n_keypoint = 68
        self.image_size = (128, 128)
        self.inference_mode = inference_mode

        self.resize_image = torchvision.transforms.Resize(self.image_size)
        self.aug_version = 1
        self.negative_sample_p = negative_sample_p
        # print('negative_sample_p:',self.negative_sample_p)

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, idx):
        image = self.batch_list[idx][1]
        keypoints = self.batch_list[idx][2]

        if self.inference_mode:
            batch_image = []
            batch_label = []
            batch_keypoints = []
            for bone_idx in range(self.num_bones - 1):
                image_item, label_item, keypoints_item = self.transform(image, keypoints, bone_idx=bone_idx)
                batch_image.append(image_item)
                batch_label.append(label_item)
                batch_keypoints.append(keypoints_item)
            batch_image = torch.stack(batch_image)
            batch_label = torch.tensor(batch_label)
            batch_keypoints = torch.stack(batch_keypoints)
            return batch_image, batch_label, batch_keypoints

        else:
            image, label, keypoints = self.transform(image, keypoints)
            return image, label, keypoints

    def transform(self, image, keypoints, bone_idx=None):
        # image: 3, h, w (tensor)
        # keypoints: 68, 2 (tensor)
        keypoints_selected, bone_idx = self.select_two_bones(keypoints, bone_idx)
        keypoints_selected, label = self.make_negative_sample(keypoints_selected, keypoints, bone_idx)
        bbox = self.make_bbox_from_keypoints(keypoints_selected, full_image_size=image.shape[1:])
        image, keypoints_selected = self.crop(image, keypoints_selected, bbox)
        image, keypoints_selected = self.resize_image_keypoitns(image, keypoints_selected)


        return image, label, keypoints_selected



    def resize_image_keypoitns(self, image, keypoints):
        c, h, w = image.shape

        image = self.resize_image(image)

        new_keypoints = torch.zeros_like(keypoints)
        new_keypoints[:, 0] = keypoints[:, 0] / h * self.image_size[0]
        new_keypoints[:, 1] = keypoints[:, 1] / w * self.image_size[1]

        return image, new_keypoints

    def make_bbox_from_keypoints(self, keypoints, full_image_size=(512,256)):
        # keypoints: (8,2), row-col 
        row_min = int(keypoints[:, 0].min().item())
        row_max = int(keypoints[:, 0].max().item()) + 1
        col_min = int(keypoints[:, 1].min().item())
        col_max = int(keypoints[:, 1].max().item()) + 1

        if row_min >= full_image_size[0]:
            row_min = full_image_size[0]-1
        if row_max <= 0:
            row_max = 1
            row_min = 0
        if col_min >= full_image_size[1]:
            col_min = full_image_size[1]-1
        if col_max <= 0:
            col_max = 1
            col_min = 0
        if row_max <= row_min:
            row_max = row_min + 1
        if col_max <= col_min:
            col_max = col_min + 1

        bbox = [row_min, row_max, col_min, col_max]
        return bbox

    def crop(self, image, keypoints, bbox):
        image = image[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]  # channel, row, col
        new_keypoints = torch.zeros_like(keypoints)
        new_keypoints[:, 0] = keypoints[:, 0] - bbox[0]
        new_keypoints[:, 1] = keypoints[:, 1] - bbox[2]
        return image, new_keypoints

    def select_two_bones(self, keypoints, bone_idx=None):
        if bone_idx is None:
            bone_idx = np.random.randint(self.num_bones - 1)

        idx = torch.arange(bone_idx * 4, (bone_idx + 2) * 4, 1)
        return keypoints[idx], bone_idx

    def make_negative_sample(self, keypoints_selected, keypoints=None, bone_idx=None, p=8 / 9):
        n_keypoint = self.n_keypoint
        label = np.zeros(8)
        if self.inference_mode:
            return keypoints_selected, label.astype(np.float32)

        n_negative_sample = np.random.choice(len(self.negative_sample_p), size=None, p=self.negative_sample_p)  # 0,1,2,3,4
        if n_negative_sample > 0:
            replace_idx_list = np.random.choice(8, size=n_negative_sample, replace=False)
            for replace_idx in replace_idx_list:
                keypoint_idx = bone_idx * 4 + replace_idx
                replace_target_idx = keypoint_idx + np.random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if replace_target_idx < 0:
                    replace_target_idx = replace_target_idx + n_keypoint
                elif replace_target_idx >= n_keypoint:
                    replace_target_idx = replace_target_idx - n_keypoint  # v2

                keypoints_selected[replace_idx] = keypoints[replace_target_idx]
                label[replace_idx] = 1
        return keypoints_selected, label.astype(np.float32)
