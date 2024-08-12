import copy

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from functools import partial
import numpy as np


def get_func_pseudo_label():
    return get_pseudo_label

def get_pseudo_label(batch_image, batch_pred_keypoint, suggest_model, pseudo_label_model, suggestion_cls_train_dataset, refine_train_dataset, max_hint=None):
    # batch_image: (b,3,512,256)
    # batch_pred_keypoint: (b, 68, 2)

    batch_suggestion = suggest_model.inference(batch_image, batch_pred_keypoint, suggestion_cls_train_dataset)
    hint_index, hint_coord, full_recon_result = pseudo_label_model.inference(batch_image, batch_pred_keypoint, batch_suggestion, refine_train_dataset, max_hint)

    return hint_index, hint_coord, full_recon_result


def heatmap2hargmax_coord(heatmap):
    b, c, row, column = heatmap.shape
    heatmap = heatmap.reshape(b, c, -1)
    max_indices = heatmap.argmax(-1)
    keypoint = torch.zeros(b, c, 2, device=heatmap.device)
    # keypoint[:, :, 0] = torch.floor(torch.div(max_indices, column)) # old environment
    keypoint[:, :, 0] = torch.div(max_indices, column, rounding_mode='floor')
    keypoint[:, :, 1] = max_indices % column
    return keypoint



class HeatmapMaker():
    def __init__(self, image_size, std):
        self.image_size = image_size
        self.heatmap_std = std

    def make_gaussian_heatmap(self, mean, size, std):
        # coord : (13,2)
        mean = mean.unsqueeze(1).unsqueeze(1)
        var = std ** 2  # 64, 1
        grid = torch.stack(torch.meshgrid([torch.arange(size[0]), torch.arange(size[1])]), dim=-1).unsqueeze(0)
        grid = grid.to(mean.device)
        x_minus_mean = grid - mean  # 13, 1024, 1024, 2

        # (x-u)^2: (13, 512, 512, 2)  inverse_cov: (1, 1, 1, 1) > (13, 512, 512)
        gaus = (-0.5 * (x_minus_mean.pow(2) / var)).sum(-1).exp()
        # (13, 512, 512)
        return gaus

    def coord2heatmap(self, coord):
        # coord : (batch, 13, 2), torch tensor, gpu
        with torch.no_grad():
            heatmap = torch.stack([
                self.make_gaussian_heatmap(coord_item, size=self.image_size, std=self.heatmap_std) for coord_item in
                coord])
        return heatmap

class PseudoLabelModel(nn.Module):
    def __init__(self, n_keypoint=68, num_bones=17, image_size=(256,128), heatmap_std=5, backbone='deeplabv3'):
        super().__init__()
        self.backbone = backbone
        self.num_bones = num_bones
        self.image_size = image_size

        self.heatmap_maker = HeatmapMaker(image_size=image_size, std=heatmap_std)

        # make backbone
        in_channels = 3 + n_keypoint

        if backbone == 'deeplabv3':
            print('Backbone: deeplab v3')
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=n_keypoint)
            self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            raise NotImplemented

    def forward(self, x, coord, gt_hint_index=None, return_input=False):
        heatmap = self.heatmap_maker.coord2heatmap(coord)

        if gt_hint_index is not None:
            for batch_idx, item_hint_idx in enumerate(gt_hint_index):
                heatmap[batch_idx, item_hint_idx] = heatmap[batch_idx, item_hint_idx] * 10

        x = torch.cat([x, heatmap], dim=1)

        if return_input:
            return x

        x = self.model(x)['out'].sigmoid() # b, n_keypoint, 256, 128
        return x


    def fit(self, train_loader, val_loader, epoch=300):
        criterion = nn.BCELoss()

        optimizer = torch.optim.AdamW(params=self.parameters(), lr=1e-3)
        best_loss = 100000
        best_model = None
        self.cuda()
        epoch_train_loss_list = []
        epoch_val_loss_list = []

        for e in (range(epoch)):
            self.train()
            train_loss = []
            for i, batch in enumerate(train_loader):
                image, label, keypoint_neg, keypoint_pos = batch
                image = image.cuda()
                keypoint_neg = keypoint_neg.cuda()
                keypoint_pos = keypoint_pos.cuda()

                y = self(image, keypoint_neg)
                optimizer.zero_grad()
                with torch.no_grad():
                    target_heatmap = self.heatmap_maker.coord2heatmap(keypoint_pos)
                loss = criterion(y, target_heatmap)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            epoch_train_loss_list.append((train_loss))
            print(
                f'Epoch: [{e + 1}/{epoch}] Train loss [{(train_loss):.4f}]')

            self.eval()
            with torch.no_grad():
                val_MRE = []
                val_loss = []
                for i, batch in enumerate(val_loader):
                    image, label, keypoint_neg, keypoint_pos = batch
                    image = image.cuda()
                    keypoint_neg = keypoint_neg.cuda()
                    keypoint_pos = keypoint_pos.cuda()
                    y = self(image, keypoint_neg)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        target_heatmap = self.heatmap_maker.coord2heatmap(keypoint_pos)
                    loss = criterion(y, target_heatmap)
                    recon_out = heatmap2hargmax_coord(y).cuda()
                    mre = torch.sqrt(
                        ((recon_out - keypoint_pos)**2).sum(-1)
                    ).mean()
                    val_MRE.append(mre.detach().cpu().item())
                    val_loss.append(loss.item())
                val_MRE = np.mean(val_MRE)
                val_loss= np.mean(val_loss)
                epoch_val_loss_list.append(val_loss)

                if val_loss < best_loss:
                    print('Saved')
                    best_loss = val_loss
                    best_model = copy.deepcopy(self.state_dict())
                    best_epoch = e
            print(f'Epoch: [{e + 1}/{epoch}] Val loss [{(val_loss):.4f}] Val MRE [{val_MRE:.4f}]')
            print()
        self.eval()

        self.load_state_dict(best_model)
        print(best_epoch, best_loss)


    def inference(self, batch_image, batch_pred_keypoint, batch_suggestion, refine_train_dataset, max_hint=None):
        # batch_pred_keypoint: (b, 68, 2)
        # batch_suggestion: (b, 16)

        dataset_util = refine_train_dataset
        dataset_util.inference_mode = True
        self.cuda()
        self.eval()

        result = []

        recon_out_list = []
        for img, kp in zip(batch_image, batch_pred_keypoint):
            img, _, _, kp = dataset_util.transform(img, kp)
            img = img[None,:].cuda()
            kp = kp[None,:].cuda()

            with torch.no_grad():
                recon_heatmap = self(img, kp)  # batch, 68, 2
                recon_out = heatmap2hargmax_coord(recon_heatmap) * 2 # 256,128 -> 512, 256
            recon_out_list.append(recon_out)
        recon_out_list = torch.cat(recon_out_list)
        recon_out = recon_out_list


        full_recon_result = []
        for batch_idx in (range(len(batch_pred_keypoint))):
            batch_recon_result = []
            pseudo_label_one_item = []
            for bone_idx in range(self.num_bones-1):
                for keypoint_idx in range(8):
                    if batch_suggestion[batch_idx, bone_idx, keypoint_idx]:
                        pseudo_label_one_item.append(
                            [bone_idx * 4 + keypoint_idx, recon_out[batch_idx, bone_idx * 4 + keypoint_idx].cpu()])
                        batch_recon_result.append(recon_out[batch_idx].cpu())

            result.append(pseudo_label_one_item)
            full_recon_result.append(batch_recon_result)

        hint_index, hint_coord = self.pseudo_label2hint_indexNhint_coord(result, max_hint)

        return hint_index, hint_coord, full_recon_result

    def pseudo_label2hint_indexNhint_coord(self, pseudo_label, max_hint):
        hint_index = []
        hint_coord = []

        for b_idx, item in enumerate(pseudo_label):
            if len(item) == 0:
                index_one_item = None
                coord_one_item = None
            else:
                index_one_item = []
                coord_one_item = []
                for index, coord in item:
                    index_one_item.append(index)
                    coord_one_item.append(coord.tolist())

                    if max_hint is not None and len(index_one_item) == max_hint:
                        break

            hint_index.append(index_one_item)
            hint_coord.append(coord_one_item)
        return hint_index, hint_coord
