import os
import time
import argparse
import datetime

import numpy as np
import random
from munch import Munch

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import copy

import matplotlib.pyplot as plt
import pickle
from PIL import Image
import json
import cv2
from tqdm.auto import tqdm

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


from PIL import ImageDraw
import torchvision


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


class SuggestionConvModel(nn.Module):
    def __init__(self, n_keypoint=8, image_size=(128,128), fake_model=False, backbone='resnet18'):
        super().__init__()
        self.fake_model = fake_model
        self.n_keypoint = n_keypoint
        self.backbone = backbone
        if fake_model:
            pass
        else:
            if backbone == 'resnet18':
                self.model = torchvision.models.resnet18(num_classes=n_keypoint)

                self.model.conv1 = nn.Conv2d(3 + n_keypoint, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif backbone == 'vit':
                from transformers import ViTConfig, ViTForImageClassification

                # Initializing a ViT vit-base-patch16-224 style configuration
                configuration = ViTConfig()
                configuration.image_size = image_size
                configuration.num_channels = 3 + n_keypoint
                configuration.num_labels = n_keypoint
                # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
                self.model = ViTForImageClassification(configuration)

            self.heatmap_maker = HeatmapMaker(image_size=image_size, std=5)

    def forward(self, x, coord=None):
        if self.fake_model:
            return torch.ones(x.shape[0], self.n_keypoint, device=x.device)
        else:
            heatmap = self.heatmap_maker.coord2heatmap(coord)
            x = torch.cat([x, heatmap], dim=1)

            # x: (b, 3, 128, 128)
            if self.backbone == 'resnet18':
                x = self.model(x)
            elif self.backbone == 'vit':
                x = self.model(x).logits
            else:
                raise NotImplemented

            x = x.sigmoid()
            return x

    def fit(self, train_loader, val_loader, epoch=300):
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=1e-3)
        best_acc = 0
        best_model = None
        self.cuda()
        epoch_loss_list = []

        for e in (range(epoch)):
            self.train()
            train_loss = []
            train_acc = []
            for i, batch in enumerate(train_loader):
                x, label, keypoint = batch
                x = x.float().cuda()
                label = label.cuda()
                keypoint = keypoint.cuda()
                y = self(x, keypoint)
                optimizer.zero_grad()
                loss = criterion(y, label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                train_acc.append(((y > 0.5) == label).float().mean().detach().cpu())
            epoch_loss_list.append(np.mean(train_loss))
            print(
                f'Epoch: [{e + 1}/{epoch}] Train loss [{np.mean(train_loss):.4f}] Train ACC [{np.mean(train_acc):.4f}]')

            self.eval()
            val_acc = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    x, label, keypoint = batch
                    x = x.float().cuda()
                    label = label.cuda()
                    keypoint = keypoint.cuda()
                    y = self(x, keypoint)
                    val_acc.append(((y > 0.5) == label).float().mean().detach().cpu())
                val_acc = np.mean(val_acc)

                if val_acc > best_acc:
                    best_model = copy.deepcopy(self.state_dict())
                    best_acc = val_acc
                    best_epoch = e
            print(f'Epoch: [{e + 1}/{epoch}] Val loss [{loss.item():.4f}] Val ACC [{np.mean(val_acc):.4f}]')
            print()
        self.eval()

        self.load_state_dict(best_model)
        print(best_epoch, best_acc)


    def inference(self, batch_image, batch_pred_keypoint, suggestion_cls_train_dataset, return_prob=False):
        # image: (b, 3, 512,256)
        # pred_keypoint: (b, 68, 2)
        dataset_util = suggestion_cls_train_dataset
        dataset_util.inference_mode = True

        self.cuda()
        self.eval()

        result = []
        prob_result =[]
        # batch 단위 인퍼런스
        for batch_idx in (range(len(batch_image))):
            image = batch_image[batch_idx]
            keypoints = batch_pred_keypoint[batch_idx]

            image_list = []
            label_list = []
            keypoints_list = []
            if dataset_util.aug_version == 1:
                for bone_idx in range(dataset_util.num_bones - 1):
                    image_item, label_item, keypoints_item = dataset_util.transform(image, keypoints, bone_idx=bone_idx)
                    image_list.append(image_item)
                    label_list.append(label_item)
                    keypoints_list.append(keypoints_item)
            elif  dataset_util.aug_version == 2:
                image_item, label_item, keypoints_item = dataset_util.transform(image, keypoints, bone_idx=None)
                image_list.append(image_item)
                label_list.append(label_item)
                keypoints_list.append(keypoints_item)
            image_list = torch.stack(image_list)
            label_list = torch.tensor(label_list)
            keypoints_list = torch.stack(keypoints_list)
            with torch.no_grad():
                suggestion_prob = self(image_list.cuda(), keypoints_list.cuda())
                result.append((suggestion_prob > 0.5).detach().cpu())
                prob_result.append(suggestion_prob.detach().cpu())

        result = torch.stack(result).detach().cpu()
        prob_result = torch.stack(prob_result)
        if return_prob:
            return result, prob_result
        else:
            return result
