import cv2
import albumentations
import numpy as np

def aug_isbi_cnn19(img_size, bbox_params=None):
    return albumentations.Compose([
        albumentations.augmentations.geometric.rotate.SafeRotate((-25, 25), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        albumentations.augmentations.geometric.resize.RandomScale((0.9, 1.2), p=0.5),
        albumentations.augmentations.transforms.ColorJitter(brightness=(0,0.25), contrast=(0,0.25), saturation=(0,0.25), hue=(0,0.25),
                                                            always_apply=False, p=0.5),
        albumentations.augmentations.geometric.resize.Resize(img_size[0], img_size[1], p=1)
    ], keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False),
       bbox_params=bbox_params)


def aug_isbi_cnn19_flip(img_size, bbox_params=None):
    return albumentations.Compose([
        albumentations.augmentations.geometric.rotate.SafeRotate((-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.augmentations.geometric.resize.RandomScale((0.9, 1.2), p=0.5),
        albumentations.RandomBrightnessContrast(),
        albumentations.augmentations.geometric.resize.Resize(img_size[0], img_size[1], p=1)
    ], keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False),
       bbox_params=bbox_params)

def aug_new1(img_size, bbox_params=None):
    return albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.augmentations.transforms.ColorJitter(brightness=(0,0.25), contrast=(0,0.25), saturation=(0,0.25), hue=(0,0.25),
                                                            always_apply=False, p=0.5),
        albumentations.augmentations.geometric.resize.Resize(img_size[0], img_size[1], p=1)
    ], keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False),
       bbox_params=bbox_params)


def aug1(p, bbox_params=None):
    return albumentations.Compose([
        albumentations.Cutout(num_holes=50, max_h_size=2, max_w_size=2, fill_value=0, p=p[0]),
        albumentations.HorizontalFlip(p=p[1]),
        albumentations.OneOf([
            albumentations.IAAAdditiveGaussianNoise(),
            albumentations.GaussNoise(),
        ], p=p[2]),
        albumentations.OneOf([
            albumentations.IAASharpen(),
            albumentations.RandomBrightnessContrast(),
        ], p=p[3])
    ], keypoint_params=albumentations.KeypointParams(format='xy'),
    bbox_params=bbox_params)




def aug2(p, bbox_params=None):
    return albumentations.Compose([
        albumentations.Cutout(num_holes=50, max_h_size=10, max_w_size=10, fill_value=0, p=p[0]),
        albumentations.HorizontalFlip(p=p[1]),
        albumentations.GaussNoise(p=p[2]),
        albumentations.IAASharpen(p=p[3]),
        albumentations.RandomBrightnessContrast(p=p[3]),
    ], keypoint_params=albumentations.KeypointParams(format='xy'),
    bbox_params=bbox_params)

def subpixel_aug(img_size, bbox_params=None):
    return albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5) ,
        albumentations.augmentations.geometric.rotate.Rotate((-15, 15), p=0.5),
        albumentations.augmentations.geometric.resize.RandomScale((0.85, 1.15), p=0.5),
        albumentations.augmentations.geometric.resize.Resize(img_size[0], img_size[1], p=1),
    ], keypoint_params=albumentations.KeypointParams(format='xy'),
    bbox_params=bbox_params)

class detecton2_aug():
    def __init__(self, img_size):
        self.augs= T.AugmentationList([
            T.RandomBrightness(0.9, 3),
            T.RandomFlip(prob=0.5),
            T.RandomRotation((-30, 30)),
            T.ResizeScale(0.85, 1.15, img_size[0], img_size[1])
        ])
    def __call__(self, image, coord):
        input = T.AugInput(image)
        transform = self.augs(input)
        image_transformed = input.image
        coord_transformed = transform.apply_coords(coord)
        return image_transformed, coord_transformed

    # def


def fake(**kwargs):
    return {**kwargs}

def points_normalize(points:np.array, dim0:float, dim1:float):
    points[..., 0] = points[..., 0] / dim0
    points[..., 1] = points[..., 1] / dim1

    return points