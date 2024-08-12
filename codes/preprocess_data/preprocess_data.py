import os
import json
import numpy as np
from PIL import Image
import scipy.io
from tqdm.auto import tqdm
import copy

from albumentations import (
     Resize, Compose, KeypointParams, BboxParams
)

# ========================== params ====================================
base_path = './AASCE_rawdata/boostnet_labeldata/' # your data path
target_path = './AASCE_processed/'
val_images_path = './AASCE_processed/val_images.json'
# ======================================================================

image_path = os.path.join(base_path, 'data')
label_path = os.path.join(base_path, 'labels')

train_image_paths = []
test_image_paths = []

for name in os.listdir(image_path + '/training/'):
    p = os.path.join(image_path, 'training/', name)
    train_image_paths.append(p)

for name in os.listdir(image_path + '/test/'):
    p = os.path.join(image_path, 'test/', name)
    test_image_paths.append(p)

train_label_paths = [(i + '.mat').replace('boostnet_labeldata/data/', 'boostnet_labeldata/labels/') for i in
                     train_image_paths]
test_label_paths = [(i + '.mat').replace('boostnet_labeldata/data/', 'boostnet_labeldata/labels/') for i in
                    test_image_paths]

train_image_paths.sort()
test_image_paths.sort()
train_label_paths.sort()
test_label_paths.sort()


# select validation dataset (val size = 128)

with open(val_images_path,'r') as f:
    val_image_names = json.load(f)

val_idx = []
for i in range(len(train_image_paths)):
    if train_image_paths[i].split('/data/')[1] in val_image_names:
        val_idx.append(i)
train_idx = [i for i in range(len(train_image_paths)) if i not in val_idx]

val_image_paths = np.array(train_image_paths)[val_idx].tolist()
train_image_paths = np.array(train_image_paths)[train_idx].tolist()

val_label_paths = np.array(train_label_paths)[val_idx].tolist()
train_label_paths = np.array(train_label_paths)[train_idx].tolist()



# make json items
def make_data(image_paths, label_paths, base_path):
    data = []

    base_image_path= os.path.join(base_path,'data')
    for idx in range(len(image_paths)):
        item = {'image':None, 'label':None, 'raw_size_row_col':None, 'pixelSpacing':[1,1]}
        # indexing
        image_path = image_paths[idx]
        label_path = label_paths[idx]

        # loda data
        img = np.repeat(np.array(Image.open(image_path))[:,:,None], 3, axis=-1) # (row, col) -> (row,col,3)
        label = scipy.io.loadmat(label_path)['p2'] # x,y

        # make items
        item['image'] = image_path.replace(base_image_path+'/', '') # remove base path
        item['label'] = label.tolist()
        item['raw_size_row_col'] = (img.shape[0], img.shape[1])
        data.append(item)
    return data

train_data = make_data(train_image_paths, train_label_paths, base_path)
val_data = make_data(val_image_paths, val_label_paths, base_path)
test_data = make_data(test_image_paths, test_label_paths, base_path)





def inference_aug(img_size):
    return Compose([
        Resize(1024,512),
        Resize(img_size[0], img_size[1]),
    ], keypoint_params=KeypointParams(format='xy'),
       bbox_params=BboxParams(format='coco', label_fields=['category_ids']))


def get_bbox(points, img_shape_row_col):
    min_x = points[:,0].min() - 30
    min_y = points[:,1].min() - 30
    max_x = points[:,0].max() + 30
    max_y = points[:,1].max() + 30
    w = max_x-min_x
    h = max_y-min_y
    bbox = np.array([min_x, min_y, w, h])
    row, column = img_shape_row_col
    bbox = [np.clip(bbox[0], 0, column),
                np.clip(bbox[1], 0, row),
                np.clip(bbox[2], 0, column),
                np.clip(bbox[3], 0, row)]
    column_sum = bbox[0] + bbox[2]
    if column_sum > column:
        bbox[2] -= (column_sum-column + 1)

    row_sum = bbox[1] + bbox[3]
    if row_sum > row:
        bbox[3] -= (row_sum-row + 1)
    return bbox


# check keypoints are inside of the corresponding image
print('train')
del_list = []
for t, item in tqdm(enumerate(train_data)):
    row, col = item['raw_size_row_col']
    coord = np.array(item['label'])
    if coord[:, 1].max() > row:
        print('error:', t, '- exceed max y')
        del_list.append(t)

    if coord[:, 0].max() > col:
        print('error:', t, '- excced max x')
        del_list.append(t)
for t in del_list:
    del train_data[t]

print('val')
del_list = []
for t, item in tqdm(enumerate(val_data)):
    row, col = item['raw_size_row_col']
    coord = np.array(item['label'])
    if coord[:, 1].max() > row:
        print('error:', t, '- exceed max y')
        del_list.append(t)

    if coord[:, 0].max() > col:
        print('error:', t, '- excced max x')
        del_list.append(t)
for t in del_list:
    del val_data[t]

print('test')
del_list = []
for t, item in tqdm(enumerate(test_data)):
    row, col = item['raw_size_row_col']
    coord = np.array(item['label'])
    if coord[:, 1].max() > row:
        print('error:', t, '- exceed max y')
        del_list.append(t)

    if coord[:, 0].max() > col:
        print('error:', t, '- excced max x')
        del_list.append(t)
for t in del_list:
    del test_data[t]

for sizes in [(512, 256)]:
    aug = inference_aug((sizes[0], sizes[1]))
    size = sizes[0]
    for split in ['train', 'val', 'test']:
        print(split)

        if split == 'train':
            table = copy.deepcopy(train_data)
        elif split == 'val':
            table = copy.deepcopy(val_data)
        else:
            table = copy.deepcopy(test_data)

        for t, item in tqdm(enumerate(table)):
            # image load
            img_path = os.path.join(base_path, 'data', item['image'])
            img = np.repeat(np.array(Image.open(img_path))[:, :, None], 3, axis=-1)  # (row, col) -> (row,col,3)

            # points load (13,2) (column, row)
            points = item['label']

            bbox = get_bbox(np.array(points), item['raw_size_row_col'])

            transformed = aug(image=img, keypoints=points, bboxes=[bbox], category_ids=[1])
            img, points, bbox = transformed["image"], transformed["keypoints"], transformed["bboxes"]

            save_img_path = os.path.join(target_path, item['image'].replace('.jpg', '.npy'))
            os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
            np.save(save_img_path, img)
            item['points_{}'.format(size)] = points
            item['bbox_{}'.format(size)] = bbox
            item['image'] = item['image'].replace('.jpg', '.npy')

        with open('{}/{}.json'.format(target_path, split), 'w') as f:
            json.dump(table, f)