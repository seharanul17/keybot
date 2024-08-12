import torch
import os
from AnomalySuggestion_get_model import get_test_data_loader

from suggest_codes.get_suggest_dataset import SuggestionDataset
from suggest_codes.get_suggest_model import SuggestionConvModel
from suggest_codes.get_pseudo_generation_dataset_image_negative_sample import RefineDataset
from suggest_codes.get_pseudo_generation_model_image_heatmap import PseudoLabelModel

dataset = 'spineweb'

def load_dataset(dataset):
    keypoint_train_loader, keypoint_val_loader, _ = get_test_data_loader(off_train_aug=True,data=dataset)
    return keypoint_train_loader, keypoint_val_loader, _

suggest_model_path = '../save_suggestion/AASCE_suggestModel.pth'
pseudo_label_model_path = '../save_refine/AASCE_refineModel.pth'

keypoint_train_loader, keypoint_val_loader, _ = load_dataset(dataset=dataset)


# Detector
batch_size = 128
suggestion_cls_train_dataset = SuggestionDataset(keypoint_train_loader, split='train')
suggestion_cls_train_dataloader = torch.utils.data.DataLoader(suggestion_cls_train_dataset, batch_size=batch_size, pin_memory=True,
                                                     shuffle=True, drop_last=True)

suggestion_cls_val_dataset = SuggestionDataset(keypoint_val_loader, split='val')
suggestion_cls_val_dataloader = torch.utils.data.DataLoader(suggestion_cls_val_dataset, batch_size=batch_size, pin_memory=True,
                                                     shuffle=False, drop_last=False)

suggest_model = SuggestionConvModel()
suggest_model.fit(suggestion_cls_train_dataloader, suggestion_cls_val_dataloader)
os.makedirs(os.path.dirname(suggest_model_path),exist_ok=True)
torch.save(suggest_model.state_dict(), suggest_model_path)
print('Model saved: {}'.format(suggest_model_path))

del suggest_model
del suggestion_cls_val_dataloader
del suggestion_cls_val_dataset
del suggestion_cls_train_dataloader
del suggestion_cls_train_dataset


# Corrector
batch_size=4
refine_train_dataset = RefineDataset(keypoint_train_loader, split='train')
refine_train_dataloader = torch.utils.data.DataLoader(refine_train_dataset, batch_size=batch_size, pin_memory=True,
                                                      shuffle=True, drop_last=True)

refine_val_dataset = RefineDataset(keypoint_val_loader, split='val')
refine_val_dataloader = torch.utils.data.DataLoader(refine_val_dataset, batch_size=batch_size, pin_memory=True,
                                                    shuffle=False, drop_last=False)

pseudo_label_model = PseudoLabelModel(n_keypoint=68, num_bones=17, image_size=(256,128))
pseudo_label_model.fit(refine_train_dataloader, refine_val_dataloader, epoch=300)
os.makedirs(os.path.dirname(pseudo_label_model_path),exist_ok=True)
torch.save(pseudo_label_model.state_dict(), pseudo_label_model_path)
print('Model saved: {}'.format(pseudo_label_model_path))
