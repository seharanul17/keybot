from tqdm.auto import tqdm
import numpy as np
import torch
import json

from AnomalySuggestion_get_model import get_test_data_loader
from AnomalySuggestion_get_model import get_keypoint_model, test

from suggest_codes.get_suggest_dataset import SuggestionDataset
from suggest_codes.get_suggest_model import SuggestionConvModel
from suggest_codes.get_pseudo_generation_dataset_image_negative_sample import RefineDataset
from suggest_codes.get_pseudo_generation_model_image_heatmap import PseudoLabelModel, get_func_pseudo_label

dataset = 'spineweb'

def load_interactive_keypoint_model(dataset):
    keypoint_train_loader, keypoint_val_loader, keypoint_test_loader = get_test_data_loader(off_train_aug=True,data=dataset)
    trainer, save_manager = get_keypoint_model(data=dataset)
    
    return keypoint_train_loader, keypoint_val_loader, keypoint_test_loader, trainer, save_manager

suggest_model_path = '../save_suggestion/AASCE_suggestModel.pth'
pseudo_label_model_path = '../save_refine/AASCE_refineModel.pth'

keypoint_train_loader, keypoint_val_loader, keypoint_test_loader, trainer, save_manager = load_interactive_keypoint_model(dataset=dataset)

# load dataset, model
suggestion_cls_train_dataset = SuggestionDataset(keypoint_train_loader, inference_mode=True)

suggest_model = SuggestionConvModel()
suggest_model.load_state_dict(torch.load(suggest_model_path,map_location='cpu'))
suggest_model.eval()

refine_train_dataset = RefineDataset(keypoint_train_loader, split='train', inference_mode=True)
pseudo_label_model = PseudoLabelModel(n_keypoint=68, num_bones=17)
pseudo_label_model.load_state_dict(torch.load(pseudo_label_model_path,map_location='cpu'))
pseudo_label_model.eval()

get_pseudo_label = get_func_pseudo_label()

with open('error_list.json','r') as f:
    _,_,test_remove_list = json.load(f)

hh = None
for uc in [3]:
    print(' ========= With KeyBot - iter {} ========= '.format(uc))
    max_suggest_iter,max_suggest_hint=uc,hh

    post_metrics= test(trainer, save_manager, keypoint_test_loader, test_remove_list,
                        max_hint=5,
                        get_pseudo_label=get_pseudo_label,
                        suggest_model=suggest_model,
                        pseudo_label_model=pseudo_label_model,
                        suggestion_cls_train_dataset=suggestion_cls_train_dataset,
                        refine_train_dataset=refine_train_dataset,
                        max_suggest_hint=max_suggest_hint,
                        max_suggest_iter=max_suggest_iter,
                      )
