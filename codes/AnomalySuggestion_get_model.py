import numpy as np
from munch import Munch

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from util import SaveManager
from model import get_model
from misc.metric import MetricManager
from misc.train import Trainer
import copy

def test(trainer, save_manager, keypoint_test_loader, test_error_index,
                max_hint=2,
                get_pseudo_label=None,
                suggest_model=None,
                pseudo_label_model=None,
                suggestion_cls_train_dataset=None,
                refine_train_dataset=None,
                max_suggest_hint=None,
                max_suggest_iter=3,
         ):
    trainer.model.eval()
    suggest_model.eval()
    pseudo_label_model.eval()

    with torch.no_grad():
        post_metric_managers = [MetricManager(save_manager) for _ in
                                range(save_manager.config.Dataset.num_keypoint + 1)]
        test_error_msg = "Skip: "

        for i, batch in enumerate(tqdm(keypoint_test_loader)):
            if i in test_error_index:
                test_error_msg+=f'batch {i}, '
                continue
            batch = Munch.fromDict(batch)
            batch.is_training = False
            batch.prev_heatmap = torch.zeros_like(trainer.model.module.heatmap_maker.coord2heatmap(batch.label.coord))
            for n_hint in range(max_hint):
                initial_forward_manager=MetricManager(copy.deepcopy(save_manager))
                _out, _batch, _post_processing_pred = trainer.forward_batch(batch, metric_flag=True,
                                                                             average_flag=False,
                                                                             metric_manager=initial_forward_manager,
                                                                             return_post_processing_pred=True,
                                                                             pseudo_hint_index=None,
                                                                             pseudo_hint_coord=None)
                pseudohint_index = [[]]
                pseudohint_coord = [[]]
                pseudohint_dict = {}

                # suggest / refine iter
                for sss in range(max_suggest_iter):

                    input_image = _batch.input_image
                    if n_hint >= 1:
                        pred_keypoints = _post_processing_pred.sargmax_coord
                    else:
                        pred_keypoints = _out.pred.sargmax_coord

                    one_pseudohint_index, one_pseudohint_coord, one_full_recon_result = get_pseudo_label(input_image,
                                                                  pred_keypoints, suggest_model,
                                                                  pseudo_label_model, suggestion_cls_train_dataset,
                                                                  refine_train_dataset, max_suggest_hint)
                    if one_pseudohint_index[0] is not None:
                        if n_hint >= 1:
                            select_list_index=[[]]
                            select_list_coord=[[]]
                            for ot, one_item in enumerate(one_pseudohint_index[0]):
                                if one_item not in batch.hint.index[0]:
                                    select_list_index[0].append(one_item)
                                    select_list_coord[0].append(one_pseudohint_coord[0][ot])
                            one_pseudohint_index, one_pseudohint_coord = select_list_index, select_list_coord


                    if one_pseudohint_index[0] is None or len(one_pseudohint_index[0]) == 0:
                        # no suggest
                        one_pseudohint_index[0] = None
                        break
                    else:
                        _batch.prev_heatmap[0, one_pseudohint_index[0]] = _out.pred.heatmap[0, one_pseudohint_index[0]]
                        pseudohint_index, pseudohint_coord, pseudohint_full_recon_result = update_result(pseudohint_dict, one_pseudohint_index, one_pseudohint_coord,
                                      one_full_recon_result)

                        out, batch, post_processing_pred = _out, _batch, _post_processing_pred
                        temp_manager=MetricManager(copy.deepcopy(save_manager))
                        _out, _batch, _post_processing_pred = trainer.forward_batch(_batch, metric_flag=True,
                                                                                 average_flag=False,
                                                                                 metric_manager=temp_manager,
                                                                                 return_post_processing_pred=True,
                                                                                 pseudo_hint_index=pseudohint_index,
                                                                                 pseudo_hint_coord=pseudohint_coord,
                                                                                 )

                # iter end

                out, batch, post_processing_pred = trainer.forward_batch(batch, metric_flag=True,
                                                                             average_flag=False,
                                                                             metric_manager=post_metric_managers[
                                                                                 n_hint],
                                                                             return_post_processing_pred=True,
                                                                             pseudo_hint_index=pseudohint_index,
                                                                             pseudo_hint_coord=pseudohint_coord,
                                                                             )

                # ============================= model hint =================================
                worst_index = trainer.find_worst_pred_index(batch.hint.index, post_metric_managers,
                                                             save_manager, n_hint)
                # hint index update
                if n_hint == 0:
                    batch.hint.index = worst_index  # (batch, 1)
                else:
                    batch.hint.index = torch.cat((batch.hint.index, worst_index.to(batch.hint.index.device)),
                                                 dim=1)  # ... (batch, max hint)


                if save_manager.config.Model.use_prev_heatmap_only_for_hint_index:
                    for j in range(len(batch.hint.index)):
                        batch.prev_heatmap[j, batch.hint.index[j, -1:]] = out.pred.heatmap[
                            j, batch.hint.index[j, -1:]]

        post_metrics = [metric_manager.average_running_metric() for metric_manager in post_metric_managers]
    # save metrics
    for t in range(max_hint):
        print('(model ) Hint {} ::: MRE: {:.2f}'.format(t, post_metrics[t].sargmax_mm_MRE))
        print()

    return post_metrics


def update_result(pseudohint_dict, one_pseudohint_index, one_pseudohint_coord, one_full_recon_result):
    for a in range(len(one_pseudohint_index[0])):  # [[1,2,3]]
        index, coord = one_pseudohint_index[0][a], one_pseudohint_coord[0][a]
        full_r = one_full_recon_result[0][a]
        pseudohint_dict[index] = {'coord': coord, 'full_r': full_r}
    # print(pseudohint_dict)
    pseudohint_index = [[]]
    pseudohint_coord = [[]]
    pseudohint_full_recon_result = [[]]
    for key, value in pseudohint_dict.items():
        pseudohint_index[0] += [key]
        pseudohint_coord[0] += [value['coord']]
        pseudohint_full_recon_result[0] += [value['full_r']]
    return pseudohint_index, pseudohint_coord, pseudohint_full_recon_result




def get_keypoint_model(data='spineweb'):
    if data=='spineweb':
        model_version = 'AASCE_interactive_keypoint_estimation'
    else:
        raise NotImplementedError
    arg=Munch({})
    arg['config']='ForTest'
    arg['only_test_version']=model_version
    arg['random_morph_angle_lambda'] = -1
    arg['random_morph_distance_lambda'] = -1
    arg['fBRS_test_mode'] = None
    arg['save_test_prediction'] = False
    arg['debug'] = False
    arg['subpixel_inference'] = 15
    arg['test_pixelspacing_one']=False
    arg['use_prev_heatmap_inference'] = True
    arg['batch_size'] = 1
    arg['use_prev_heatmap_only_for_hint_index'] = True
    arg['revision_type']='worst'
    arg['save_test_prediction']=True

    save_manager = SaveManager(arg)
    save_manager.config.Model.bbox_predictor = False

    # model initialization
    device_ids = list(range(len(save_manager.config.MISC.gpu.split(','))))

    model = nn.DataParallel(get_model(save_manager), device_ids=device_ids)

    best_param, best_epoch, best_metric = save_manager.load_model()

    model.cpu()
    model.load_state_dict(best_param)
    model.to(save_manager.config.MISC.device)

    metric_manager = MetricManager(save_manager)

    trainer = Trainer(model, metric_manager)

    return trainer, save_manager


def get_test_data_loader(off_train_aug=False, data='spineweb'):
    if data == 'spineweb':
        return get_spineweb_data_loader(off_train_aug=off_train_aug)

def get_spineweb_data_loader(off_train_aug=False):
    from munch import Munch
    from util import SaveManager
    from dataset import get_dataloader

    arg = Munch({})
    arg['seed'] = "42"
    arg['gpu'] = '6'
    arg['config'] = 'ForTest'
    arg['random_morph_angle_lambda'] = -1
    arg['random_morph_distance_lambda'] = -1
    arg['fBRS_test_mode'] = None
    arg['save_test_prediction'] = False
    arg['debug'] = False
    arg['only_test_version'] = 'AASCE_interactive_keypoint_estimation'
    arg['subpixel_inference'] = 15
    arg['test_pixelspacing_one'] = False
    arg['use_prev_heatmap_inference'] = True
    arg['batch_size'] = 1
    arg['use_prev_heatmap_only_for_hint_index'] = False
    arg['revision_type'] = 'worst'
    save_manager = SaveManager(arg)
    save_manager.config.Model.bbox_predictor = False
    train_loader = get_dataloader(save_manager.config, 'train', off_train_aug)
    val_loader = get_dataloader(save_manager.config, 'val')
    test_loader = get_dataloader(save_manager.config, 'test')
    return train_loader, val_loader, test_loader
