import yaml
import json
import pickle
from munch import Munch
import os
import torch
from misc.metric import heatmap2hargmax_coord
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

random_distance_lambda = [0.02694842405617237,
 0.09876708686351776,
 0.07010390609502792,
 0.05175469443202019,
 0.031244121491909027,
 0.0634990856051445,
 0.09300514310598373,
 0.07078634202480316,
 0.06182816997170448,
 0.03700973838567734]
random_angle_lambda = [0.0006802229909226298,
 0.0018467222107574344,
 0.000733348832000047,
 0.0004263218434061855,
 0.006054277066141367,
 0.005081675481051207,
 0.0003568786196410656,
 0.005725877825170755,
 0.009861857630312443,
 0.0027906245086342096]

class SaveManager(object):
    def __init__(self, arg):
        self.read_config(arg)
        self.unspecified_configs_to_default()
        self.save_config()



        self.config.MISC.device = torch.device('cuda')
        self.config_name2value()

    def config_name2value(self):
        # name -> values
        with open('./morph_pairs/{}/{}.json'.format(self.config.Dataset.NAME, self.config.Morph.pairs), 'r') as f:
            self.config.Morph.pairs = json.load(f)

        if self.config.Hint.num_dist == 'isbi1':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096, 0, 0, 0, 0, 0, 0]
        elif self.config.Hint.num_dist == 'datset16_1':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096]+[0 for _ in range(68-13)]
        elif self.config.Hint.num_dist == 'cep_1':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096]
        elif self.config.Hint.num_dist == 'dataset16':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512,
                                         1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096] + [0 for _ in range(68 - 13)]
        elif self.config.Hint.num_dist == 'buu':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512,
                                         1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096, 0, 0, 0, 0, 0, 0, 0]
        elif self.config.Hint.num_dist == 'buula':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512,
                                         1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif self.config.Hint.num_dist == 'isbi':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512,
                                         1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096, 0, 0, 0, 0, 0, 0]
        elif self.config.Hint.num_dist == 'cep':
            self.config.Hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512,
                                         1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096]
        else:
            raise NotImplementedError

    def read_config(self, arg):
        # load config
        if arg.only_test_version:
            config_path = '../save/{}/config.yaml'.format(arg.only_test_version)
        else:
            config_path = './config/{}.yaml'.format(arg.config)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.config = Munch.fromDict(config)

        if not arg.only_test_version:
            os.makedirs('../save/', exist_ok=True)
            self.config.version = "AASCE_interactive_keypoint_estimation"
            os.makedirs('../save/{}'.format(self.config.version), exist_ok=True)
        else:
            self.config.version = arg.only_test_version

        # set path and write version
        self.set_config_path()
        # self.write_log('Version : {}'.format(self.config.version), n_mark=16)

        # update arg to config
        items = arg.__dict__.items() if 'namespace' in str(arg.__class__).lower() else arg.items()
        for key, value in items:
            self.config[key] = value

        # random search
        if arg.random_morph_distance_lambda >=0:
            self.config.Morph.distance_lambda = random_distance_lambda[arg.random_morph_distance_lambda]
            self.write_log('distance lambda replaced : [{}]'.format(self.config.Morph.distance_lambda), n_mark=4)
        if arg.random_morph_angle_lambda >= 0:
            self.config.Morph.angle_lambda = random_angle_lambda[arg.random_morph_angle_lambda]
            self.write_log('angle lambda replaced : [{}]'.format(self.config.Morph.angle_lambda), n_mark=4)

        # only for f-BRS
        if arg.fBRS_test_mode is not None:
            self.config.Model.mode = arg.fBRS_test_mode
            self.write_log('Test f-BRS : {} mode'.format(self.config.Model.mode), n_mark=8)

        if arg.save_test_prediction:
            self.predictions_for_save = Munch({})

        if arg.subpixel_inference > 0:
            self.config.Dataset.subpixel_decoding_patch_size = arg.subpixel_inference
            self.config.Dataset.subpixel_decoding = True
            # self.write_log('Subpixel inference - patch size: {}'.format(arg.subpixel_inference), n_mark=8)

        if arg.use_prev_heatmap_inference:
            self.config.Model.use_prev_heatmap = True
        else:
            self.config.Model.use_prev_heatmap = False
        # if self.config.Model.use_prev_heatmap:
        #     self.write_log('Use Previous Heatmap when inference time', n_mark=8)

        if arg.use_prev_heatmap_only_for_hint_index:
            self.config.Model.use_prev_heatmap_only_for_hint_index = True
        else:
            self.config.Model.use_prev_heatmap_only_for_hint_index = False

        # if self.config.Model.use_prev_heatmap_only_for_hint_index:
        #     self.write_log('Use previous heatmap only for hint index, else zero', n_mark=8)


        self.config.Revision = Munch({})
        if arg.revision_type == 'random':
            self.config.Revision.type = 'random'
            self.write_log('Inference Revision Type - random')
        elif arg.revision_type == 'worst_3':
            self.config.Revision.type = 'worst_3'
            self.write_log('Inference Revision Type - worst 3')
        else:
            self.config.Revision.type = 'worst'


        if arg.batch_size is not None:
            self.config.Train.batch_size = arg.batch_size

    def write_log(self, text, n_mark=0, save_flag=True):
        log = '{} {} {}'.format('='*n_mark, text, '='*n_mark)
        print(log)
        if save_flag:
            with open(self.config.PATH.LOG_PATH, 'a+') as f:
                f.write('{}\n'.format(log))
        return

    def set_config_path(self):
        self.config.PATH.LOG_PATH = '../save/{}/log.txt'.format(self.config.version)
        self.config.PATH.CONFIG_PATH = '../save/{}/config.yaml'.format(self.config.version)
        self.config.PATH.MODEL_PATH = '../save/{}/model.pth'.format(self.config.version)
        self.config.PATH.post_RESULT_PATH = '../save/{}/post_result.pickle'.format(self.config.version)
        self.config.PATH.manual_RESUAL_PATH = '../save/{}/manual_result.pickle'.format(self.config.version)
        self.config.PATH.PREDICTION_RESULT_PATH = '../save/{}/predictions.pickle'.format(self.config.version)

    def save_config(self):
        with open(self.config.PATH.CONFIG_PATH, 'w') as f:
            yaml.dump(self.config.toDict(), f)
        return

    def load_model(self):
        best_save = torch.load(self.config.PATH.MODEL_PATH, map_location=torch.device('cpu'))
        best_param = best_save['model']
        best_epoch = best_save['epoch']
        return best_param, best_epoch, None

    def save_model(self, epoch, param, metric):
        save_dict = {
            'model': param,
            'epoch':epoch,
            'metric':metric
        }
        torch.save(save_dict, self.config.PATH.MODEL_PATH)
        return

    def save_metric(self, metric, manual_metric=None):
        with open(self.config.PATH.post_RESULT_PATH, 'wb') as f:
            pickle.dump(metric, f)
        if manual_metric is not None:
            with open(self.config.PATH.manual_RESUAL_PATH, 'wb') as f:
                pickle.dump(manual_metric, f)

    def save_test_prediction(self, save_predictions_path=None):
        if save_predictions_path:
            path = f'{save_predictions_path}/predictions.pickle'
        else:
            path = self.config.PATH.PREDICTION_RESULT_PATH
        with open(path, 'wb') as f:
            pickle.dump(self.predictions_for_save, f)

    def add_test_prediction_for_save(self, batch, post_processing_pred, manual_pred, n_hint, post_metric_manager, manual_metric_manager):
        manual_pred.hargmax_coord = heatmap2hargmax_coord(manual_pred.heatmap).detach()
        post_sargmax_mm_MRE = post_metric_manager.running_metric['sargmax_mm_MRE'][-1].detach().cpu().numpy()
        post_hargmax_mm_MRE = post_metric_manager.running_metric['hargmax_mm_MRE'][-1].detach().cpu().numpy()
        manual_sargmax_mm_MRE = manual_metric_manager.running_metric['sargmax_mm_MRE'][-1].detach().cpu().numpy()
        manual_hargmax_mm_MRE = manual_metric_manager.running_metric['hargmax_mm_MRE'][-1].detach().cpu().numpy()
        for b in range(len(batch.label.coord)):
            name = 'batch{}_hint{}'.format(batch.index[b], n_hint)
            self.predictions_for_save[name] = Munch({})
            self.predictions_for_save[name].post = Munch({})
            self.predictions_for_save[name].post.sargmax_coord = post_processing_pred.sargmax_coord[b].detach().cpu().numpy()
            self.predictions_for_save[name].post.hargmax_coord = post_processing_pred.hargmax_coord[b].detach().cpu().numpy()
            self.predictions_for_save[name].manual = Munch({})
            self.predictions_for_save[name].manual.sargmax_coord = manual_pred.sargmax_coord[b].detach().cpu().numpy()
            self.predictions_for_save[name].manual.hargmax_coord = manual_pred.hargmax_coord[b].detach().cpu().numpy()

            self.predictions_for_save[name].label = Munch({})
            self.predictions_for_save[name].label.coord = batch.label.coord[b].detach().cpu().numpy()
            self.predictions_for_save[name].pspace = batch.pspace[b].detach().cpu().numpy()

            self.predictions_for_save[name].hint = Munch({})
            if torch.is_tensor(batch.hint.index[b]):
                self.predictions_for_save[name].hint.index = batch.hint.index[b].detach().cpu().numpy()
            else:
                self.predictions_for_save[name].hint.index = batch.hint.index[b]

            self.predictions_for_save[name].metric = Munch({})
            self.predictions_for_save[name].metric.post = Munch({})
            self.predictions_for_save[name].metric.post.sargmax_mm_MRE = post_sargmax_mm_MRE[b]
            self.predictions_for_save[name].metric.post.hargmax_mm_MRE = post_hargmax_mm_MRE[b]
            self.predictions_for_save[name].metric.manual = Munch({})
            self.predictions_for_save[name].metric.manual.sargmax_mm_MRE = manual_sargmax_mm_MRE[b]
            self.predictions_for_save[name].metric.manual.hargmax_mm_MRE = manual_hargmax_mm_MRE[b]



    def get_new_exp_num(self):
        save_path = '../save/'
        save_folder_names = os.listdir(save_path)
        max_exp_num = 0
        new_exp_num = '{:05d}'.format(max_exp_num+1)

        return new_exp_num

    def unspecified_configs_to_default(self):
        if self.config.Dataset.get('subpixel_decoding',None) is None:
            self.config.Dataset.subpixel_decoding = False
        if self.config.Dataset.get('subpixel_decoding_patch_size',None) is None:
            self.config.Dataset.subpixel_decoding_patch_size = 5
        if self.config.Dataset.get('heatmap_encoding_maxone',None) is None:
            self.config.Dataset.heatmap_encoding_maxone = False
        if self.config.Dataset.get('subpixel_heatmap_encoding', None) is None:
            self.config.Dataset.subpixel_heatmap_encoding = False
        if self.config.Model.get('subpixel_decoding_coord_loss', None) is None:
            self.config.Model.subpixel_decoding_coord_loss = False
        if self.config.Model.get('facto_heatmap', None) is None:
            self.config.Model.facto_heatmap = False
        if self.config.Model.get('HintEncoder', None) is None:
            self.config.Model.HintEncoder = Munch({})
        if self.config.Model.HintEncoder.get('dilation', None) is None:
            self.config.Model.HintEncoder.dilation = 5
        if self.config.Model.get('Decoder', None) is None:
            self.config.Model.Decoder = Munch({})
        if self.config.Model.Decoder.get('dilation', None) is None:
            self.config.Model.Decoder.dilation = 5
        if self.config.Dataset.get('label_smoothing', None) is None:
            self.config.Dataset.label_smoothing = False
        if self.config.Model.get('MSELoss', None) is None:
            self.config.Model.MSELoss = 0.0
        if self.config.Dataset.get('heatmap_max_norm',None) is None:
            self.config.Dataset.heatmap_max_norm = False
        if self.config.Model.get('input_padding') is None:
            self.config.Model.input_padding = None
        if self.config.Morph.get('cosineSimilarityLoss') is None:
            self.config.Morph.cosineSimilarityLoss = False
        if self.config.Morph.get('threePointAngle') is None:
            self.config.Morph.threePointAngle = False
        if self.config.MISC.get('free_memory',None) is None:
            self.config.MISC.free_memory = False
        if self.config.Model.get('bbox_predictor', None) is None:
            self.config.Model.bbox_predictor = False
        if self.config.Morph.get('distance_l1', None) is None:
            self.config.Morph.distance_l1 = False
        if self.config.Model.get('SE_maxpool',None) is None:
            self.config.Model.SE_maxpool = False
        if self.config.Model.get('SE_softmax', None) is None:
            self.config.Model.SE_softmax = False
        if self.config.Model.get('use_prev_heatmap', None) is None:
            self.config.Model.use_prev_heatmap = False
        if self.config.Model.get('no_iterative_training', None) is None:
            self.config.Model.no_iterative_training = False # RITM에만 적용
        if self.config.Morph.get('coord_use', None) is None:
            self.config.Morph.coord_use = False
        if self.config.Model.get('L1Loss', None) is None:
            self.config.Model.L1Loss = False
        if self.config.Model.get('L2Loss', None) is None:
            self.config.Model.L2Loss = False


class TensorBoardManager():
    def __init__(self, save_manager):
        tensorboard_path = '../tensorboard/{}/{}/'.format(save_manager.config.Dataset.NAME, save_manager.config.version)
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)

        self.n_iter = {'train':0, 'val':0, 'test':0}
        self.n_epoch = {'train':0, 'val':0, 'test':0}

    def plot_image_heatmap(self, image, pred_heatmap, label_heatmap, epoch):
        # image: (batch, 3, H, W) -1 <= x <= 1
        # heatmap: (batch, num_keypoint, Height, Width) 0 <= x <= 1
        image_unNorm = image * 0.5 + 0.5

        pred_label_heatmap = torch.cat((pred_heatmap[:,:,None,:,:], label_heatmap[:,:,None,:,:], torch.zeros_like(pred_heatmap)[:,:,None,:,:]), dim=2)

        grids = [torchvision.utils.make_grid(
                                               torch.cat([image_unNorm[i].unsqueeze(0), pred_label_heatmap[i]], dim=0)
                                            )
                                        for i in range(pred_label_heatmap.shape[0])] # (1,3,H,W), (num_keypoint,3,H,W)

        for i, grid in enumerate(grids):
            text = 'Epoch [{}] - {} ::: Pred(red) Label(green)'.format(epoch, i+1)
            self.writer.add_image(text, grid)

    def plot_outlier_image_heatmap(self, image, pred_heatmap, label_heatmap, epoch):
        # image: (batch, 3, H, W) -1 <= x <= 1
        # heatmap: (batch, num_keypoint, Height, Width) 0 <= x <= 1
        image_unNorm = image * 0.5 + 0.5

        pred_label_heatmap = torch.cat((pred_heatmap[:,:,None,:,:], label_heatmap[:,:,None,:,:], torch.zeros_like(pred_heatmap)[:,:,None,:,:]), dim=2)

        grids = [torchvision.utils.make_grid(
                                               torch.cat([image_unNorm[i].unsqueeze(0), pred_label_heatmap[i]], dim=0)
                                            )
                                        for i in range(pred_label_heatmap.shape[0])] # (1,3,H,W), (num_keypoint,3,H,W)

        for i, grid in enumerate(grids):
            text = 'iter [{}] - outlier - {}'.format(self.n_iter['train'], i+1)
            self.writer.add_image(text, grid)

    def plot_model_param_histogram(self, model, epoch):
        for k, v in model.named_parameters():
            self.writer.add_histogram(k, v.data.cpu().reshape(-1), epoch)
        return

    def write_loss(self, loss, split):
        # loss: scalar
        # split: 'train', 'val', 'test'
        self.n_iter[split] += 1
        self.writer.add_scalar('Loss/{}'.format(split), loss, self.n_iter[split])

    def write_metric(self, metric, split):
        self.n_epoch[split] += 1
        for key in metric:
            self.writer.add_scalar('{}/{}'.format(key, split), metric[key], self.n_epoch[split])

