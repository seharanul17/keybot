from pytz import timezone
import time
import argparse
import datetime

import numpy as np
import random
import torch
import torch.nn as nn

from util import SaveManager, TensorBoardManager
from model import get_model
from dataset import get_dataloader
from misc.metric import MetricManager
from misc.optimizer import get_optimizer
from misc.train import Trainer

torch.set_num_threads(4)

def set_seed(config):
    if config.use_deterministic:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    return

def parse_args():
    parser = argparse.ArgumentParser(description='TMI experiments')
    parser.add_argument('--config', type=str, help='config name, required')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--random_morph_distance_lambda', type=int, default=-1)
    parser.add_argument('--random_morph_angle_lambda', type=int, default=-1)
    parser.add_argument('--only_test_version', type=str, default=None, help='If activated, there is no training. The number is the experiment number. => load & test model')
    parser.add_argument('--save_test_prediction', action='store_true', default=False, help='If activated, save test predictions at save path')
    parser.add_argument('--fBRS_test_mode', type=str, default=None, help='Use with --only_test_version and f-BRS model. ')
    parser.add_argument('--subpixel_inference', type=int, default=0, help='subpixel inference patch size')
    parser.add_argument('--test_pixelspacing_one', action='store_true', default=False, help='If activated, MRE_mm -> MRE_pixel at original image size')
    parser.add_argument('--debug', action='store_true', default=False, help='If activated, dataset -> data[:10]')
    parser.add_argument('--batch_size', default=None, type=int, help='if activated, batch size changes. WARNING: USE ONLY FOR BRS, F-BRS')
    parser.add_argument('--split_test_dataset', default=None, type=int, help='split dataset into 10 subsets. choose 0~9')
    parser.add_argument('--use_deterministic', action='store_true', default=False, help='use deterministic algorithms. This argument only works for f-BRS. CAUTION::: Change interpolate "bilinear" to "nearest"')
    parser.add_argument('--use_prev_heatmap_inference', action='store_true', default=False)
    parser.add_argument('--use_prev_heatmap_only_for_hint_index', action='store_true', default=False)
    parser.add_argument('--BRS_noBackprop', action='store_true', default=False)
    parser.add_argument('--revision_type', type=str, default='worst')
    arg = parser.parse_args()
    set_seed(arg)
    return arg



def main(save_manager):
    if save_manager.config.MISC.TB and save_manager.config.only_test_version is None:
        writer = TensorBoardManager(save_manager)
    else:
        writer = None

    # model initialization
    device_ids = list(range(len(save_manager.config.MISC.gpu.split(','))))
    model = nn.DataParallel(get_model(save_manager), device_ids=device_ids)
    model.to(save_manager.config.MISC.device)

    # calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():
        n_params += v.reshape(-1).shape[0]
    save_manager.write_log('Number of model parameters : {}'.format(n_params), 0)

    # optimizer initialization
    optimizer = get_optimizer(save_manager.config.Optimizer, model)

    metric_manager = MetricManager(save_manager)
    trainer = Trainer(model, metric_manager)


    # if training
    if not save_manager.config.only_test_version:
        # dataloader
        train_loader = get_dataloader(save_manager.config, 'train')
        val_loader = get_dataloader(save_manager.config, 'val')

        # training
        save_manager.write_log('Start Training...'.format(n_params), 4)
        trainer.train(save_manager=save_manager,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         optimizer=optimizer,
                         writer=writer)
        # deallocate data loaders from the memory
        del train_loader
        del val_loader


    trainer.best_param, trainer.best_epoch, trainer.best_metric = save_manager.load_model()

    save_manager.write_log('Start Test Evaluation...'.format(n_params), 4)
    test_loader = get_dataloader(save_manager.config, 'test')
    trainer.test(save_manager=save_manager, test_loader=test_loader, writer=writer)
    del test_loader


if __name__ == '__main__':
    start_time = time.time()

    arg = parse_args()
    save_manager = SaveManager(arg)
    save_manager.write_log('Process Start ::: {}'.format(datetime.datetime.now(timezone('Asia/Seoul')), n_mark=16))

    main(save_manager)

    end_time = time.time()
    save_manager.write_log('Process End ::: {} {:.2f} hours'.format(datetime.datetime.now(timezone('Asia/Seoul')), (end_time - start_time) / 3600), n_mark=16)
    save_manager.write_log('Version ::: {}'.format(save_manager.config.version), n_mark=16)
