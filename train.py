#!/usr/bin/env Python
# coding=utf-8
'''
https://blog.csdn.net/Q1u1NG/article/details/107463417
https://github.com/ahangchen/windy-afternoon/blob/master/ml/pratice/torch_best_practice.md
'''
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import yaml
import torch
from torch.utils import data
import options
import time
import logging

from model.SGEH_base import SGEH

from utils.logger import log_para
from data.data_entry import get_dataset, get_loader
from utils.metrics import similarity_matrix, similarity_matrix_wiki


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
    with open(args.scripts) as f:
        scripts = yaml.load(f, Loader=yaml.FullLoader)  # load hyps, as a dictionary
    print('Called with args:')
    print(args)

    # result dir for each exp
    if args.exp_name == '':
        args.exp_name = '{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.code_len, args.cluster_num, args.pn_num)
    # auto-generate snapshot path if not specified
    if args.snapshot_dir == '':
        args.snapshot_dir = osp.join(args.result_dir, args.exp_name)
    if not osp.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    _init_fn = None
    if not args.random_train:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        def _init_fn(worker_id):
            np.random.seed(args.random_seed + worker_id)

    logger = logging.getLogger('train')
    log_para(logger, args)

    if args.model == 'SGEH':
        model = SGEH(args, logger)

    else:
        raise Exception("undefined model!", args.model)
    num_epoch = 0
    if args.eval == True:
        model.load_checkpoints()
        model.eval()

    else:
        for epoch in range(args.num_epoch):
            # train the Model
            model.train(epoch)
            # eval the Model
            if (epoch + 1) % args.eval_interval == 0:
                num_epoch = model.eval(step=epoch + 1, num_epoch=num_epoch)
            if num_epoch > 10:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for training")
    parser.add_argument('--scripts', type=str, default='/home/cs4007/data/zy/master/CCTV-new/new/scripts/para_config.yml',
                        help='hyperparameters', )
    parser.add_argument('--sim_map', default='/home/cs4007/data/zy/dataset/CrossModal/res/')

    parser.add_argument('--lr_img', type=float, default=1e-3)
    parser.add_argument('--lr_txt', type=float, default=1e-2)
    parser.add_argument('--code_len', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='wiki', help='MSCOCO NUS WIKI MIR')
    parser.add_argument('--dataset_path', type=str, default='/home/cs4007/data/zy/dataset/CrossModal')
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--pre_num_epoch', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epoch_interval', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='./checkpoint', help='model save path')
    parser.add_argument('--model', type=str, default='SGEH')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--result_dir', default='./result')
    parser.add_argument('--snapshot_dir', default='', help='result_dir/exp_name')
    parser.add_argument('--random_train', type=bool, default=True)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--eval', type=bool, default=False, help='only eval or not')
    parser.add_argument('--cluster_num', type=int, default=10)
    parser.add_argument('--pn_num', type=int, default=20)
    args = parser.parse_args()
    main(args)
