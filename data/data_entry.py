import torch
from data.MSCOCO import MSCOCO, MSCOCO_map
from data.WIKI import WIKI, WIKI_map
from data.NUS_WIDE import NUS, NUS_map
from data.MIR_Flickr import MIR, MIR_map
import os.path as osp
import numpy as np

import data.base_dataset

def get_dataset(args):

    if args.dataset == 'coco':
        train_path = osp.join(args.dataset_path, 'coco_train.mat')
        test_path = osp.join(args.dataset_path, 'coco_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = MSCOCO_map(args)
        test_dataset = MSCOCO(dataset, False)
        base_dataset = MSCOCO(dataset, True)

    if args.dataset == 'wiki':
        train_path = osp.join(args.dataset_path, 'wiki_train.mat')
        test_path = osp.join(args.dataset_path, 'wiki_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = WIKI_map(args)
        test_dataset = WIKI(dataset, False)
        base_dataset = WIKI(dataset, True)

    if args.dataset == 'nus':
        train_path = osp.join(args.dataset_path, 'nus_train.mat')
        test_path = osp.join(args.dataset_path, 'nus_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = NUS_map(args)
        test_dataset = NUS(dataset, False)
        base_dataset = NUS(dataset, True)

    if args.dataset == 'mir':
        train_path = osp.join(args.dataset_path, 'mir_train.mat')
        test_path = osp.join(args.dataset_path, 'mir_test.mat')
        dataset = data.base_dataset.BaseDataset(train_path, test_path)
        train_dataset = MIR_map(args)
        test_dataset = MIR(dataset, False)
        base_dataset = MIR(dataset, True)

    return train_dataset, test_dataset, base_dataset

def collate_fn(data, train_dataset):
    '''

    :param data: list of tuple(x,y,dis)
    :return:
    '''
    x, y, dis = zip(*data)
    # x: tuple, size: batchsize
    nodes = np.array(
        [np.array(x, dtype=int), np.array(y, dtype=int)]).transpose()  # https://bbs.csdn.net/topics/394371579
    unq_keys, key_idx = np.unique(nodes, return_inverse=True)
    key_idx = key_idx.reshape(-1, 2)

    num = len(unq_keys)
    adj = np.ones((num, num), dtype=float)
    adj[key_idx[:, 0], key_idx[:, 1]] = np.array(list(dis))
    adj = adj - np.diag(np.diag(adj))

    # train_txt = np.array(set_train['text_train'], dtype=np.float)  # 1000*512
    # train_img = np.array(set_train['image_train'], dtype=np.float)  # 1000*512

    image_batch = train_img_set[unq_keys, :]
    text_batch = train_txt_set[unq_keys, :]

    return adj, image_batch, text_batch

def get_loader(args):
    train_dataset, test_dataset, base_dataset = get_dataset(args)

    # base数据集用于初始化聚类中心以及验证时计算指标, shuffle = False
    static_loader = torch.utils.data.DataLoader(dataset=base_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=False,
                                         num_workers=args.workers,
                                         drop_last=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers,
                                         drop_last=False,
                                         collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.workers,
                                         drop_last=False)

    return static_loader, train_loader, test_loader
