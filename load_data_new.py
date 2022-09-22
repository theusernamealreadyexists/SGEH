from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import pdb
import torch
import scipy.io as scio
import os.path as osp
from scipy.sparse import csr_matrix, triu, find
torch.multiprocessing.set_sharing_strategy('file_system')
import h5py

class SIMI_train(torch.utils.data.Dataset):
    def __init__(self, dataset, cluster_num, sim_map):
        if dataset == 'wiki':
            simi_file = 'wiki_train_uni_{}_5.mat'.format(cluster_num)
        else:
            simi_file = '{}_train_uni_{}.mat'.format(dataset, cluster_num)
        try:
            sim_map = scio.loadmat(osp.join(sim_map, simi_file))
            sim = np.asarray(find(sim_map['U'])).T

        except Exception:
            sim_map = h5py.File(osp.join(sim_map, simi_file), 'r')
            sim = np.asarray(find(sim_map['U'])).T
            sim_map.close()

        self.sim = sim

    def __getitem__(self, index):
        [x, y, dis] = self.sim[index]
        return x, y, dis

    def __len__(self):
        return len(self.sim)

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count


def get_loader(data_name, batch_size):
    p = '/home/cs4007/data/zy/dataset/CrossModal'
    train_path = osp.join(p, '{}_train.mat'.format(data_name))
    test_path = osp.join(p, '{}_test.mat'.format(data_name))
    set_train = scio.loadmat(train_path)
    set_test = scio.loadmat(test_path)

    test_index = np.array(set_test['id_test'], dtype=np.int).squeeze()  # 1*2000
    test_label_set = np.array(set_test['label_test'], dtype=np.int8).squeeze()  # 2000*81
    test_txt_set = np.array(set_test['text_test'], dtype=np.float).squeeze()  # 2000*512
    test_img_set = np.array(set_test['image_test'], dtype=np.float32).squeeze()  # 2000*4096

    train_index = np.array(set_train['id_train'], dtype=np.uint8).squeeze()  # 1*65000
    train_label_set = np.array(set_train['label_train'], dtype=np.int8).squeeze()  # 65000*81
    train_txt_set = np.array(set_train['text_train'], dtype=np.float).squeeze()  # 65000*512
    train_img_set = np.array(set_train['image_train'], dtype=np.float32).squeeze()  # 65000*4096

    if data_name == 'wiki':
        test_label_set = np.eye(10)[test_label_set.astype(dtype=np.int8) - 1]
        train_label_set = np.eye(10)[train_label_set.astype(dtype=np.int8) - 1]
    elif data_name == 'coco' or data_name == 'nus':
        train_label_set = train_label_set[:20000,:]
        train_txt_set = train_txt_set[:20000,:]
        train_img_set = train_img_set[:20000,:]

    query_L = test_label_set
    query_x = test_img_set
    query_y = test_txt_set
    validation_L = test_label_set
    validation_x = test_img_set
    validation_y = test_txt_set

    # query_L = test_label_set[231:]
    # query_x = test_img_set[231:]
    # query_y = test_txt_set[231:]
    # validation_L = test_label_set[:231]
    # validation_x = test_img_set[:231]
    # validation_y = test_txt_set[:231]

    retrieval_L = retrieval_vL = train_label_set
    retrieval_x = retrieval_vx = train_img_set
    retrieval_y = retrieval_vy = train_txt_set

    imgs = {'train': train_img_set, 'query': query_x, 'database': retrieval_x, 'databasev': retrieval_vx,
            'validation': validation_x}
    texts = {'train': train_txt_set, 'query': query_y, 'database': retrieval_y, 'databasev': retrieval_vy,
             'validation': validation_y}
    labels = {'train': train_label_set, 'query': query_L, 'database': retrieval_L, 'databasev': retrieval_vL,
              'validation': validation_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database', 'databasev', 'validation']}

    shuffle = {'query': False, 'train': True, 'database': False, 'validation': False, 'databasev': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in
                  ['query', 'train', 'database', 'databasev', 'validation']}

    return dataloader, (train_img_set, train_txt_set, train_label_set), test_label_set
