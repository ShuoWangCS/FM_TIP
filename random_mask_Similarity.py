import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import h5py
import json
import argparse
import os

import shutil
import torch.nn.functional as F
import pdb
import random
import time
seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

class SimpleHDF5Dataset:
    def __init__(self, file_handle):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats']
        self.all_labels = self.f['all_labels']
        self.total = self.f['count']

    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

# a dataset to allow for category-uniform sampling of base and novel classes.
# also incorporates hallucination
class LowShotDataset:
    def __init__(self, base_feats, novel_feats, base_classes, novel_classes, params):
        self.f = base_feats
        self.all_base_feats_dset = self.f['all_feats'][...] # (38400, 512, 5, 5)
        self.all_base_labels_dset = self.f['all_labels'][...]

        self.novel_feats = novel_feats['all_feats'] # (25, 512, 5, 5)
        self.novel_labels = novel_feats['all_labels'] 

        self.base_classes = base_classes
        self.novel_classes = novel_classes

        self.frac = 0.5
        self.all_classes = np.concatenate((base_classes, novel_classes))
   
        self.novel_feats_tensor = torch.FloatTensor(self.novel_feats)
        self.novel_feat_dim = self.novel_feats_tensor.view(self.novel_feats_tensor.shape[0], self.novel_feats_tensor.shape[1], -1).mean(dim=2) # (25, 512)
        self.base_feats_tensor = torch.FloatTensor(self.all_base_feats_dset)
        self.base_feat_dim = self.base_feats_tensor.view(self.base_feats_tensor.shape[0], self.base_feats_tensor.shape[1], -1).mean(dim=2) # (38400, 512)

        self.relation = F.normalize(self.novel_feat_dim,dim=-1).mm(F.normalize(self.base_feat_dim,dim=-1).t()) # (25, 38400)

        _, self.index = torch.topk(self.relation, params.beta, dim=-1) #  select topk similar images from all the 38400 images for each query image

    def sample_base_class_examples(self, num):
        sampled_idx = np.sort(np.random.choice(len(self.all_base_labels_dset), num, replace=False))
        return torch.Tensor(self.all_base_feats_dset[sampled_idx,:]), torch.LongTensor(self.all_base_labels_dset[sampled_idx].astype(int))

    def sample_novel_class_examples(self, num): # select num from 25
        sampled_idx = np.random.choice(len(novel_feats['all_labels']), num)
        return torch.Tensor(self.novel_feats[sampled_idx,:]), torch.LongTensor(self.novel_labels[sampled_idx].astype(int)), sampled_idx

    def get_sample(self, batchsize):
        num_base = round(self.frac*batchsize)
        num_novel = batchsize - num_base
        base_feats, base_labels = self.sample_base_class_examples(int(num_base))
        novel_feats, novel_labels = self.sample_novel_class_examples(int(num_novel))
        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def get_sample_similarity(self, batchsize):
        novel_feats, novel_labels, sampled_idx = self.sample_novel_class_examples(int(batchsize))
        base_sample_idx = self.index[sampled_idx,:] # (batchsize, params.beta=1000), the top 1000 similar with each query image
        random_matrix = torch.rand(base_sample_idx.size()) # (batchsize, params.beta=1000)
        sample_idxx = torch.diag(base_sample_idx[:,random_matrix.max(-1)[-1]]) 
        base_feats, base_labels = torch.Tensor(self.all_base_feats_dset[sample_idxx,:]), torch.LongTensor(self.all_base_labels_dset[sample_idxx].astype(int))

        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def get_index_base_novel_sample(self, batchsize, cur_novel_l, related_index):
        novel_feats, novel_labels = self.sample_novel_class_examples(int(batchsize))
        base_sample_index = []
        for curent_label in novel_labels:
            idx = np.where(cur_novel_l == curent_label.data.numpy())[0][0]
            idy = related_index.data.numpy()[idx]
            idxx = np.random.choice(idy, 1)[0]
            idyy = np.random.choice(np.where(self.all_base_labels_dset==idxx)[0], 1)[0]
            base_sample_index.append(idyy)

        base_feats, base_labels = torch.Tensor(self.all_base_feats_dset[base_sample_index,:]), torch.LongTensor(self.all_base_labels_dset[base_sample_index].astype(int))
        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def featdim(self):
        return self.novel_feats.shape[1]

# simple data loader for test
def get_test_loader(file_handle, batch_size=1000):
    testset = SimpleHDF5Dataset(file_handle)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return data_loader

def construct_patchmix(x, rotated_x, chunks = 5, alpha=0.1):
    therold = torch.rand(chunks, chunks)
    therold[therold >= alpha] = 1
    therold[therold < alpha] = 0

    new_x = therold.cuda() * x + (1 - therold.cuda()) * rotated_x
    return new_x

def CrossEntropy(pred, target, scale = False):
    pred = pred.softmax(-1)
    loss = -torch.log(pred) * target
    if scale:
        loss = loss.sum() / ((target > 0).sum() + 0.000001)
    else:
        loss = loss.sum() / (target.sum() + 0.000001)

    return loss

def BinaryEntropy(pred, target, scale = False):
    pred = pred.sigmoid()
    loss = -torch.log(pred + 0.0000001) * target
    if scale:
        loss = loss.sum() / ((target > 0).sum() + 0.000001)
    else:
        loss = loss.sum() / (target.sum() + 0.000001)
    return loss

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def training_loop(lowshot_dataset,novel_test_feats, num_classes, params, batchsize=1000, maxiters=1000, nTimes = 0):
    if os.path.exists('Model_SHOT1/' + params.name + '/' + params.dataset_name + '/' + params.backbone_name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '_' + str(params.lamda) + '/') == False:
        os.makedirs('Model_SHOT1/' + params.name + '/' + params.dataset_name + '/' + params.backbone_name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '_' + str(params.lamda) + '/')
    if os.path.exists('Model_SHOT1/' + params.name + '/' + params.dataset_name + '/' + params.backbone_name + '/' + str(nTimes) + '_' + str(params.alpha) + '_' + str(params.beta) + '_' + str(params.lamda)) == False:
        os.makedirs('Model_SHOT1/' + params.name + '/' + params.dataset_name + '/' + params.backbone_name + '/' + str(nTimes) + '_' + str(params.alpha) + '_' + str(params.beta) + '_' + str(params.lamda))

    featdim = params.feat_dim
    model = torch.nn.utils.weight_norm(nn.Linear(featdim, num_classes), dim=0)
    model = model.cuda()
    
    lamda = params.lamda
    test_loader = get_test_loader(novel_test_feats)

    best_ACC = 0.0
    tmp_epoach = 0
    tmp_count = 0
    tmp_rate = params.lr
    recode_reload = {}
    reload_model = False
    max_tmp_count = 10
    optimizer = torch.optim.Adam(model.parameters(), tmp_rate, weight_decay=params.wd)

    novel_labels = lowshot_dataset.novel_labels
    for epoch in range(maxiters):

        optimizer.zero_grad()

        (x,y) = lowshot_dataset.get_sample_similarity(batchsize)


        x = Variable(x.cuda())
        y = Variable(y.cuda())

        x_base, x_novel = torch.chunk(x, 2, dim = 0)
        y_base, y_novel = torch.chunk(y, 2, dim = 0)

        # lam = np.random.beta(2.0, 2.0)
        lam = lamda
        new_x = construct_patchmix(lam * x_novel, (1 - lam) * x_base, chunks = params.chunks, alpha=params.alpha) 

        novel_feat_x = x_novel.view(x_novel.shape[0], x_novel.shape[1], -1).mean(dim=2)
        mixup_feat = new_x.view(new_x.shape[0], new_x.shape[1], -1).mean(dim=2)

        all_feats = torch.cat((novel_feat_x, mixup_feat),0)

        scores_novel, scores_mixup = torch.chunk(model(all_feats), 2, dim = 0)

        loss_1 = F.cross_entropy(scores_novel, y_novel)
        loss_3 = F.cross_entropy(scores_mixup, y_novel) * lam

        loss = loss_1  + loss_3

        loss.backward()
        optimizer.step()
        
    return model

def perelement_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def eval_loop(data_loader, model, base_classes, novel_classes):
    model = model.eval()
    top1 = None
    top5 = None
    no_novel_class = list(set(range(100)).difference(set(novel_classes)))
    all_labels = None
    for i, (x,y) in enumerate(data_loader):
        x = Variable(x.cuda())
        feat_x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        scores = model(feat_x).softmax(-1)
        scores[:,no_novel_class] = -0.0
        top1_this, _ = perelement_accuracy(scores.data, y)
        top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
        all_labels = y.numpy() if all_labels is None else np.concatenate((all_labels, y.numpy()))

    is_novel = np.in1d(all_labels, novel_classes)
    top1_novel = np.mean(top1[is_novel])
    return [top1_novel]


def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--name', default='random_mask_Similarity_time', type=str)
    parser.add_argument('--numclasses', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--maxiters', default=2002, type=int)
    parser.add_argument('--batchsize', default=10, type=int)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--beta', default=512, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--lamda', default=0.1, type=float)
    parser.add_argument('--backbone_name', default='meta-baseline', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    # np.random.seed(0)
    # seed = 0
    # torch.manual_seed(seed)            # 为CPU设置随机种子
    # torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    params = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)

    max_epoach_fsl = 100

    # for dataset_name in os.listdir('Features/'):
    #     for backbone_name in os.listdir('Features/' + dataset_name):
    #         print(dataset_name, backbone_name)
    #         if 'BDC' in backbone_name:
    #             continue
    params.dataset_name = 'mini-ImageNet'
    dataset_name = params.dataset_name
    backbone_name = params.backbone_name
    train_feats = h5py.File('Features/' + dataset_name + '/' + backbone_name+ '/train.hdf5', 'r')
    test_feats = h5py.File('Features/' + dataset_name + '/' + backbone_name+ '/val.hdf5', 'r')
    all_feats_dset = test_feats['all_feats'][...]
    all_labels = test_feats['all_labels'][...]

    params.feat_dim = np.shape(all_feats_dset[0])[0]
    params.chunks = np.shape(all_feats_dset[0])[1]

    base_classes =  list(set(train_feats['all_labels'][...]))
    novel_classes =  list(set(test_feats['all_labels'][...]))

    n_shot = 1
    for alpha in [0.8]:
        start_ = 0
        end_ = max_epoach_fsl 
        params.alpha = alpha
        if os.path.exists('Model_SHOT1/' + params.name + '/' + params.dataset_name + '/' + params.backbone_name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '_' + str(params.lamda) + '/'):
            len_results = len(os.listdir('Model_SHOT1/' + params.name + '/' + params.dataset_name + '/' + params.backbone_name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '_' + str(params.lamda) + '/'))
            if len_results >= max_epoach_fsl:
                continue
            else:
                start_ = len_results

        lowshot_dataset = None
        for nTime in range(start_, end_):
                    # print(nTime)

            selected = np.random.choice(novel_classes, 5, replace=False)

            novel_train_feats = []
            novel_train_labels = []
            novel_test_feats = []
            novel_test_labels = []

            for K in selected:
                is_K = np.in1d(all_labels, K)

                current_idx = np.random.choice(np.sum(is_K), 15 + n_shot, replace=False)
                novel_train_feats.append(all_feats_dset[is_K][current_idx[:n_shot]])
                novel_test_feats.append(all_feats_dset[is_K][current_idx[n_shot:]])

                for _ in range(n_shot):
                    novel_train_labels.append(K)
                for _ in range(15):
                    novel_test_labels.append(K)

            novel_train_feats  =  np.vstack(novel_train_feats)
            novel_train_labels =  np.array(novel_train_labels)
            novel_test_feats   =  np.vstack(novel_test_feats)
            novel_test_labels  =  np.array(novel_test_labels)

            novel_feats = {}
            novel_feats['all_feats'] = novel_train_feats
            novel_feats['all_labels'] = novel_train_labels
            novel_feats['count'] = len(novel_train_labels)

            novel_val_feats = {}
            novel_val_feats['all_feats'] = novel_test_feats
            novel_val_feats['all_labels'] = novel_test_labels
            novel_val_feats['count'] = len(novel_test_labels)

            if lowshot_dataset is not None:
                lowshot_dataset.novel_feats = novel_feats['all_feats']
                lowshot_dataset.novel_labels = novel_feats['all_labels']
                lowshot_dataset.novel_classes = novel_classes
                lowshot_dataset.all_classes = np.concatenate((base_classes, novel_classes))

                lowshot_dataset.novel_feats_tensor = torch.FloatTensor(lowshot_dataset.novel_feats)

                lowshot_dataset.novel_feat_dim = lowshot_dataset.novel_feats_tensor.view(lowshot_dataset.novel_feats_tensor.shape[0], lowshot_dataset.novel_feats_tensor.shape[1], -1).mean(dim=2) # (25, 512)
                # lowshot_dataset.base_feats_tensor = torch.FloatTensor(lowshot_dataset.all_base_feats_dset)
                # lowshot_dataset.base_feat_dim = lowshot_dataset.base_feats_tensor.view(lowshot_dataset.base_feats_tensor.shape[0], lowshot_dataset.base_feats_tensor.shape[1], -1).mean(dim=2) # (38400, 512)

                lowshot_dataset.relation = F.normalize(lowshot_dataset.novel_feat_dim,dim=-1).mm(F.normalize(lowshot_dataset.base_feat_dim,dim=-1).t()) # (25, 38400)

                _, lowshot_dataset.index = torch.topk(lowshot_dataset.relation, params.beta, dim=-1) #  select topk similar images from all the 38400 images for each query image

            else:
                lowshot_dataset = LowShotDataset(train_feats, novel_feats, base_classes, novel_classes, params)

            model = training_loop(lowshot_dataset, novel_val_feats, params.numclasses, params, params.batchsize, params.maxiters, nTimes = nTime)

