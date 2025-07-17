# Adapted from https://github.com/alanqrwang/nwhead
# No license specified in original repository

import numpy as np
import random
import torch
from torch.utils.data import Dataset, ConcatDataset
from .utils import DatasetMetadata, InfiniteUniformClassLoader, FullDataset, compute_clusters

class SupportSet:
    '''Support set base class for NW.'''
    def __init__(self, 
                 support_set, 
                 n_classes,
                 env_array=None,
                 ):
        self.y_array = np.array(support_set.targets)
        self.n_classes = n_classes

        # If env_array is provided, then support dataset should be a single
        # Pytorch Dataset. 
        if env_array is not None:
            self.env_array = env_array
            support_set = DatasetMetadata(support_set, self.env_array)
            self.combined_dataset = support_set
            self.env_datasets = self._separate_env_datasets(support_set)
        # Otherwise, it should be a list of Datasets.
        elif env_array is None and all(isinstance(d, Dataset) for d in support_set):
            assert all(isinstance(d, Dataset) for d in support_set)
            self.env_array = []
            for i, ds in enumerate(support_set):
                self.env_array += [i for _ in range(len(ds))]
            support_set = DatasetMetadata(support_set, self.env_array)
            self.env_datasets = support_set
            self.combined_dataset = self._combine_env_datasets(support_set)
        # Simplest case, no environment info and single support dataset
        else:
            self.env_array = np.zeros(len(support_set))
            support_set = DatasetMetadata(support_set, self.env_array)
            self.combined_dataset = support_set
            self.env_datasets = self._separate_env_datasets(support_set)

    def _combine_env_datasets(self, env_datasets):
        self.env_map = {i:i for i in range(len(env_datasets))}
        combined_dataset = ConcatDataset(env_datasets)
        combined_dataset.targets = np.concatenate([env.targets for env in self.env_datasets])
        assert len(combined_dataset) == len(combined_dataset.targets)
        return combined_dataset

    def _separate_env_datasets(self, combined_dataset):
        env_datasets = []
        self.env_map = {}
        for i, attr in enumerate(np.unique(self.env_array)):
            self.env_map[attr] = i
            indices = (self.env_array==attr).nonzero()[0]
            env_dataset = torch.utils.data.Subset(combined_dataset, indices)
            env_dataset.targets = self.y_array[indices]
            env_datasets.append(env_dataset)
        return env_datasets

class SupportSetTrain(SupportSet):
    '''Support set for NW training.'''
    def __init__(self, 
                 support_set, 
                 n_classes,
                 train_type,
                 n_shot,
                 n_way=None,
                 env_array=None,
                 ):
        super().__init__(support_set, n_classes, env_array)
        self.train_type = train_type
        self.n_shot = n_shot
        self.n_way = n_way
        self.train_iter = self._build_iter()

    def get_support(self, y):
        '''Samples a support for training.'''
        if self.train_type == 'irm':
            train_iter = np.random.choice(self.train_iter)
            sx, sy, sm = train_iter.next()
        else:
            sx, sy, sm = self.train_iter.next(y)

        return sx, sy

    def _build_iter(self):
        '''Iterators for random sampling during training.
        Samples images from dataset.'''
        if self.train_type == 'random':
            train_iter = InfiniteUniformClassLoader(
                self.combined_dataset, self.n_shot, 
                self.n_way)
        else:
            train_iter = [iter(InfiniteUniformClassLoader(env, self.n_shot)) for env in self.env_datasets]
        return train_iter

class SupportSetEval(SupportSet):
    '''Support set for NW evaluation.'''
    def __init__(self, 
                 support_set, 
                 n_classes,
                 n_shot_random,
                 n_shot_full, 
                 n_shot_cluster=3,
                 n_neighbors=20,
                 env_array=None,
                 num_workers=4,
                 ):
        super().__init__(support_set, n_classes, env_array)
        self.n_shot_random = n_shot_random
        self.n_shot_full = n_shot_full
        self.n_shot_cluster = n_shot_cluster
        self.n_neighbors = n_neighbors
        self.support_loaders = self._build_full_loader(num_workers)

    def _extract_img_from_batch(self, i, img_batch, batch_idx, in_batch_idx):
        assert i in batch_idx
        indices_in_this_batch = np.where(batch_idx == i)[0]
        return [x for x in img_batch[in_batch_idx[indices_in_this_batch]]]


    def build_infer_iters(self, sfeat, sy, smeta, sfeat_env, sy_env, smeta_env):
        # Full
        self.full_feat = sfeat
        self.full_y = sy
        self.full_meta = smeta
        self.full_feat_sep = sfeat_env
        self.full_y_sep = sy_env
        self.full_meta_sep = smeta_env

        # Cluster
        self.cluster_feat, self.cluster_y, self.cluster_indices = compute_clusters(self.full_feat, self.full_y, self.n_shot_cluster)
        assert len(self.support_loaders) == 1, 'Only one environment supported for now.'
        loader = self.support_loaders[0]
        batchsize = loader.batch_size
        batches = self.cluster_indices // batchsize
        in_batch_idx = self.cluster_indices % batchsize
        self.cluster_imgs = []
        for i, (img_batch, _, _) in enumerate(loader):
            if i in batches:
                self.cluster_imgs += self._extract_img_from_batch(i, img_batch, batches, in_batch_idx)


    def transfer_to_device(self, device):
        self.full_feat = [x.to(device) for x in self.full_feat] if isinstance(self.full_feat, list) else self.full_feat.to(device)
        self.full_y = [x.to(device) for x in self.full_y] if isinstance(self.full_y, list) else self.full_y.to(device)
        self.full_meta = [x.to(device) for x in self.full_meta] if isinstance(self.full_meta, list) else self.full_meta.to(device)
        self.full_feat_sep = [x.to(device) for x in self.full_feat_sep] if isinstance(self.full_feat_sep, list) else self.full_feat_sep.to(device)
        self.full_y_sep = [x.to(device) for x in self.full_y_sep] if isinstance(self.full_y_sep, list) else self.full_y_sep.to(device)
        self.full_meta_sep = [x.to(device) for x in self.full_meta_sep] if isinstance(self.full_meta_sep, list) else self.full_meta_sep.to(device)
        self.cluster_feat = [x.to(device) for x in self.cluster_feat] if isinstance(self.cluster_feat, list) else self.cluster_feat.to(device)
        self.cluster_y = [x.to(device) for x in self.cluster_y] if isinstance(self.cluster_y, list) else self.cluster_y.to(device)
        self.cluster_imgs = [x.to(device) for x in self.cluster_imgs] if isinstance(self.cluster_imgs, list) else self.cluster_imgs.to(device)


    def get_support(self, mode, x=None, expl_mode=False):
        '''Samples a support for inference depending on mode.'''
        try:
            if mode == 'random':
                sfeat, sy, _ = self.random_iter.next()
            elif mode == 'full':
                sfeat, sy = self.full_feat, self.full_y
            elif mode == 'cluster':
                sfeat, sy = self.cluster_feat, self.cluster_y
            elif mode == 'ensemble':
                sfeat, sy = self.full_feat_sep, self.full_y_sep
            elif mode == 'knn':
                sfeat, sy = self.knn(x)
            elif mode == 'hnsw':
                sfeat, sy = self.hnsw(x)
            else:
                raise NotImplementedError
            if mode == 'cluster' and expl_mode:
                return sfeat, sy, self.cluster_imgs
            else:
                return sfeat, sy
        except AttributeError:
            raise AttributeError('Did you run precompute()?')

    def _build_full_loader(self, num_workers):
        '''Full loader for precomputing features during evaluation.
        Because the model assumes balanced classes during training and
        test, the support loader samples evenly across classes.
        '''
        self.full_datasets = []
        for env in self.env_datasets:
            self.full_datasets.append(FullDataset(env, self.n_shot_full))
        return [torch.utils.data.DataLoader(
                env, batch_size=128, shuffle=False, num_workers=num_workers) for env in self.full_datasets]