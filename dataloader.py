import torch
import torch.utils.data as tdata
from torch.nn import functional as F

import numpy as np
import geopandas as gpd
import pandas as pd

from datetime import datetime
import os
import json
import collections.abc
import re



class PASTIS_Dataset(tdata.Dataset):
    def __init__(self, folder, norm=True, folds=None, reference_date='2018-09-01', cache=False,
                 class_mapping=None, target='semantic',  sats=['S2'], mono_date=None):
        """

        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed channel-wise
            means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date based on which all observation dates
            are expressed. Along with the image time series and the target tensor, this dataloader yields the sequence
            of observation dates (in terms of number of days since the reference date). This sequence of dates is used
            for instance for the positional encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is returned by the dataloader.
            If 'semantic' the target tensor is a tensor containing the class of each pixel.
            If 'instance' the target tensor is the concatenation of several signals, necessary to train the
            Parcel-as-Points module:
                - the centerness heatmap,
                - the instance ids,
                - the voronoi partitioning of the patch with regards to the parcels' centers,
                - the (height, width) size of each parcel
                - the semantic label of each parcel
                - the semantic label of each pixel
            cache (bool): If True the loaded samples stay in RAM, default False.
            folds (list, optional): List of ints specifying which of the 5 official folds to load.
            By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional):
            mono_date (int or str, optional):
            sats (list):
        """
        super(PASTIS_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split('-')))
        self.cache = cache
        self.mono_date = datetime(*map(int, mono_date.split('-'))) if isinstance(mono_date, str) else mono_date
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = np.vectorize(lambda x: class_mapping[x]) if class_mapping is not None else class_mapping
        self.target = target
        self.sats = sats

        # Get metadata and check dataset completeness
        print('Reading patch metadata . . .')
        self.meta_patch = gpd.read_file(os.path.join(folder, 'metadata.geojson'))
        self.meta_patch.index = self.meta_patch['ID_PATCH'].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch['dates-{}'.format(s)]
            date_table = pd.DataFrame(index=self.meta_patch.index, columns=self.date_range, dtype=int)
            for pid, date_seq in dates.iteritems():
                d = pd.DataFrame().from_dict(date_seq, orient='index')
                d = d[0].apply(
                    lambda x: (datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])) - self.reference_date).days)
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {index: np.array(list(d.values())) for index, d in
                                   date_table.to_dict(orient='index').items()}

        print('Done.')

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat([self.meta_patch[self.meta_patch['Fold'] == f] for f in folds])

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(os.path.join(folder, 'NORM_{}_patch.json'.format(s)), 'r') as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals['Fold_{}'.format(f)]['mean'] for f in selected_folds]
                stds = [normvals['Fold_{}'.format(f)]['std'] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = torch.from_numpy(self.norm[s][0]).float(), torch.from_numpy(self.norm[s][1]).float()
        else:
            self.norm = None
        print('Dataset ready.')

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {satellite: np.load(
                os.path.join(self.folder, 'DATA_{}'.format(satellite),
                             '{}_{}.npy'.format(satellite, id_patch))).astype(np.float32)
                    for satellite in self.sats}  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}
            if self.norm is not None:
                data = {s: (d - self.norm[s][0][None, :, None, None]) / self.norm[s][1][None, :, None, None] for s, d in
                        data.items()}

            if self.target == 'semantic':
                target = np.load(os.path.join(self.folder, 'ANNOTATIONS', 'TARGET_{}.npy'.format(id_patch)))
                target = torch.from_numpy(target[0].astype(int))
                if self.class_mapping is not None:
                    target = self.class_mapping(target)
            elif self.target == 'instance':
                heat = np.load(os.path.join(self.folder, 'INSTANCE_ANNOTATIONS', 'HEATMAP_{}.npy'.format(id_patch)))

                inst = np.load(os.path.join(self.folder, 'INSTANCE_ANNOTATIONS', 'INSTANCES_{}.npy'.format(id_patch)))
                zone = np.load(os.path.join(self.folder, 'INSTANCE_ANNOTATIONS', 'ZONES_{}.npy'.format(id_patch)))

                sem_pix = np.load(os.path.join(self.folder, 'ANNOTATIONS', 'TARGET_{}.npy'.format(id_patch)))
                if self.class_mapping is not None:
                    sem_pix = self.class_mapping(sem_pix[0])
                else:
                    sem_pix = sem_pix[0]

                size = np.zeros((*inst.shape, 2))
                sem_obj = np.zeros(inst.shape)
                for x in np.unique(inst):
                    if x != 0:
                        h = (inst == x).any(axis=-1).sum()
                        w = (inst == x).any(axis=-2).sum()
                        size[zone == x] = (h, w)
                        sem_obj[zone == x] = sem_pix[inst == x][0]

                target = torch.from_numpy(
                    np.concatenate([heat[:, :, None], inst[:, :, None], zone[:, :, None], size,
                                    sem_obj[:, :, None], sem_pix[:, :, None]], axis=-1)).float()


            if self.cache:
                self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]


        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats}
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = data[self.mono_date]
                dates = dates[self.mono_date]
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = (dates - mono_delta).abs().argmin()
                data = data[mono_date]
                dates = dates[mono_date]

        return (data, dates), target


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient='index')
    d = d[0].apply(lambda x: (datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])) - reference_date).days)
    return d.values


def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = PASTIS_Dataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            print('{}/{}'.format(i, len(dt)), end='\r')
            data = b[0][0][sat]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals['Fold_{}'.format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, 'NORM_{}_patch.json'.format(sat)), 'w') as file:
        file.write(json.dumps(norm_vals, indent=4))


np_str_obj_array_pattern = re.compile(r'[SaUO]')

def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError('Format not managed : {}'.format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError('Format not managed : {}'.format(elem_type))