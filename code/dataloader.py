"""
Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
License MIT
"""

import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata


class PASTIS_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.

        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target

        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use. If you are using PASTIS-R, you have access to
                Sentinel-2 imagery and Sentinel-1 observations in Ascending and Descending orbits,
                respectively S2, S1A, and S1D.
                For example use sats=['S2', 'S1A'] for Sentinel-2 + Sentinel-1 ascending time series,
                or sats=['S2', 'S1A','S1D'] to retrieve all time series.
                If you are using PASTIS, only  S2 observations are available.
        """
        super(PASTIS_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                    os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None
        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                    / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.mem16:
            return ({k: v.float() for k, v in data.items()}, dates), target
        else:
            return (data, dates), target


def prepare_dates(date_dict, reference_date):
    """Date formating."""
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values

