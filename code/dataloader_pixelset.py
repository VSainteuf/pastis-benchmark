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
import torch.utils.data as data
from scipy.interpolate import interp1d
from torch.nn import functional as F


class PASTIS_Dataset_PixelSet(data.Dataset):
    def __init__(
            self,
            folder,
            norm=True,
            cache=False,
            sats=["S2"],
            reference_date="2018-09-01",
            n_pixel=32,
            geom_features=True,
            folds=None,
            ignore_label=None,
            label_offset=1,
            class_mapping=None,
            interpolate=False,
            drop_temp_s2=0,
            drop_temp_s1=0,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS-R dataset in pixel-set
        format for parcel-based classification.

        The Dataset yields a data dictionary for each sample.
        For example, assuming the dataset was set to load
        Sentinel-2 observations (sats=['S2']),
        it will have the following keys:
            'S2': A tuple containing the Sentinel-2 time series of pixel-sets and
                the mask of true and repeated pixels. Formally, this key contains
                a tuple (X, M) with X a tensor of shape T_s2 x C_s2 x n_pixel and
                M a tensor of shape n_pixel. (T: number of available observations for S2,
                and C: number of spectral channels of S2)
            'dates-S2': A tensor containing the T dates of observation of the loaded
                S2 sequence. The dates are expressed as number of days since the
                reference date.
            'geomfeat' : A vector containing the parcel's 4 geometric features used
                by the Pixel-Set Encoder.
            'label' : Crop type label of the parcel.

        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            cache (bool): If True, the loaded samples stay in RAM, default False.
            sats (list): defines the satellites to use. In PASTIS-R you have access to
                Sentinel-2 imagery and Sentinel-1 observations in Ascending and Descending orbits,
                respectively S2, S1A, and S1D.
                For example use sats=['S2', 'S1A'] for Sentinel-2 + Sentinel-1 ascending time series,
                or sats=['S2', 'S1A','S1D'] to retrieve all time series.
                If you are using PASTIS, only  S2 observations are available.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            n_pixel (int): number of pixels randomly sampled from each parcel (default 32).,
                geom_features (bool): If False no geometric descriptors of the parcel's shape
                are used in the Pixel-Set Encoder (default True).
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            ignore_label (int): If not None, the parcels annotated with this label are discarded from
                the dataset.
            label_offset (int): If not None this offset is substracted from the labels contained in the dataset
                (default to 1 because PASTIS crop type labels start at 1).
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            interpolate (bool): if True, the Sentinel-1 observations are temporally interpolated to the
                dates of Sentinel-2 acquisitions (default False).
            drop_temp_s2 (float): probability of temporal dropout for Sentinel-2 time series (default 0.0),
            drop_temp_s1: probability of temporal dropout for Sentinel-1 time series (default 0.0),
        """
        assert sum([s in {"S1A", "S1D", "S2"} for s in sats]) == len(
            sats
        ), "Unknown satellite name (available: S2/S1A/S1D)"

        super(PASTIS_Dataset_PixelSet, self).__init__()
        self.folder = folder
        self.sats = sats
        self.norm = norm
        self.n_pixel = n_pixel
        self.geom_features = geom_features
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.label_offset = label_offset
        self.cache = cache
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = class_mapping
        self.interpolate = interpolate
        self.drop_temp_s2 = drop_temp_s2
        self.drop_temp_s1 = drop_temp_s1

        # Get metadata
        print("Reading parcel metadata . . .")
        self.meta = pd.read_csv(os.path.join(folder, "metadata_parcel.csv"))
        self.meta.index = self.meta["ID_PARCEL"].astype(int)
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
            for pid, date_seq in dates.iteritems():
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
            self.meta = pd.concat([self.meta[self.meta["Fold"] == f] for f in folds])
        # Remove unwanted parcels
        if ignore_label is not None:
            self.meta = self.meta[self.meta["Label"] != ignore_label]

        self.meta.sort_index(inplace=True)
        self.len = self.meta.shape[0]
        self.id_parcels = self.meta.index
        self.labels = self.meta["Label"].to_dict()
        self.id_patches = self.meta["ID_PATCH"].to_dict()

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                        os.path.join(folder, "NORM_PARCEL_{}_set.json".format(s)), "r"
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
            self.norm = {s: None for s in sats}

        # Prepare geometric features and their normalisation values for pixel-set
        if geom_features:
            print("Getting geometric features . . .")
            self.geomfeat = self.meta[[k for k in self.meta.columns if "GF_" in k]]
            self.geom_m, self.geom_s = (
                self.geomfeat.mean(axis=0).values,
                self.geomfeat.std(axis=0).values,
            )
            self.geomfeat = {
                index: np.array(list(d.values()))
                for index, d in self.geomfeat.to_dict(orient="index").items()
            }

        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_parcel = self.id_parcels[item]
        id_patch = self.id_patches[id_parcel]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_parcel),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}
            data = {
                s: repeat_pixel_and_norm(a, self.n_pixel, norm=self.norm[s])
                for s, a in data.items()
            }

            if self.cache:
                self.memory[item] = data

        else:
            data = self.memory[item]

        data = {
            sat: (sample_pixels(tensor, self.n_pixel), mask)
            for sat, (tensor, mask) in data.items()
        }  # Random pixel sampling

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                "dates-{}".format(s): torch.from_numpy(self.get_dates(id_patch, s))
                for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        # Temporal dropout
        if self.drop_temp_s1 + self.drop_temp_s2 > 0:
            selected_dates = {}
            for sat in self.sats:
                if "S1" in sat:
                    selected_dates[sat] = (
                            np.random.rand(data[sat][0].shape[0]) > self.drop_temp_s1
                    )
                elif "S2" in sat:
                    selected_dates[sat] = (
                            np.random.rand(data["S2"][0].shape[0]) > self.drop_temp_s2
                    )
            data = {sat: (d[0][selected_dates[sat]], d[1]) for sat, d in data.items()}
            dates = {s: d[selected_dates[s.split("-")[1]]] for s, d in dates.items()}

        data.update(dates)

        if self.interpolate and len(self.sats) > 1:
            for sat in self.sats:
                if sat == "S2":
                    continue
                interpolation = interp1d(
                    data["dates-{}".format(sat)].numpy(),
                    data[sat][0].numpy(),
                    axis=0,
                    fill_value="extrapolate",
                )
                data[sat] = (
                    torch.from_numpy(interpolation(data["dates-S2"].numpy())).float(),
                    data[sat][1],
                )
                data["dates-{}".format(sat)] = data["dates-S2"]

        # Retrieve label
        label = self.labels[id_parcel]
        if self.class_mapping is not None:
            label = self.class_mapping[label]
        data["label"] = torch.from_numpy(np.array(label - self.label_offset, dtype=int))

        # Retrieve geometric features
        if self.geom_features:
            gf = (self.geomfeat[id_parcel] - self.geom_m) / self.geom_s
            data["geomfeat"] = torch.from_numpy(gf).float()

        return data


def repeat_pixel_and_norm(pixels, n_pixel, norm=None):
    """
    Repeats a pixel if the parcel has fewer pixels than n_pixel.
    Normalises the channels.
    """
    if pixels.shape[-1] < n_pixel:
        if pixels.shape[-1] == 0:
            x = torch.zeros((*pixels.shape[:2], n_pixel))
            mask = np.array([0 for _ in range(n_pixel)])
            mask[0] = 1
        else:
            x = F.pad(pixels, [0, n_pixel - pixels.shape[-1]], mode="replicate")
            mask = np.array(
                [1 for _ in range(pixels.shape[-1])]
                + [0 for _ in range(pixels.shape[-1], n_pixel)]
            )
    else:
        x = pixels
        mask = np.array([1 for _ in range(n_pixel)])

    if norm is not None:
        m, s = norm
        x = (x - m[None, :, None]) / s[None, :, None]  # channelwise normalisation

    return (x, torch.from_numpy(mask))


def sample_pixels(pixels, n_pixel):
    """
    Random sampling of pixels within a parcel.
    """
    if pixels.shape[-1] > n_pixel:
        idx = np.random.choice(
            list(range(pixels.shape[-1])), size=n_pixel, replace=False
        )
        x = pixels[:, :, idx]
    else:
        x = pixels
    return x


def prepare_dates(date_dict, reference_date):
    """Transforms date of observation to number of days since
    the reference date.
    """
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
                datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                - reference_date
        ).days
    )
    return d.values

