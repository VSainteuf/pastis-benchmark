"""
Baseline Pytorch Dataset
"""

import os

import geopandas as gpd
import numpy as np
import torch


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, mini_dataset: bool = False):
        super(BaselineDataset, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        json_name = "metadata_mini.geojson" if mini_dataset else "metadata.geojson"
        self.meta_patch = gpd.read_file(os.path.join(folder, json_name))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        print("Done.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        print("Dataset ready.")

    def __len__(self):
        return self.len

    def __getitem__(self, item: int):
        id_patch = self.id_patches[item]

        # Open and prepare satellite data into T x C x H x W arrays
        path_patch = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(path_patch).astype(np.float32)
        # Move channel dim to last
        # data = np.moveaxis(data, 1, -1)
        # data = {"S2": torch.from_numpy(data).permute((0, 2, 3, 1))}
        data = {"S2": torch.from_numpy(data)}

        # If you have other modalities, add them as fields of the `data` dict ...
        # data["radar"] = ...

        # Open and prepare targets
        target = np.load(
            os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        )
        target = torch.from_numpy(target[0].astype(int))

        return data, target
