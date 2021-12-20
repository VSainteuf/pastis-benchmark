"""
Code used to compute the centerness ground truth.
This code is not fully functional, we show how to compute
the ground truth for one patch to illustrate how Eq. 6 of the 
ICCV2021 paper was implemented.

Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
License MIT
"""

import os

import matplotlib.patches as ptch
import numpy as np


def bbox(img):
    """
    Get the bounding box of a binary mask.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if rmin > 0:
        rmin -= 1
    if cmin > 0:
        cmin -= 1

    rmax = rmax + 1 if rmax < img.shape[1] - 1 else rmax
    cmax = cmax + 1 if cmax < img.shape[1] - 1 else cmax
    return (rmin, rmax, cmin, cmax)


def splat(shape, center, dims, stride=None, r=3, alpha=0.7, homo=False):
    """
    Compute gaussian kernel.
    """
    h, w = shape
    hc, wc = center
    height, width = dims

    if stride is not None:
        h, w = h // stride, w // stride
        hc, wc = hc // stride, wc // stride
        height, width = height / stride, width / stride

    hcords = np.stack([np.arange(h) for _ in range(w)], axis=1)
    wcords = np.stack([np.arange(w) for _ in range(h)], axis=0)

    if homo:
        sigma_h = (1 - alpha) / (1 + alpha) * np.mean([height, width]) / r
        sigma_w = sigma_h
    else:
        sigma_h, sigma_w = (1 - alpha) / (1 + alpha) * height / r, (1 - alpha) / (
            1 + alpha
        ) * width / r
    out = (hcords - hc) ** 2 / (2 * sigma_h ** 2) + (wcords - wc) ** 2 / (
        2 * sigma_w ** 2
    )
    if np.isnan(out).any():
        print(height, width)
    out = np.exp(-out)
    return out


#### Compute centerness ground truth for one patch

folder = "PATH/TO/SEMSEG_ANNOTATION/FOLDER"
patch_id = 10000

instance = np.load(os.path.join(folder, "ParcelIDs_{}.npy".format(patch_id)))
semantic = np.load(os.path.join(folder, "TARGET_{}.npy".format(patch_id)))
instance = instance * (semantic[0] != 0)

paid = np.unique(instance)  # Get the list of parcelIDs
mapp = {paid[k]: k for k in range(len(paid))}  # Mapping ParcelIDS to instance codes
instance = np.vectorize(lambda x: mapp[x])(instance)

splats = []
boxes = []
codes = []

centers = []
for i, (inst_ID, inst_code) in enumerate(mapp.items()):
    if inst_code != 0:
        mask = instance == inst_code

        rmin, rmax, cmin, cmax = bbox(mask)  # get parcel bbox
        center_r = rmin + (rmax - rmin) // 2
        center_c = cmin + (cmax - cmin) // 2  # parcel center coordinates
        height = rmax - rmin
        width = cmax - cmin  # parcel dimensions

        box = ptch.Rectangle(
            (cmin, rmin), width, height, linewidth=1, edgecolor="r", facecolor="none"
        ) ## these objects can be used for plotting with matplotlib for a visual check.
        boxes.append(box)
        centers.append((center_c, center_r))
        s = splat(
            shape=mask.shape,
            center=(center_r, center_c),
            dims=(height, width),
            r=3,
        )
        splats.append(s)
        codes.append(inst_code)

heatmap = np.stack(splats).max(
    axis=0
)  ## The cenerness ground truth, aka target heatmap
zones = np.argmax(splats, axis=0)  ## The mapping between pixel and instance codes.
zones = np.vectorize(lambda x: {i: c for i, c in enumerate(codes)}[x])(zones)

### Count the number of colisions between object centers
count_colisions = np.zeros((128, 128))
for i, b in enumerate(boxes):
    count_colisions[centers[i][0], centers[i][1]] += 1

collision = np.where(count_colisions > 1)
collisions = len(collision[0])
if collisions > 0:
    print(collisions, " collisions")
