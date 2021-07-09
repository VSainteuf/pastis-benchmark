# PAnoptic Segmentation of satellite image TIme Series - PASTIS Benchmark

![](images/predictions.png)

## Dataset presentation
PASTIS is a benchmark dataset for panoptic and semantic segmentation of agricultural parcels from 
satellite image time series. It is composed of 2433 one square kilometer-patches in the French metropolitan territory for which sequences 
of satellite observations are assembled into a four-dimensional spatio-temporal tensor. 
The dataset contains both semantic and instance annotations, assigning to each pixel a semantic label and an instance id.
There is an official 5 fold split provided in the dataset's metadata.
## Usage 
- **DOWNLOAD** 

The dataset can be downloaded from [zenodo](https://zenodo.org/record/5012942).
- **DATALOADER** 

This repository also contains a PyTorch dataset class in `code/dataloader.py` 
that can be readily used to load data for training.
- **METRICS** 

A PyTorch implementation is also given in `code/panoptic_metrics.py` to compute
the panoptic metrics. In order to use these metrics, the model's output should contain an instance prediction
and a semantic prediction. The first one allocates an instance id to each pixel of the image, 
and the latter a semantic label.

## Leaderboard
Please open an issue to submit new entries.

### Semantic Segmentation
| Model name         | #Params| OA  |  mIoU | Published |
| ------------------ |---- |---- | ---| --- |
| U-TAE   |   1.1M|  83.2%    | 63.1%|  :heavy_check_mark:|
| Unet-3d*   | 1.6M|    81.3%    |  58.4%| :heavy_check_mark:|
| Unet-ConvLSTM* |1.5M  |     82.1%    |  57.8%| :heavy_check_mark:|
| FPN-ConvLSTM*  | 1.3M|    81.6%   |  57.1%|:heavy_check_mark: |

Models that we re-implemented are denoted with a star (*).


### Panoptic Segmentation
| Model name         | SQ  | RQ | PQ|
| ------------------ |--- | --- |--- |
| U-TAE + PaPs       | 82.0|51.0 |42.2   |



## Documentation
The agricultural parcels are grouped into 18 different crop classes as shown in the 
table below. 
<img src="images/Nomenclature.png" alt="drawing" width="300"/>

Additional information about the dataset can be found in the `documentation/Doc.pdf` document.

## References
If you use PASTIS please cite the related paper:
```
@article{garnot2021panoptic,
  title={Panoptic Segmentation of Satellite Image Time Series
with Convolutional Temporal Attention Networks},
  author={Sainte Fare Garnot, Vivien  and Landrieu, Loic },
  journal={arxiv},
  year={2021}
}
```

## Credits

- The satellite imagery used in PASTIS was retrieved from [THEIA](www.theia.land.fr): 
"Value-added data processed by the CNES for the Theia www.theia.land.fr data cluster using Copernicus data.
The treatments use algorithms developed by Theia’s Scientific Expertise Centres. "

- The annotations used in PASTIS stem from the French [land parcel identification system](https://www.data.gouv.fr/en/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/) produced
 by IGN, the French mapping agency.
 
- This work was partly supported by [ASP](https://www.asp-public.fr), the French Payment Agency. 