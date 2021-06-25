# PAnoptic Segmentation of satellite image TIme Series - PASTIS Benchmark


## Dataset presentation
PASTIS is a benchmark dataset for panoptic and semantic segmentation of agricultural parcels from 
satellite image time series. It is composed of 2433 one square kilometer-patches in the French metropolitan territory for which sequences 
of satellite observations are assembled into a four-dimensional spatio-temporal tensor. 
The dataset contains both semantic and instance annotations, assigning to each pixel a semantic label and an instance id.
There is an official 5 fold split provided in the dataset's metadata.
## Usage 
The dataset can be downloaded from [zenodo]().

This repository also contains a PyTorch dataset class in `dataloader.py` that can be readily used to load data for training.

## Leaderboard

### Semantic Segmentation
| Model name         | #Params| OA  |  mIoU |
| ------------------ |---- |---- | ---|
| U-TAE   |   1.1M|  83.2%    | 63.1%|
| Unet-3d   | 1.6M|    81.3%    |  58.4%|
| Unet-ConvLSTM |1.5M  |     82.1%    |  57.8%|
| FPN-ConvLSTM  | 1.3M|    81.6%   |  57.1%|



### Panoptic Segmentation
| Model name         | SQ  | RQ | PQ|
| ------------------ |--- | --- |--- |
| U-TAE + PaPs       | | |   |


## Documentation
### Nomenclature
![](images/Nomenclature.jp2)

## References