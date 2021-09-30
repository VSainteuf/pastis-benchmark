# :ear_of_rice: PASTIS :ear_of_rice: Panoptic Agricultural Satellite TIme Series

![](images/predictions.png)

## The PASTIS Dataset

- **Dataset presentation**

PASTIS is a benchmark dataset for panoptic and semantic segmentation of agricultural parcels from satellite time series. 
It contains 2,433 patches within the French metropolitan territory with panoptic annotations (instance index + semantic labelfor each pixel). Each patch is a Sentinel-2 multispectral image time series of variable lentgh. 

We propose an official 5 fold split provided in the dataset's metadata, and evaluated several of the top-performing image time series networks. You are welcome to use our numbers and to submit your own entries to the leaderboard!

- **Dataset in numbers**

:arrow_forward: 2,433 time series             |  :arrow_forward: 124,422 individual parcels         | :arrow_forward: 18 crop types   
:-------------------------------------------- | :-------------------------------------------------- | :------------------------------
:arrow_forward: **128x128 pixels / images**   | :arrow_forward:  **38-61 acquisitions / series**    | :arrow_forward:  **10m / pixel** 
:arrow_forward:  **10 spectral bands**        | :arrow_forward: **covers ~4,000 km²**                       | :arrow_forward: **over 2B pixels**


## Usage 
- **Download** 

The dataset can be downloaded from [zenodo](https://zenodo.org/record/5012942).
- **Dataloader** 

This repository also contains a PyTorch dataset class in `code/dataloader.py` 
that can be readily used to load data for training.
- **Metrics** 

A PyTorch implementation is also given in `code/panoptic_metrics.py` to compute
the panoptic metrics. In order to use these metrics, the model's output should contain an instance prediction
and a semantic prediction. The first one allocates an instance id to each pixel of the image, 
and the latter a semantic label.

## Leaderboard
Please open an issue to submit new entries. Do mention if the work has been published and wether the code accessible for reproducibility. We require that at least a preprint is available to present the method used.

### Semantic Segmentation
| Model name         | #Params| OA  |  mIoU | Published |
| ------------------ |---- |---- | ---| --- |
| U-TAE   |   1.1M|  83.2%    | 63.1%|  :heavy_check_mark: [link](https://arxiv.org/pdf/2107.07933.pdf)|
| Unet-3d*   | 1.6M|    81.3%    |  58.4%| :heavy_check_mark: [link](http://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.html)|
| Unet-ConvLSTM* |1.5M  |     82.1%    |  57.8%| :heavy_check_mark: [link](http://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.html)|
| FPN-ConvLSTM*  | 1.3M|    81.6%   |  57.1%|:heavy_check_mark: [link](https://www.sciencedirect.com/science/article/pii/S0924271620303142?casa_token=uhkmVE-Lk94AAAAA:r6USZEEFMFE2qc2uYZSrqTzy1_DSI9hflG2cVeay-2Bd-PHFIg3CPwgisf7jatDDfRnR4ROzN9k)|

Models that we re-implemented ourselves are denoted with a star (*).

### Panoptic Segmentation
| Model name         | SQ  | RQ | PQ|
| ------------------ |--- | --- |--- |
| U-TAE + PaPs       | 81.3|49.2 |40.4|

## Documentation
The agricultural parcels are grouped into 18 different crop classes as shown in the 
table below. The backgroud class corresponds to non-agricultural land, and the void label for parcels that are mostly outside their patch.
<img src="images/Nomenclature.png" alt="drawing" width="300"/>

Additional information about the dataset can be found in the `documentation/pastis-documentation.pdf` document.

## References
If you use PASTIS please cite the [related paper](https://arxiv.org/abs/2107.07933):
```
@article{garnot2021panoptic,
  title={Panoptic Segmentation of Satellite Image Time Series
with Convolutional Temporal Attention Networks},
  author={Sainte Fare Garnot, Vivien  and Landrieu, Loic },
  journal={ICCV},
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
