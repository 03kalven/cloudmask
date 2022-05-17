# cloudmask

## Overview
This repository contains all the code to train and use the cloudmask model. This cloudmask model identifies water-ice clouds in Mars Daily Global Maps. 

### Dependencies:
Use Python >=3.9.0. Dependencies for this model are stored in [requirements.txt](requirements.txt).

### Quick breakdown of the files and folders:
- [data](data) contains a sample of training and testing data
- [save_cloud_model](save_cloud_model) contains the pretrained model's weights
- [model.py](model.py) holds the model's architecture and loss function
- [predict.ipynb](predict.ipynb) is a Jupyter Notebook that allows users to quickly create a cloud mask for a given MDGM image. Automatically applies smooth tiling processes. Can additionally create file directories of many predictions for larger jobs
- [training.ipynb](training.ipynb) contains all the necessary elements to train the cloudMask model
- [preprocessing_training_data.ipynb](preprocessing_training_data.ipynb) performs preprocessing and data augmentation on the training data. You must preprocess the training data before using "training.ipynb"
- [utils.py](utils.py) contains a collection of useful functions for other files in this repo
- [view_masks.ipynb](view_masks.ipynb) has a few handy methods to plot cloud masks and MDGMs

## Using the model
The parameters for the pretrained model are in the [save_cloud_model](save_cloud_model) folder. [predict.ipynb](predict.ipynb) is made for loading and using the model. For each MDGM, use `subdivs_compute()` to get the cloud mask as an array and then `make_netCDF()` to convert the numpy array into a NetCDF4 file.

MDGMs can be computed in bulk via the final cell in [predict.ipynb](predict.ipynb). Be sure to store your MDGMs in a filesystem as shown:
```
B
│
└───B01
│   │   B01_ls.txt
│   │
│   │   B01_day01_zequat.jpg
│   │   B01_day02_zequat.jpg
│   │   B01_day32_zequat.jpg
│   │   ...
│   │
│   └───list
│       │   B01_day01.list
│       │   B01_day02.list
│       │   B01_day03.list
│       │   ...
│
└───B02
│
└───B03
...
```
This format should be satisfied if you are downloading MDGMs directly from [here](https://doi.org/10.7910/DVN/U3766S). The folder should contain one or more years of data.

## Training the model
The model has already been trained, with an accuracy of about 97%. However, if you desire you may train the model yourself. Training data for this model is generated from the cloud masks by [Wang and González Abad [2021]](https://doi.org/10.3390/geosciences11080324). The data should be stored in [data/train](data/train).

Before training, the downloaded MDGMs and cloud masks must pre preprocessed via [preprocessing_training_data.ipynb](preprocessing_training_data.ipynb). Run the cells in the notebook. The folder [data/train_processed](data/train_processed) should now contain subdivided MDGMs and cloud masks.

Now run [training.ipynb](training.ipynb) to train the model. You must have an NVIDIA GPU compliant with Tensorflow to train the model. Depending on VRAM constraints, adjust the `BATCH_SIZE` parameter accordingly. I trained the model with one NVIDIA RTX 3060 12GB. Be sure to save the model weights during and after training.