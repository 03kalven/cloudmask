# cloudMask

This repository contains all the code to train and use the cloudMask model.

A breakdown of the files and folders:
- "data" contains a sample of training and testing data
- "save_cloudModel" contains the trained model's weights
- "model.py" holds the model's architecture and loss function
- "predict.ipynb" is a Jupyter Notebook that allows users to quickly create a cloud mask for a given MDGM image. Automatically applies smooth tiling processes. Can additionally create file directories of many predictions for larger jobs
- "training.ipynb" contains all the necessary elements to train the cloudMask model
- "preprossingTrainingData.ipynb" performs preprocessing and data augmentation on the training data. You must preprocess the training data before using "training.ipynb"
- "utils.py" contains a collection of useful functions for other files in this repo
- "viewMasks.ipynb" has a few handy methods to plot cloud masks and MDGMs