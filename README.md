# City classification and index prediction

This repository contains codes and data for performing city classification and sustainiable city index (SCI) prediction tasks.

[Original dataset](https://drive.google.com/drive/folders/1HEPV9hhT_X9OWVvjEFNQRQLxbo5A_R9C?usp=sharing) contains raw 1585 images of 4800 x 4800 resolution (16 GB). 

Dataset consists of 45 cities from various locations, and mostly chosen from [Arcadis Index 2022](https://www.arcadis.com/en/knowledge-hub/perspectives/global/sustainable-cities-index). Additional 8 cities (Almaty, Ankara, Ashgabat, Astana, Baku, Bishkek, Shymkent, and Tashkent) underrepresented in Arcadis Index is also part of the dataset.

# Downloading the repository

```
$ git clone https://github.com/IS2AI/city-classification-and-index-prediction
```

# Pre-processing 

Prior to training it is necessary to perform pre-processing on raw images. To generate patches out of raw images needed for training, and to perform train-val-test split launch the following script:

```
python3 preprocessing.py
```

# Training

To launch training for city classification use the following script:
```
python3 train_classification.py
```

Similarly, to launch training for city index predition use:
```
python3 train_regression.py
```

# Inference

For testing out city classificaiton model performance on unseen patches run:
```
python3 test_classification.py
```

Similarly, to test city index prediciton on unseen patches run:
```
python3 test_regression.py
```

# Sustainability map creation

To create sustainability color map, there is a available another script:

```
python3 make_sustainability_map.py
```

