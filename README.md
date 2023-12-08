# Deep learning-based identification of similar cities using satellite images

![image](https://github.com/IS2AI/city-similarity/assets/5821328/330f7d3d-e5a5-4f42-8631-e3bf837bd8df)
<h6 align="center"> Figure 1: Cities represented by their International Air Transport Association (IATA) codes and corresponding raw sattelite views from one of the sites. Raw site images are part of the dataset and were used for city identification model training. </h6>

This repository contains codes and data for performing city classification task. Original dataset contains raw 1585 images of 4800 x 4800 resolution (16 GB). 

# Access

Raw data, processed dataset, and models altogether can be downloaded by filling out [this form](https://forms.gle/vsg8SqTB1V6iqXx3A).
Alternatively, models only can be accessed at [this google drive link](https://drive.google.com/drive/folders/1-7C7YY3ejCsLZlXKM5o0E8kT5IY2ROyK?usp=sharing).

Dataset consists of 45 cities from various locations, and mostly chosen from [Arcadis Index 2022](https://www.arcadis.com/en/knowledge-hub/perspectives/global/sustainable-cities-index). Additional 8 cities (Almaty, Ankara, Ashgabat, Astana, Baku, Bishkek, Shymkent, and Tashkent) underrepresented in Arcadis Index is also part of the dataset.

# Downloading the repository

```
$ git clone https://github.com/IS2AI/city-classification-and-index-prediction
```

# Data pre-processing 

Prior to training it is necessary to perform pre-processing on raw images. To generate patches out of raw images needed for training, and to perform train-val-test split launch the following script:

```
python3 preprocessing.py
```

# City Identification Model: Training

To launch training for city classification use the following script:
```
python3 train_classification.py
```

# City Identification Model: Inference

For testing out city classificaiton model performance on unseen patches run:
```
python3 test_classification.py
```

# Saliency maps

Salient features of each city is vizualized via depth-wise heatmaps and masked images, full set of saliency maps for pre-processed sattelite patches is available [here](https://drive.google.com/drive/folders/1ryIsorRSUBuroRSG3gmCJCwrGWvK6uxQ?usp=sharing)
