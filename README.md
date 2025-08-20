# DL-SLICER: Deep Learning for Satellite-Based Identification of Cities with Enhanced Resemblance

![image](https://github.com/IS2AI/city-similarity/assets/5821328/330f7d3d-e5a5-4f42-8631-e3bf837bd8df)
<h6 align="center"> Figure 1: Cities represented by their International Air Transport Association (IATA) codes and corresponding raw sattelite views from one of the sites. Raw site images are part of the dataset and were used for city identification model training. </h6>

This repository contains codes and data for performing city classification task. Original dataset contains raw 1585 images of 4800 x 4800 resolution (16 GB). 

Published Paper is [here](https://www.mdpi.com/2682968).

# Access

Raw data, processed dataset can be downloaded from our Hugging Face: https://huggingface.co/datasets/issai/DL-SLICER-dataset
Alternatively, models only can be accessed at [this google drive link](https://drive.google.com/drive/folders/1-7C7YY3ejCsLZlXKM5o0E8kT5IY2ROyK?usp=sharing).

Dataset consists of 45 cities from various locations, and mostly chosen from [Arcadis Index 2022](https://www.arcadis.com/en/knowledge-hub/perspectives/global/sustainable-cities-index). Additional 8 cities (Almaty, Ankara, Ashgabat, Astana, Baku, Bishkek, Shymkent, and Tashkent) underrepresented in Arcadis Index is also part of the dataset.

# Downloading the repository

```
git clone https://github.com/IS2AI/city-identification
```

# Data pre-processing 

Prior to training it is necessary to perform pre-processing on raw images. To generate patches out of raw images needed for training, and to perform train-val-test split launch the following script:

```
python3 preprocessing.py
```
Raw images are of 4800 x 4800 pixels resolution. To capture more specific features of satellite imagery, pre-proceesed images of size 480 × 480 resolution (which corresponds to 200 m by 200 m area) and with an overlap of 240 pixels are generated.

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

# Class Activation Maps

We forked Relevance-CAM from [this](https://github.com/mongeoroo/Relevance-CAM) repo. Modified it for our model and produced the Saliency maps for 10 random patches for each city. Salient features of each city is vizualized via depth-wise heatmaps and masked images, full set of saliency maps for randomly selected pre-processed sattelite patches is available [here](https://drive.google.com/drive/folders/1ryIsorRSUBuroRSG3gmCJCwrGWvK6uxQ?usp=sharing)

# If you use the dataset/source code/pre-trained models in your research, please cite our work:

```
@Article{buildings14020551,
AUTHOR = {Bissarinova, Ulzhan and Tleuken, Aidana and Alimukhambetova, Sofiya and Varol, Huseyin Atakan and Karaca, Ferhat},
TITLE = {DL-SLICER: Deep Learning for Satellite-Based Identification of Cities with Enhanced Resemblance},
JOURNAL = {Buildings},
VOLUME = {14},
YEAR = {2024},
NUMBER = {2},
ARTICLE-NUMBER = {551},
URL = {https://www.mdpi.com/2075-5309/14/2/551},
ISSN = {2075-5309},
ABSTRACT = {This paper introduces a deep learning (DL) tool capable of classifying cities and revealing the features that characterize each city from a visual perspective. The study utilizes city view data captured from satellites and employs a methodology involving DL-based classification for city identification, along with an Explainable Artificial Intelligence (AI) tool to unveil definitive features of each city considered in this study. The city identification model implemented using the ResNet architecture yielded an overall accuracy of 84%, featuring 45 cities worldwide with varied geographic locations, Human Development Index (HDI), and population sizes. The portraying attributes of urban locations have been investigated using an explanatory visualization tool named Relevance Class Activation Maps (CAM). The methodology and findings presented by the current study enable decision makers, city managers, and policymakers to identify similar cities through satellite data, understand the salient features of the cities, and make decisions based on similarity patterns that can lead to effective solutions in a wide range of objectives such as urban planning, crisis management, and economic policies. Analyzing city similarities is crucial for urban development, transportation strategies, zoning, improvement of living conditions, fostering economic success, shaping social justice policies, and providing data for indices and concepts such as sustainability and smart cities for urban zones sharing similar patterns.},
DOI = {10.3390/buildings14020551}
}
```

