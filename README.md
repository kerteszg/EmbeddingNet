# Siamese and Triplet networks for image classification

Original framework is available at https://github.com/RocketFlash/EmbeddingNet. 

This repository contains the modifications and materials used in experiments described in ***Metric Embedding Learning on Multi-Directional Projections***, submitted to *MDPI Algorithms* to be published as an Open Access research article (*currently under review*). Result logs are available in https://github.com/kerteszg/EmbeddingNet/tree/master/experiment-results.

# Results

The method presented in the paper is based on the MDIPFL transformation, which is applied as a dimensionality reduction technique to compress input images efficiently. These transformed images are fed to Siamese and Triplet Networks, and performance were measured for metric learning with and without multiclass classification pretraining, and for different triplet mining methods. Results are compared with raw image input based approaches.

Scripts for preprocessing and result collection are available in the [notebooks](https://github.com/kerteszg/EmbeddingNet/tree/master/notebooks) dir.

## NIST SD19

To prove the discriminative ability of the proposed method by experiments, the NIST SD19 dataset was used. After highlighting and transformation, different experiments were performed using a similar backbone architecture.

Results showed, that the MDIPFL based approach achieves similar performance, despite of the significantly lower number of parameters.

## ATS-CVPR2016

To apply the method in a real-life problem of object re-identification, the dataset published in the *International Workshop on Automatic Traffic Surveillance of CVPR 2016* is processed similarly. As a backbone architecture, ResNet-18 was applied.

On 10-way one-shot classification, the model trained with only with triplet loss on semi-hard negatives achieved a decent performance of 75.7% for one-shot classification on the validation dataset.

# Installation

Below are the original instructions to setup the environment. To setup the original version of the framework, please refer to the installation notes on [the original repository](https://github.com/RocketFlash/EmbeddingNet).

Clone the repository using:

```bash
git clone git@github.com:kerteszg/EmbeddingNet.git
```

## Install dependencies

Creating a virtual environment is recommended, but not necessary:

```bash
pip install --upgrade pip
venv env
source env/bin/activate
```

The dependencies are more or less the same as for the original *EmbeddingNet*.

- keras
- tensorflow==1.14.0
- tensorflow-gpu==1.14.0 - if applicable, strongly advised
- scikit-learn
- opencv
- matplotlib
- plotly - for interactive t-SNE plot visualization
- [albumentations](https://github.com/albu/albumentations) - for online augmentation during training
- [image-classifiers](https://github.com/qubvel/classification_models) - for different backbone models
- [keras-rectified-adam](https://github.com/CyberZHG/keras-radam) - for cool state-of-the-art optimization


```bash
pip install -r requirements.txt
```

# Train

Before training the dataset should be prepared. This environment handles two directories: **train** and **valid**, and all subdirs below these are representing the categories.

In the paper *Metric Embedding Learning on Multi-Directional Projections*, the [NIST SD19](https://www.nist.gov/srd/nist-special-database-19) and the [ATS-CVPR2016](https://medusa.fit.vutbr.cz/traffic/datasets/) datasets were used. To create the necessary directory structure and for preprocessing, refer to the Jupyter Notebooks.

The prepared configuration files can be found in the **configs** folder. 

```bash
$ python train.py [path to configuration_file]
```

# Test one-shot classification accuracy on all available configs

After training all configs in the **configs** dir, measuring performance can be done using:

```bash
$ python test_allconfigs.py
```

To filter out some of the configs (e.g. by name), refer to the source code.

Once again, kudos to the authors of the original EmbeddingNet framework [1].

# References

[1] Rauf Yagfarov, Vladislav Ostankovich, Aydar Akhmetzyanov. [Traffic Sign Classification Using Embedding Learning Approach for Self-driving Cars](https://link.springer.com/chapter/10.1007/978-3-030-44267-5_27), IHIET–AI 2020

[2] Kertész, Gábor, Sándor Szénási, and Zoltán Vámossy. [Multi-directional image projections with fixed resolution for object matching](https://www.researchgate.net/profile/Gabor_Kertesz/publication/324775359_Multi-Directional_Image_Projections_with_Fixed_Resolution_for_Object_Matching/links/5ae1bdee458515c60f668f9c/Multi-Directional-Image-Projections-with-Fixed-Resolution-for-Object-Matching.pdf) Acta Polytechnica Hungarica 15.2 (2018): 211-229.