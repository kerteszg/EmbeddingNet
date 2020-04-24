# Siamese and Triplet networks for image classification

Original framework is available at https://github.com/RocketFlash/EmbeddingNet. 

This repository contains the modifications and materials used in experiments described in *Metric Embedding Learning on Multi-Directional Projections*, submitted to *MDPI Algorithms*. Result logs are available in https://github.com/kerteszg/EmbeddingNet/tree/master/experiment-results.

Below are the original instructions to setup the environment. To setup the original version of the framework, please refer to the installation notes on [the original repository](https://github.com/RocketFlash/EmbeddingNet).

# Installation

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

Once again, kudos to the authors of the original framework, referred to 

# References

[0] Rauf Yagfarov, Vladislav Ostankovich, Aydar Akhmetzyanov. [Traffic Sign Classification Using Embedding Learning Approach for Self-driving Cars](https://link.springer.com/chapter/10.1007/978-3-030-44267-5_27), IHIET–AI 2020

[1] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering.](https://arxiv.org/abs/1503.03832) CVPR 2015

[2] Alexander Hermans, Lucas Beyer, Bastian Leibe, [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737), 2017

[3] Adam Bielski [Siamese and triplet networks with online pair/triplet mining in PyTorch](https://github.com/adambielski/siamese-triplet)

[4] Olivier Moindrot [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)
