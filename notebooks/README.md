# Notebooks

This directory contains scripts which were used to preprocess the [NIST SD19](https://www.nist.gov/srd/nist-special-database-19) and the [ATS-CVPR2016](https://medusa.fit.vutbr.cz/traffic/datasets/) datasets.

## NIST SD19

The images are cropped to square and inverted. *imcroplib.py* contains the functions, *ProcessDirPrepare.ipynb* restructures the files of the dataset to a structure which is accepted by the framework, *ProcessDirImCrop.ipynb* was used for highlighting and finally *ProcessDirImInvert.ipynb* inverts all images.

## ATS-CVPR2016

*extract.py* was used to clip images from the videos, *rearrange_images.py* creates the dataset of extracted observations in a structure that is accepted by the framework.

For the experiments all images were transformed to MDIPFL50 format, explained in [Multi-directional image projections with fixed resolution for object matching](https://www.researchgate.net/profile/Gabor_Kertesz/publication/324775359_Multi-Directional_Image_Projections_with_Fixed_Resolution_for_Object_Matching/links/5ae1bdee458515c60f668f9c/Multi-Directional-Image-Projections-with-Fixed-Resolution-for-Object-Matching.pdf)

## Experiment runtimes from TensorBoard logs

*RuntimesFromLogs.ipynb* processes the CSV exports of TensorBoard to show the step numbers and total processing times from the saved unix timestamp differences. Logs are available in https://github.com/kerteszg/EmbeddingNet/tree/master/experiment-results