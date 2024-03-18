# Clustering of unknown sounds
## Description
This repository gathers python and scripts (R and python) used to generate the output of the publication Calonge et al. (2024) Revised clusters of annotated unknown sounds in the Belgian part of the North Sea. 

## Installation
Necessary python packages can be installed using conda with the environment.yml file, or with pip using requirements.txt.

## Data
To reproduce the results obtained, raw data can be downloaded from https://doi.org/10.14284/659 

## Usage
The python scripts are to extract the acoustic features, train the CAE, and apply a grid search function as explained in the paper. 
This can be re-used for any Raven selection table, and parameters to be used can be defined in a config file following the same structure than the example config.json

## License
When using this code in your own experiments, please cite the corresponding paper.

## References
This work includes some scripts from https://gitlab.lis-lab.fr/paul.best/repertoire_embedder to extract the CAE features. 
> Best P, Paris S, Glotin H, Marxer R (2023) Deep audio embeddings for vocalisation clustering. PLOS ONE 18(7): e0283396. https://doi.org/10.1371/journal.pone.0283396

These script include the filterbank script, from https://github.com/f0k/ismir2015. Copyright (c) 2017 Jan Schlüter
> "Exploring Data Augmentation for Improved Singing Voice Detection with Neural Networks" by Jan Schlüter and Thomas Grill at the 16th International Society for Music Information Retrieval Conference (ISMIR 2015)
