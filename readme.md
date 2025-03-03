# multisesh

## Overview
multisesh is a python project for loading complex image datasets and metadata from various microscopy formats into a uniform data structure suitable for scalebale processing and analysis. It is designed for use in jupyter notebooks with an emphasis on allowing simple custom processing pipelines with 

## Features
- **Memory management**: load the data chunks you want by specifying positions along any of the 7 common microscopy dimensions: Time, Well, Montage tile, Z, Channel, Y, X.
- **Data visualisation**: interactively display image data and associated segmentations in-notebook
- **Processing toolbox**: many common image analysis tasks are already included (segmentation, projections, stabilisation, flat field correction, tile stitching, tracking...) and it is easily extensible

## Installation
Create a conda environment:
```bash
conda create -n multisesh_env python=3.11.4
conda activate multisesh_env
```

To install the package and dependencies, navigate to the location of the packages pyproject.toml file and:
```bash
pip install -e .
```


## Usage
See the notebook 'MultiSeshTutorial.ipynb'


## Supported datatypes/microscopes:
* czi
* lif
* nd2
* Opera
* Incucyte
* Andor
* OME
* Micromanager
* General tiffs

BUT: within these filetypes there are some things that aren't supported yet. E.g. I haven't ever worked with a lif file containing multiple time-points so haven't been able to see how to read that properly. These things are fast to add so let me know if you have examples!