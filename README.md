# Octo+: A Suite for Automatic Open-Vocabulary Object Placement in Mixed Reality
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/octo-pearl/octo-pearl/blob/main/demo.ipynb) [![Homepage](https://img.shields.io/badge/🌐-Homepage-blue)](https://octo-pearl.github.io/) [![arXiv](https://img.shields.io/badge/📖-arXiv-b31b1b)](https://octo-pearl.github.io/)

This repo contains the code and data for the paper "[Octo+: A Suite for Automatic Open-Vocabulary Object Placement in Mixed Reality](https://octo-pearl.github.io/)".

## Overview
![Teaser Image](assets/figure.png)

Augmented reality involves overlaying virtual objects on the real world. To enhance the realism, these virtual objects should be placed in natural locations. For example, it would be more natural to place a cupcake on a plate than on the wall. In this repo, we include:
- OCTO+ (and other methods we experimented with):
  - Input: image (e.g. an AR camera frame) and text naming an object (e.g. "cupcake")
  - Output: 2D location where the object should be placed (ray casting can be used to determine the corresponding 3D location)
- PEARL, a benchmark for **P**lacement **E**valuation of **A**ugmented **R**eality E**L**ements
  - Dataset containing pairs of images[^1] and text naming an object to be placed in the image, as well as a segmentation mask indicating which (x, y) pixel locations in the image are valid and which are not
  - Automated metrics to evaluate how well the objects are placed in the image

[^1]: The images (and segmentation masks used by PEARL) come from [RMRC 2014](https://cs.nyu.edu/~silberman/rmrc2014/indoor.php), which itself uses images from the [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [Sun3D Dataset](https://sun3d.cs.princeton.edu/).

## Installation
```
git clone https://github.com/octo-pearl/octo-pearl.git
cd octo-pearl
pip install -e .
```

The included placement techniques require model weights to be downloaded to `octo-pearl/weights`. Weights are not required to run the PEARL benchmark. To download all weights, run the following commands from the project root, `octo-pearl` (each individual pipeline/stage only requires a subset of the following):
```
mkdir -p weights
wget -q -O weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -q -O weights/ram_plus_swin_large_14m.pth https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
wget -q -O weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Usage
For examples on how to use OCTO+ and some other placement techniques, and how to use PEARL on a single image-object pair, see the [Colab demo](https://colab.research.google.com/github/octo-pearl/octo-pearl/blob/main/demo.ipynb).

To run the PEARL on the full dataset, create a JSON file with the (x, y) placement location for every object for every image in [pearl_imgs_objs.json](octo_pearl/eval/data/pearl_imgs_objs.json). A subset of the image-object pairs can also be used. The JSON file should be in the same format as [example_placements.json](octo_pearl/eval/data/example_placements.json). Then, run the benchmark with the following command (it may take several seconds if using the entire dataset):

```
python octo_pearl/eval/pearl.py --placements octo_pearl/eval/data/example_placements.json 
```
Replace `octo_pearl/eval/data/example_placements.json` with the path to the JSON file containing the placement locations.