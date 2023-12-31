# Octo+: A Suite for Automatic Open-Vocabulary Object Placement in Mixed Reality
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/octo-pearl/octo-pearl/blob/main/demo.ipynb) [![Homepage](https://img.shields.io/badge/🌐-Homepage-blue)](https://octo-pearl.github.io/) [![arXiv](https://img.shields.io/badge/📖-arXiv-b31b1b)](https://octo-pearl.github.io/)

This repo contains the code and data for the paper "[Octo+: A Suite for Automatic Open-Vocabulary Object Placement in Mixed Reality](https://octo-pearl.github.io/)".

## Overview
![Teaser Image](assets/teaser.png)

Augmented reality involves overlaying virtual objects on the real world. To enhance the realism, these virtual objects should be placed in natural locations. For example, it would be more natural to place a cupcake on a plate than on the wall. In this repo, we include:
- OCTO+ (and other methods we experimented with):
  - Input: image (e.g. an AR camera frame) and text naming an object (e.g. "cupcake")
  - Output: 2D location where the object should be placed (ray casting can be used to determine the corresponding 3D location)
- (TODO) PEARL, a benchmark for **P**lacement **E**valuation of **A**ugmented **R**eality E**L**ements
  - Dataset containing pairs of images and text naming an object to be placed in the image, as well as a segmentation mask indicating which (x, y) pixel locations in the image are valid and which are not
  - Automated metrics to evaluate how well the objects are placed in the image


## Installation
```
git clone https://github.com/octo-pearl/octo-pearl.git
cd octo-pearl
pip install -e .
```
