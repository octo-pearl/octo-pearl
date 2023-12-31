from os import system
from sys import executable

from setuptools import find_packages, setup

PREREQS = ["torch"]  # GroundingDINO assumes torch is installed
for prereq in PREREQS:
    system(f"{executable} -m pip install {prereq}")

setup(
    name="octo-pearl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flair",
        "numpy",
        "Pillow",
        "clip-text-decoder",
        "ram @ git+https://github.com/xinyu1205/recognize-anything.git",
        "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git",
        "clip @ git+https://github.com/openai/CLIP.git",
        "salesforce-lavis @ git+https://github.com/salesforce/LAVIS.git@4ad7b8d040eaeb3a83bb2b76a636c964bbbeaedb",
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git",
    ],
)
