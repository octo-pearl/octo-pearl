from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import (
    ViltForQuestionAnswering,
    ViltProcessor,
)

from octo_pearl.utils import get_clipseg_heatmap, get_gdino_result

vqa_model = None
vqa_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def filter_tags_clipseg(image: Image.Image, tags: List[str], K: int = 20) -> List[str]:
    tag_to_max = {}
    for tag in tags:
        heatmap = get_clipseg_heatmap(image, tag)
        tag_to_max[tag] = np.max(heatmap)
    sorted_tags = sorted(tag_to_max.items(), key=lambda x: x[1], reverse=True)[:K]
    return [tag for tag, _ in sorted_tags]


def filter_tags_vilt(image: Image.Image, tags: List[str]) -> List[str]:
    global vqa_model, vqa_processor
    if vqa_model is None:
        vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vqa_model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        ).to(device)

    filtered_tags = []
    for tag in tags:
        question = f"Is there a {tag} in the image?"
        encoding = vqa_processor(image, question, return_tensors="pt").to(device)
        outputs = vqa_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        in_image = vqa_model.config.id2label[idx]
        if "yes" in in_image.lower():
            filtered_tags.append(tag)

    return filtered_tags


def filter_tags_gdino(image: Image.Image, tags: List[str]) -> List[str]:
    detections, phrases = get_gdino_result(image, tags)
    filtered_tags = []
    for tag in tags:
        for (
            phrase,
            area,
        ) in zip(phrases, detections.area):
            if area < 0.9 * image.size[0] * image.size[1] and tag in phrase:
                filtered_tags.append(tag)
                break
    return filtered_tags
