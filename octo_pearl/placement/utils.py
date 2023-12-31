from typing import List, Tuple

import groundingdino.config.GroundingDINO_SwinT_OGC
import numpy as np
import torch
from groundingdino.util.inference import (
    Model,
)
from PIL import Image
from segment_anything import sam_model_registry
from supervision import Detections
from transformers import (
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
clipseg_processor = None
clipseg_model = None

gdino_model = None

sam_model = None


def get_clipseg_heatmap(image: Image.Image, prompt: str) -> np.ndarray:
    global clipseg_processor, clipseg_model
    if clipseg_processor is None or clipseg_model is None:
        clipseg_processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        clipseg_model.to(device)

    inputs = clipseg_processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = clipseg_model(**inputs)
        preds = outputs.logits

    heatmap = preds.cpu().numpy()
    return heatmap


def get_gdino_result(
    image: Image.Image,
    classes: List[str],
    box_threshold: float = 0.25,
    weights_folder="weights",
) -> Tuple[Detections, List[str]]:
    global gdino_model

    if gdino_model is None:
        config_path = groundingdino.config.GroundingDINO_SwinT_OGC.__file__
        gdino_model = Model(
            model_config_path=config_path,
            model_checkpoint_path=f"{weights_folder}/groundingdino_swint_ogc.pth",
            device=device,
        )

    detections, phrases = gdino_model.predict_with_caption(
        image=np.array(image),
        caption=", ".join(classes),
        box_threshold=box_threshold,
        text_threshold=0.25,
    )

    return detections, phrases


def get_sam_model(weights_folder="weights"):
    global sam_model
    if sam_model is None:
        sam_checkpoint = f"{weights_folder}/sam_vit_h_4b8939.pth"
        sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam_model.to(device=device)
    return sam_model
