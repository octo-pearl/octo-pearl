import os
from typing import List

import numpy as np
import torch
from clip_text_decoder.model import ImageCaptionInferenceModel
from flair.data import Sentence
from flair.models import SequenceTagger
from PIL import Image
from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram_plus
from segment_anything import SamAutomaticMaskGenerator

from octo_pearl.placement.utils import get_sam_model, read_file_to_string, gpt4v

USR_PATH = "octo_pearl/placement/tagging_user_template.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
ram_model = None
clip_decoder_model = None
pos_tagger_model = None
ram_threshold_multiplier = 1


def get_tags_ram(
    image: Image.Image, threshold_multiplier=0.8, weights_folder="weights"
) -> List[str]:
    global ram_model, ram_threshold_multiplier
    if ram_model is None:
        ram_model = ram_plus(
            pretrained=f"{weights_folder}/ram_plus_swin_large_14m.pth",
            vit="swin_l",
            image_size=384,
        )
        ram_model.eval()
        ram_model = ram_model.to(device)

    ram_model.class_threshold *= threshold_multiplier / ram_threshold_multiplier
    ram_threshold_multiplier = threshold_multiplier
    transform = get_transform()

    image = transform(image).unsqueeze(0).to(device)
    res = inference(image, ram_model)
    return [s.strip() for s in res[0].split("|")]


def get_tags_scp(image: Image.Image, weights_folder="weights") -> List[str]:
    global clip_decoder_model, pos_tagger_model
    sam_model = get_sam_model(weights_folder)
    if clip_decoder_model is None or pos_tagger_model is None:
        clip_decoder_checkpoint = f"{weights_folder}/pretrained-model-1.4.0.pt"

        if os.path.exists(clip_decoder_checkpoint):
            clip_decoder_model = ImageCaptionInferenceModel.load(
                clip_decoder_checkpoint
            ).to(device)
        else:
            clip_decoder_model = ImageCaptionInferenceModel.download_pretrained(
                clip_decoder_checkpoint
            ).to(device)
        pos_tagger_model = SequenceTagger.load("flair/pos-english").to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=10_000,
    )
    masks = mask_generator.generate(np.array(image))
    boxes = [mask["bbox"] for mask in masks]

    words = []
    for x, y, w, h in boxes:
        cropped = image.crop((x, y, x + w, y + h))
        caption = clip_decoder_model(cropped, beam_size=1)
        box_words = caption.lower().replace(".", "").split(" ")
        words.extend(box_words)

    sentence = Sentence(words)
    pos_tagger_model.predict(sentence)
    nouns = {
        words[i]
        for i, annotation in enumerate(sentence.annotation_layers["pos"])
        if annotation._value in {"NN", "NNS"}
    }

    return list(nouns)


def get_tags_gpt4v(image_path: str, api_key: str = "") -> List[str]:
    user_prompt = read_file_to_string(USR_PATH)
    response = gpt4v(image_path, user_prompt, api_key=api_key)
    tags = [t.strip().lower() for t in response.replace(".", "").split(",")]
    return sorted(list(set(tags)))
