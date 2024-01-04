from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from segment_anything import SamPredictor

from octo_pearl.placement.utils import (
    get_clipseg_heatmap,
    get_gdino_result,
    get_sam_model,
)


def get_location_clipseg(image: Image.Image, prompt: str) -> Tuple[int, int]:
    heatmap = get_clipseg_heatmap(image, prompt)
    cy, cx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    cx = int(image.size[0] * cx / 352)
    cy = int(image.size[1] * cy / 352)
    return cx, cy


def get_location_gsam(
    image: Image.Image, prompt: str, weights_folder="weights"
) -> Tuple[int, int]:
    BOX_TRESHOLD = 0.25
    RESIZE_RATIO = 3

    detections, phrases = get_gdino_result(
        image=image,
        classes=[prompt],
        box_threshold=BOX_TRESHOLD,
    )

    while len(detections.xyxy) == 0:
        BOX_TRESHOLD -= 0.02
        detections, phrases = get_gdino_result(
            image=image,
            classes=[prompt],
            box_threshold=BOX_TRESHOLD,
        )

    sam_model = get_sam_model(weights_folder)
    sam_predictor = SamPredictor(sam_model)

    sam_predictor.set_image(np.array(image))
    result_masks = []
    for box in detections.xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    detections.mask = np.array(result_masks)

    combined_mask = detections.mask[0]
    for mask in detections.mask[1:]:
        combined_mask += mask
    combined_mask[combined_mask > 1] = 1
    mask = cv2.resize(
        combined_mask.astype("uint8"),
        (
            combined_mask.shape[1] // RESIZE_RATIO,
            combined_mask.shape[0] // RESIZE_RATIO,
        ),
    )

    mask_2_pad = np.pad(mask, pad_width=2, mode="constant", constant_values=0)
    mask_1_pad = np.pad(mask, pad_width=1, mode="constant", constant_values=0)

    windows = np.lib.stride_tricks.sliding_window_view(mask_2_pad, (3, 3))
    windows_all_zero = (windows == 0).all(axis=(2, 3))

    # if the 3x3 window around a pixel is all 0, it is irrelevant, so we set it to 2
    result = np.where(windows_all_zero, 2, mask_1_pad)
    mask_0_coordinates = np.argwhere(result == 0)
    mask_1_coordinates = np.argwhere(result == 1)
    distances = cdist(mask_1_coordinates, mask_0_coordinates, "euclidean")
    max_min_distance_index = np.argmax(np.min(distances, axis=1))
    y, x = mask_1_coordinates[max_min_distance_index]

    return int(x) * RESIZE_RATIO, int(y) * RESIZE_RATIO
