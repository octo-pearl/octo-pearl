import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image

dir = os.path.dirname(__file__)
ground_truth = json.load(open(f"{dir}/data/pearl_ground_truth.json"))
label_to_ids = json.load(open(f"{dir}/data/label_to_ids.json"))


def create_mask(img_name: str, obj_name: str) -> np.ndarray:
    gt_locs = ground_truth[img_name][obj_name]
    mask = Image.open(f"{dir}/data/masks/{img_name}")
    np_mask = np.array(mask)
    valid_mask = np.zeros_like(np_mask)
    for gt_loc in gt_locs:
        for id in label_to_ids[gt_loc]:
            valid_mask += np.where(np_mask == id, 1, 0)
    return valid_mask


def placement_score(
    img_name: str, obj_name: str, x: float, y: float
) -> Tuple[int, float]:
    valid_mask = create_mask(img_name, obj_name)
    in_mask = valid_mask[round(y), round(x)]

    coords = np.argwhere(valid_mask == 1 - in_mask)
    distances = np.linalg.norm(coords - np.array([round(y), round(x)]), axis=1)
    closest_dist = np.min(distances)
    diff = pow(-1, 1 - in_mask) * closest_dist
    return in_mask, diff


def overall_placement_score(placements: Dict[str, Dict[str, Tuple[float, float]]]):
    num_placements = 0
    skipped_placements = 0
    mask_score = 0
    pearl_score = 0
    for img, objs in ground_truth.items():
        if img not in placements:
            skipped_placements += len(ground_truth[img])
            continue

        for obj in objs:
            if obj not in placements[img]:
                skipped_placements += 1
                continue

            x, y = placements[img][obj]
            in_mask, diff = placement_score(img, obj, x, y)

            mask_score += in_mask
            pearl_score += diff
            num_placements += 1

    mask_score /= num_placements
    pearl_score /= num_placements
    return {
        "num_placements": num_placements,
        "skipped_placements": skipped_placements,
        "mask_score": mask_score,
        "pearl_score": pearl_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Placements")
    parser.add_argument("--placements", type=str, help="Path to placements json file")
    args = parser.parse_args()

    placements = json.load(open(args.placements))
    print(overall_placement_score(placements))
