import os
from typing import List, Tuple

import base64
import groundingdino.config.GroundingDINO_SwinT_OGC
import json
import numpy as np
import openai
import requests
import torch
from dotenv import load_dotenv
from groundingdino.util.inference import Model
from PIL import Image
from segment_anything import sam_model_registry
from supervision import Detections
from tenacity import retry, wait_fixed
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

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


def read_file_to_string(file_path: str) -> str:
    content = ""

    try:
        with open(file_path, "r", encoding="utf8") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")

    return content


"""Open AI Utils."""


@retry(wait=wait_fixed(2))
def completion_with_backoff(**kwargs):
    print("Trying")
    return openai.ChatCompletion.create(**kwargs)


def gpt4(
    usr_prompt: str, sys_prompt: str = "", api_key: str = "", model: str = "gpt-4"
) -> str:
    if api_key != "":
        openai.api_key = api_key
    else:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    message = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]

    response = completion_with_backoff(
        model=model,
        messages=message,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0,
    )

    return response["choices"][0]["message"]["content"]


# Function to encode the image for gpt4v
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt4v(
    image_path: str,
    usr_prompt: str,
    sys_prompt: str = "",
    api_key: str = "",
    model: str = "gpt-4-vision-preview",
) -> str:
    if api_key != "":
        openai.api_key = api_key
    else:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }

    usr_content = [
        {"type": "text", "text": usr_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ]

    message = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_content},
    ]

    payload = {
        "model": model,
        "messages": message,
        "max_tokens": 200,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    return (response.json())["choices"][0]["message"]["content"]
