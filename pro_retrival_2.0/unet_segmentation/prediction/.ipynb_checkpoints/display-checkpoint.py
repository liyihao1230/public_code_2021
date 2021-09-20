from PIL import Image

import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from unet_segmentation.unet import Unet
from unet_segmentation.prediction.post_processing import (
    prediction_to_classes,
    mask_from_prediction,
    remove_predictions
)

DEEP_FASHION2_CUSTOM_CLASS_MAP = {
    1: "trousers",
    2: "skirt",
    3: "top",
    4: "dress",
    5: "outwear",
    6: "shorts"
}


def display_prediction(
        unet: Unet,
        image_path: str,
        image_resize: int = 512,
        label_map: dict = DEEP_FASHION2_CUSTOM_CLASS_MAP,
        device: str = 'cuda',
        min_area_rate: float = 0.05) -> None:

    # Load image tensor
    img = Image.open(image_path)
    img_tensor = _preprocess_image(img, image_size=image_resize).to(device)

    # Predict classes from model
    prediction_map = \
        torch.argmax(unet(img_tensor), dim=1).squeeze(0).cpu().numpy() # .cuda().data.cpu.numpy()

    # Remove spurious classes
    classes = prediction_to_classes(prediction_map, label_map)
    predicted_classes = list(
        filter(lambda x: x.area_ratio >= min_area_rate, classes))
    spurious_classes = list(
        filter(lambda x: x.area_ratio < min_area_rate, classes))
    clean_prediction_map = remove_predictions(spurious_classes, prediction_map)

    # Get masks for each of the predictions
    masks = [
        mask_from_prediction(predicted_class, clean_prediction_map)
        for predicted_class in predicted_classes
    ]

    # Display predictions on top of original image
    plt.imshow(np.array(img))
    for mask in masks:
        plt.imshow(mask.resize(img).binary_mask, cmap='jet', alpha=0.65)
    plt.show()
    return masks

def get_masks(
        unet: Unet,
        image: np.ndarray,
        image_resize: int = 512,
        label_map: dict = DEEP_FASHION2_CUSTOM_CLASS_MAP,
        device: str = 'cuda',
        min_area_rate: float = 0.05) -> list:

    # Load image tensor
    img = Image.fromarray(image)
    if torch.cuda.is_available():
        # gpu版本
        img_tensor = _preprocess_image(img, image_size=image_resize).to(device)
    else:
        # cpu版本
        img_tensor = _preprocess_image(img, image_size=image_resize).cpu()

    if torch.cuda.is_available():
        # gpu版本
        prediction_map = \
            torch.argmax(unet(img_tensor), dim=1).squeeze(0).cuda().data.cpu().numpy() # .cpu().numpy()
    else:
        # cpu版本
        prediction_map = \
            torch.argmax(unet(img_tensor), dim=1).squeeze(0).data.numpy() # .cpu().numpy()

    # Remove spurious classes
    classes = prediction_to_classes(prediction_map, label_map)
    predicted_classes = list(
        filter(lambda x: x.area_ratio >= min_area_rate, classes))
    spurious_classes = list(
        filter(lambda x: x.area_ratio < min_area_rate, classes))
    clean_prediction_map = remove_predictions(spurious_classes, prediction_map)

    # Get masks for each of the predictions
    masks = [
        mask_from_prediction(predicted_class, clean_prediction_map)
        for predicted_class in predicted_classes
    ]
    for i in range(len(masks)):
        masks[i] = masks[i].resize(img)
    return masks


def _preprocess_image(image: Image, image_size: int) -> torch.Tensor:
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,))
    ])
    return preprocess_pipeline(image).unsqueeze(0)
