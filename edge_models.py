"""Wrappers for edge detection models.

This module contains placeholder wrappers for multiple edge detection models.
Each model exposes a `process_folder` method that takes an input directory and an
output directory. Images are processed sequentially with basic edge detection
for demonstration purposes.
"""

import logging
import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


VALID_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}


def is_image_file(filename: str) -> bool:
    """Return True if the filename has a valid image extension."""
    return filename.split(".")[-1].lower() in VALID_EXTENSIONS


@dataclass
class EdgeModel:
    """Base class for edge detection models."""

    name: str

    def process_image(self, image_path: str, output_path: str, device: str = "cpu") -> None:
        """Process a single image and save the edge map."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite(output_path, edges)

    def process_folder(self, input_dir: str, output_dir: str, device: str = "cpu") -> None:
        """Process all images in ``input_dir`` and save them under ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)
        files: List[str] = [f for f in os.listdir(input_dir) if is_image_file(f)]
        if not files:
            logging.warning("%s: No images found in %s", self.name, input_dir)
            return
        for filename in files:
            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
            try:
                self.process_image(src, dst, device)
            except Exception as exc:  # noqa: BLE001
                logging.error("%s: failed to process %s (%s)", self.name, filename, exc)


class HED(EdgeModel):
    name = "HED"


class RCF(EdgeModel):
    name = "RCF"


class BDCN(EdgeModel):
    name = "BDCN"


class DexiNed(EdgeModel):
    name = "DexiNed"


class PiDiNet(EdgeModel):
    name = "PiDiNet"


class EDTER(EdgeModel):
    name = "EDTER"


class UAED(EdgeModel):
    name = "UAED"


class DiffusionEdge(EdgeModel):
    name = "DiffusionEdge"


class RankED(EdgeModel):
    name = "RankED"


class MuGE(EdgeModel):
    name = "MuGE"


class SAUGE(EdgeModel):
    name = "SAUGE"
