"""Utilities for image representation in Jupyter.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import imagehash
from typing import Union, Tuple

def average_hash_mp(_input):
    path, hash_size = _input  # path, int
    image = Image.open(path)
    _hash = imagehash.average_hash(image, hash_size)
    return (_hash, path)

def average_hash_mp_o(_input):
    (dataset, path), hash_size = _input  # path, int
    image = Image.open(path)
    _hash = imagehash.average_hash(image, hash_size)
    return (_hash, dataset, path)


def show_images(images: list, titles: list = [], resize: int = None, crop=False,
                imsize: tuple = (7, 5), cmap=None, per_row: int = 2,
                keep_ticks=False, font_size=16):
    """Returns a grid with provided images as subplots.

    args:
    images: list of images (file paths)
    titles: list of image titles
    resize: resize shortest edge of images to value in pixels
    crop: crop images to square around center
    imsize: plot-size of images in grid
    per_row: number of images per row
    cmap, keep_ticks, font_size: standard plt options
    """

    # Make sure titles is same length as ls_img
    if len(titles):
        assert len(titles) == len(images), \
            "Error: number of provided images and titles must be the same."
    else:
        titles = [''] * len(images)

    # Prepare images
    imgs = []
    for file_path in images:
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape

        if resize is not None:
            px = resize
            img = resize_shortest_edge(img, px, interpolation='auto')
        else:
            px = min(height, width)

        if crop is True:
            img = center_crop(img, (px, px))

        imgs.append(img)

    # Prepare figure
    num_rows = len(imgs) // per_row + ((len(imgs) % per_row) > 0)
    fig, ax = plt.subplots(num_rows, per_row, figsize=(
        imsize[0] * per_row, imsize[1] * num_rows))
    if type(ax) == np.ndarray:
        ax = ax.flatten()
    else:
        ax = np.array([ax])

    # Populate figure
    for i, img in enumerate(imgs):
        this_cmap = cmap
        if this_cmap is None and (len(img.shape) == 2 or img.shape[-1] == 1):
            this_cmap = 'gray'
        ax[i].imshow(img, cmap=this_cmap, vmin=0, vmax=255)
        ax[i].set_title(titles[i], fontdict={'fontsize': font_size})
        if not keep_ticks:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    return fig, ax


def center_crop(img: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop image to square around center.

    Args:
    img: image to be cropped
    crop_size: size of crop; must be smaller than or equal to the image size
    """
    assert (img.shape[0] >= crop_size[0]) and (img.shape[1] >= crop_size[1])
    y0 = img.shape[0] // 2 - crop_size[0] // 2
    y1 = y0 + crop_size[0]
    x0 = img.shape[1] // 2 - crop_size[1] // 2
    x1 = x0 + crop_size[1]
    return img.copy()[y0:y1, x0:x1]


def resize_shortest_edge(
        img: np.ndarray, length: int,
        interpolation: Union[int, str] = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image with locked aspect ratio to shortest edge.

    args:
    length: length of shortest edge
    interpolation: specifies the cv2 interpolation type and defaults to
    cv2.INTER_LINEAR; it may be set to 'auto' which sets the interpolation
    to cv2.INTER_AREA or cv2.INTER_CUBIC depending on whether the image
    is down- or upsized
    """
    f = length / np.min(img.shape[:2])
    if isinstance(interpolation, str):
        assert interpolation == 'auto', \
            "If `interpolation` is a str it can only be 'auto'"
        interpolation = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (0, 0), fx=f, fy=f, interpolation=interpolation)