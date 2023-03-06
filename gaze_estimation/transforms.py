from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import yacs.config

from .types import GazeEstimationMethod


def create_transform(config: yacs.config.CfgNode) -> Any:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        return _create_mpiigaze_transform(config)
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        return _create_mpiifacegaze_transform(config)
    else:
        raise ValueError

def tmp_fun1(x):
    return x.astype(np.float32) / 255

def _create_mpiigaze_transform(config: yacs.config.CfgNode) -> Any:
    scale = torchvision.transforms.Lambda(tmp_fun1)
    transform = torchvision.transforms.Compose([
        scale,
        torch.from_numpy,
        torchvision.transforms.Lambda(tmp_fun6),
    ])
    return transform

def tmp_fun2(x):
    return cv2.cvtColor(
            cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.
            COLOR_GRAY2BGR)

def tmp_fun3(x):
    return  x.transpose(2, 0, 1)

def tmp_fun4(x):
    return x

# def tmp_fun5(x, size):
#     return cv2.resize(x, (size, size))

def tmp_fun6(x):
    return x[None, :, :]

def _create_mpiifacegaze_transform(config: yacs.config.CfgNode) -> Any:
    scale = torchvision.transforms.Lambda(tmp_fun1)
    identity = torchvision.transforms.Lambda(tmp_fun4)
    size = config.transform.mpiifacegaze_face_size
    if size != 448:
        resize = torchvision.transforms.Lambda(lambda x: cv2.resize(x, (size, size)))
    else:
        resize = identity
    if config.transform.mpiifacegaze_gray:
        to_gray = torchvision.transforms.Lambda(tmp_fun2)
    else:
        to_gray = identity

    transform = torchvision.transforms.Compose([
        resize,
        to_gray,
        torchvision.transforms.Lambda(tmp_fun3),
        scale,
        torch.from_numpy,
        torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                         std=[0.225, 0.224, 0.229]),
    ])
    return transform
