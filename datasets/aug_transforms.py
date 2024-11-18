import albumentations as A
import cv2
import torch

from torchvision.transforms import v2 as T

def labels_getter(x):
    return x[-1]


basic = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# train transform
hflip = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

lsj = T.Compose([
    T.ScaleJitter(target_size=(1024, 1024), antialias=True),
    T.RandomCrop(size=(1024, 1024), pad_if_needed=True, fill=(123.0, 117.0, 104.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # T.SanitizeBoundingBox(labels_getter=labels_getter),
])

lsj_1536 = T.Compose([
    T.ScaleJitter(target_size=(1536, 1536), antialias=True),
    T.RandomCrop(size=(1536, 1536), pad_if_needed=True, fill=(123.0, 117.0, 104.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # T.SanitizeBoundingBox(labels_getter=labels_getter),
])

scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

multiscale = T.Compose([
    T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

detr = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomChoice([
        T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        T.Compose([
            T.RandomShortestSize([400, 500, 600], antialias=True),
            T.RandomCrop(384, 600),
            T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        ]),
    ]),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # T.SanitizeBoundingBox(labels_getter=labels_getter),
])

ssd = T.Compose([
    T.RandomPhotometricDistort(),
    T.RandomZoomOut(fill=[123.0, 117.0, 104.0]),
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # T.SanitizeBoundingBox(labels_getter=labels_getter),
])

ssdlite = T.Compose([
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # T.SanitizeBoundingBox(labels_getter=labels_getter),
])


rtdetr_transform = T.Compose([
    T.RandomPhotometricDistort(p=0.8),
    T.RandomZoomOut(p=0.5, fill=0, side_range=(1.0, 4.0)),
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.Resize(size=[640, 640], antialias=True),
    # T.ToImageTensor(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # T.SanitizeBoundingBox(labels_getter=labels_getter),
])