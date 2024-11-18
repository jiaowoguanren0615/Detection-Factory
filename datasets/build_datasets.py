from .aug_transforms import detr, ssd, basic
from .coco import CocoDetection
from .voc import VOCDataSet


def build_transform(args, is_train_mode=False):
    if is_train_mode:
        if args.detector == 'detr':
            transforms = detr
        else:
            transforms = ssd
    else:
        transforms = basic
    return transforms




def build_dataset(args):
    ## TODO: For your own dataset, should ignore this following code
    assert args.dataset.lower() in ['coco', 'voc'], 'No support training dataset!'
    train_transform = build_transform(args, is_train_mode=True)
    valid_transform = build_transform(args)

    train_set = valid_set = None
    if args.dataset.lower() == 'coco':
        train_set = CocoDetection(args.data_root, dataset="train", transforms=train_transform)
        valid_set = CocoDetection(args.data_root, dataset="val", transforms=valid_transform)
    elif args.dataset.lower() == 'voc':
        train_set = VOCDataSet(args.data_root, transforms=train_transform)
        valid_set = VOCDataSet(args.data_root, transforms=valid_transform)

    assert train_set is not None, 'Train-set is None! Please check your data root setting!'
    assert valid_set is not None, 'Valid-set is None! Please check your data root setting!'

    return train_set, valid_set