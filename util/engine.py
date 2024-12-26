import math
import sys
import time

import torch
import io
import contextlib
import logging
from terminaltables import AsciiTable

from datasets.coco_utils import get_coco_api_from_dataset
from util.coco_eval import CocoEvaluator
import util.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler,
                    clip_grad, clip_mode,
                    print_freq=50, loss_scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)


    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(losses, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

        lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return mloss, now_lr


@torch.inference_mode()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # for collect detection numbers
    category_det_nums = [0] * (max(coco.getCatIds()) + 1)
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)


        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # update detection number
        cat_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
        for cat_name in cat_names:
            cat_id = coco.getCatIds(catNms=cat_name)
            cat_det_num = len(coco_evaluator.coco_eval["bbox"].cocoDt.getAnnIds(catIds=cat_id))
            category_det_nums[cat_id[0]] += cat_det_num


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # print category-wise evaluation results
    cat_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
    table_data = [["class", "imgs", "gts", "dets", "recall", "ap"]]

    # table data for show, each line has the number of image, annotations, detections and metrics
    bbox_coco_eval = coco_evaluator.coco_eval["bbox"]
    for cat_idx, cat_name in enumerate(cat_names):
        cat_id = coco.getCatIds(catNms=cat_name)
        num_img_id = len(coco.getImgIds(catIds=cat_id))
        num_ann_id = len(coco.getAnnIds(catIds=cat_id))
        row_data = [cat_name, num_img_id, num_ann_id, category_det_nums[cat_id[0]]]
        row_data += [f"{bbox_coco_eval.eval['recall'][0, cat_idx, 0, 2].item():.3f}"]
        row_data += [f"{bbox_coco_eval.eval['precision'][0, :, cat_idx, 0, 2].mean().item():.3f}"]
        table_data.append(row_data)

    # get the final line of mean results
    cat_recall = coco_evaluator.coco_eval["bbox"].eval["recall"][0, :, 0, 2]
    valid_cat_recall = cat_recall[cat_recall >= 0]
    mean_recall = valid_cat_recall.sum() / max(len(valid_cat_recall), 1)
    cat_ap = coco_evaluator.coco_eval["bbox"].eval["precision"][0, :, :, 0, 2]
    valid_cat_ap = cat_ap[cat_ap >= 0]
    mean_ap50 = valid_cat_ap.sum() / max(len(valid_cat_ap), 1)
    mean_data = ["mean results", "", "", "", f"{mean_recall:.3f}", f"{mean_ap50:.3f}"]
    table_data.append(mean_data)

    # show results
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True

    # print info
    print(table.table)

    metric_names = ["mAP", "AP@50", "AP@75", "AP-s", "AP-m", "AP-l"]
    metric_names += ["AR_1", "AR_10", "AR_100", "AR-s", "AR-m", "AR-l"]
    metric_dict = dict(zip(metric_names, coco_evaluator.coco_eval["bbox"].stats))

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_evaluator, coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types