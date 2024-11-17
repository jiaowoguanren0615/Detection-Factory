"""
This script has not yet been completed now, do not run
"""
import argparse
import datetime
import os
import numpy as np
import re
import time
import logging

import torch
from torch.utils import data

from timm.utils import NativeScaler

from util.collate_fn import collate_fn
from util.engine import evaluate, train_one_epoch
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import encode_labels, fixed_generator, seed_worker
from util.distributed_utils import load_checkpoint, load_state_dict, init_distributed_mode, get_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--accumulate-steps", type=int, default=1, help="Steps to accumulate gradients"
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    dynamo_backend = ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser"]
    dynamo_backend += ["cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "ipex", "tvm"]
    parser.add_argument(
        "--dynamo-backend",
        type=str,
        default="no",
        choices=dynamo_backend,
        help="""
        Set to one of the possible dynamo backends to optimize the training with torch dynamo.
        See https://pytorch.org/docs/stable/torch.compiler.html and
        https://huggingface.co/docs/accelerate/main/en/package_reference/utilities#accelerate.utils.DynamoBackend
        """,
    )

    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    print(args)
    init_distributed_mode(args)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    torch.backends.cudnn.benchmark = True

    cfg = Config(args.config_file, partials=("lr_scheduler", "optimizer", "param_dicts"))
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)

    # modify output directory
    if getattr(cfg, "output_dir", None) is None:
        if hasattr(cfg, "resume_from_checkpoint") and os.path.isdir(str(cfg.resume_from_checkpoint)):
            # default path: xxxx-xx-xx-yy_yy_yy/checkpoints/{checkpoint_1}
            if "checkpoints" in os.listdir(cfg.resume_from_checkpoint):
                # if given output_dir, find the newest checkpoint under checkpoints directory
                output_dir = os.path.join(cfg.resume_from_checkpoint, "checkpoints")
                folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
                folders.sort(
                    key=lambda folder:
                    list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]
                )
                cfg.resume_from_checkpoint = folders[-1]

            if "checkpoints" in os.path.dirname(cfg.resume_from_checkpoint):
                cfg.output_dir = os.path.dirname(os.path.dirname(cfg.resume_from_checkpoint))
        else:
            # make sure all processes have same output directory
            cfg.output_dir = os.path.join(
                "checkpoints",
                os.path.basename(cfg.model_path).split(".")[0],
                "train",
                datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
            )


    # instantiate dataset
    params = dict(num_workers=cfg.num_workers, collate_fn=collate_fn)
    params.update(dict(pin_memory=cfg.pin_memory, persistent_workers=True))
    if args.use_deterministic_algorithms:
        # set using deterministic algorithms
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        params.update({"worker_init_fn": seed_worker, "generator": fixed_generator()})

    # we use group_based sampler, which increases training speed slightly
    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(
        data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
    )
    train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)
    test_loader = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)

    # instantiate model, optimizer and lr_scheduler
    model = Config(cfg.model_path).model

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    lr_scheduler = cfg.lr_scheduler(optimizer)

    # load from a pretrained weight and fine-tune on it
    weight_path = getattr(cfg, "resume_from_checkpoint", None)
    if weight_path is not None and os.path.isfile(weight_path):
        checkpoint = load_checkpoint(cfg.resume_from_checkpoint)
        load_state_dict(model, checkpoint)
        logger.info(f"load pretrained from {cfg.resume_from_checkpoint}")

    # register dataset class information into the model, useful for inference
    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # TODO prepare for distributed training


    # load from a directory, which means resume training
    if weight_path is not None and os.path.isdir(weight_path):
        path = os.path.basename(cfg.resume_from_checkpoint)
        cfg.starting_epoch = int(path.split("_")[-1]) + 1
        logger.info(f"resume training of {cfg.output_dir}, from {path}")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("model parameters: {}".format(n_params))


    loss_scalar = NativeScaler()
    start_time = time.perf_counter()
    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=cfg.device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            print_freq=cfg.print_freq,
            clip_grad=cfg.clip_grad,
            clip_mode=cfg.clip_mode,
            loss_scaler=loss_scalar
        )

        coco_evaluator = evaluate(model, test_loader, epoch)

        # save best results
        cur_ap, cur_ap50 = coco_evaluator.coco_eval["bbox"].stats[:2]

    total_time = time.perf_counter() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time: {}".format(total_time))


if __name__ == '__main__':
    train()