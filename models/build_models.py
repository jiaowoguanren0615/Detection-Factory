from .backbones import *
from .bricks import *
from .detectors import *
from .necks import *
import torch.nn as nn
import torch
import os

from util.set_criterion import HybridSetCriterion
from util.hungarian_matcher import HungarianMatcher


# ========================================
embed_dim = 256
num_classes = 91
num_queries = 900
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048
# ========================================


class DetectionModel(nn.Module):
    def __init__(self, backbone, detector_arch, num_classes, pretrained_backbone, **kwargs):
        super(DetectionModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.detector_arch = detector_arch
        self.pretrained_backbone = pretrained_backbone

        self.position_embedding = PositionEmbeddingSine(embed_dim // 2, temperature=10000, normalize=True, offset=-0.5)
        self.postprocessor = PostProcess(select_box_nums_for_evaluation=300)
        self.neck = ChannelMapper(
            in_channels=self.backbone.num_channels,
            out_channels=embed_dim,
            num_outs=num_feature_levels,
        )

    def _set_detr(self):
        criterion, foreground_criterion = self._set_criterion()
        transformer = self._set_transformer()
        model = SalienceDETR(
            backbone=self.backbone,
            neck=self.neck,
            position_embedding=self.position_embedding,
            transformer=transformer,
            criterion=criterion,
            focus_criterion=foreground_criterion,
            postprocessor=self.postprocessor,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=True,
            min_size=800,
            max_size=1333,
        )
        return model


    def forward(self, inputs):
        model_arch = self._set_detr()
        output = model_arch(inputs)
        return output

    def _set_criterion(self):
        matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

        weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
        weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
        weight_dict.update({
            k + f"_{i}": v
            for i in range(transformer_dec_layers - 1)
            for k, v in weight_dict.items()
        })
        weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
        weight_dict.update({"loss_salience": 2})

        criterion = HybridSetCriterion(self.num_classes, matcher=matcher, weight_dict=weight_dict, alpha=0.25, gamma=2.0)
        foreground_criterion = SalienceCriterion(noise_scale=0.0, alpha=0.25, gamma=2.0)
        return criterion, foreground_criterion

    def _set_transformer(self):
        transformer = SalienceTransformer(
            encoder=SalienceTransformerEncoder(
                encoder_layer=SalienceTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    n_heads=num_heads,
                    dropout=0.0,
                    activation=nn.ReLU(inplace=True),
                    n_levels=num_feature_levels,
                    n_points=4,
                    d_ffn=dim_feedforward,
                ),
                num_layers=transformer_enc_layers,
            ),
            neck=RepVGGPluXNetwork(
                in_channels_list=self.neck.num_channels,
                out_channels_list=self.neck.num_channels,
                norm_layer=nn.BatchNorm2d,
                activation=nn.SiLU,
                groups=4,
            ),
            decoder=SalienceTransformerDecoder(
                decoder_layer=SalienceTransformerDecoderLayer(
                    embed_dim=embed_dim,
                    n_heads=num_heads,
                    dropout=0.0,
                    activation=nn.ReLU(inplace=True),
                    n_levels=num_feature_levels,
                    n_points=4,
                    d_ffn=dim_feedforward,
                ),
                num_layers=transformer_dec_layers,
                num_classes=num_classes,
            ),
            num_classes=num_classes,
            num_feature_levels=num_feature_levels,
            two_stage_num_proposals=num_queries,
            level_filter_ratio=(0.4, 0.8, 1.0, 1.0) if num_feature_levels == 4 else (0.4, 0.6, 0.8, 1.0, 1.0),
            layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        )
        return transformer