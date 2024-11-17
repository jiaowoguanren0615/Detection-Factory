from timm.models import register_model

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone
from models.backbones.mobilenetv4_config import MODEL_SPECS

from functools import partial

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.stochastic_depth import StochasticDepth

from util.lazy_load import LazyCall as L
from util.lazy_load import instantiate
from util.distributed_utils import load_checkpoint


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"

    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.append(nn.BatchNorm2d(out_channels))
    if act:
        conv.append(nn.ReLU6())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, act=False, squeeze_exactation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(in_channels * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module("exp_1x1", conv2d(in_channels, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_exactation:
            self.block.add_module("conv_3x3", conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module("res_1x1", conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample,
                 stride, expand_ratio):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super(UniversalInvertedBottleneckBlock, self).__init__()
        # starting depthwise conv
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_, groups=in_channels, act=False)
        # expansion with 1x1 convs
        expand_filters = make_divisible(in_channels * expand_ratio, 8)
        self._expand_conv = conv2d(in_channels, expand_filters, kernel_size=1)
        # middle depthwise conv
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        # projection with 1x1 convs
        self._proj_conv = conv2d(expand_filters, out_channels, kernel_size=1, stride=1, act=False)

        # expand depthwise conv (not used)
        # _end_dw_kernel_size = 0
        # self._end_dw = conv2d(out_channels, out_channels, kernel_size=_end_dw_kernel_size, stride=stride, groups=in_channels, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size=3, dropout=0.0):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.

        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super(MultiQueryAttentionLayerWithDownSampling, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = self.key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(in_channels)
        self._query_proj = conv2d(in_channels, self.num_heads * self.key_dim, 1, 1, norm=False, act=False)

        if self.kv_strides > 1:
            self._key_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                       norm=True, act=False)
            self._value_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                         norm=True, act=False)
        self._key_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._output_proj = conv2d(num_heads * key_dim, in_channels, 1, 1, norm=False, act=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        bs, seq_len, _, _ = x.size()
        # print(x.size())
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_strides, self.query_w_strides)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(bs, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_len, key_dim]

        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(bs, 1, self.key_dim, -1)   # [batch_size, 1, key_dim, seq_length]
        v = v.view(bs, 1, -1, self.key_dim)    # [batch_size, 1, seq_length, key_dim]

        # calculate attention score
        # print(q.shape, k.shape, v.shape)
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        # context = torch.einsum('bnhm,bmv->bnhv', attn_score, v)
        # print(attn_score.shape, v.shape)
        context = torch.matmul(attn_score, v)
        context = context.view(bs, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        # print(output.shape)
        return output


class MNV4layerScale(nn.Module):
    def __init__(self, init_value):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super(MNV4layerScale, self).__init__()
        self.init_value = init_value

    def forward(self, x):
        gamma = self.init_value * torch.ones(x.size(-1), dtype=x.dtype, device=x.device)
        return x * gamma


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides,
                 kv_strides, use_layer_scale, use_multi_query, use_residual=True):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual
        self._input_norm = nn.BatchNorm2d(in_channels)

        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(in_channels, num_heads, kdim=key_dim)

        if use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4layerScale(self.layer_scale_init_value)

    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            # print(x.size())
            x = self.multi_query_attention(x)
            # print(x.size())
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x


def build_blocks(layer_spec, drop_path=0.2):
    global msha
    if not layer_spec.get("block_name"):
        return nn.Sequential()
    block_names = layer_spec["block_name"]
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ["in_channels", "out_channels", "kernel_size", "stride"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            layers.add_module(f"convbn_{i}", conv2d(**args))
    elif block_names == "uib":
        schema_ = ["in_channels", "out_channels", "start_dw_kernel_size", "middle_dw_kernel_size", "middle_dw_downsample",
                   "stride", "expand_ratio", "msha"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            msha = args.pop("msha") if "msha" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if msha:
                msha_schema_ = [
                    "in_channels", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides",
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(msha_schema_, [args["out_channels"]] + (msha)))
                layers.add_module(
                    f"msha_{i}", MultiHeadSelfAttentionBlock(**args)
                )
    elif block_names == "fused_ib":
        schema_ = ["in_channels", "out_channels", "stride", "expand_ratio", "act"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError

    stochastic_depth = StochasticDepth(drop_path, mode="row")
    layers.add_module('stochastic_depth', stochastic_depth)
    return layers


class MobileNetV4(nn.Module):
    def __init__(self, model, num_classes=1000, **kwargs):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
        """Params to initiate MobilenNetV4
        Args:
            model : support 5 types of models as indicated in
            "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"
        """
        super(MobileNetV4, self).__init__()
        # print(MODEL_SPECS.keys(), model not in MODEL_SPECS.keys())
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.num_classes = num_classes
        self.spec = MODEL_SPECS[self.model]

        first_channel = self.spec["layer1"]["block_specs"][-1][1]
        second_channel = self.spec["layer2"]["block_specs"][-1][1]
        third_channel = self.spec["layer3"]["block_specs"][-1][1]
        forth_channel = self.spec["layer4"]["block_specs"][-1][1]
        fifth_channel = self.spec["layer5"]["block_specs"][-1][1]
        self.channels = [first_channel, second_channel, third_channel, forth_channel, fifth_channel]

        # conv0
        self.conv0 = build_blocks(self.spec["conv0"])
        # layer1
        self.layer1 = build_blocks(self.spec["layer1"])
        # layer2
        self.layer2 = build_blocks(self.spec["layer2"])
        # layer3
        self.layer3 = build_blocks(self.spec["layer3"])
        # layer4
        self.layer4 = build_blocks(self.spec["layer4"])
        # layer5
        self.layer5 = build_blocks(self.spec["layer5"])

        # classify [optional]
        self.head = nn.Linear(1280, num_classes)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        # x5 = F.adaptive_avg_pool2d(x5, 1)
        # out = self.head(x5.flatten(1))

        return [x1, x2, x3, x4, x5]
        # return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        """
        :param in_channels_list: List of input channels for P3, P4, P5, P6, P7
        :param out_channels: Number of output channels for each FPN layer (default: 256)
        """
        super(FPN, self).__init__()
        # 1x1 卷积用于调整输入通道到 out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])

        # 3x3 深度可分离卷积
        self.fpn_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, inputs):
        """
        :param inputs: List of feature maps [C3, C4, C5, C6, C7]
        :return: List of FPN outputs [P3, P4, P5, P6, P7]
        """
        # Step 1: 横向连接 (Lateral Connections)
        lateral_features = [lateral_conv(x) for x, lateral_conv in zip(inputs, self.lateral_convs)]

        # Step 2: 自顶向下融合 (Top-Down Pathway)
        fpn_features = {}
        for i in range(len(lateral_features) - 1, 0, -1):
            upsampled = F.interpolate(lateral_features[i], size=lateral_features[i - 1].shape[2:], mode='nearest')
            lateral_features[i - 1] += upsampled

        # Step 3: 通过 3x3 深度可分离卷积进一步处理特征
        for stage_index, (feature, fpn_conv) in enumerate(zip(lateral_features, self.fpn_convs)):
            fpn_features[f"layer{stage_index}"] = fpn_conv(feature)

        return fpn_features


class MobileNetV4BackBone(BaseBackbone):
    model_arch = {
        'mobilenetv4_samll': L(MobileNetV4)(
            model='MobileNetV4ConvSmall'
        ),
        'mobilenetv4_medium': L(MobileNetV4)(
            model='MobileNetV4ConvMedium'
        ),
        'mobilenetv4_large': L(MobileNetV4)(
            model='MobileNetV4ConvLarge'
        ),
        'mobilenetv4_hybrid_medium': L(MobileNetV4)(
            model='MobileNetV4HybridMedium'
        ),
        'mobilenetv4_hybrid_large': L(MobileNetV4)(
            model='MobileNetV4HybridLarge'
        )
    }

    def __new__(
            self,
            arch: str,
            weights: Dict = None,
            return_indices: Tuple[int] = (0, 1, 2, 3, 4),
            **kwargs,
    ):
        # get parameters and instantiate backbone
        model_config = self.get_instantiate_config(self, MobileNetV4, arch, kwargs)
        default_weight = model_config.pop("url", None)
        mobilenetv4 = instantiate(model_config)

        # load state dict
        weights = load_checkpoint(default_weight if weights is None else weights)
        if isinstance(weights, Dict):
            weights = weights["model"] if "model" in weights else weights
        self.load_state_dict(mobilenetv4, weights)

        fpn = FPN(in_channels_list=mobilenetv4.channels, out_channels=256)

        backbone = nn.Sequential(mobilenetv4, fpn)
        backbone.num_channels = [256] * len(return_indices)
        return backbone



# if __name__ == '__main__':
#     from torchinfo import summary
#     img = torch.randn(1, 3, 384, 384)
#     model = MobileNetV4('MobileNetV4ConvMedium')
#     summary(model, input_size=(1, 3, 384, 384))
    # y = model(img)
    # for i in y:
    #     print(i.size())
    # # print(model.channels)
    # fpn = FPN(in_channels_list=model.channels, out_channels=256)
    # outputs = fpn([y[0], y[1], y[2], y[3], y[4]])
    # for i in outputs:
    #     print(i.size())