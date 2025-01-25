import math
import torch
from torch import Tensor
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19
from typing import Any, List, Optional, Type, Union




class VGG19(nn.Module):
    """
    Custom version of VGG19 with the maxpool layers replaced with avgpool as per the paper
    """
    def __init__(self):
        """
        If True, the gradients for the VGG params are turned off
        """
        super(VGG19, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = vgg19(weights=VGG19_Weights(VGG19_Weights.DEFAULT)).to(device)
        # note: added one extra maxpool (layer 36) from the vgg... worked well so kept it in
        self.output_layers = [9, 18, 27, 36]
        self.layer_weights = [10**9] * len(self.output_layers)
        for layer in self.output_layers[1:]:  # convert the maxpool layers to an avgpool
            self.model.features[layer] = nn.AvgPool2d(kernel_size=2, stride=2)

        self.feature_maps = []

    def forward(self, x):
        """
        Take in image, pass it through the VGG, capture feature maps at each of the output layers of VGG
        """
        self.feature_maps = []
        for index, layer in enumerate(self.model.features):
            # print(layer)
            x = layer(x)  # pass the img through the layer to get feature maps of the img
            if index in self.output_layers:
                self.feature_maps.append(x)
            if index == self.output_layers[-1]:
                # stop VGG execution as we've captured the feature maps from all the important layers
                break


    def get_gram_matrices(self, x):
        """
        Convert the featuremaps captured by the call method into gram matrices
        """
        gram_matrices = []
        self(x)
        '''for fm in self.feature_maps:
            _ , n, x, y = fm.size()  # num filters n and (filter dims x and y)
            F = fm.reshape( n, x * y)  # reshape filterbank into a 2D mat before doing auto correlation
            gram_mat = (F @ F.t()) / (4. * n * x * y)  # auto corr + normalize by layer output dims
            gram_matrices.append(gram_mat)'''
        #return gram_matrices  # if want to return gram matrix
        return self.feature_maps

    def style_loss(self):
        gram_matrices_ideal = self.get_gram_matrices(self.image)
        gram_matrices_pred =  self.get_gram_matrices(self.pred)
        loss_cnn = 0.  # (w1*E1 + w2*E2 + ... + wl*El)
        for i in range(len(self.layer_weights)):
            # E_l = w_l * ||G_ideal_l - G_pred_l||^2
            E = self.layer_weights[i] * ((gram_matrices_ideal[i] - gram_matrices_pred[i]) ** 2.).sum()
            loss_cnn += E
        return loss_cnn

    def content_loss(self):
        content_loss = 0.
        self(self.image)
        image_fet_map = self.feature_maps
        self(self.pred)
        pred_fet_map = self.feature_maps
        for i in range(len(self.feature_maps)):
            content_loss += nn.functional.mse_loss(image_fet_map[i], pred_fet_map[i])

        return content_loss



"""
For easier copy-pasting.
No need to extract stuff from a super-coupled codebase :)

Note that if you want to use the pretrained model weights,
then please keep in mind that the outputs might be different from the original implementation
as this uses some slightly different ops than with which the weights were trained.

ResNet-50 implementation modified for B-cos from 
Modified from https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf05446bb8a2/torchvision/models/resnet.py
"""

class AddInverse(torch.nn.Module):
    """To a [B, C, H, W] input add the inverse channels of the given one to it.
    Results in a [B, 2C, H, W] output. Single image [C, H, W] is also accepted.

    Args:
        dim (int): where to add channels to. Default: -3
    """

    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: Tensor) -> Tensor:
        return torch.cat([in_tensor, 1 - in_tensor], dim=self.dim)
    



class BcosConv2d(nn.Conv2d):
    def __init__(self, *args, b: float = 2.0, **kwargs):
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.b = b
        assert self.dilation == (1, 1), "Dilation > 1 is not supported."
        self.detach = False

    def calc_patch_norms(self, in_tensor: Tensor) -> Tensor:
        squares = in_tensor**2
        norms = (
            squares.sum(1, keepdim=True)
            if self.groups == 1
            else squares.unflatten(
                1, (self.groups, self.in_channels // self.groups)
            ).sum(2)
        )

        norms = (
            F.avg_pool2d(
                norms,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                divisor_override=1,  # sum, not avg
            )
            + 1e-6  # stabilizing term
        ).sqrt_()

        if self.groups > 1:
            norms = torch.repeat_interleave(
                norms, repeats=self.out_channels // self.groups, dim=1
            )

        return norms

    def set_explanation_mode(self, on: bool):
        self.detach = on

    def forward(self, in_tensor: Tensor) -> Tensor:
        # this is better, as it's from torch + it has a stabilizing term (eps)
        normed_weights = F.normalize(self.weight, dim=(1, 2, 3))  # type: ignore
        out = self._conv_forward(in_tensor, normed_weights, self.bias)

        if self.b == 1:
            return out

        norms = self.calc_patch_norms(in_tensor)

        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norms = norms.detach()

        dynamic_scaling = (maybe_detached_out / norms).abs()
        if self.b != 2:
            dynamic_scaling = (dynamic_scaling + 1e-6).pow_(self.b - 1)

        return dynamic_scaling * out

    # for compatibility with weights
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "linear.weight" in state_dict:
            state_dict[prefix + "weight"] = state_dict.pop(prefix + "linear.weight")
        if prefix + "linear.bias" in state_dict:
            state_dict[prefix + "bias"] = state_dict.pop(prefix + "linear.bias")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class BatchNorm2dUncenteredNoBias(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None
        self.detach = False

    def set_explanation_mode(self, on: bool):
        self.detach = on

    def forward(self, in_tensor: Tensor) -> Tensor:
        if self.training:
            x = in_tensor.detach() if self.detach else in_tensor
            var = x.var(dim=(0, 2, 3), unbiased=False)

            if self.running_var is not None:
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var.detach()
                )
        else:  # evaluation mode
            var = self.running_var

        # might be slightly faster as it avoids division
        rstd = (var + self.eps).rsqrt()[None, ..., None, None]

        result = in_tensor * rstd

        if self.weight is not None:
            result = self.weight[None, ..., None, None] * result
        if self.bias is not None:
            result = result + self.bias[None, ..., None, None]

        return result


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
):
    """3x3 convolution with padding"""
    return BcosConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
):
    """1x1 convolution"""
    return BcosConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = BatchNorm2dUncenteredNoBias(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dUncenteredNoBias(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    
class LayerWithIntermediateOutputs(nn.Module):
    """
    Custom module to store outputs of each block in the layer.
    """
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs  

class BcosResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 1000,
        in_chans: int = 6,
        groups: int = 1,
        width_per_group: int = 64,
        inplanes: int = 64,
        small_inputs: bool = False,
        logit_bias: Optional[float] = None,
    ):
        super().__init__()
        self.inplanes = inplanes
        self.groups = groups
        self.base_width = width_per_group

        if small_inputs:
            self.conv1 = conv3x3(in_chans, self.inplanes)
            self.pool = None
        else:
            self.conv1 = BcosConv2d(
                in_chans, self.inplanes, kernel_size=7, stride=1, padding=3
            )
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = BatchNorm2dUncenteredNoBias(self.inplanes)

        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        try:
            self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
            last_ch = inplanes * 8
        except IndexError:
            self.layer4 = None
            last_ch = inplanes * 4

        self.num_features = last_ch * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes
        self.fc = BcosConv2d(
            self.num_features,
            self.num_classes,
            kernel_size=1,
        )
        self.logit_bias = (
            logit_bias
            if logit_bias is not None
            else math.log(1 / (self.num_classes - 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2dUncenteredNoBias(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
        return LayerWithIntermediateOutputs(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        fmap5 = None
        x = self.conv1(x)
        x = self.bn1(x)
        if self.pool is not None:
           fmap1  = self.pool(x)
        fmap2 = self.layer1(fmap1)
        fmap3 = self.layer2(fmap2[1])
        #fmap4 = self.layer3(fmap3[1])
        #if self.layer4 is not None:
        #    fmap5 = self.layer4(fmap4[1])
        return  fmap3[1] , fmap3[0] , fmap2[1] , fmap2[0] , fmap1

    def forward(self, x: Tensor) -> Tensor:
        fmap5, fmap4, fmap3,fmap2,fmap1 = self.forward_features(x)
        '''if fmap5 is not None:
            x = self.fc(fmap5)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = x + self.logit_bias'''
        return fmap5, fmap4, fmap3,fmap2,fmap1


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock]],
    layers: List[int],
    pretrained: bool = False,
    progress: bool = True,
    inplanes: int = 64,
    **kwargs: Any,
) -> BcosResNet:
    model = BcosResNet(block, layers, inplanes=inplanes, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            URLS[arch],
            progress=progress,
        )
        # load with changed keys
        model = load_state_dict_with_key_mapping(model, state_dict)

        #model.load_state_dict(state_dict)
    return model





def resnet18(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


# ---------------------
# Model URLs
# ---------------------
BASE = "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights"
URLS = {
    "resnet18": f"{BASE}/resnet_18-68b4160fff.pth",
    "resnet34": f"{BASE}/resnet_34-a63425a03e.pth",
    "resnet50": f"{BASE}/resnet_50-ead259efe4.pth",
    "resnet101": f"{BASE}/resnet_101-84c3658278.pth",
    "resnet152": f"{BASE}/resnet_152-42051a77c1.pth",
    "resnext50_32x4d": f"{BASE}/resnext_50_32x4d-57af241ab9.pth",
    "resnet50_long": f"{BASE}/resnet_50_long-ef38a88533.pth",
    "resnet152_long": f"{BASE}/resnet_152_long-0b4b434939.pth",
}

from typing import Optional
import re
def load_state_dict_with_key_mapping(model: nn.Module, state_dict: dict):
    """
    Load a state_dict into the model, mapping old keys to the new structure.
    :param model: The model to load weights into.
    :param state_dict: The state dictionary with old-style keys.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Replace the first occurrence of '.blocks.0' or '.blocks.1'
        new_key = re.sub(r"\.(0|1)", r".blocks.\1", key, count=1)

        new_state_dict[new_key] = value

    # Load the remapped state_dict
    model.load_state_dict(new_state_dict, strict=True)
    return model

class resnet18_bcos(nn.Module):
    """
    Custom version of VGG19 with the maxpool layers replaced with avgpool as per the paper
    """
    def __init__(self, img: Optional[object] = None, pred: Optional[object] = None):
        """
        If True, the gradients for the VGG params are turned off
        """
        super(resnet18_bcos, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet18(pretrained=True , progress=True).to(self.device)
        self.image = img
        self.pred = pred
        # note: added one extra maxpool (layer 36) from the vgg... worked well so kept it in
        #self.layer_weights = [10**9] * 5 
        #for layer in self.output_layers[1:]:  # convert the maxpool layers to an avgpool
        #    self.model.features[layer] = nn.AvgPool2d(kernel_size=2, stride=2)

        self.feature_maps = []

    def forward(self, x):
        """
        Take in image, pass it through the VGG, capture feature maps at each of the output layers of VGG
        """
        self.feature_maps = self.model(AddInverse()(x))

    def get_gram_matrices(self, x):
        """
        Convert the featuremaps captured by the call method into gram matrices
        """
        gram_matrices = []
        self(x)
        '''for fm in self.feature_maps:
            _ , n, x, y = fm.size()  # num filters n and (filter dims x and y)
            F = fm.reshape( n, x * y)  # reshape filterbank into a 2D mat before doing auto correlation
            gram_mat = (F @ F.t()) / (4. * n * x * y)  # auto corr + normalize by layer output dims
            gram_matrices.append(gram_mat)
        return gram_matrices  # if want to return gram matrix'''
        return self.feature_maps

    def style_loss(self):
        gram_matrices_ideal = self.get_gram_matrices(self.image)
        gram_matrices_pred =  self.get_gram_matrices(self.pred)
        loss_cnn = 0.  # (w1*E1 + w2*E2 + ... + wl*El)
        for i in range(len(self.layer_weights)):
            # E_l = w_l * ||G_ideal_l - G_pred_l||^2
            E = self.layer_weights[i] * ((gram_matrices_ideal[i] - gram_matrices_pred[i]) ** 2.).sum()
            loss_cnn += E
        return loss_cnn

    def content_loss(self):
        content_loss = 0.
        self(self.image)
        image_fet_map = self.feature_maps
        self(self.pred)
        pred_fet_map = self.feature_maps
        for i in range(len(self.feature_maps)):
            content_loss += nn.functional.mse_loss(image_fet_map[i], pred_fet_map[i])

        return content_loss




