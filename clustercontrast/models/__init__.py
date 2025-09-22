from __future__ import absolute_import

from .resnet import *
from .resnet_ibn import *
#from .vision_transformer import * #RWM added 
from .ensemble_resnet import * #RWM added 
from .xception import * #RWM added 
from .xxception import *
from .hexception import * 
from .sam import * 
from .unet2 import * #change between unet versions 
from .ViT import *
from .draft import *
from .moe import *
# from .beitv3dp import *
from .beitv3my import *
# from .beitv3gpt import *

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'EnsembleModel': EnsembleModel, #RWM 
    'xception': xception,
    'xxception':xxception, 
    'hexception':hexception, 
    'sam':sam, #RWM
    'unet':unet,
    'vit':vit, 
    'vit_base': vitbase,
    'moe': moe,
    # 'beitv3dp': beit_encoder,
    'beitv3my': beit_encoder,
    # 'beitv3gpt': beit_encoder,
    
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
