from models.drn import drn_d_54
import torch
from torch import nn
from models.modules import *

class Model(nn.Module):
    def __init__(self, options, pretrained=True):
        super(Model, self).__init__()
        
        self.options = options        
        self.drn = drn_d_54(pretrained=pretrained, out_map=32, num_classes=-1, out_middle=False)
        self.pyramid = PyramidModule(options, 512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.cbam_conv = CbamBlock(512, 512, use_cbam=True)
        self.segmentation_pred = nn.Conv2d(512, NUM_CORNERS + NUM_ICONS + 2 + NUM_ROOMS + 2, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(options.height, options.width), mode='bilinear', align_corners=False)
        return

    def forward(self, inp):
        features = self.drn(inp)
        features = self.pyramid(features)
        features = self.feature_conv(features)
        features = self.cbam_conv(features)
        segmentation = self.upsample(self.segmentation_pred(features))
        segmentation = segmentation.transpose(1, 2).transpose(2, 3).contiguous()
        return segmentation[:, :, :, :NUM_CORNERS], segmentation[:, :, :, NUM_CORNERS:NUM_CORNERS + NUM_ICONS + 2], segmentation[:, :, :, -(NUM_ROOMS + 2):]
