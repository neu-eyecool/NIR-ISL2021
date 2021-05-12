import torch
import torch.nn.functional as F
from torch import nn

from .efficient_net import EfficientNet


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class LocAttention(nn.Module):
    def __init__(self, channels):
        super(LocAttention, self).__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(channels, 1, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.loc(x)


class EfficientUNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientUNet, self).__init__()
        self.feat_extract = EfficientNet.from_pretrained('efficientnet-b5', weights_path=None)
        # self.feat_extract = EfficientNet()
        self.center = _DecoderBlock(176, 2048, 176)
        self.dec4 = _DecoderBlock(352, 176, 64)
        self.dec3 = _DecoderBlock(128, 64, 40)
        self.dec2 = _DecoderBlock(80, 40, 24)
        self.dec1 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.heatmap4 = LocAttention(64)
        self.heatmap3 = LocAttention(40)
        self.heatmap2 = LocAttention(24)

        self.final_head = nn.Conv2d(24, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        feat = self.feat_extract.extract_endpoints(x)
        enc1, enc2, enc3, enc4 = feat['reduction_1'], feat['reduction_2'], feat['reduction_3'], feat['reduction_4']
        center = self.center(enc4)

        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec4_heatmap = self.heatmap4(dec4)
        dec4 = dec4 + dec4*dec4_heatmap

        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec3_heatmap = self.heatmap3(dec3)
        dec3 = dec3 + dec3*dec3_heatmap

        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec2_heatmap = self.heatmap2(dec2)
        dec2 = dec2 + dec2*dec2_heatmap

        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
        
        pred = self.final_head(dec1)

        return {
            'pred_mask': F.interpolate(pred[:,0:1,:,:], x.size()[2:], mode='bilinear', align_corners=True),
            'pred_iris_mask': F.interpolate(pred[:,1:2,:,:], x.size()[2:], mode='bilinear', align_corners=True),
            'pred_pupil_mask': F.interpolate(pred[:,2:3,:,:], x.size()[2:], mode='bilinear', align_corners=True),
            'heatmap':[dec4_heatmap, dec3_heatmap, dec2_heatmap]
        }


if __name__ == '__main__':
    x = torch.randn(4,3,400,400)
    net = EfficientUNet(3)
    y = net(x)
    print(y['pred_mask'].size())