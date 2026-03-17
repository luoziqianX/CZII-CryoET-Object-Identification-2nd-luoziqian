import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .decoder import MyUnetDecoder3d

ENCODER_DIMS = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet18d": [64, 64, 128, 256, 512],
    "resnet34d": [64, 64, 128, 256, 512],
    "resnet50d": [64, 256, 512, 1024, 2048],
    "seresnext26d_32x4d": [64, 256, 512, 1024, 2048],
    "convnext_small.fb_in22k": [96, 192, 384, 768],
    "convnext_tiny.fb_in22k": [96, 192, 384, 768],
    "convnext_base.fb_in22k": [128, 256, 512, 1024],
    "tf_efficientnet_b4.ns_jft_in1k": [32, 56, 160, 448],
    "tf_efficientnet_b5.ns_jft_in1k": [40, 64, 176, 512],
    "tf_efficientnet_b6.ns_jft_in1k": [40, 72, 200, 576],
    "tf_efficientnet_b7.ns_jft_in1k": [48, 80, 224, 640],
    "pvt_v2_b1": [64, 128, 320, 512],
    "pvt_v2_b2": [64, 128, 320, 512],
    "pvt_v2_b4": [64, 128, 320, 512],
}


def encode_for_resnet(e, x, B, depth_scaling=[2, 2, 2, 2, 1]):

    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(
            x1,
            kernel_size=(depth_scaling, 1, 1),
            stride=(depth_scaling, 1, 1),
            padding=0,
        )
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode = []
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)

    x = F.avg_pool2d(x, kernel_size=2, stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])
    encode.append(x1)

    return encode


class Net(nn.Module):

    def __init__(
        self,
        pretrained=False,
        cfg=None,
        arch="resnet34d",
        decoder_dim=[256, 128, 64, 32, 16],
        out_channels=7,
    ):
        super().__init__()

        self.register_buffer("D", torch.tensor(0))

        self.arch = arch
        if cfg is not None:
            self.arch = cfg.arch

        encoder_dim = ENCODER_DIMS.get(self.arch, [768])

        self.encoder = timm.create_model(
            model_name=self.arch,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
            features_only=True,
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1], out_channels, kernel_size=1)

    def forward(self, image):
        with torch.no_grad():
            B, C, D, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            x = image.expand(-1, 3, -1, -1)

        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2, 2, 2, 2, 1])

        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None],
            depth_scaling=[1, 2, 2, 2, 2],
        )

        logit = self.mask(last)
        return logit
