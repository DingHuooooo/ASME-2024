import torch.nn as nn
import torch


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape the input tensor from (B, C, H, W) to (B, C, H*W)
        x = x.view(B, C, -1)

        # Transpose x to shape (B, H*W, C) to perform attention
        x = x.transpose(1, 2)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_weights = self.softmax(q @ k.transpose(-2, -1) / (self.query.out_features ** 0.5))
        output = attention_weights @ v

        # Transpose output back to shape (B, C, H*W)
        output = output.transpose(1, 2)

        # Reshape output to original shape (B, C, H, W)
        output = output.view(B, C, H, W)

        return output



class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.rgb_to_grayscale = nn.Conv2d(in_channels, 1, 1)

        self.dconv_down1 = self.double_conv(1, 32)
        self.dconv_down2 = self.double_conv(32, 64)
        self.dconv_down3 = self.double_conv(64, 128)
        self.dconv_down4 = self.double_conv(128, 256)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = self.double_conv(128 + 256, 128)
        self.dconv_up2 = self.double_conv(64 + 128, 64)
        self.dconv_up1 = self.double_conv(32 + 64, 32)

        self.conv_last = nn.Conv2d(32, out_channels, 1)
        self.sigmoid = self.thresholding_approx

        self.cnn_out = None

    def forward(self, x):
        x = self.rgb_to_grayscale(x)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        cnn_out = self.conv_last(x)
        self.cnn_out = cnn_out
        
        out = self.sigmoid(self.maxpool(self.maxpool(cnn_out)))

        return out

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def thresholding_approx(x, a=10.0, b=0.5):
        return torch.sigmoid(a * (x - b))


