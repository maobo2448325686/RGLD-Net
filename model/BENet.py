from decoder.decoder import Decoder

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from encoder.resnet import ResNet50


class RPAM(nn.Module):
    def __init__(self):
        super(RPAM, self).__init__()

        self.spatial_attention = SpatialAttention()
        self.pixel_attention = PixelAttention()

    def forward(self, x):
        spatial_att = self.spatial_attention(x)
        pixel_att = self.pixel_attention(x)
        return spatial_att * pixel_att * x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channels = in_planes // ratio
        self.fc1 = nn.Conv2d(in_planes, self.channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(self.channels, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PixelAttention(nn.Module):
    def __init__(self):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        pixel_att = self.conv1(combined)
        pixel_att = self.sigmoid(pixel_att)
        return pixel_att


def gram_schmidt(input):
    def projection(u, v):
        return (torch.dot(u.view(-1), v.view(-1)) / torch.dot(u.view(-1), u.view(-1))) * u

    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x / x.norm(p=2)
        output.append(x)
    return torch.stack(output)


def initialize_orthogonal_filters(c, h, w):
    if h * w < c:
        n = c // (h * w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))


class GramSchmidtTransform(torch.nn.Module):
    instance = {}  # 初始化字典
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if (c, h) not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.to(self.device).detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)


class OCAM(nn.Module):
    def __init__(self, channels: int, height: int):
        """
        初始化Orthogonal_Channel_Attention模块
        :param channels: 输入张量的通道数 (C)
        :param height: Gram-Schmidt 变换所需的高度 (假设 H=W)
        """
        super(OCAM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channels = channels
        self.height = height
        # Gram-Schmidt 变换初始化
        self.F_C_A = GramSchmidtTransform.build(channels, height)

        # 通道注意力映射（SE Block 结构）
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 输入张量 (B, C, H, W)
        :return: 输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 如果输入的 H 和 W 与初始化不匹配，则自适应池化
        if H != self.height or W != self.height:
            x = nn.functional.adaptive_avg_pool2d(x, (self.height, self.height))
        # Gram-Schmidt 变换
        transformed = self.F_C_A(x)  # (B, C, 1, 1)
        # 去除空间维度，进入通道注意力网络
        compressed = transformed.view(B, C)
        # 通道注意力生成
        excitation = self.channel_attention(compressed).view(B, C, 1, 1)
        # 加权原始输入特征
        output = x * excitation
        return output


class OPAM(nn.Module):
    def __init__(self, channel, weight):
        super(OPAM, self).__init__()
        self.channel = channel
        self.weight = weight
        self.ocam = OCAM(self.channel, self.weight)
        self.rpam = RPAM()
        self.conv1 = nn.Conv2d(self.channel, self.channel, 1)

    def forward(self, x):
        out = x * self.ocam(x)
        res = out * self.rpam(out)
        return self.conv1(res)


class BENet(nn.Module):
    ### RGLD-Net ###

    def __init__(self):
        super(BENet, self).__init__()
        weights = [8, 16, 32, 64, 128, 256]
        channels = [2048, 1024, 512, 256, 128, 64, 32]
        self.backbone = ResNet50(output_stride=16, in_c=3)

        self.opam4 = OPAM(channels[0], weights[1])
        self.opam3 = OPAM(channels[1], weights[2])
        self.opam2 = OPAM(channels[2], weights[3])
        self.opam1 = OPAM(channels[3], weights[4])

        self.decoder = Decoder()



    def forward(self, hr):
        # x4, x1, x2, x3  torch.Size([4, 2048, 16, 16]) torch.Size([4, 256, 128, 128]) torch.Size([4, 512, 64, 64]) torch.Size([4, 1024, 32, 32])
        x4, x1, x2, x3 = self.backbone(hr)

        x1, x2, x3, x4 = self.opam1(x1), self.opam2(x2), self.opam3(x3), self.opam4(x4)

        res = self.decoder(x4, x3, x2, x1)

        return torch.sigmoid(res)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BENet().to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)