import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class TargetFC(nn.Module):
    """
    定义目标网络的全连接操作
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


class Myvgg(nn.Module):
    """提取语义特征的vgg网络"""
    def __init__(self, in_chans=3):
        super(Myvgg, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        vgg = vgg16.features
        for param in vgg.parameters():
            param.requires_grad_(False)

        self.in_chans = in_chans
        self.vgg_L = vgg
        self.vgg_R = vgg

        self.linear = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU()
        )

    def forward(self, xl, xr):
        xl = self.vgg_L(xl)
        xr = self.vgg_R(xr)
        out = 0.5 * xl + 0.5 * xr
        return out


class HyperNet(nn.Module):
    """
    定义超网络
    Hyper network for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self):
        super(HyperNet, self).__init__()

        self.target_in_size = 2048
        self.hyperInChn = 112
        self.f1 = 784
        self.f2 = 3
        self.feature_size = 7

        self.vgg = Myvgg()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / 7 ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / 7 ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)
        self.linear = nn.Linear(49*128, 2048)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, xl_224, xr_224, fea_q):

        vgg_out = self.vgg(xl_224, xr_224)
        # input vector for target net
        target_in_vec = self.linear(fea_q.view(fea_q.size()[0], -1)).view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(vgg_out)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b

        return out


class LargeKernel(nn.Module):
    def __init__(self, channels_in, channels_out, large_kernel, small_kernel):
        super(LargeKernel, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.large_kernel = large_kernel
        self.small_kernel = small_kernel
        self.L_Conv = nn.Conv2d(self.channels_in, self.channels_out, self.large_kernel, padding=self.large_kernel // 2)
        self.S_Conv = nn.Conv2d(self.channels_in, self.channels_out, self.small_kernel, padding=self.small_kernel // 2)
        self.BN1 = nn.BatchNorm2d(self.channels_out, momentum=0.1, affine=True)
        self.BN2 = nn.BatchNorm2d(self.channels_out, momentum=0.1, affine=True)

    def forward(self, x):
        x_l =  F.relu(self.BN1(self.L_Conv(x)))
        x_s =  F.relu(self.BN2(self.S_Conv(x)))
        out = x_l + x_s
        return out


class Self_atten(nn.Module):
    def __init__(self, dropout):
        super(Self_atten, self).__init__()
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, q, k, v):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context


class StereoNet(nn.Module):
    def __init__(self):
        super(StereoNet, self).__init__()
        self.dropout_rate = 0.5
        self.n_class = 3
        self.conv_13_L = LargeKernel(channels_in=3, channels_out=32, large_kernel=13, small_kernel=5)
        self.conv_13_R = LargeKernel(channels_in=3, channels_out=32, large_kernel=13, small_kernel=5)
        self.conv_7_L = LargeKernel(channels_in=32, channels_out=64, large_kernel=7, small_kernel=3)
        self.conv_7_R = LargeKernel(channels_in=32, channels_out=64, large_kernel=7, small_kernel=3)
        self.conv_3_L1 = nn.Conv2d(64, 64, 3, padding=0)
        self.conv_3_R1 = nn.Conv2d(64, 64, 3, padding=0)
        self.conv_3_L2 = nn.Conv2d(64, 128, 3, padding=0)
        self.conv_3_R2 = nn.Conv2d(64, 128, 3, padding=0)
        self.max_pool1_L = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool1_R = nn.MaxPool2d(kernel_size=2, stride=2)
        self.self_attention_L = Self_atten(self.dropout_rate)
        self.self_attention_R = Self_atten(self.dropout_rate)
        self.conv_3_L3 = nn.Conv2d(128, 256, 3, padding=0)
        self.conv_3_R3 = nn.Conv2d(128, 256, 3, padding=0)
        self.max_pool2_L = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2_R = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_diff2 = nn.Linear(25088, 2624)
        self.linear_sum2 = nn.Linear(25088, 2624)
        self.linear_fuse = nn.Linear(9216, 1024)
        self.conv_1_L = nn.Conv2d(256, 256, 1, padding=0)
        self.BN_L = nn.BatchNorm2d(256)
        self.conv_1_R = nn.Conv2d(256, 256, 1, padding=0)
        self.BN_R = nn.BatchNorm2d(256)
        self.linear_L = nn.Linear(9216, 1)
        self.linear_R = nn.Linear(9216, 1)
        self.classifier = nn.Sequential(
            nn.Linear(5120, 1024),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(1024, self.n_class),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xl, xr):
        xl1 = self.conv_13_L(xl)
        xr1 = self.conv_13_R(xr)
        xl2 = self.conv_7_L(xl1)
        xr2 = self.conv_7_R(xr1)
        xl3 = F.relu(self.conv_3_L1(xl2))
        xr3 = F.relu(self.conv_3_R1(xr2))
        xl3 = F.relu(self.max_pool1_L(self.conv_3_L2(xl3)))
        xr3 = F.relu(self.max_pool1_R(self.conv_3_R2(xr3)))
        diff = torch.sub(xl3, xr3)
        diff = diff.view(-1, 128, diff.size()[2] * diff.size()[3])
        sum = torch.add(xl3, xr3)
        sum = sum.view(-1, 128, sum.size()[2] * sum.size()[3])

        diff_linear = self.linear_diff2(diff.view(diff.size()[0], -1))
        diff_linear = F.dropout(diff_linear, p=self.dropout_rate, training=True, inplace=False)
        sum_linear = self.linear_sum2(sum.view(sum.size()[0], -1))
        sum_linear = F.dropout(sum_linear, p=self.dropout_rate, training=True, inplace=False)

        xl3 = xl3.view(-1, 128, xl3.size()[2] * xl3.size()[3])
        xr3 = xr3.view(-1, 128, xr3.size()[2] * xr3.size()[3])
        xl4 = self.self_attention_L(diff, sum, xl3)
        xr4 = self.self_attention_R(diff, sum, xr3)
        xl4 = xl4.view(-1, 128, int(math.sqrt(xl4.size()[2])), int(math.sqrt(xl4.size()[2])))
        xr4 = xr4.view(-1, 128, int(math.sqrt(xr4.size()[2])), int(math.sqrt(xr4.size()[2])))
        xl5 = F.relu(self.max_pool2_L(self.conv_3_L3(xl4)))
        xr5 = F.relu(self.max_pool2_R(self.conv_3_R3(xr4)))

        w_l = F.sigmoid(self.linear_L(self.BN_L(self.conv_1_L(xl5)).view(xl5.size()[0], -1))).view(-1, 1, 1, 1)
        w_r = F.sigmoid(self.linear_R(self.BN_R(self.conv_1_R(xr5)).view(xr5.size()[0], -1))).view(-1, 1, 1, 1)

        x_fuse = w_l * xl5 + w_r * xr5
        x_fuse = x_fuse.view(x_fuse.size()[0], -1)  # 拉伸成一行
        x_fuse = self.linear_fuse(x_fuse)
        x_fuse = F.dropout(x_fuse, p=self.dropout_rate, training=True, inplace=False)

        x = F.relu(torch.cat((x_fuse, diff_linear, sum_linear), 1))
        out = x.reshape(-1, 128, 7, 7)
        return out


class TargetNet(nn.Module):
    """
    最终分类网络
    Target network for quality prediction.
    """

    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            # nn.Sigmoid(),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        classes = self.l1(x)
        z = classes.view(classes.size(0), -1)
        # q = F.dropout(q)
        classes = self.l2(classes)
        classes = classes.view(classes.size(0), -1)
        num = self.softmax(classes)
        return classes, num, z


if __name__ == "__main__":
    xl_224 = torch.rand(1, 3, 224, 224).cuda()
    xr_224 = torch.rand(1, 3, 224, 224).cuda()
    xl_32 = torch.rand(1, 3, 32, 32).cuda()
    xr_32 = torch.rand(1, 3, 32, 32).cuda()

    model_hyper = HyperNet().cuda()
    model_img = StereoNet().cuda()

    inf = model_img(xl_32, xr_32)
    paras = model_hyper(xl_224, xr_224, inf)
    model_target = TargetNet(paras).cuda()
    for param in model_target.parameters():
        param.requires_grad = False

    # Quality prediction
    # pred, num = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
    # print(pred.shape)











