import torch
from torch import nn
from torch.nn.functional import elu
import warnings
import time

# 不显示警告
warnings.filterwarnings('ignore')


class Expression(torch.nn.Module):
    """
    Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
                self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
                self.__class__.__name__
                + "("
                + "expression="
                + str(expression_str)
                + ")"
        )


def identity(x):
    """
    No activation function
    """
    return x


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose1(x):
    return x.permute(0, 3, 2, 1)


def _transpose2(x):
    return x.permute(0, 1, 3, 2)


def self_padding(x):
    """
    pariodic padding after the wavelet convolution, defined by formula (3) in the paper

    Parameters
    ----------
    x : input feature
    """
    return torch.cat((x[:, :, :, -3:], x, x[:, :, :, 0:3]), 3)


class SELayer(nn.Module):
    """
    the Squeeze and Excitation layer, defined by formula (4)(5) in the paper

    Parameters
    ----------
    channel: the input channel number
    reduction: the reduction ratio r
    """

    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EEG_encoder(nn.Module):
    def __init__(self, in_chans=62,
                 n_classes=5,
                 reduction_ratio=8,
                 conv_stride=1,
                 pool_stride=3,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.2,
                 input_size=256,
                 hidden_dim=256,
                 attention_dim=256,
                 ):
        super(EEG_encoder, self).__init__()

        self.in_chans = in_chans
        self.n_classes = n_classes
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.reduction_ratio = reduction_ratio
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # the Spatio-Temporal Block
        self.transpose1 = Expression(_transpose1)
        self.conv_time = nn.Conv2d(2, 32, (1, self.in_chans), stride=(1, 1), bias=not self.batch_norm)
        self.conv_spa = nn.Conv2d(2, 64, (256, 1), stride=(1, 1), bias=not self.batch_norm)
        self.bn0 = nn.BatchNorm2d(32, momentum=self.batch_norm_alpha, affine=True)
        self.bn0_1 = nn.BatchNorm2d(64, momentum=self.batch_norm_alpha, affine=True)
        self.conv_nonlinear0 = nn.ELU()
        self.conv_nonlinear0_1 = nn.ELU()

        # the 1-st Temporal Conv Unit
        self.drop1 = nn.Dropout(p=self.drop_prob)
        self.conv1 = nn.Conv2d(32, 64, (11, 1), stride=(conv_stride, 1), padding=(0, 0), bias=not self.batch_norm)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlinear1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.pool_nonlinear1 = Expression(identity)

        # the 2-nd Temporal Conv Unit
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.conv2 = nn.Conv2d(64, 128, (9, 1), stride=(conv_stride, 1), padding=(0, 0), bias=not self.batch_norm)
        self.bn2 = nn.BatchNorm2d(128, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlinear2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.pool_nonlinear2 = Expression(identity)

        # the 3-rd Temporal Conv Unit
        self.drop3 = nn.Dropout(p=self.drop_prob)
        self.conv3 = nn.Conv2d(128, 256, (7, 1), stride=(conv_stride, 1), padding=(0, 0), bias=not self.batch_norm)
        self.bn3 = nn.BatchNorm2d(256, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlinear3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.pool_nonlinear3 = Expression(identity)

        # the 4-rd Temporal Conv Unit
        self.drop4 = nn.Dropout(p=self.drop_prob)
        self.conv4 = nn.Conv2d(256, 512, (5, 1), stride=(conv_stride, 1), padding=(0, 0), bias=not self.batch_norm)
        self.bn4 = nn.BatchNorm2d(512, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlinear4 = nn.ELU()
        self.pool_nonlinear4 = Expression(identity)

        # *********change***********
        self.conv_spa2 = nn.Conv2d(64, 128, (1, 1), stride=(1, 1), bias=not self.batch_norm)
        self.bn0_2 = nn.BatchNorm2d(128, momentum=self.batch_norm_alpha, affine=True)
        self.conv_nonlinear0_2 = nn.ELU()
        self.transpose2 = Expression(_transpose2)
        # the large size Spatio Conv Unit
        self.drop_1 = nn.Dropout(p=self.drop_prob)
        self.conv_1 = nn.Conv2d(128, 512, (1, 62), stride=(1, conv_stride), padding=(0, 0), bias=not self.batch_norm)
        self.bn_1 = nn.BatchNorm2d(512, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlinear_1 = nn.ELU()
        self.pool_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.pool_nonlinear_1 = Expression(identity)

        # the middle size Spatio Conv Unit
        self.drop_2 = nn.Dropout(p=self.drop_prob)
        self.conv_2 = nn.Conv2d(128, 512, (1, 30), stride=(1, conv_stride), padding=(0, 0), bias=not self.batch_norm)
        self.bn_2 = nn.BatchNorm2d(512, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlinear_2 = nn.ELU()
        self.pool_2 = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 3))
        self.pool_nonlinear_2 = Expression(identity)

        # the small size Spatio Conv Unit
        self.drop_3 = nn.Dropout(p=self.drop_prob)
        self.conv_3 = nn.Conv2d(128, 512, (1, 15), stride=(1, conv_stride), padding=(0, 0), bias=not self.batch_norm)
        self.bn_3 = nn.BatchNorm2d(512, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlinear_3 = nn.ELU()
        self.pool_3 = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 2))
        self.pool_nonlinear_3 = Expression(identity)

        # The SEC Unit for Spatio features
        self.SElayer1 = SELayer(512, self.reduction_ratio)
        self.SEconv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 7), stride=(1, 2))
        self.SEbn1 = nn.BatchNorm2d(512)
        self.SEpooling1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        # The SEC Unit for Deep-Temporal features
        self.SElayer2 = SELayer(512, self.reduction_ratio)
        self.SEconv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1))
        self.SEbn2 = nn.BatchNorm2d(512)
        self.SEpooling2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.elu = nn.ELU(inplace=True)

        # the Classifier
        self.conv_classifier = nn.Conv2d(512, 100, (10, 1), bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.squeeze_output = Expression(_squeeze_final_output)

        self.linear1 = nn.Linear(2000, 512)
        self.drop_4 = nn.Dropout(p=self.drop_prob)
        self.linear2 = nn.Linear(100, self.n_classes)
        # initializing parameters
        self.initialize()

        # LSTM
        self.lstm = nn.LSTM(62, 62, 1, batch_first=True, dropout=0.2,
                            bidirectional=True)

        self.lstm_s = nn.LSTM(62, 62, 1, batch_first=True, dropout=0.2,
                              bidirectional=True)

        # self-attention
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_t = x.permute(0, 3, 2, 1)
        x_lstm = x_t.view(-1, 256, 62)
        x_lstm = self.lstm(x_lstm)[0]
        x_lstm = x_lstm.unsqueeze(3)
        x_lstm = x_lstm.reshape(-1, 2, 256, 62)
        # Temporal
        x_t = x_lstm
        x1 = self.conv_nonlinear0(self.bn0(self.conv_time(x_t)))
        # the Deep-Temporal Convolution Block
        # the 1-st Temporal Conv Unit
        x_t = self.conv_nonlinear1(self.bn1(self.conv1(x1)))
        x_t = self.pool_nonlinear1(self.pool1(x_t))
        # the 2-nd Temporal Conv Unit
        x_t = self.conv_nonlinear2(self.bn2(self.conv2(x_t)))
        x_t = self.pool_nonlinear2(self.pool2(x_t))
        # the 3-rd Temporal Conv Unit
        x_t = self.conv_nonlinear3(self.bn3(self.conv3(self.drop3(x_t))))
        x_t = self.pool_nonlinear3(self.pool3(x_t))
        # the 4-nd Temporal Conv Unit
        x_t = self.conv_nonlinear4(self.bn4(self.conv4(self.drop4(x_t))))
        x_t = self.pool_nonlinear4(x_t)
        # The SEC Unit for Deep-Temporal features
        x_t = self.SElayer2(x_t)
        x_t = self.elu(self.SEbn2(self.SEconv2(x_t)))
        x_t = self.SEpooling2(x_t)

        # Spatio
        x_s = x_lstm
        x_s = self.conv_nonlinear0_1(self.bn0_1(self.conv_spa(x_s)))
        x2 = self.conv_nonlinear0_2(self.bn0_2(self.conv_spa2(x_s)))
        # the Spatio Convolution Block
        # the 1-st Spatio Conv Unit
        out_l = self.conv_nonlinear_1(self.bn_1(self.conv_1(self.drop_1(x2))))
        out_l = self.pool_nonlinear_1(out_l)
        # the 2-nd Spatio Conv Unit
        out_m = self.conv_nonlinear_2(self.bn_2(self.conv_2(self.drop_2(x2))))
        out_m = self.pool_nonlinear_2(self.pool_2(out_m))
        # the 3-rd Spatio Conv Unit
        out_s = self.conv_nonlinear_3(self.bn_3(self.conv_3(self.drop_3(x2))))
        out_s = self.pool_nonlinear_3(self.pool_3(out_s))
        # 三个尺度拼在一起
        x_spat_feature = torch.cat((out_l, out_m, out_s), dim=3)

        # The SEC Unit for Multi-Spectral features
        x_spat_feature = self.SElayer1(x_spat_feature)
        x_spat_feature = self.elu(self.SEbn1(self.SEconv1(x_spat_feature)))
        x_spat_feature = self.SEpooling1(x_spat_feature)
        x_spat_feature = self.transpose2(x_spat_feature)

        # the Classifier
        x = torch.cat((x_t, x_spat_feature), dim=2)
        x = self.conv_classifier(x)
        x = torch.tanh(x)
        x = x.view(x.size()[0], -1)  # 拉伸成一行
        y = self.linear2(self.drop_4(x))
        z = self.softmax(y)
        y = y.view(y.size()[0], -1)  # 拉伸成一行
        return y, x, z


if __name__ == "__main__":
    x = torch.rand(4, 62, 256, 1).cuda()
    model = EEG_encoder(62, 3).cuda()
    
    torch.cuda.synchronize()
    time_start = time.time()
    predict = model(x)
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
