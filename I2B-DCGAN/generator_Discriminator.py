import torch
import torch.nn as nn
from torch.nn import functional as F
import hiddenlayer as hl
from torch.autograd import Variable

def trans(x):
        return x.permute(0, 2, 1, 3)


class Generator(nn.Module):
    """定义生成器"""
    def __init__(self, eegfeature_size, imgfeature_size, imgfeature_reduced_size):
        super(Generator, self).__init__()

        self.eegfeature_size = eegfeature_size
        self.imgfeature_size = imgfeature_size
        self.imgfeature_reduced_size = imgfeature_reduced_size

        # 减小图像特征的维度
        self.enlarge = nn.Sequential(
            nn.Linear(imgfeature_size, imgfeature_reduced_size),
            nn.LeakyReLU(),
        )
        # Defining the generator network architecture
        self.LRelu = nn.LeakyReLU()
        self.downsample_block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 4), 4, (0, 0)),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(),
        )
        self.downsample_block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 4), 4, (0, 2)),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
        )
        self.downsample_block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 4), 4, (0, 1)),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
        )
        self.upsample_block1 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, (1, 5), 2, (0, 2)),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
            # GLU()
        )
        self.upsample_block2 = nn.Sequential(
            nn.ConvTranspose2d(100, 200, (1, 6), 2, (0, 2)),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            # GLU()
        )
        self.upsample_block3 = nn.Sequential(
            nn.ConvTranspose2d(200, 400, (1, 6), 2, (0, 2)),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(),
            # GLU()
        )
        self.trans = trans
        self.conv = nn.Conv2d(1, 1, (400, 1))

    def forward(self, imgfeature):
        concat = imgfeature.view(-1, 1, 1, 784)
        concat = self.downsample_block1(concat)
        concat = self.downsample_block2(concat)
        concat = self.downsample_block3(concat)
        concat = self.upsample_block1(concat)
        concat = self.upsample_block2(concat)
        concat = self.upsample_block3(concat)

        concat = self.conv(self.trans(concat))
        output = self.trans(concat)
        output = output.view(output.size()[0], -1)
        output = torch.tanh(output)
        return output


class Discriminator(nn.Module):
    """定义鉴别器"""
    def __init__(self, eegfeature_size):
        super(Discriminator, self).__init__()

        self.eegfeature_size = eegfeature_size

        # Defining the discriminator network architecture
        self.fc = nn.Sequential(
            nn.Linear(784, 100),
            nn.LeakyReLU(),
        )
        self.downsample_block1 = nn.Sequential(
            nn.Conv2d(2, 7, (1, 9), 1, (0, 4)),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Conv2d(7, 15, (1, 9), 2, (0, 4)),
            nn.BatchNorm2d(15),
            nn.LeakyReLU(),
        )
        self.downsample_block2 = nn.Sequential(
            nn.Conv2d(15, 30, (1, 9), 1, (0, 4)),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Conv2d(30, 75, (1, 9), 2, (0, 4)),
            nn.BatchNorm2d(75),
            nn.LeakyReLU(),
        )
        self.downsample_block3 = nn.Sequential(
            nn.Conv2d(75, 150, (1, 9), 1, (0, 4)),
            nn.BatchNorm2d(150),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Conv2d(150, 349, (1, 9), 2, (0, 4)),
            nn.BatchNorm2d(349),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(349*13, 1)

    def forward(self, eegfeature, cond):
        cond = self.fc(cond)
        cond = cond.view(-1, 1, 1, 100)
        eegfeature = eegfeature.view(-1, 1, 1, 100)
        eegfeature = torch.cat((eegfeature, cond), 1)
        d_net_out = self.downsample_block1(eegfeature)  
        d_net_out = self.downsample_block2(d_net_out)  
        output = self.downsample_block3(d_net_out)
        output = self.linear(output.view(output.size()[0], -1))
        output = torch.sigmoid(output)

        return output


class GLU(nn.Module):
    """门控线性单元激活函数"""
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])  # 输入分成两份，一份线性变换，一份sigmoid


if __name__ == "__main__":
    model = Generator(100, 784, 100)
    # model = Discriminator(100)
    imgfea = torch.zeros(1,784)
    out = model(imgfea)
    print(out.shape)