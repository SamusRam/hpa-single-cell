import torch
import torch.nn as nn
import torch.nn.functional as F


class BestfittingEncodingsModel(nn.Module):
    def __init__(self, densenet121_model):
        super(BestfittingEncodingsModel, self).__init__()
        self.densenet121_model = densenet121_model

    def forward(self, x):
        mean = [0.074598, 0.050630, 0.050891, 0.076287]  # rgby
        std = [0.122813, 0.085745, 0.129882, 0.119411]
        for i in range(self.densenet121_model.in_channels):
            x[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]

        x = self.densenet121_model.conv1(x)
        if self.densenet121_model.large:
            x = self.densenet121_model.maxpool(x)
        e2 = self.densenet121_model.encoder2(x)
        e3 = self.densenet121_model.encoder3(e2)
        e4 = self.densenet121_model.encoder4(e3)
        e5 = self.densenet121_model.encoder5(e4)
        e5 = F.relu(e5, inplace=True)
        if self.densenet121_model.dropout:
            x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
            x = x.view(x.size(0), -1)
            x = self.densenet121_model.bn1(x)
            x = F.dropout(x, p=0.5, training=self.densenet121_model.training)
            x = self.densenet121_model.fc1(x)
            return x
        else:
            raise NotImplementedError()