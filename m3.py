import torch.nn as nn
import torch.nn.functional as activation


class M3(nn.Module):
    def __init__(self, num_classes):
        super(M3, self).__init__()
        self.first_conv_layer = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=80, stride=4)
        self.first_batch_norm_layer = nn.BatchNorm1d(num_features=256)
        self.pooling_layer = nn.MaxPool1d(kernel_size=4)
        self.second_conv_layer = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.second_batch_norm_layer = nn.BatchNorm1d(num_features=256)

        self.avg_pool = nn.AvgPool1d(138)
        self.softmax_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pooling_layer(activation.relu(self.first_batch_norm_layer(self.first_conv_layer(x))))
        x = self.pooling_layer(activation.relu(self.second_batch_norm_layer(self.second_conv_layer(x))))

        # Global avg pooling
        x = self.avg_pool(x) # [batch_size, 256, 1]

        # Dense
        x = x.view(x.size(0), -1) # [batch_size, 256*1=256]
        x = self.softmax_layer(x) # [batch_size, 10]
        return x
