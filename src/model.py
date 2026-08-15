import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .dataset import IMG_HEIGHT, IMG_WIDTH


class CRNN(nn.Module):
    """
    ResNet-18 + CNN feature refinement + two BiGRU layers + CTC output.

    Input:  [B, 3, H, W]
    Output: [T, B, num_chars]
    """

    def __init__(self, num_chars: int, rnn_hidden: int = 256):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn1 = nn.Sequential(*list(backbone.children())[:-3])

        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 6), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        feature_size = self._get_cnn_output_size()
        self.fc1 = nn.Linear(feature_size, rnn_hidden)

        self.gru1 = nn.GRU(
            rnn_hidden,
            rnn_hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            rnn_hidden,
            rnn_hidden,
            bidirectional=True,
            batch_first=True,
        )

        self.fc2 = nn.Linear(rnn_hidden * 2, num_chars)

        # Initialize only layers added on top of the pretrained ResNet.
        self._initialize_custom_layers()

    def _get_cnn_output_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            features = self.cnn2(self.cnn1(dummy))
            _, channels, height, _ = features.shape
        return channels * height

    @staticmethod
    def _init_layer(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)

    def _initialize_custom_layers(self):
        self.cnn2.apply(self._init_layer)
        self.fc1.apply(self._init_layer)
        self.fc2.apply(self._init_layer)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)

        batch, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, width, channels * height)
        x = self.fc1(x)

        x, _ = self.gru1(x)
        half = x.size(2) // 2
        x = x[:, :, :half] + x[:, :, half:]

        x, _ = self.gru2(x)
        x = self.fc2(x)

        return x.permute(1, 0, 2)
