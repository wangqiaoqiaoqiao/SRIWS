import torch
import torch.nn as nn
from encoders import get_encoder
from base import initialization as init
from typing import Optional
from base import ClassificationHead


class ClassificationModel(torch.nn.Module):
    def __init__(self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            pooling: str = "avg",
            dropout: float = 0.2,):
        super(ClassificationModel, self).__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], classes=classes, pooling=pooling, dropout=dropout, activation=activation)
        self.initialize() # initialize head

    def initialize(self):
        init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        labels = self.classification_head(features[-1])
        return labels, features[-1]


if __name__=="__main__":
    net = ClassificationModel(
        encoder_name="timm-regnety_040",
        in_channels=3,
        classes=1,
        encoder_weights="imagenet"
    ).cuda()
    print(net)