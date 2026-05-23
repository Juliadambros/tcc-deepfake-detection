import torch
import torch.nn as nn
from torchvision import models


class ResNeXtLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        freeze_backbone=True,
    ):
        super().__init__()

        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        resnext = models.resnext50_32x4d(weights=weights)

        self.feature_dim = resnext.fc.in_features
        resnext.fc = nn.Identity()

        self.backbone = resnext

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 3, 256, 256)
        """
        batch_size, seq_len, c, h, w = x.shape

        x = x.view(batch_size * seq_len, c, h, w)

        features = self.backbone(x)

        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)

        last_output = lstm_out[:, -1, :]

        logits = self.classifier(last_output)

        return logits