import torch.nn as nn


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_output: int = 3,
        dim_last: int = 64,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_output, 1))

    def forward(self, batch):
        x = self.encoder(batch)
        y = self.decoder(x)
        z = self.to_logits(y)

        return z
