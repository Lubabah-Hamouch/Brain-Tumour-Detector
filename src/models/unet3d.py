"""3D U-Net for BraTS region segmentation (WT, TC, ET) with deep supervision.

Architecture (defaults, for 4 x 128^3 input):

    encoder   128^3 x 32 -> 64^3 x 64 -> 32^3 x 128 -> 16^3 x 256
    bottleneck                                       8^3 x 320
    decoder   16^3 x 256 -> 32^3 x 128 -> 64^3 x 64 -> 128^3 x 32 -> 1x1x1 conv -> 3

Each stage is two 3x3x3 convolutions with instance norm and LeakyReLU.
Downsampling uses stride-2 convolutions, upsampling uses 2x2x2 transposed
convolutions, and skip connections concatenate encoder features.

Output activation is sigmoid, not softmax: WT, TC and ET are nested regions
(ET within TC within WT), so a voxel can belong to several at once and each
channel is an independent binary prediction. forward() returns logits so the
loss can apply sigmoid internally (BCE-with-logits is numerically stable where
log(sigmoid(x)) is not); predict() returns probabilities.

Deep supervision: with deep_supervision=2, forward() returns
[logits at 32^3, logits at 64^3, logits at 128^3], lowest resolution first and
the full-resolution output last. Planned loss use: compare each output with
the target downsampled to its resolution and combine with weights that halve
per level and are normalised to sum to 1, e.g. 1, 1/2, 1/4 -> 0.57, 0.29, 0.14
for final, 64^3 and 32^3. The auxiliary heads give the deeper decoder stages a
direct gradient signal; at inference only the final output is used. With
deep_supervision=0, forward() returns the final logits tensor alone.
"""
import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """(Conv3d -> InstanceNorm3d -> LeakyReLU) x 2; the first conv may downsample.

    Instance norm instead of batch norm: a 4 x 128^3 patch only fits in 8 GB at
    batch size 1-2, and batch statistics estimated from one or two samples are
    noisy and make training unstable. Instance norm normalises each sample and
    channel on its own, so it behaves the same at any batch size and at
    inference time.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class UpBlock(nn.Module):
    """Transposed-conv upsampling, concatenation with the skip, then a ConvBlock."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class UNet3D(nn.Module):
    """Configurable 3D U-Net.

    Args:
        in_channels: input modalities (4: FLAIR, T1, T1ce, T2).
        out_channels: output regions (3: WT, TC, ET).
        widths: encoder channel widths, one per level; len(widths) sets the depth.
            Spatial size must be divisible by 2 ** len(widths).
        bottleneck: channel width at the lowest resolution.
        deep_supervision: number of auxiliary outputs taken from the decoder
            stages just above the final one (0 disables them).
    """

    def __init__(self, in_channels=4, out_channels=3, widths=(32, 64, 128, 256), bottleneck=320, deep_supervision=2):
        super().__init__()
        if not 0 <= deep_supervision < len(widths):
            raise ValueError(f"deep_supervision must be in [0, {len(widths) - 1}]")
        self.widths = tuple(widths)
        self.deep_supervision = deep_supervision

        self.encoder = nn.ModuleList()
        prev = in_channels
        for level, width in enumerate(widths):
            self.encoder.append(ConvBlock(prev, width, stride=1 if level == 0 else 2))
            prev = width
        self.bottleneck = ConvBlock(prev, bottleneck, stride=2)

        # Decoder runs from the deepest skip back up to full resolution.
        self.decoder = nn.ModuleList()
        prev = bottleneck
        for width in reversed(widths):
            self.decoder.append(UpBlock(prev, width, width))
            prev = width
        self.head = nn.Conv3d(widths[0], out_channels, 1)
        # Auxiliary heads on the decoder outputs one, two, ... levels above full resolution.
        self.aux_heads = nn.ModuleList(
            nn.Conv3d(widths[k], out_channels, 1) for k in range(deep_supervision, 0, -1)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.bottleneck(x)

        decoder_outputs = []
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
            decoder_outputs.append(x)

        final = self.head(decoder_outputs[-1])
        if not self.deep_supervision:
            return final
        # decoder_outputs[-1 - k] is k levels above full resolution.
        aux = [head(decoder_outputs[-1 - k]) for head, k in zip(self.aux_heads, range(self.deep_supervision, 0, -1))]
        return aux + [final]

    @torch.no_grad()
    def predict(self, x):
        """Per-region probabilities at full resolution, shape (N, 3, D, H, W)."""
        out = self(x)
        return torch.sigmoid(out[-1] if isinstance(out, list) else out)


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
