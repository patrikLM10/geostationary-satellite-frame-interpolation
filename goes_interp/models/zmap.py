"""Physics-guided depth (Z) map.

Satellite radiance is inverse-correlated with brightness temperature: colder
(hence higher/deeper) cloud tops are brighter in Band-13 normalized space.
Z is initialized as (1 - frame) and refined by a small residual CNN, giving
softmax splatting a physically meaningful ordering signal.
"""

import torch.nn as nn


class PhysicsZMapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, frame):
        z_prior = 1.0 - frame
        return z_prior + self.net(frame)
