class MSCA(nn.Module):
    def __init__(self, dim, scales=(3, 5, 7)):
        super().__init__()
        self.scales = scales
        self.fc = nn.Sequential(
            nn.Conv2d(dim * len(scales), dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pools = [F.adaptive_avg_pool2d(x, s) for s in self.scales]
        pools = [F.interpolate(p, size=x.shape[-2:], mode='bilinear', align_corners=False) for p in pools]
        return x * self.fc(torch.cat(pools, dim=1))