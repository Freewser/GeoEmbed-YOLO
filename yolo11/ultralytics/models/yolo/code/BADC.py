class BADC(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    @torch.no_grad()
    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                         device=x.device).view(1, 1, 3, 3), padding=1)
        gy = F.conv2d(gray, torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                                         device=x.device).view(1, 1, 3, 3), padding=1)
        ang = torch.atan2(gy, gx).abs().view(x.size(0), -1)
        ang = ang % math.pi

        centers = torch.linspace(0, math.pi, self.k + 2, device=x.device)[1:-1]
        for _ in range(5):
            d = torch.abs(ang.unsqueeze(-1) - centers.view(1, 1, -1))
            d = torch.minimum(d, math.pi - d)
            idx = d.argmin(dim=-1)
            for i in range(self.k):
                m = idx == i
                if m.any():
                    centers[i] = ang[m].mean()
        return centers