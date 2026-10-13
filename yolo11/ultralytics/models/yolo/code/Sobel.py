class Sobel2D(nn.Module):
    def __init__(self, in_ch, out_ch=4, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv_x = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.conv_y = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self._init_sobel()
        self.act_i = nn.Softplus()
        self.act_d = nn.Tanh()
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch * 2, 1, groups=out_ch * 2, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
        )

    def _init_sobel(self):
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.],
                           [0., 0., 0.],
                           [1., 2., 1.]])
        for conv, k in [(self.conv_x, kx), (self.conv_y, ky)]:
            w = k.view(1, 1, 3, 3).repeat(conv.out_channels, 1, 1, 1)
            conv.weight.data.copy_(w)
            conv.weight.requires_grad = False

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        I2D = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        D2D = torch.atan2(gx + self.eps, gy + self.eps)
        I2D = self.act_i(I2D)
        D2D = self.act_d(D2D)
        out = self.fuse(torch.cat([I2D, D2D], dim=1))
        return out, I2D, D2D


class Sobel3D(nn.Module):
    def __init__(self, in_ch, out_ch=16, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv_xy = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.conv_z = nn.Conv2d(out_ch, out_ch, 1, groups=out_ch, bias=False)
        self.act = nn.Softplus()
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self._init_sobel_xy()

    def _init_sobel_xy(self):
        k = torch.tensor([[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]])
        w = k.view(1, 1, 3, 3).repeat(self.conv_xy.out_channels, 1, 1, 1)
        self.conv_xy.weight.data.copy_(w)
        self.conv_xy.weight.requires_grad = False

    def forward(self, x, I2D, D2D):
        v = torch.stack([x, I2D, D2D], dim=2)
        B, C, M, H, W = v.shape
        xy = self.conv_xy(v.view(B * C * M, 1, H, W)).view(B, C, M, H, W)
        z = xy[:, :, 2] - xy[:, :, 0]
        z = self.conv_z(z)
        z = self.act(z)
        out = self.fuse(torch.cat([xy[:, :, 1], z], dim=1))
        return out


class AMFCEU(nn.Module):
    def __init__(self, channels, targets=(0.7, 0.2, 0.1), eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer('targets', torch.tensor(targets).view(1, 3, 1, 1, 1))
        self.proj = nn.Conv2d(channels * 3, channels, 1, bias=False)

    def forward(self, f_sem, f_2d, f_3d):
        feats = torch.stack([f_sem, f_2d, f_3d], dim=1)
        mav = feats.abs().mean(dim=(2, 3, 4), keepdim=True) + self.eps
        w = self.targets / mav
        w = w / (w.sum(dim=1, keepdim=True) + self.eps)
        out = (feats * w).sum(dim=1)
        return self.proj(out)

