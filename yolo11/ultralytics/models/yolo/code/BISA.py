class BISA(nn.Module):
    def __init__(self, dim, num_heads=4, k=64, window=8):
        super().__init__()
        self.num_heads = num_heads
        self.k = k
        self.window = window
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.E = nn.Linear(k, k, bias=False)
        self.F = nn.Linear(k, k, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, num_heads))

    def forward(self, x, edge_intensity=None):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).view(B, 3, self.num_heads, C // self.num_heads, N)
        q, k_, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q_low = F.adaptive_avg_pool1d(q, self.k)
        k_low = F.adaptive_avg_pool1d(k_, self.k)
        k_low = self.E(k_low.transpose(-1, -2)).transpose(-1, -2)
        attn = torch.einsum('bhdk,bhdk->bhk', q_low, k_low) / (C // self.num_heads) ** 0.5

        if edge_intensity is not None:
            mask = F.interpolate(edge_intensity, size=(H, W), mode='bilinear', align_corners=False)
            mask = mask.view(B, 1, N).mean(dim=1, keepdim=True)
            mask = F.adaptive_avg_pool1d(mask, self.k)
            attn = attn + mask * (-1e4)

        attn = attn.softmax(dim=-1)
        v_low = F.adaptive_avg_pool1d(v, self.k)
        v_low = self.F(v_low.transpose(-1, -2)).transpose(-1, -2)
        out = torch.einsum('bhk,bhdk->bhd', attn, v_low)
        out = out.view(B, C, H, W)
        return self.proj(out)