class TDCBLoss(nn.Module):
    def __init__(self, lambda_bg=0.5, eps=1e-6):
        super().__init__()
        self.lambda_bg = lambda_bg
        self.eps = eps

    def forward(self, pred, target, A95):
        px1, py1, px2, py2 = pred.unbind(-1)
        tx1, ty1, tx2, ty2 = target.unbind(-1)

        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)
        iw = (ix2 - ix1).clamp(min=0)
        ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih

        area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        union = area_p + area_t - inter + self.eps
        iou = inter / union
        L_s = 1 - iou

        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2
        L_l = ((pcx - tcx) ** 2 + (pcy - tcy) ** 2) / (area_t + self.eps)

        wp = (px2 - px1).clamp(min=0)
        hp = (py2 - py1).clamp(min=0)
        wt = (tx2 - tx1).clamp(min=0)
        ht = (ty2 - ty1).clamp(min=0)
        L_v = torch.abs(torch.atan2(wp, hp + self.eps) - torch.atan2(wt, ht + self.eps)) / (math.pi / 2)

        L_bg = (area_p - inter) / (area_p + self.eps)

        beta_s = A95 / (area_p + self.eps)
        beta_l = area_p / (A95 + self.eps)
        beta_v = L_v.detach()

        return (beta_s * L_s + beta_l * L_l + beta_v * L_v + self.lambda_bg * L_bg).mean()
