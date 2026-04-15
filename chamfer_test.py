import torch
import torch.nn as nn

class ChamferDistanceTorch(nn.Module):
    """
    Differentiable Chamfer Distance using torch.cdist
    Inputs: x (B, N, 3), y (B, M, 3)
    Returns: scalar loss
    """
    def __init__(self, variant="sum"):  
        # variant: "sum" -> mean(d1)+mean(d2)
        #          "max" -> max(mean(d1), mean(d2))
        super().__init__()
        assert variant in {"sum", "max"}
        self.variant = variant

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        y = y.contiguous()

        # (B, N, M) pairwise distances
        d = torch.cdist(x, y, p=2)

        # 每个点到对方点云最近距离
        d1 = d.min(dim=2).values  # (B, N)
        d2 = d.min(dim=1).values  # (B, M)

        m1 = d1.mean(dim=1)       # (B,)
        m2 = d2.mean(dim=1)       # (B,)

        if self.variant == "sum":
            loss = (m1 + m2).mean()      # 标量
        else:  # "max"
            loss = torch.max(m1, m2).mean()

        return loss