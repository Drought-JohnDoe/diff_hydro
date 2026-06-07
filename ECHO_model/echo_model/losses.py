import torch


class RmseLossComb(torch.nn.Module):
    def __init__(self, alpha: float, beta: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ny = target.shape[2]
        loss = 0.0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask = torch.isfinite(t0)
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t) ** 2).mean())
            mask1 = torch.isfinite(t1)
            pa = p1[mask1]
            ta = t1[mask1]
            loss2 = torch.sqrt(((pa - ta) ** 2).mean())
            loss = loss + (1.0 - self.alpha) * loss1 + self.alpha * loss2
        return loss
