import torch
from ..ops_base import GPU_Operator

class CsZscore(GPU_Operator):
    def _compute(self, x):
        mean = torch.nanmean(x, dim=1, keepdim=True)
        # 用方差开根号近似实现 nanstd
        var = torch.nanmean((x - mean)**2, dim=1, keepdim=True)
        std = torch.sqrt(var)
        return torch.where(std != 0, (x - mean) / std, torch.tensor(0.0, device=x.device))

class CsRank(GPU_Operator):
    def _compute(self, x):
        # 沿列排序生成截面百分位排名 (0 到 1 之间)
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
        valid_counts = (~torch.isnan(x)).sum(dim=1, keepdim=True).float()
        return ranks / (valid_counts - 1.0 + 1e-8)