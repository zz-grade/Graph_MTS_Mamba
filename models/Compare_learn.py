import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeTokenContrastiveLoss(nn.Module):
    """
    两组节点 token 的双向 InfoNCE。

    支持输入：
        [B, N, D]
        [B, N, L, D]
        [B, L*N, 1, D]

    要求两组特征形状相同、token 顺序一致。
    """

    def __init__(
        self,
        input_dim,
        projection_dim=128,
        temperature=0.2,
    ):
        super().__init__()

        self.temperature = temperature

        self.projector1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

        self.projector2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(
        self,
        node_token_feats,
        node_token_feat_topk,
        valid_mask=None,
    ):
        if node_token_feats.shape != node_token_feat_topk.shape:
            raise ValueError(
                "Contrastive features must have the same shape, but got "
                f"{node_token_feats.shape} and "
                f"{node_token_feat_topk.shape}"
            )

        if node_token_feats.dim() < 3:
            raise ValueError(
                "Expected features shaped [B,...,D], "
                f"but got {node_token_feats.shape}"
            )

        feature_dim = node_token_feats.size(-1)

        # [B,...,D] -> [M,D]
        features = node_token_feats.reshape(-1, feature_dim)
        features_topk = node_token_feat_topk.reshape(-1, feature_dim)

        if valid_mask is not None:
            mask = valid_mask.reshape(-1).bool()

            if mask.numel() != features.size(0):
                raise ValueError(
                    "valid_mask must correspond to every token"
                )

            features = features[mask]
            features_topk = features_topk[mask]

        if features.size(0) <= 1:
            return node_token_feats.sum() * 0.0

        # 两个分支共享投影头
        z = self.projector1(features)
        z_topk = self.projector2(features_topk)

        z = F.normalize(z, p=2, dim=-1)
        z_topk = F.normalize(z_topk, p=2, dim=-1)

        # logits[i,j] 表示普通 token i 与 top-k token j 的相似度
        logits = torch.matmul(z, z_topk.transpose(0, 1))
        logits = logits / self.temperature

        # 对角线位置是正样本
        labels = torch.arange(
            logits.size(0),
            device=logits.device,
        )

        # 普通分支 -> top-k 分支
        loss_forward = F.cross_entropy(logits, labels)

        # top-k 分支 -> 普通分支
        loss_backward = F.cross_entropy(
            logits.transpose(0, 1),
            labels,
        )

        return 0.5 * (loss_forward + loss_backward)