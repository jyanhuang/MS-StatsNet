# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from attention import (SEAttention2D,
                        ECAAttention2D,
                        BandBranchAttention,
                        SpectralStatChannelAttention,
                        SpectralStatChannelAttentionv2,
                        TABRA,
                        TECAAttention2D,
                        TECAAttention2Dv2,
                        TECAAttention2Dv3,
                        TRECAAttention2D)


#  260420 baseline model == best model
# =========================
# Non-attention pooling
# =========================
class StatsPool2D(nn.Module):
    """
    Input: x shape [B, C, 1, T]
    Output:
        mean_std: [B, 2C]
        mean:     [B, C]
        max:      [B, C]
        mean_max: [B, 2C]
    """
    def __init__(self, pool_type="mean_std"):
        super().__init__()
        self.pool_type = pool_type

        if pool_type in ("mean_std", "mean_max"):
            self.out_multiplier = 2
        elif pool_type in ("mean", "max"):
            self.out_multiplier = 1
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def forward(self, x):
        x = x.squeeze(2)                  # (B, C, T)
        mean = x.mean(dim=-1)             # (B, C)

        if self.pool_type == "mean":
            return mean
        if self.pool_type == "max":
            return x.max(dim=-1).values
        if self.pool_type == "mean_max":
            return torch.cat([mean, x.max(dim=-1).values], dim=1)

        std = x.std(dim=-1, unbiased=False)  # (B, C)
        return torch.cat([mean, std], dim=1)  # (B, 2C)


# =========================
# Original non-attention modules
# =========================
class TemporalDSBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=9, dilation=1, dropout=0.3, gn_groups=4):
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation

        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            dilation=(1, dilation),
            groups=in_ch,
            bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False)
        self.gn = nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.dw(x)
        out = self.pw(out)
        out = self.gn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + identity

class FDNetCNN_CrossSubject(nn.Module):
    def __init__(
        self,
        classes=2,
        sampleChannel=30,
        sampleLength=384,
        fs=128,
        stem_channels=16,
        branch_channels=24,
        emb_dim=64,
        dropout=0.3,
        return_log_probs=False,
        active_branches=("s", "m", "l"),
        use_temporal_block=True,
        pool_type="mean_std"
    ):
        super().__init__()
        self.sampleChannel = sampleChannel
        self.sampleLength = sampleLength
        self.return_log_probs = return_log_probs
        self.active_branches = tuple(active_branches)

        valid_branches = {"s", "m", "l"}
        if not self.active_branches:
            raise ValueError("active_branches must contain at least one branch.")
        unknown = set(self.active_branches) - valid_branches
        if unknown:
            raise ValueError(f"Unknown active branches: {sorted(unknown)}")

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, kernel_size=(sampleChannel, 15), padding=(0, 7), bias=False),
            nn.InstanceNorm2d(stem_channels, affine=True),
            nn.GELU()
        )

        self.branch_s = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=(1, 7), padding=(0, 3),
                      groups=stem_channels, bias=False),
            nn.Conv2d(stem_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(num_groups=4, num_channels=branch_channels),
            nn.GELU(),
            nn.AvgPool2d((1, 2))
        )

        self.branch_m = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=(1, 31), padding=(0, 15),
                      groups=stem_channels, bias=False),
            nn.Conv2d(stem_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(num_groups=4, num_channels=branch_channels),
            nn.GELU(),
            nn.AvgPool2d((1, 2))
        )

        self.branch_l = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=(1, 15), padding=(0, 14),  # k=15
                      dilation=(1, 2), groups=stem_channels, bias=False),   # d=2
            nn.Conv2d(stem_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(num_groups=4, num_channels=branch_channels),
            nn.GELU(),
            nn.AvgPool2d((1, 2))
        )

        # self.spec_attn = SpectralStatChannelAttention(
        #     channels=stem_channels,
        #     fs=fs,
        #     reduction=8,
        #     bands=((4, 8), (8, 13), (13, 30)),
        #     use_time_stats=True
        # )
        # self.spec_attnv2 = SpectralStatChannelAttentionv2(
        #     channels=stem_channels,
        #     fs=fs,
        #     reduction=8,
        #     bands=((4, 8), (8, 13), (13, 30)),
        #     use_time_stats=True,
        #     use_relative_power=True,
        #     use_band_ratio=True,
        #     use_log_power=True,
        #     norm_desc=True,
        #     gate_scale=0.5
        # )
        # self.tab_attn = TABRA(
        #     channels=stem_channels,
        #     fs=fs,
        #     reduction=8,
        #     bands=((4, 8), (8, 13), (13, 30)),
        #     use_time_stats=True,
        #     use_log_ratio=True,
        #     norm_desc=True,
        #     residual_gate=True,
        #     gate_scale=0.5
        # )

        # self.branch_attn = BandBranchAttention(
        #     branch_channels=branch_channels,
        #     num_branches=3
        # )

        fusion_ch = branch_channels * len(active_branches)


        # self.se = SEAttention2D(fusion_ch, reduction=8)
        # self.eac = TECAAttention2D(fusion_ch, k_size=3)
        # self.eac = TECAAttention2Dv3(fusion_ch, k_size=3)
        # self.eca = TRECAAttention2D(channels=30, c_k=3, t_k=5)
        # self.attn = nn.Identity()

        # self.block1 = TemporalDSBlock2D(
        #     fusion_ch, fusion_ch, kernel_size=9, dilation=1, dropout=dropout
        # )
        if use_temporal_block:
            self.block2 = TemporalDSBlock2D(
                fusion_ch, fusion_ch, kernel_size=9, dilation=2, dropout=dropout
            )
        else:
            self.block2 = nn.Identity()

        self.pool = StatsPool2D(pool_type=pool_type)

        self.embed = nn.Sequential(
            nn.Linear(fusion_ch * self.pool.out_multiplier, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(emb_dim, classes)

        self.out_act = nn.LogSoftmax(dim=1) if return_log_probs else nn.Identity()

    def forward(self, x):
        x = self.stem(x)

        feats = []
        if "s" in self.active_branches:
            feats.append(self.branch_s(x))
        if "m" in self.active_branches:
            feats.append(self.branch_m(x))
        if "l" in self.active_branches:
            feats.append(self.branch_l(x))
        x = torch.cat(feats, dim=1)

        # x = self.se(x)
        # x = self.eac(x)
        # x = self.block1(x)
        x = self.block2(x)

        feat = self.pool(x)
        feat = self.embed(feat)
        logits = self.fc(feat)
        return self.out_act(logits)
