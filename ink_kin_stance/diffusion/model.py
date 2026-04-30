import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        t = t.float()
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings
        )
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, num_steps=20, num_joints=12, condition_dim=6,
                 hidden_dim=512, time_dim=256, num_blocks=4):
        super().__init__()
        self.num_steps = num_steps
        self.num_joints = num_joints
        self.traj_dim = num_steps * num_joints

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(self.traj_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for _ in range(num_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, self.traj_dim)

    def forward(self, x, t, condition):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        t_emb = self.time_mlp(t)
        c_emb = self.condition_mlp(condition)
        x_emb = self.input_proj(x_flat)

        h = x_emb
        for block in self.blocks:
            h_in = torch.cat([h, t_emb, c_emb], dim=-1)
            h = h + block(h_in)

        out = self.output_proj(h)
        return out.view(batch_size, self.num_steps, self.num_joints)
