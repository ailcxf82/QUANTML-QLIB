"""
日内 Deep Factor GRU（B3 架构 B）。

结构对齐 qlib.contrib.model.pytorch_gru_ts.GRUModel：
  GRU(d_feat → hidden_size, num_layers, batch_first) → Linear(hidden_size, 1)
  预测取最后一根 bar 的 hidden：out[:, -1, :]。

与 qlib 版的唯一差异：本类 forward 直接接受 [N, T, F] 张量（日内池化后的形状），
而 qlib 版接受 [N, F*T] 再内部 reshape。Deep Factor 提取即取 extract_hidden()
返回的 out[:, -1, :]（hidden_size 维），与 forward 内部完全一致，保证一致性。

train（scripts/train_intraday_model.py）与 extract（scripts/extract_deep_factor.py）
共用本类，避免结构漂移导致 hidden 不可复现。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IntradayGRU(nn.Module):
    """日内 GRU，forward 返回标量预测；extract_hidden 返回末根 hidden。"""

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze(-1)

    @torch.no_grad()
    def extract_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """返回末根 bar 的 hidden 表示 out[:, -1, :]，形状 [N, hidden_size]。"""
        out, _ = self.rnn(x)
        return out[:, -1, :]
