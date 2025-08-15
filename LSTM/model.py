import torch
import torch.nn as nn
import torch.nn.functional as F

# Original simple LSTM model preserved
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, layers=1, dropout=0.1):
        super(LSTM, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        return self.output_proj(out[:, -1, :])


# New cross-attention module
class CrossAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        # x: (batch, seq_len, num_groups, feature_dim)
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, seq_len, num_groups, num_groups)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch, seq_len, num_groups, hidden_dim)
        return attn_output


# LSTM model with cross-attention between groups
class LSTMWithCrossAttention(nn.Module):
    def __init__(self, num_groups, features_per_group, hidden_size, output_dim, layers=1, dropout=0.1):
        super().__init__()
        self.num_groups = num_groups
        self.features_per_group = features_per_group
        self.hidden_size = hidden_size

        self.cross_attention = CrossAttention(feature_dim=features_per_group, hidden_dim=hidden_size)
        self.lstm = nn.LSTM(
            input_size=num_groups * hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, num_groups, features_per_group)
        attn_out = self.cross_attention(x)  # (batch, seq_len, num_groups, hidden_size)
        batch, seq_len, num_groups, hidden_dim = attn_out.shape
        lstm_input = attn_out.reshape(batch, seq_len, num_groups * hidden_dim)
        out, _ = self.lstm(lstm_input)
        return self.output_proj(out[:, -1, :])
