import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # 调整维度顺序，适应 Transformer 的输入 (seq_length, batch_size, input_dim)
        output = self.transformer_encoder(x)
        output = output.permute(1, 0, 2)  # 调整维度顺序，恢复为 (batch_size, seq_length, input_dim)
        return output


# 定义 MLP 部分
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.mean(x, dim=1)  # 对序列维度取平均
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    