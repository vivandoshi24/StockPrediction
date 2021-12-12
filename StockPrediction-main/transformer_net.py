import torch
import torch.nn as nn
import torch.nn.functional as f


def scaled_dot_product_attention(query, key, value):
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    attention = softmax.bmm(value)
    return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, key_dim, value_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, key_dim)
        self.key = nn.Linear(input_dim, key_dim)
        self.value = nn.Linear(input_dim, value_dim)
        self.num_heads = num_heads

        self.linear = nn.Linear(num_heads * value_dim, input_dim)

    def forward(self, query, key, value):
        multiheads_out = [
            scaled_dot_product_attention(self.query(query), self.key(key), self.value(value)) for _ in
            range(self.num_heads)
        ]
        out = self.linear(torch.cat(multiheads_out, dim=-1))
        return out


def positioning_encoding(seq_length, model_dim):
    position = torch.arange(seq_length, dtype=torch.float).reshape(1, -1, 1)
    frequencies = 1e-4 ** (2 * (torch.arange(model_dim, dtype=torch.float) // 2) / model_dim).reshape(1, 1, -1)
    pos_enc = position * frequencies
    pos_enc[:, ::2] = torch.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = torch.sin(pos_enc[:, 1::2])
    return pos_enc


def forward(input_dim=512, forward_dim=2048):
    forward_layer = nn.Sequential(
        nn.Linear(input_dim, forward_dim),
        nn.ReLU(),
        nn.Linear(forward_dim, input_dim)
    )
    return forward_layer


class ResidualConnection(nn.Module):

    def __init__(self, layer, dimension, dropout=0.2):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *X):
        return self.norm(X[-1] + self.dropout(self.layer(*X)))


class Encoder(nn.Module):

    def __init__(self, n_layers=6, model_dim=512, num_heads=8, forward_dim=2048, dropout=0.2):
        super().__init__()

        self.n_layers = n_layers
        key_dim = value_dim = model_dim // num_heads

        # Multihead attention layer with normalized residual connections and dropout
        self.multihead_attention = ResidualConnection(
            MultiHeadAttention(num_heads, model_dim, key_dim, value_dim),
            dimension=model_dim,
            dropout=dropout
        )
        # Feed-forward layer with normalized residual connections and dropout
        self.feed_forward = ResidualConnection(
            forward(model_dim, forward_dim),
            dimension=model_dim,
            dropout=dropout
        )

    def forward(self, X):
        seq_length, dimension = X.size(1), X.size(2)
        out = X
        # Computes the positional encodings
        out += positioning_encoding(seq_length, dimension)
        # Feeds the input to the multihead attention layer followed by the feed-forward
        # layer for 'n_layers' many layers
        for _ in range(self.n_layers):
            att_out = self.multihead_attention(out, out, out)
            out = self.feed_forward(att_out)
        return out


class TransformerModel(nn.Module):
    def __init__(self, n_layers=6, model_dim=512, output_dim=512,
                 num_heads=6, forward_dim=2048, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(n_layers, model_dim, num_heads, forward_dim, dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        enc_out = self.encoder(X)
        #flat = self.flatten(enc_out)
        out = self.relu(self.linear(enc_out[:, -1, :]))
        return out

#X = torch.rand(32, 64, 32)
#out = TransformerModel(model_dim=32, output_dim=50)(X)
#print(out.shape)
