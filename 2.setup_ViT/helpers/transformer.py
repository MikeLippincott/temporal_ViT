import math

import torch
import torch.nn as nn


# code from https://towardsdatascience.com/a-complete-guide-to-write-your-own-transformers-29e23f371ddd
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        """
        input_dim: Dimensionality of the input.
        num_heads: The number of attention heads to split the input into.
        """
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num heads"
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)  # the Value part
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)  # the Key part
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)  # the Query part
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False)  # the output layer

    def check_sdpa_inputs(self, x):
        assert (
            x.size(1) == self.num_heads
        ), f"Expected size of x to be ({-1, self.num_heads, -1, self.hidden_dim // self.num_heads}), got {x.size()}"
        assert x.size(3) == self.hidden_dim // self.num_heads

    def scaled_dot_product_attention(
        self, query, key, value, attention_mask=None, key_padding_mask=None
    ):
        """
        query : tensor of shape (batch_size, num_heads, query_sequence_length, hidden_dim//num_heads)
        key : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        value : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)


        """
        self.check_sdpa_inputs(query)
        self.check_sdpa_inputs(key)
        self.check_sdpa_inputs(value)

        d_k = query.size(-1)
        tgt_len, src_len = query.size(-2), key.size(-2)

        # logits = (B, H, tgt_len, E) * (B, H, E, src_len) = (B, H, tgt_len, src_len)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Attention mask here
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                assert attention_mask.size() == (tgt_len, src_len)
                attention_mask = attention_mask.unsqueeze(0)
                logits = logits + attention_mask
            else:
                raise ValueError(f"Attention mask size {attention_mask.size()}")

        # Key mask here
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # Broadcast over batch size, num heads
            logits = logits + key_padding_mask

        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(
            attention, value
        )  # (batch_size, num_heads, sequence_length, hidden_dim)

        return output, attention

    def split_into_heads(self, x, num_heads):
        batch_size, seq_length, hidden_dim = x.size()
        x = x.view(batch_size, seq_length, num_heads, hidden_dim // num_heads)

        return x.transpose(
            1, 2
        )  # Final dim will be (batch_size, num_heads, seq_length, , hidden_dim // num_heads)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, head_hidden_dim = x.size()
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, num_heads * head_hidden_dim)
        )

    def forward(self, q, k, v, attention_mask=None, key_padding_mask=None):
        """
        q : tensor of shape (batch_size, query_sequence_length, hidden_dim)
        k : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        v : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)

        """
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, self.num_heads)
        k = self.split_into_heads(k, self.num_heads)
        v = self.split_into_heads(v, self.num_heads)

        # attn_values, attn_weights = self.multihead_attn(q, k, v, attn_mask=attention_mask)
        attn_values, attn_weights = self.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        grouped = self.combine_heads(attn_values)
        output = self.Wo(grouped)

        self.attention_weigths = attn_weights

        return output


#  Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_padding_mask=None):
        assert x.ndim == 3, "Expected input to be 3-dim, got {}".format(x.ndim)
        att_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(att_output))

        ff_output = self.ff(x)
        output = x + self.norm2(ff_output)

        return output


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_dim: int,
        dropout: float,
        n_encoder_blocks: int,
        n_heads: int,
    ):

        super(Encoder, self).__init__()
        self.n_dim = n_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_dim)
        self.positional_encoding = PositionalEncoding(d_model=n_dim, dropout=dropout)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(n_dim, dropout, n_heads) for _ in range(n_encoder_blocks)]
        )

    def forward(self, x, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.n_dim)
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x=x, src_padding_mask=padding_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(DecoderBlock, self).__init__()

        # The first Multi-Head Attention has a mask to avoid looking at the future
        self.self_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)

        # The second Multi-Head Attention will take inputs from the encoder as key/value inputs
        self.cross_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm2 = nn.LayerNorm(n_dim)

        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm3 = nn.LayerNorm(n_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_padding_mask=None,
        memory_padding_mask=None,
    ):

        masked_att_output = self.self_attention(
            q=tgt,
            k=tgt,
            v=tgt,
            attention_mask=tgt_mask,
            key_padding_mask=tgt_padding_mask,
        )
        x1 = tgt + self.norm1(masked_att_output)

        cross_att_output = self.cross_attention(
            q=x1,
            k=memory,
            v=memory,
            attention_mask=None,
            key_padding_mask=memory_padding_mask,
        )
        x2 = x1 + self.norm2(cross_att_output)

        ff_output = self.ff(x2)
        output = x2 + self.norm3(ff_output)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_dim: int,
        dropout: float,
        n_decoder_blocks: int,
        n_heads: int,
    ):

        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_dim, padding_idx=0
        )
        self.positional_encoding = PositionalEncoding(d_model=n_dim, dropout=dropout)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_dim, dropout, n_heads) for _ in range(n_decoder_blocks)]
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_padding_mask=None,
        memory_padding_mask=None,
    ):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        for block in self.decoder_blocks:
            x = block(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )
        return x


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask. From PyTorch docs."""
    mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()

        for k, v in kwargs.items():
            print(f" * {k}={v}")

        self.vocab_size = kwargs.get("vocab_size")
        self.model_dim = kwargs.get("model_dim")
        self.dropout = kwargs.get("dropout")
        self.n_encoder_layers = kwargs.get("n_encoder_layers")
        self.n_decoder_layers = kwargs.get("n_decoder_layers")
        self.n_heads = kwargs.get("n_heads")
        self.batch_size = kwargs.get("batch_size")
        self.PAD_IDX = kwargs.get("pad_idx", 0)

        self.encoder = Encoder(
            self.vocab_size,
            self.model_dim,
            self.dropout,
            self.n_encoder_layers,
            self.n_heads,
        )
        self.decoder = Decoder(
            self.vocab_size,
            self.model_dim,
            self.dropout,
            self.n_decoder_layers,
            self.n_heads,
        )
        self.fc = nn.Linear(self.model_dim, self.vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(size: int):
        """Generate a triangular (size, size) mask. From PyTorch docs."""
        mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input
            x: (B, S) with elements in (0, C) where C is num_classes
        Output
            (B, S, E) embedding
        """

        mask = (x == self.PAD_IDX).float()
        encoder_padding_mask = mask.masked_fill(mask == 1, float("-inf"))

        # (B, S, E)
        encoder_output = self.encoder(x, padding_mask=encoder_padding_mask)

        return encoder_output, encoder_padding_mask

    def decode(
        self, tgt: torch.Tensor, memory: torch.Tensor, memory_padding_mask=None
    ) -> torch.Tensor:
        """
        B = Batch size
        S = Source sequence length
        L = Target sequence length
        E = Model dimension

        Input
            encoded_x: (B, S, E)
            y: (B, L) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """

        mask = (tgt == self.PAD_IDX).float()
        tgt_padding_mask = mask.masked_fill(mask == 1, float("-inf"))

        decoder_output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)),
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
        output = self.fc(decoder_output)  # shape (B, L, C)
        return output

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """

        # Encoder output shape (B, S, E)
        encoder_output, encoder_padding_mask = self.encode(x)

        # Decoder output shape (B, L, C)
        decoder_output = self.decode(
            tgt=y, memory=encoder_output, memory_padding_mask=encoder_padding_mask
        )

        return decoder_output


def predict(
    self, x: torch.Tensor, sos_idx: int = 1, eos_idx: int = 2, max_length: int = None
) -> torch.Tensor:
    """
    Method to use at inference time. Predict y from x one token at a time. This method is greedy
    decoding. Beam search can be used instead for a potential accuracy boost.

    Input
        x: str
    Output
        (B, L, C) logits
    """

    # Pad the tokens with beginning and end of sentence tokens
    x = torch.cat([torch.tensor([sos_idx]), x, torch.tensor([eos_idx])]).unsqueeze(0)

    encoder_output, mask = self.transformer.encode(x)  # (B, S, E)

    if not max_length:
        max_length = x.size(1)

    outputs = torch.ones((x.size()[0], max_length)).type_as(x).long() * sos_idx
    for step in range(1, max_length):
        y = outputs[:, :step]
        probs = self.transformer.decode(y, encoder_output)
        output = torch.argmax(probs, dim=-1)

        # Uncomment if you want to see step by step predicitons
        # print(f"Knowing {y} we output {output[:, -1]}")

        if output[:, -1].detach().numpy() in (eos_idx, sos_idx):
            break
        outputs[:, step] = output[:, -1]

    return outputs
