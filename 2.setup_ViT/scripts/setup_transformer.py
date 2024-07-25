#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

sys.path.append("../helpers")


# ## load data

# In[2]:


# load data
cls_file_path = pathlib.Path(
    "../../1.scDINO_analysis/1.scDINO_run/outputdir/mnist_photos/CLS_features/channel_binary_model_dino_deitsmall16_pretrain_full_checkpoint_features.csv"
).resolve(strict=True)

image_paths_file_path = pathlib.Path(
    "../../1.scDINO_analysis/1.scDINO_run/outputdir/mnist_photos/CLS_features/image_paths.csv"
).resolve(strict=True)

# load in the image paths
image_paths = pd.read_csv(image_paths_file_path, header=None)
print(image_paths.shape)

# load in the the data to a csv
cls_features = pd.read_csv(cls_file_path, header=None)
print(cls_features.shape)
cls_features.head()


# In[3]:


# rename columns
cls_features.columns = [f"CLS_{i}" for i in range(cls_features.shape[1])]
cls_features.head()


# In[4]:


# rename the image paths columns
image_paths.columns = ["Metadata_image_paths"]
# make metadata columns for the image paths
cls_features["Metadata_label"] = image_paths["Metadata_image_paths"].apply(
    lambda x: pathlib.Path(x).stem.split("_")[1]
)
cls_features["Metadata_cell_idx"] = image_paths["Metadata_image_paths"].apply(
    lambda x: pathlib.Path(x).stem.split("_")[3]
)
cls_features["Metadata_Time"] = image_paths["Metadata_image_paths"].apply(
    lambda x: pathlib.Path(x).stem.split("_")[5]
)
# reorder the columns so that the metadata columns are first
cls_features = cls_features[
    ["Metadata_label", "Metadata_cell_idx", "Metadata_Time"]
    + cls_features.columns[:-3].tolist()
]

# make all columns floats
cls_features = cls_features.astype(float)
cls_features.head()


# In[5]:


cls_tensor = torch.tensor(cls_features.iloc[:, 3:].values)
# reshape the data from (cell_idx, time, features) to (cell_idx, features*time)
print(cls_tensor.shape)
cls_tensor = torch.tensor(cls_features.iloc[:, 3:].values)
cls_tensor = cls_tensor.reshape(
    -1, cls_features.Metadata_Time.nunique(), cls_tensor.shape[1]
)

print(cls_tensor.shape)


# In[6]:


import torch.nn

# make a Dataset
import torch.utils


class CLSDataset(torch.utils.data.Dataset):
    def __init__(self, cls_features: pd.DataFrame):
        super(CLSDataset, self).__init__()
        self.cls_features = cls_features

    def __len__(self):
        return self.cls_features.shape[0]

    @staticmethod
    def mask_timepoints(tensor: torch.Tensor, timepoints: int):
        # mask one timepoint in the tensor
        # random value from 0 to timepoints
        random_timepoint = np.random.randint(0, timepoints)
        tensor[random_timepoint, :] = 1
        return tensor

    def __getitem__(self, idx):
        y = self.cls_features[idx, :, :]
        x = self.mask_timepoints(y, y.shape[0])
        return x, y


# make a DataLoader
cls_dataset = CLSDataset(cls_tensor)
cls_loader = torch.utils.data.DataLoader(cls_dataset, batch_size=20, shuffle=False)
# get the first batch size
x, y = next(iter(cls_loader))
x.shape, y.shape


# ## Define the model

# Code adapted from https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6

# In[7]:


class PositionWiseEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )  # Classification Token

        # Creating positional encoding
        pe = torch.zeros(max_seq_length + 1, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = torch.sin(
                        pos / 10000 ** (2 * i / torch.tensor(d_model))
                    )
                else:
                    pe[pos][i] = torch.cos(
                        pos / 10000 ** (2 * i / torch.tensor(d_model))
                    )

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch, x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe[:, : x.size(1), :]

        return x


# test the PositionWiseEncoding
d_model = torch.tensor(512)
max_seq_length = torch.tensor(10)
pe = PositionWiseEncoding(d_model, max_seq_length)
pe.forward(torch.randn(2, 10, 512))


# In[8]:


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2, -1)

        # Scaling
        attention = attention / (self.head_size**0.5)

        attention = torch.softmax(attention, dim=-1)

        attention = attention @ V

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)]
        )

    def forward(self, x):
        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        out = self.W_o(out)

        return out


# test the MultiHeadAttention
d_model = torch.tensor(512)
n_heads = torch.tensor(8)
mha = MultiHeadAttention(d_model, n_heads)
mha.forward(torch.randn(2, 10, 512))


# In[9]:


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model),
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))

        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))

        return out


# test the TransformerEncoder
d_model = 384
n_heads = 8
r_mlp = 4
te = TransformerEncoder(d_model, n_heads, r_mlp)
te.forward(torch.randn(2, 10, 384, dtype=torch.float32))


# In[10]:


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, n_classes, n_heads, n_layers, seq_len):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model  # Dimensionality of model
        self.n_classes = n_classes  # Number of classes
        self.n_heads = n_heads  # Number of attention heads
        self.seq_len = seq_len  # Sequence length

        self.positional_encoding = PositionWiseEncoding(self.d_model, self.seq_len)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)]
        )

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes), nn.Softmax(dim=-1)
        )

    def forward(self, x):

        x = self.positional_encoding(x)

        x = self.transformer_encoder(x)

        # x = self.classifier(x[:,0])
        # remove the class token
        x = x[:, 1:, :]

        return x


# test the Transformer
d_model = 384
n_classes = 10
n_heads = 8
n_layers = 4
seq_len = 10
tt = TemporalTransformer(d_model, n_classes, n_heads, n_layers, seq_len)
tt.forward(torch.randn(2, 10, 384))


# ## Test to see if the model works?

# In[17]:


d_model = 384
n_classes = 10
n_heads = 12
n_layers = 8
batch_size = 32
epochs = 50
alpha = 0.05

seq_len = 25
# training data loader is cls_loader


# In[19]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Using device: ",
    device,
    f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
)

transformer = TemporalTransformer(d_model, n_classes, n_heads, n_layers, seq_len).to(
    device
)
# import adam
from torch.optim import Adam

optimizer = Adam(transformer.parameters(), lr=alpha)
criterion = nn.MSELoss()

for epoch in range(epochs):

    training_loss = 0.0
    for i, data in enumerate(cls_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        outputs = transformer(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(cls_loader) :.3f}")


# In[20]:


list_of_reconstructed = []
list_of_original = []
with torch.no_grad():
    for data in cls_loader:
        x, y = data
        x, y = x.to(device).float(), y.to(device).float()

        # reshape (B, T, d) -> (B*T, d)
        x = transformer(x)
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        list_of_reconstructed.append(x.cpu().numpy())
        list_of_original.append(y.cpu().numpy())


# In[21]:


# make two different dfs for the reconstructed and original data
reconstructed_df = pd.DataFrame(np.concatenate(list_of_reconstructed))
original_df = pd.DataFrame(np.concatenate(list_of_original))

# add the metadata columns
reconstructed_df = pd.concat([cls_features.iloc[:, :3], reconstructed_df], axis=1)
original_df = pd.concat([cls_features.iloc[:, :3], original_df], axis=1)

# add label for reconstructed and original
reconstructed_df["Metadata_reconstructed"] = "reconstructed_df"
original_df["Metadata_reconstructed"] = "original_df"

# combine the two dataframes
combined_df = pd.concat([reconstructed_df, original_df])
combined_df.head()
# rename columns if int columns
combined_df.rename(
    columns={
        col: f"CLS_{col}" if isinstance(col, int) else col
        for col in combined_df.columns
    },
    inplace=True,
)
# metadata columns
metadata_columns = [col for col in combined_df.columns if "Metadata" in col]
combined_df.reset_index(drop=True, inplace=True)
features_df = combined_df.drop(metadata_columns, axis=1)

features_df.head()


# In[22]:


# umap
import umap

reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
embedding = reducer.fit_transform(features_df)
embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
embedding_df["Metadata_reconstructed"] = combined_df["Metadata_reconstructed"]
embedding_df["Metadata_label"] = combined_df["Metadata_label"]
embedding_df["Metadata_cell_idx"] = combined_df["Metadata_cell_idx"]
embedding_df.head()


# In[ ]:


# plot the umap
import matplotlib.pyplot as plt
import seaborn as sns

# randomize the rows for plotting
embedding_df = embedding_df.sample(frac=1)
# two subplots
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.scatterplot(
    data=embedding_df, x="UMAP1", y="UMAP2", hue="Metadata_reconstructed", ax=ax[0]
)
sns.scatterplot(data=embedding_df, x="UMAP1", y="UMAP2", hue="Metadata_label", ax=ax[1])
plt.show()


# In[ ]:
