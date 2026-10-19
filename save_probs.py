# save_probs.py
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
import sys

log_file = "D:/dataset/log/save_probs.log"

logger = logging.getLogger('SAVEPROBS')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

with open("D:/dataset/token_to_id.pkl", 'rb') as f:
    token_to_id = pickle.load(f)

class InsiderDataset(Dataset):
    def __init__(self, df):
        self.sequences = torch.LongTensor(np.vstack(df['padded'].values))
        self.masks = torch.FloatTensor(np.stack(df['mask'].values))
        self.labels = torch.FloatTensor(df['has_anomaly'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx], self.labels[idx]


class InsiderBiLSTM(nn.Module):
    def __init__(self, vocab_size, token_weights=None, embedding_dim=64, hidden_dim=120, num_layers=2, dropout=0.4):
        super(InsiderBiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=token_to_id['PAD'])

        if token_weights is not None:
            self.register_buffer('token_weights', torch.FloatTensor(token_weights))
        else:
            self.register_buffer('token_weights', torch.ones(vocab_size))

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2 + 1, 1)

    def forward(self, x, mask):
        embedded = self.embedding(x)
        weights = self.token_weights[x].unsqueeze(-1)
        embedded = embedded * weights

        lengths = mask.sum(dim=1).long().clamp(min=1)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        attn_scores = self.attn(lstm_out)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_out = (lstm_out * attn_weights).sum(dim=1)

        lengths_norm = lengths.float().unsqueeze(-1) / 250.0
        combined = torch.cat([attn_out, lengths_norm], dim=1)

        combined = self.dropout(combined)
        out = self.fc(combined)
        return out.squeeze(-1)


def get_probs(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for sequences, masks, labels in loader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            outputs = model(sequences, masks)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Устройство: {device}")

train_df = pd.read_pickle("D:/dataset/train_sequences.pkl")
val_df = pd.read_pickle("D:/dataset/val_sequences.pkl")
test_df = pd.read_pickle("D:/dataset/test_sequences.pkl")

token_weights = np.load("D:/dataset/token_weights.npy")

logger.info(f"Train: {len(train_df)} сессий")
logger.info(f"Val: {len(val_df)} сессий")
logger.info(f"Test: {len(test_df)} сессий")

vocab_size = len(token_to_id)
model = InsiderBiLSTM(
    vocab_size=vocab_size,
    token_weights=token_weights,
    embedding_dim=64,
    hidden_dim=120,
    num_layers=2,
    dropout=0.4
).to(device)

state_dict = torch.load("D:/dataset/bilstm_best.pt", map_location=device)
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

logger.info("Модель загружена")

BATCH_SIZE = 128

logger.info("Инференс на train")
train_loader = DataLoader(InsiderDataset(train_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
train_probs = get_probs(model, train_loader, device)
train_df_copy = train_df.copy()
train_df_copy['prob'] = train_probs
train_df_copy.to_pickle("D:/dataset/train_with_probs.pkl")
logger.info(f"Сохранено: train_with_probs.pkl ({len(train_probs)} записей)")

logger.info("Инференс на val")
val_loader = DataLoader(InsiderDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
val_probs = get_probs(model, val_loader, device)
val_df_copy = val_df.copy()
val_df_copy['prob'] = val_probs
val_df_copy.to_pickle("D:/dataset/val_with_probs.pkl")
logger.info(f"Сохранено: val_with_probs.pkl ({len(val_probs)} записей)")

logger.info("Инференс на test")
test_loader = DataLoader(InsiderDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_probs = get_probs(model, test_loader, device)
test_df_copy = test_df.copy()
test_df_copy['prob'] = test_probs
test_df_copy.to_pickle("D:/dataset/test_with_probs.pkl")
logger.info(f"Сохранено: test_with_probs.pkl ({len(test_probs)} записей)")

logger.info("ГОТОВО")
logger.info("Файлы для HMM:")
logger.info("D:/dataset/train_with_probs.pkl")
logger.info("D:/dataset/val_with_probs.pkl")
logger.info("D:/dataset/test_with_probs.pkl")