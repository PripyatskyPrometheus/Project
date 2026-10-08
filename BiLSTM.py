# BiLSTM.py
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score
import time
import matplotlib.pyplot as plt


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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for sequences, masks, labels in loader:
        sequences = sequences.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for sequences, masks, labels in loader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(sequences, masks)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss, np.array(all_probs), np.array(all_labels)


def calculate_metrics(labels, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return p, r, f1, auroc, auprc


print("ПЕРВЫЙ ЭТАП: ПРОВЕРКА УСТРОЙСТВА И ЗАГРУЗКА ДАННЫХ")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.backends.cudnn.benchmark = True
print(f"PyTorch: {torch.__version__}")

train_df = pd.read_pickle("D:/dataset/train_sequences.pkl")
val_df = pd.read_pickle("D:/dataset/val_sequences.pkl")
test_df = pd.read_pickle("D:/dataset/test_sequences.pkl")

with open("D:/dataset/token_to_id.pkl", 'rb') as f:
    token_to_id = pickle.load(f)

token_weights = np.load("D:/dataset/token_weights.npy")
pos_weight = np.load("D:/dataset/pos_weight.npy")[0]
pos_weight_sqrt = np.sqrt(pos_weight)

print(f"\nTrain: {len(train_df)} сессий ({train_df['has_anomaly'].sum()} аномалий)")
print(f"Val: {len(val_df)} сессий ({val_df['has_anomaly'].sum()} аномалий)")
print(f"Test: {len(test_df)} сессий ({test_df['has_anomaly'].sum()} аномалий)")
print(f"Размер словаря: {len(token_to_id)}")
print(f"pos_weight: {pos_weight:.1f} -> sqrt(pos_weight): {pos_weight_sqrt:.1f}")

print("\nВТОРОЙ ЭТАП: СОЗДАНИЕ DATASET И DATALOADER")

train_dataset = InsiderDataset(train_df)
val_dataset = InsiderDataset(val_df)
test_dataset = InsiderDataset(test_df)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"\nTrain-батчи: {len(train_loader)}")
print(f"Val-батчи: {len(val_loader)}")
print(f"Test-батчи: {len(test_loader)}")

sample_seq, sample_mask, sample_label = next(iter(train_loader))
print(f"\nПример батча:")
print(f"Sequence shape: {sample_seq.shape}")
print(f"Mask shape: {sample_mask.shape}")
print(f"Label shape: {sample_label.shape}")

print("\nТРЕТИЙ ЭТАП: СОЗДАНИЕ МОДЕЛИ BiLSTM")

vocab_size = len(token_to_id)
model = InsiderBiLSTM(
    vocab_size=vocab_size,
    token_weights=token_weights,
    embedding_dim=64,
    hidden_dim=120,
    num_layers=2,
    dropout=0.4
).to(device)

print(f"Модель на устройстве: {next(model.parameters()).device}")
print(f"Всего памяти GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nВсего параметров: {total_params}")
print(f"Обучаемых параметров: {trainable_params}")

print("\nАрхитектура модели:")
print(model)

pos_weight_tensor = torch.FloatTensor([pos_weight_sqrt]).to(device)
criterion_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
criterion_eval = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print(f"\nLoss: BCEWithLogitsLoss (pos_weight={pos_weight_sqrt:.1f})")
print(f"Optimizer: Adam (lr=0.001, weight_decay=1e-5)")

print("\nЧЕТВЕРТЫЙ ЭТАП: ОБУЧЕНИЕ")
N_EPOCHS = 30
PATIENCE = 5

best_val_auprc = 0
best_epoch = 0
patience_counter = 0

train_losses = []
val_losses = []
val_f1s = []
val_auprcs = []

start_time = time.time()

for epoch in range(N_EPOCHS):
    epoch_start = time.time()

    train_loss = train_epoch(model, train_loader, criterion_train, optimizer, device)

    val_loss, val_probs, val_labels = eval_epoch(model, val_loader, criterion_eval, device)
    val_p, val_r, val_f1, val_auroc, val_auprc = calculate_metrics(val_labels, val_probs)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1s.append(val_f1)
    val_auprcs.append(val_auprc)
    torch.cuda.empty_cache()

    epoch_time = time.time() - epoch_start

    print(f"\nEpoch {epoch+1:2d}/{N_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print(f"Val AUPRC: {val_auprc:.4f}")
    print(f"Время: {epoch_time:.1f} секунд")
    print(f"Максимальная загрузка памяти GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if val_auprc > best_val_auprc:
        best_val_auprc = val_auprc
        best_epoch = epoch + 1
        patience_counter = 0
        state_dict = model.state_dict()
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        torch.save(state_dict, "D:/dataset/bilstm_best.pt")
        print(f"Новая лучшая модель (AUPRC={val_auprc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping на эпохе {epoch+1}")
            break

total_time = time.time() - start_time
print(f"\nОбщее время обучения: {total_time/60:.1f} минут")
print(f"Лучшая эпоха: {best_epoch} (Val AUPRC: {best_val_auprc:.4f})")

print("\nПЯТЫЙ ЭТАП: ОЦЕНКА")

model_eval = InsiderBiLSTM(
    vocab_size=vocab_size,
    token_weights=token_weights,
    embedding_dim=64,
    hidden_dim=120,
    num_layers=2,
    dropout=0.4
).to(device)

model_eval.load_state_dict(torch.load("D:/dataset/bilstm_best.pt"))
model_eval.eval()

test_loss, test_probs, test_labels = eval_epoch(model_eval, test_loader, criterion_eval, device)
test_p, test_r, test_f1, test_auroc, test_auprc = calculate_metrics(test_labels, test_probs)

print(f"\nРЕЗУЛЬТАТЫ НА ТЕСТЕ (порог 0.5):")
print(f"Precision: {test_p:.4f}")
print(f"Recall: {test_r:.4f}")
print(f"F1: {test_f1:.4f}")
print(f"AUROC: {test_auroc:.4f}")
print(f"AUPRC: {test_auprc:.4f}")

print("\nПОДБОР ОПТИМАЛЬНОГО ПОРОГА НА VAL")

val_loss, val_probs, val_labels = eval_epoch(model_eval, val_loader, criterion_eval, device)

precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1s[:-1])
best_threshold = thresholds[best_idx]

print(f"\nЛучший порог: {best_threshold:.4f}")

np.save("D:/dataset/best_threshold.npy", np.array([best_threshold]))
print(f"Порог сохранён: D:/dataset/best_threshold.npy")

p_best, r_best, f1_best, _, _ = calculate_metrics(test_labels, test_probs, threshold=best_threshold)
print(f"\nПри лучшем пороге {best_threshold:.4f}:")
print(f"Precision: {p_best:.4f}")
print(f"Recall: {r_best:.4f}")
print(f"F1: {f1_best:.4f}")

print("\nМЕТРИКИ ПО СЕССИЯМ (TEST)")

print(f"\n{'Порог':>8} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    p, r, f1, _, _ = calculate_metrics(test_labels, test_probs, threshold=th)
    marker = " <- лучший" if abs(th - best_threshold) < 0.02 else ""
    print(f"{th:>8.2f} | {p:>10.4f} | {r:>10.4f} | {f1:>10.4f}{marker}")

print("\nМЕТРИКИ ПО ПОЛЬЗОВАТЕЛЯМ (TEST)")

test_df_copy = test_df.copy()
test_df_copy['prob'] = test_probs

user_probs = test_df_copy.groupby('user')['prob'].max()
user_labels = test_df_copy.groupby('user')['has_anomaly'].max()

print(f"\nВсего пользователей: {len(user_probs)}")
print(f"Пользователей с аномалиями: {user_labels.sum()}")

for th in [0.3, 0.5, best_threshold]:
    preds = (user_probs > th).astype(int)
    p = precision_score(user_labels, preds, zero_division=0)
    r = recall_score(user_labels, preds, zero_division=0)
    f1 = f1_score(user_labels, preds, zero_division=0)
    caught = preds[user_labels == 1].sum()
    total = user_labels.sum()
    print(f"\nПорог {th:.4f}:")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Поймано: {caught} / {total}")

print("\nАНАЛИЗ ОШИБОК")

test_df_copy['pred'] = (test_probs > best_threshold).astype(int)

fn = test_df_copy[(test_df_copy['has_anomaly'] == 1) & (test_df_copy['pred'] == 0)]
fp = test_df_copy[(test_df_copy['has_anomaly'] == 0) & (test_df_copy['pred'] == 1)]
tp = test_df_copy[(test_df_copy['has_anomaly'] == 1) & (test_df_copy['pred'] == 1)]

print(f"\nПропущено аномалий: {len(fn)} из {test_df_copy['has_anomaly'].sum()}")
print(f"Ложных тревог: {len(fp)}")
print(f"Поймано аномалий: {len(tp)}")

print(f"\nПримеры ПРОПУЩЕННЫХ аномалий:")
for _, row in fn.head(5).iterrows():
    print(f"user={row['user']}, len={row['sequence_length']}, prob={row['prob']:.4f}")
    print(f"  {row['sequence'][:15]}...")

print(f"\nПримеры ПОЙМАННЫХ аномалий (первые 5):")
for _, row in tp.head(5).iterrows():
    print(f"user={row['user']}, len={row['sequence_length']}, prob={row['prob']:.4f}")
    print(f"  {row['sequence'][:15]}...")

print("\nПОЛЬЗОВАТЕЛИ С АНОМАЛИЯМИ - СТАТУС")

user_max_prob = test_df_copy.groupby('user')['prob'].max()
user_anom_count = test_df_copy[test_df_copy['has_anomaly'] == 1].groupby('user').size()
user_has_anom = test_df_copy.groupby('user')['has_anomaly'].max()

bad_users_df = pd.DataFrame({
    'max_prob': user_max_prob,
    'n_anomalies': user_anom_count,
    'is_bad': user_has_anom
})
bad_users_df = bad_users_df[bad_users_df['is_bad'] == 1].sort_values('n_anomalies', ascending=False)

print(f"\n{'user':>12} | {'аномалий':>10} | {'max_prob':>10} | {'пойман?':>8}")
for user, row in bad_users_df.iterrows():
    caught = "ДА" if row['max_prob'] > best_threshold else "нет"
    print(f"{user:>12} | {int(row['n_anomalies']):>10} | {row['max_prob']:>10.4f} | {caught:>8}")

print("\nШЕСТОЙ ЭТАП: СОХРАНЕНИЕ")

results = {
    'model': 'BiLSTM',
    'precision_sessions': test_p,
    'recall_sessions': test_r,
    'f1_sessions': test_f1,
    'auroc': test_auroc,
    'auprc': test_auprc,
    'best_threshold': best_threshold,
}
pd.DataFrame([results]).to_csv("D:/dataset/bilstm_results.csv", index=False)
print(f"Результаты сохранены: D:/dataset/bilstm_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses, label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(val_f1s, label='Val F1', color='green')
axes[1].plot(val_auprcs, label='Val AUPRC', color='blue')
axes[1].axhline(y=best_val_auprc, color='r', linestyle='--', label=f'Best AUPRC: {best_val_auprc:.4f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Metric')
axes[1].set_title('Validation Metrics')
axes[1].legend()
axes[1].grid(True)

axes[2].hist(test_probs[test_labels == 0], bins=50, alpha=0.5, label='Normal', density=True)
axes[2].hist(test_probs[test_labels == 1], bins=50, alpha=0.5, label='Anomaly', density=True)
axes[2].set_xlabel('Probability')
axes[2].set_ylabel('Density')
axes[2].set_title('Distribution of Predictions')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("D:/dataset/bilstm_training.png", dpi=150)
plt.close()
print("Графики сохранены: D:/dataset/bilstm_training.png")