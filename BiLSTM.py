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
from sklearn.isotonic import IsotonicRegression
import time
import matplotlib.pyplot as plt
import logging
import sys
from save_run import *

log_file = "D:/dataset/log/bilstm.log"

logger = logging.getLogger('BiLSTM')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Дополнительные признаки дня (кроме последовательности токенов)
EXTRA_COLS = ['full_length_norm']
N_EXTRA = len(EXTRA_COLS)


class InsiderDataset(Dataset):
    def __init__(self, df):
        self.sequences = torch.LongTensor(np.vstack(df['padded'].values))
        self.masks = torch.FloatTensor(np.stack(df['mask'].values))
        self.labels = torch.FloatTensor(df['has_anomaly'].values)
        self.extra = torch.FloatTensor(df[EXTRA_COLS].values.astype(np.float32))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx], self.extra[idx], self.labels[idx]


class InsiderBiLSTM(nn.Module):
    def __init__(self, vocab_size, token_to_id, token_weights=None, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.4,
                 max_len=250, n_extra=1):
        super(InsiderBiLSTM, self).__init__()

        self.max_len = max_len
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
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        # вход классификатора: attention-вектор + нормированная длина + доп. признаки
        self.fc = nn.Linear(hidden_dim * 2 + 1 + n_extra, 1)

    def forward(self, x, mask, extra):
        embedded = self.embedding(x)
        weights = self.token_weights[x].unsqueeze(-1)
        embedded = embedded * weights

        lengths = mask.sum(dim=1).long().clamp(min=1)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        # attention по реальным токенам (PAD замаскирован)
        attn_scores = self.attn(lstm_out)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_out = (lstm_out * attn_weights).sum(dim=1)

        lengths_norm = lengths.float().unsqueeze(-1) / float(self.max_len)
        combined = torch.cat([attn_out, lengths_norm, extra], dim=1)

        combined = self.dropout(combined)
        out = self.fc(combined)
        return out.squeeze(-1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for sequences, masks, extra, labels in loader:
        sequences = sequences.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        extra = extra.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(sequences, masks, extra)
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
        for sequences, masks, extra, labels in loader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            extra = extra.to(device)
            labels = labels.to(device)

            outputs = model(sequences, masks, extra)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss, np.array(all_probs), np.array(all_labels)


def calculate_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return p, r, f1, auroc, auprc


def add_rolling(df, window=3):
    """Скользящее среднее вероятности по дням внутри пользователя (дни по возрастанию)."""
    df = df.sort_values(['user', 'day']).copy()
    df[f'roll{window}_mean'] = (
        df.groupby('user')['prob']
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df


def user_level_probs(df, prob_col):
    g = df.groupby('user')
    return g[prob_col].max(), g['has_anomaly'].max()


def select_f1_threshold(labels, scores, plateau_delta=0.01):
    """Возвращает лучший порог, F1, диапозон, топ-5 (по уникальным порогам)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {'best_threshold': 0.5, 'best_f1': 0.0, 'plateau': (0.5, 0.5), 'plateau_width': 0, 'top5': []}

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )

    best_idx = int(np.argmax(f1))
    best_th = float(thresholds[best_idx])
    best_f1 = float(f1[best_idx])

    plateau_mask = f1 >= (best_f1 - plateau_delta)
    plateau_th = thresholds[plateau_mask]
    plateau = (float(plateau_th.min()), float(plateau_th.max())) if plateau_th.size else (best_th, best_th)

    order = np.argsort(-f1)
    top5, seen = [], set()
    for i in order:
        t = float(thresholds[i])
        if t in seen:
            continue
        seen.add(t)
        top5.append((t, float(f1[i])))
        if len(top5) == 5:
            break

    return {'best_threshold': best_th, 'best_f1': best_f1, 'plateau': plateau, 'plateau_width': int(plateau_th.size), 'top5': top5,}

def select_day_threshold(val_labels, val_probs):
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx])

def report_user_rule(name, val_probs_series, val_labels_series, test_probs_series, test_labels_series, logger):
    """Подбор порога на val, отчёт топ-5/диапозон и метрик на test."""
    info = select_f1_threshold(val_labels_series, val_probs_series)
    th = info['best_threshold']

    preds = (test_probs_series >= th).astype(int)
    t_p = precision_score(test_labels_series, preds, zero_division=0)
    t_r = recall_score(test_labels_series, preds, zero_division=0)
    t_f1 = f1_score(test_labels_series, preds, zero_division=0)
    caught = int(preds[test_labels_series == 1].sum())
    total = int(test_labels_series.sum())

    logger.info(f"\n[{name}]")
    logger.info(f"  Лучший порог (val): {th:.4f} (val F1 users = {info['best_f1']:.4f})")
    logger.info(f"  Плато: {info['plateau'][0]:.4f}–{info['plateau'][1]:.4f} (ширина {info['plateau_width']})")
    logger.info(f"  Топ-5 порогов на val:")
    for t, f in info['top5']:
        logger.info(f"    {t:.4f}: {f:.4f}")
    logger.info(f"  TEST @ {th:.4f}: F1 users = {t_f1:.4f}, P = {t_p:.4f}, R = {t_r:.4f}, caught = {caught}/{total}")

    return {
        'rule': name,
        'best_threshold': float(th),
        'val_f1_users': float(info['best_f1']),
        'plateau_lo': float(info['plateau'][0]),
        'plateau_hi': float(info['plateau'][1]),
        'plateau_width': int(info['plateau_width']),
        'test_f1_users': float(t_f1),
        'test_precision': float(t_p),
        'test_recall': float(t_r),
        'caught': f"{caught}/{total}",
        'top5': info['top5'],
    }


logger.info("ПЕРВЫЙ ЭТАП: ПРОВЕРКА УСТРОЙСТВА И ЗАГРУЗКА ДАННЫХ")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Используется устройство: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

torch.backends.cudnn.benchmark = True
logger.info(f"PyTorch: {torch.__version__}")

train_df = pd.read_pickle("D:/dataset/train_sequences.pkl")
val_df = pd.read_pickle("D:/dataset/val_sequences.pkl")
test_df = pd.read_pickle("D:/dataset/test_sequences.pkl")

with open("D:/dataset/token_to_id.pkl", 'rb') as f:
    token_to_id = pickle.load(f)

token_weights = np.load("D:/dataset/token_weights.npy")
pos_weight = np.load("D:/dataset/pos_weight.npy")[0]
pos_weight_sqrt = np.sqrt(pos_weight)

logger.info(f"\nTrain: {len(train_df)} (user, day) ({train_df['has_anomaly'].sum()} аномалий)")
logger.info(f"Val: {len(val_df)} (user, day) ({val_df['has_anomaly'].sum()} аномалий)")
logger.info(f"Test: {len(test_df)} (user, day) ({test_df['has_anomaly'].sum()} аномалий)")
logger.info(f"Размер словаря: {len(token_to_id)}")
logger.info(f"pos_weight: {pos_weight:.1f} -> sqrt(pos_weight): {pos_weight_sqrt:.1f}")
logger.info(f"Доп. признаки ({N_EXTRA}): {EXTRA_COLS}")
for c in EXTRA_COLS:
    logger.info(f"  {c}: mean={train_df[c].mean():.3f}, std={train_df[c].std():.3f}")

logger.info("\nВТОРОЙ ЭТАП: СОЗДАНИЕ DATASET И DATALOADER")

train_dataset = InsiderDataset(train_df)
val_dataset = InsiderDataset(val_df)
test_dataset = InsiderDataset(test_df)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

logger.info(f"\nTrain-батчи: {len(train_loader)}")
logger.info(f"Val-батчи: {len(val_loader)}")
logger.info(f"Test-батчи: {len(test_loader)}")

sample_seq, sample_mask, sample_extra, sample_label = next(iter(train_loader))
logger.info(f"\nПример батча:")
logger.info(f"Sequence shape: {sample_seq.shape}")
logger.info(f"Mask shape: {sample_mask.shape}")
logger.info(f"Extra shape: {sample_extra.shape}")
logger.info(f"Label shape: {sample_label.shape}")

logger.info("\nТРЕТИЙ ЭТАП: СОЗДАНИЕ МОДЕЛИ BiLSTM")

vocab_size = len(token_to_id)
MAX_LEN = 250
HIDDEN_DIM = 128

model = InsiderBiLSTM(
    vocab_size=vocab_size,
    token_to_id=token_to_id,
    token_weights=token_weights,
    embedding_dim=64,
    hidden_dim=HIDDEN_DIM,
    num_layers=2,
    dropout=0.4,
    max_len=MAX_LEN,
    n_extra=N_EXTRA,
).to(device)

logger.info(f"Модель на устройстве: {next(model.parameters()).device}")
logger.info(f"Всего памяти GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"\nВсего параметров: {total_params}")
logger.info(f"Обучаемых параметров: {trainable_params}")
logger.info(f"hidden_dim={HIDDEN_DIM}, n_extra={N_EXTRA}")

logger.info("\nАрхитектура модели:")
logger.info(model)

pos_weight_tensor = torch.FloatTensor([pos_weight_sqrt]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

logger.info(f"\npos_weight={pos_weight_sqrt:.1f}")

logger.info("\nЧЕТВЕРТЫЙ ЭТАП: ОБУЧЕНИЕ")
N_EPOCHS = 20
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

    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

    val_loss, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device)
    val_p, val_r, val_f1, val_auroc, val_auprc = calculate_metrics(val_labels, val_probs)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1s.append(val_f1)
    val_auprcs.append(val_auprc)
    torch.cuda.empty_cache()

    epoch_time = time.time() - epoch_start

    logger.info(f"\nEpoch {epoch+1:2d}/{N_EPOCHS}")
    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Val Loss: {val_loss:.4f}")
    logger.info(f"Val F1: {val_f1:.4f}")
    logger.info(f"Val AUPRC: {val_auprc:.4f}")
    logger.info(f"Время: {epoch_time:.1f} секунд")
    logger.info(f"Максимальная загрузка памяти GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if val_auprc > best_val_auprc:
        best_val_auprc = val_auprc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "D:/dataset/bilstm_best.pt")
        logger.info(f"Новая лучшая модель (AUPRC={val_auprc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            logger.info(f"\nEarly stopping на эпохе {epoch+1}")
            break

total_time = time.time() - start_time
logger.info(f"\nОбщее время обучения: {total_time/60:.1f} минут")
logger.info(f"Лучшая эпоха: {best_epoch} (Val AUPRC: {best_val_auprc:.4f})")

logger.info("\nПЯТЫЙ ЭТАП: ОЦЕНКА")

model_eval = InsiderBiLSTM(
    vocab_size=vocab_size,
    token_to_id=token_to_id,
    token_weights=token_weights,
    embedding_dim=64,
    hidden_dim=HIDDEN_DIM,
    num_layers=2,
    dropout=0.4,
    max_len=MAX_LEN,
    n_extra=N_EXTRA,
).to(device)

model_eval.load_state_dict(torch.load("D:/dataset/bilstm_best.pt"))
model_eval.eval()

test_loss, test_probs, test_labels = eval_epoch(model_eval, test_loader, criterion, device)
test_p, test_r, test_f1, test_auroc, test_auprc = calculate_metrics(test_labels, test_probs)

logger.info(f"\nРЕЗУЛЬТАТЫ НА ТЕСТЕ (порог 0.5, дни):")
logger.info(f"Precision: {test_p:.4f}")
logger.info(f"Recall: {test_r:.4f}")
logger.info(f"F1: {test_f1:.4f}")
logger.info(f"AUROC: {test_auroc:.4f}")
logger.info(f"AUPRC: {test_auprc:.4f}")

logger.info("\nПОДБОР ПОРОГА НА VAL (user-level, PR-кривая, полный диапазон)")

val_loss, val_probs, val_labels = eval_epoch(model_eval, val_loader, criterion, device)

val_df_eval = val_df.copy()
val_df_eval['prob'] = val_probs
test_df_eval = test_df.copy()
test_df_eval['prob'] = test_probs

# max(prob)
val_user_probs, val_user_labels = user_level_probs(val_df_eval, 'prob')
test_user_probs, test_user_labels = user_level_probs(test_df_eval, 'prob')
rule_max = report_user_rule("max(prob)", val_user_probs, val_user_labels, test_user_probs, test_user_labels, logger)

# hybrid: пик + среднее за 3 дня (два @: оптимум F1 и recall-режим)
val_df_eval = add_rolling(val_df_eval, window=3)
test_df_eval = add_rolling(test_df_eval, window=3)

rule_hybrid_04 = None
rule_hybrid_05 = None
for alpha, tag in [(0.4, '@=0.4'), (0.5, '@=0.5')]:
    v = val_df_eval.copy()
    t = test_df_eval.copy()
    v['hybrid'] = v['prob'] + alpha * v['roll3_mean']
    t['hybrid'] = t['prob'] + alpha * t['roll3_mean']
    vp, vl = user_level_probs(v, 'hybrid')
    tp, tl = user_level_probs(t, 'hybrid')
    r = report_user_rule(f"hybrid {tag} (max + {alpha}*roll3_mean)", vp, vl, tp, tl, logger)
    if alpha == 0.4:
        rule_hybrid_04 = r
    else:
        rule_hybrid_05 = r

# isotonic-калибровка (precision-режим)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_df_eval['prob'].values, val_df_eval['has_anomaly'].values)
val_df_eval['prob_iso'] = iso.predict(val_df_eval['prob'].values)
test_df_eval['prob_iso'] = iso.predict(test_df_eval['prob'].values)
vp, vl = user_level_probs(val_df_eval, 'prob_iso')
tp, tl = user_level_probs(test_df_eval, 'prob_iso')
rule_iso = report_user_rule("isotonic(prob)", vp, vl, tp, tl, logger)

day_threshold = select_day_threshold(val_labels, val_probs)
logger.info(f"\nDay-level порог (для справки): {day_threshold:.4f}")

rules = {
    'max':       rule_max,
    'hybrid_04': rule_hybrid_04,
    'hybrid_05': rule_hybrid_05,
    'isotonic':  rule_iso,
}
best_rule_name = max(rules, key=lambda k: rules[k]['val_f1_users'])
best_rule = rules[best_rule_name]
best_threshold = best_rule['best_threshold']

logger.info(f"\nЛучшее правило по val F1: {best_rule_name} "
            f"(val={best_rule['val_f1_users']:.4f}, "
            f"test={best_rule['test_f1_users']:.4f})")

np.save("D:/dataset/best_threshold.npy", np.array([best_threshold]))
logger.info(f"best_threshold.npy (rule={best_rule_name}): {best_threshold:.4f}")

# Анализ ошибок — по лучшему правилу
score_col_map = {
    'max': 'prob',
    'hybrid_04': 'hybrid_04',
    'hybrid_05': 'hybrid_05',
    'isotonic': 'prob_iso',
}
if best_rule_name == 'hybrid_04':
    test_df_eval['hybrid_04'] = test_df_eval['prob'] + 0.4 * test_df_eval['roll3_mean']
elif best_rule_name == 'hybrid_05':
    test_df_eval['hybrid_05'] = test_df_eval['prob'] + 0.5 * test_df_eval['roll3_mean']

score_col = score_col_map[best_rule_name]
test_df_eval['best_score'] = test_df_eval[score_col]
test_df_eval['pred'] = (test_df_eval['best_score'] > best_threshold).astype(int)

logger.info(f"\nАНАЛИЗ ОШИБОК (rule={best_rule_name}, threshold={best_threshold:.4f})")

fn = test_df_eval[(test_df_eval['has_anomaly'] == 1) & (test_df_eval['pred'] == 0)]
fp = test_df_eval[(test_df_eval['has_anomaly'] == 0) & (test_df_eval['pred'] == 1)]
tp_ = test_df_eval[(test_df_eval['has_anomaly'] == 1) & (test_df_eval['pred'] == 1)]

logger.info(f"\nПропущено аномалий: {len(fn)} из {test_df_eval['has_anomaly'].sum()}")
logger.info(f"Ложных тревог: {len(fp)}")
logger.info(f"Поймано аномалий: {len(tp_)}")

logger.info(f"\nПримеры ПРОПУЩЕННЫХ аномалий:")
for _, row in fn.head(5).iterrows():
    logger.info(f"user={row['user']}, day={row['day']}, len={row['sequence_length']}, prob={row['prob']:.4f}, score={row['best_score']:.4f}")

logger.info(f"\nПримеры ПОЙМАННЫХ аномалий (первые 5):")
for _, row in tp_.head(5).iterrows():
    logger.info(f"user={row['user']}, day={row['day']}, len={row['sequence_length']}, prob={row['prob']:.4f}, score={row['best_score']:.4f}")

logger.info(f"\nПОЛЬЗОВАТЕЛИ С АНОМАЛИЯМИ - СТАТУС (rule={best_rule_name})")

user_best_max = test_df_eval.groupby('user')['best_score'].max()
user_anom_count = test_df_eval[test_df_eval['has_anomaly'] == 1].groupby('user').size()
user_has_anom = test_df_eval.groupby('user')['has_anomaly'].max()

bad_users_df = pd.DataFrame({
    'best_score_max': user_best_max,
    'n_anomalies': user_anom_count,
    'is_bad': user_has_anom
})
bad_users_df = bad_users_df[bad_users_df['is_bad'] == 1].sort_values('n_anomalies', ascending=False)

logger.info(f"\n{'user':>12} {'аномалий':>10} {'best_score':>12} {'пойман?':>8}")
for user, row in bad_users_df.iterrows():
    caught = "ДА" if row['best_score_max'] >= best_threshold else "нет"
    logger.info(f"{user:>12} {int(row['n_anomalies']):>10} {row['best_score_max']:>12.4f} {caught:>8}")

logger.info("\nШЕСТОЙ ЭТАП: СОХРАНЕНИЕ")

results = {
    'model': 'BiLSTM+full_length_norm',
    'hidden_dim': HIDDEN_DIM,
    'n_extra': N_EXTRA,
    'best_rule': best_rule_name,
    'auroc': test_auroc,
    'auprc': test_auprc,
    'precision_at_05_days': test_p,
    'recall_at_05_days': test_r,
    'f1_at_05_days': test_f1,
    'day_threshold_for_reference': day_threshold,
}
for r in [rule_max, rule_hybrid_04, rule_hybrid_05, rule_iso]:
    key = (r['rule'].replace(' ', '_').replace('(', '').replace(')', '').replace('*', 'x').replace('+', 'plus').replace('.', '_')
        .replace('@', 'a').replace('=', ''))
    results[f'{key}_threshold'] = r['best_threshold']
    results[f'{key}_val_f1'] = r['val_f1_users']
    results[f'{key}_test_f1'] = r['test_f1_users']
    results[f'{key}_caught'] = r['caught']
pd.DataFrame([results]).to_csv("D:/dataset/bilstm_results.csv", index=False)
logger.info(f"Результаты сохранены: D:/dataset/bilstm_results.csv")

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
logger.info("Графики сохранены: D:/dataset/bilstm_training.png")

logger.info('')
logger.info('Сохраняем версию прогона')
run_dir = make_run_dir("bilstm")

config = {
    "model": "BiLSTM+full_length_norm",
    "embedding_dim": 64,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": 2,
    "dropout": 0.4,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "pos_weight": float(pos_weight),
    "pos_weight_sqrt": float(np.sqrt(pos_weight)),
    "max_len": MAX_LEN,
    "n_epochs": N_EPOCHS,
    "patience": PATIENCE,
    "threshold_selection": "pr_curve_f1_users_full_range",
    "plateau_delta": 0.01,
    "hybrid_alpha_main": 0.4,
    "hybrid_alpha_recall": 0.5,
    "hybrid_window": 3,
    "extra_cols": EXTRA_COLS,
    "best_rule": best_rule_name,
    "random_seed": 42,
}
save_config(run_dir, config)

data_files = {
    "train_sequences": "D:/dataset/train_sequences.pkl",
    "val_sequences": "D:/dataset/val_sequences.pkl",
    "test_sequences": "D:/dataset/test_sequences.pkl",
    "token_to_id": "D:/dataset/token_to_id.pkl",
    "token_weights": "D:/dataset/token_weights.npy",
    "pos_weight": "D:/dataset/pos_weight.npy",
}
code_files = {
    "BiLSTM.py": "C:/Users/User/Project/BiLSTM.py",
    "data_encoding.py": "C:/Users/User/Project/data_encoding.py",
}
save_versions(run_dir, data_files, code_files)

train_users = train_df['user'].unique()
val_users = val_df['user'].unique()
test_users = test_df['user'].unique()
save_split(run_dir, train_users, val_users, test_users)

val_pred = val_df_eval[['user', 'day', 'prob', 'has_anomaly']].copy()
save_predictions(run_dir, "val_probs", val_pred)

test_pred = test_df_eval[['user', 'day', 'prob', 'has_anomaly']].copy()
save_predictions(run_dir, "test_probs", test_pred)

metrics = {
    "days": {
        "precision": float(test_p),
        "recall": float(test_r),
        "f1": float(test_f1),
        "auroc": float(test_auroc),
        "auprc": float(test_auprc),
    },
    "users_max": {
        "best_threshold": float(rule_max['best_threshold']),
        "val_f1_users": float(rule_max['val_f1_users']),
        "test_f1_users": float(rule_max['test_f1_users']),
        "caught": rule_max['caught'],
        "plateau": f"{rule_max['plateau_lo']:.4f}-{rule_max['plateau_hi']:.4f}",
    },
    "users_hybrid_04": {
        "best_threshold": float(rule_hybrid_04['best_threshold']),
        "val_f1_users": float(rule_hybrid_04['val_f1_users']),
        "test_f1_users": float(rule_hybrid_04['test_f1_users']),
        "caught": rule_hybrid_04['caught'],
        "plateau": f"{rule_hybrid_04['plateau_lo']:.4f}-{rule_hybrid_04['plateau_hi']:.4f}",
    },
    "users_hybrid_05": {
        "best_threshold": float(rule_hybrid_05['best_threshold']),
        "val_f1_users": float(rule_hybrid_05['val_f1_users']),
        "test_f1_users": float(rule_hybrid_05['test_f1_users']),
        "caught": rule_hybrid_05['caught'],
        "plateau": f"{rule_hybrid_05['plateau_lo']:.4f}-{rule_hybrid_05['plateau_hi']:.4f}",
    },
    "users_isotonic": {
        "best_threshold": float(rule_iso['best_threshold']),
        "val_f1_users": float(rule_iso['val_f1_users']),
        "test_f1_users": float(rule_iso['test_f1_users']),
        "caught": rule_iso['caught'],
        "plateau": f"{rule_iso['plateau_lo']:.4f}-{rule_iso['plateau_hi']:.4f}",
    },
    "best_rule": best_rule_name,
    "best_epoch": best_epoch,
    "best_val_auprc": float(best_val_auprc),
    "training_time_min": float(total_time / 60),
}
save_metrics(run_dir, metrics)

threshold_info = {
    "selection_metric": "pr_curve_f1_users_full_range",
    "day_threshold_for_reference": float(day_threshold),
    "best_rule": best_rule_name,
    "rules": {
        "max": {
            "best_threshold": float(rule_max['best_threshold']),
            "val_f1_users": float(rule_max['val_f1_users']),
            "test_f1_users": float(rule_max['test_f1_users']),
            "plateau": f"{rule_max['plateau_lo']:.4f}-{rule_max['plateau_hi']:.4f}",
            "top5_val": rule_max['top5'],
        },
        "hybrid_04": {
            "alpha": 0.4,
            "window": 3,
            "best_threshold": float(rule_hybrid_04['best_threshold']),
            "val_f1_users": float(rule_hybrid_04['val_f1_users']),
            "test_f1_users": float(rule_hybrid_04['test_f1_users']),
            "plateau": f"{rule_hybrid_04['plateau_lo']:.4f}-{rule_hybrid_04['plateau_hi']:.4f}",
            "top5_val": rule_hybrid_04['top5'],
        },
        "hybrid_05": {
            "alpha": 0.5,
            "window": 3,
            "best_threshold": float(rule_hybrid_05['best_threshold']),
            "val_f1_users": float(rule_hybrid_05['val_f1_users']),
            "test_f1_users": float(rule_hybrid_05['test_f1_users']),
            "plateau": f"{rule_hybrid_05['plateau_lo']:.4f}-{rule_hybrid_05['plateau_hi']:.4f}",
            "top5_val": rule_hybrid_05['top5'],
        },
        "isotonic": {
            "best_threshold": float(rule_iso['best_threshold']),
            "val_f1_users": float(rule_iso['val_f1_users']),
            "test_f1_users": float(rule_iso['test_f1_users']),
            "plateau": f"{rule_iso['plateau_lo']:.4f}-{rule_iso['plateau_hi']:.4f}",
            "top5_val": rule_iso['top5'],
        },
    },
}
save_threshold(run_dir, threshold_info)

copy_file(run_dir, "D:/dataset/bilstm_best.pt")
copy_file(run_dir, log_file)
copy_file(run_dir, "D:/dataset/bilstm_training.png", "plots")