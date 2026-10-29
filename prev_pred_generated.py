# prev_pred_generated.py
import pandas as pd
import numpy as np
import pickle
import json
import os
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

log_file = "D:/dataset/log/bilstm_prev_pred.log"
logger = logging.getLogger('prev_pred_exp')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(log_file, mode='w', encoding='utf-8'); fh.setFormatter(formatter); logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout); ch.setFormatter(formatter); logger.addHandler(ch)

BASE_MODEL = "D:/dataset/bilstm_best.pt"
TOKEN_PATH = "D:/dataset/token_to_id.pkl"
WEIGHTS_PATH = "D:/dataset/token_weights.npy"
POS_WEIGHT_PATH = "D:/dataset/pos_weight.npy"
TRAIN_PKL = "D:/dataset/train_sequences.pkl"
VAL_PKL = "D:/dataset/val_sequences.pkl"
TEST_PKL = "D:/dataset/test_sequences.pkl"
RUNS_DIR = "D:/dataset/runs/"

BASE_EXTRA = ['full_length_norm']
PREV_EXTRA = ['full_length_norm', 'prev_pred']
MAX_LEN = 250
HIDDEN_DIM = 128
BATCH_SIZE = 128
N_EPOCHS = 20
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_threshold_from_last_run():
    runs = sorted([d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))])
    if not runs:
        return 0.5
    last = os.path.join(RUNS_DIR, runs[-1], "threshold.json")
    if not os.path.exists(last):
        return 0.5
    with open(last, 'r', encoding='utf-8') as f:
        data = json.load(f)
    th = data.get("day_threshold_for_reference", 0.5)
    logger.info(f"Day-level порог из {runs[-1]}: {th:.4f}")
    return float(th)


class InsiderDataset(Dataset):
    def __init__(self, df, extra_cols):
        self.sequences = torch.LongTensor(np.vstack(df['padded'].values))
        self.masks = torch.FloatTensor(np.stack(df['mask'].values))
        self.labels = torch.FloatTensor(df['has_anomaly'].values)
        self.extra = torch.FloatTensor(df[extra_cols].values.astype(np.float32))
    def __len__(self): 
        return len(self.labels)
    def __getitem__(self, i): 
        return self.sequences[i], self.masks[i], self.extra[i], self.labels[i]


class InsiderBiLSTM(nn.Module):
    def __init__(self, vocab_size, token_to_id, token_weights=None, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.4, max_len=250, n_extra=1):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=token_to_id['PAD'])
        if token_weights is not None:
            self.register_buffer('token_weights', torch.FloatTensor(token_weights))
        else:
            self.register_buffer('token_weights', torch.ones(vocab_size))
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2 + 1 + n_extra, 1)

    def forward(self, x, mask, extra):
        embedded = self.embedding(x)
        weights = self.token_weights[x].unsqueeze(-1)
        embedded = embedded * weights
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=x.size(1))
        attn_scores = self.attn(lstm_out).masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_out = (lstm_out * attn_weights).sum(dim=1)
        lengths_norm = lengths.float().unsqueeze(-1) / float(self.max_len)
        combined = torch.cat([attn_out, lengths_norm, extra], dim=1)
        return self.fc(self.dropout(combined)).squeeze(-1)

def predict_probs(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for seq, mask, extra, _ in loader:
            seq, mask, extra = seq.to(DEVICE), mask.to(DEVICE), extra.to(DEVICE)
            probs.extend(torch.sigmoid(model(seq, mask, extra)).cpu().numpy())
    return np.array(probs)

def add_prev_pred(df, probs, threshold):
    df = df.copy().reset_index(drop=True)
    df['prob'] = probs
    df = df.sort_values(['user', 'day']).reset_index(drop=True)
    df['prev_pred'] = 0
    for _, idx in df.groupby('user').groups.items():
        idx = list(idx)
        for i in range(1, len(idx)):
            df.loc[idx[i], 'prev_pred'] = int(df.loc[idx[i-1], 'prob'] >= threshold)
    return df

def train_epoch(model, loader, criterion, optimizer):
    model.train(); total_loss = 0; n = 0
    for seq, mask, extra, labels in loader:
        seq, mask, extra, labels = seq.to(DEVICE), mask.to(DEVICE), extra.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(seq, mask, extra)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / n

def eval_epoch(model, loader, criterion):
    model.eval(); total_loss = 0; n = 0; probs = []; labels_all = []
    with torch.no_grad():
        for seq, mask, extra, labels in loader:
            seq, mask, extra, labels = seq.to(DEVICE), mask.to(DEVICE), extra.to(DEVICE), labels.to(DEVICE)
            out = model(seq, mask, extra)
            loss = criterion(out, labels)
            probs.extend(torch.sigmoid(out).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            total_loss += loss.item(); n += 1
    return total_loss / n, np.array(probs), np.array(labels_all)

def calculate_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return p, r, f1, roc_auc_score(labels, probs), average_precision_score(labels, probs)

def add_rolling(df, window=3):
    df = df.sort_values(['user', 'day']).copy()
    df[f'roll{window}_mean'] = df.groupby('user')['prob'].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return df

def user_level_probs(df, prob_col):
    g = df.groupby('user')
    return g[prob_col].max(), g['has_anomaly'].max().astype(int)


def select_f1_threshold(labels, scores, plateau_delta=0.02):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom != 0)
    best_idx = int(np.argmax(f1)); best_th = float(thresholds[best_idx]); best_f1 = float(f1[best_idx])
    plateau_mask = f1 >= (best_f1 - plateau_delta)
    plateau_th = thresholds[plateau_mask]
    plateau = (float(plateau_th.min()), float(plateau_th.max())) if plateau_th.size else (best_th, best_th)
    order = np.argsort(-f1); top5, seen = [], set()
    for i in order:
        t = float(thresholds[i])
        if t in seen: continue
        seen.add(t); top5.append((t, float(f1[i])))
        if len(top5) == 5: break
    return {'best_threshold': best_th, 'best_f1': best_f1, 'plateau': plateau, 'plateau_width': int(plateau_th.size), 'top5': top5}

def select_day_threshold(val_labels, val_probs):
    precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1s[:-1])])


def report_user_rule(name, vp, vl, tp, tl, logger):
    info = select_f1_threshold(vl, vp); th = info['best_threshold']
    preds = (tp >= th).astype(int)
    t_p = precision_score(tl, preds, zero_division=0); t_r = recall_score(tl, preds, zero_division=0)
    t_f1 = f1_score(tl, preds, zero_division=0)
    caught = int(preds[tl == 1].sum()); total = int(tl.sum())
    logger.info(f"\n[{name}]")
    logger.info(f"  Лучший порог (val): {th:.4f} (val F1 users = {info['best_f1']:.4f})")
    logger.info(f"  Плато: {info['plateau'][0]:.4f}–{info['plateau'][1]:.4f} (ширина {info['plateau_width']})")
    for t, f in info['top5']: logger.info(f"    {t:.4f}: {f:.4f}")
    logger.info(f"  TEST {th:.4f}: F1 users = {t_f1:.4f}, P = {t_p:.4f}, R = {t_r:.4f}, caught = {caught}/{total}")
    return {'rule': name, 'best_threshold': float(th), 'val_f1_users': float(info['best_f1']),
            'plateau_lo': float(info['plateau'][0]), 'plateau_hi': float(info['plateau'][1]),
            'plateau_width': int(info['plateau_width']), 'test_f1_users': float(t_f1),
            'test_precision': float(t_p), 'test_recall': float(t_r), 'caught': f"{caught}/{total}", 'top5': info['top5']}


def main():
    logger.info("ЗАГРУЗКА")
    train_df = pd.read_pickle(TRAIN_PKL); val_df = pd.read_pickle(VAL_PKL); test_df = pd.read_pickle(TEST_PKL)
    with open(TOKEN_PATH, 'rb') as f: token_to_id = pickle.load(f)
    token_weights = np.load(WEIGHTS_PATH); pos_weight = np.load(POS_WEIGHT_PATH)[0]
    vocab_size = len(token_to_id)

    logger.info("ГЕНЕРАЦИЯ prev_pred ПО БАЗЕ")
    base_model = InsiderBiLSTM(vocab_size, token_to_id, token_weights=token_weights, hidden_dim=HIDDEN_DIM, max_len=MAX_LEN, n_extra=1).to(DEVICE)
    base_model.load_state_dict(torch.load(BASE_MODEL, map_location=DEVICE))
    base_model.eval()
    threshold = load_threshold_from_last_run()

    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        ds = InsiderDataset(df, BASE_EXTRA)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        probs = predict_probs(base_model, loader)
        df2 = add_prev_pred(df, probs, threshold)
        if name == 'train': train_df = df2
        elif name == 'val': val_df = df2
        else: test_df = df2
        logger.info(f"{name}: prev_pred=1 {int(df2['prev_pred'].sum())}/{len(df2)}")

    logger.info("ОБУЧЕНИЕ С prev_pred")
    train_ds = InsiderDataset(train_df, PREV_EXTRA); val_ds = InsiderDataset(val_df, PREV_EXTRA); test_ds = InsiderDataset(test_df, PREV_EXTRA)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = InsiderBiLSTM(vocab_size, token_to_id, token_weights=token_weights, hidden_dim=HIDDEN_DIM, max_len=MAX_LEN, n_extra=2).to(DEVICE)
    pos_w = torch.FloatTensor([np.sqrt(pos_weight)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_auprc = 0; best_epoch = 0; pat = 0; best_path = "D:/dataset/bilstm_prev_pred_best.pt"
    start = time.time()
    for ep in range(N_EPOCHS):
        tl = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vp, vl = eval_epoch(model, val_loader, criterion)
        _, _, v_f1, v_auroc, v_auprc = calculate_metrics(vl, vp)
        logger.info(f"Epoch {ep+1:2d}/{N_EPOCHS} | train={tl:.4f} val={vl_loss:.4f} F1={v_f1:.4f} AUPRC={v_auprc:.4f}")
        if v_auprc > best_auprc:
            best_auprc = v_auprc; best_epoch = ep + 1; pat = 0
            torch.save(model.state_dict(), best_path)
        else:
            pat += 1
            if pat >= PATIENCE:
                logger.info(f"Early stop на {ep+1}"); break
    logger.info(f"Лучшая: эпоха {best_epoch}, AUPRC={best_auprc:.4f}, время { (time.time()-start)/60:.1f} мин")

    logger.info("ОЦЕНКА")
    model.load_state_dict(torch.load(best_path))
    model.eval()
    _, tp_raw, tl_raw = eval_epoch(model, test_loader, criterion)
    test_p, test_r, test_f1, test_auroc, test_auprc = calculate_metrics(tl_raw, tp_raw)

    val_df_eval = val_df.copy(); val_df_eval['prob'] = vp
    test_df_eval = test_df.copy(); test_df_eval['prob'] = tp_raw

    vp_u, vl_u = user_level_probs(val_df_eval, 'prob'); tp_u, tl_u = user_level_probs(test_df_eval, 'prob')
    rule_max = report_user_rule("max(prob)", vp_u, vl_u, tp_u, tl_u, logger)

    val_df_eval = add_rolling(val_df_eval); test_df_eval = add_rolling(test_df_eval)
    rules = {'max': rule_max}
    for alpha, tag in [(0.4, 'hybrid_04'), (0.5, 'hybrid_05')]:
        v = val_df_eval.copy(); t = test_df_eval.copy()
        v['hybrid'] = v['prob'] + alpha * v['roll3_mean']; t['hybrid'] = t['prob'] + alpha * t['roll3_mean']
        vph, vlh = user_level_probs(v, 'hybrid'); tph, tlh = user_level_probs(t, 'hybrid')
        rules[tag] = report_user_rule(f"{tag}", vph, vlh, tph, tlh, logger)
        if tag == 'hybrid_04': test_df_eval['hybrid_04'] = t['hybrid'].values
        if tag == 'hybrid_05': test_df_eval['hybrid_05'] = t['hybrid'].values

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_df_eval['prob'].values, val_df_eval['has_anomaly'].values)
    val_df_eval['prob_iso'] = iso.predict(val_df_eval['prob'].values)
    test_df_eval['prob_iso'] = iso.predict(test_df_eval['prob'].values)
    vp_i, vl_i = user_level_probs(val_df_eval, 'prob_iso'); tp_i, tl_i = user_level_probs(test_df_eval, 'prob_iso')
    rules['isotonic'] = report_user_rule("isotonic", vp_i, vl_i, tp_i, tl_i, logger)

    best_rule_name = max(rules, key=lambda k: rules[k]['val_f1_users'])
    best = rules[best_rule_name]
    logger.info(f"Лучшее правило: {best_rule_name} (val={best['val_f1_users']:.4f}, test={best['test_f1_users']:.4f})")

    run_dir = make_run_dir("bilstm_prev_pred")
    config = {
        "experiment": "prev_pred", "hidden_dim": HIDDEN_DIM, "n_extra": 2, "extra_cols": PREV_EXTRA,
        "base_model": BASE_MODEL, "prev_pred_threshold": threshold, "best_rule": best_rule_name, "random_seed": 42,
    }
    save_config(run_dir, config)
    save_split(run_dir, train_df['user'].unique(), val_df['user'].unique(), test_df['user'].unique())
    save_metrics(run_dir, {
        "days": {"precision": float(test_p), "recall": float(test_r), "f1": float(test_f1),
                 "auroc": float(test_auroc), "auprc": float(test_auprc)},
        "users": {k: {"best_threshold": v['best_threshold'], "val_f1": v['val_f1_users'],
                      "test_f1": v['test_f1_users'], "caught": v['caught']} for k, v in rules.items()},
        "best_rule": best_rule_name, "best_epoch": best_epoch, "best_val_auprc": float(best_auprc),
        "training_time_min": float((time.time()-start)/60),
    })
    save_predictions(run_dir, "val_probs", val_df_eval[['user', 'day', 'prob', 'has_anomaly']])
    save_predictions(run_dir, "test_probs", test_df_eval[['user', 'day', 'prob', 'has_anomaly']])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(tp_raw[tl_raw == 0], bins=50, alpha=0.5, label='Normal', density=True)
    axes[0].hist(tp_raw[tl_raw == 1], bins=50, alpha=0.5, label='Anomaly', density=True)
    axes[0].set_title('Distribution (prev_pred)'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot([r['val_f1_users'] for r in rules.values()], label='val F1')
    axes[1].plot([r['test_f1_users'] for r in rules.values()], label='test F1')
    axes[1].set_xticks(range(len(rules))); axes[1].set_xticklabels(list(rules.keys()), rotation=15)
    axes[1].set_title('Rules: val vs test F1'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.savefig("D:/dataset/bilstm_prev_pred_training.png", dpi=150); plt.close()

    logger.info(f"Прогон сохранён в {run_dir}")


if __name__ == "__main__":
    main()