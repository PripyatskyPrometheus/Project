# hmm_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import sys
import os
from save_run import *

log_file = "D:/dataset/log/hmm_model.log"

logger = logging.getLogger('HMM_Model')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class SimpleHMM:
    def __init__(self, trans_matrix=None, window=20, pi=None):
        if trans_matrix is None:
            self.A = np.array([
                [0.95, 0.05],
                [0.05, 0.95]
            ])
        else:
            self.A = np.array(trans_matrix)

        if pi is None:
            self.pi = np.array([0.99, 0.01])
        else:
            self.pi = np.array(pi)

        self.window = window

    def forward_score(self, probs_sequence):
        probs_sequence = np.clip(probs_sequence, 1e-6, 1 - 1e-6)
        n = len(probs_sequence)

        if n == 0:
            return 0.0

        alpha = np.zeros((n, 2))
        alpha[0, 0] = self.pi[0] * (1 - probs_sequence[0])
        alpha[0, 1] = self.pi[1] * probs_sequence[0]
        s = alpha[0].sum()
        if s > 0:
            alpha[0] /= s

        max_score = alpha[0, 1]

        for t in range(1, n):
            for j in range(2):
                emission = (1 - probs_sequence[t]) if j == 0 else probs_sequence[t]
                alpha[t, j] = (alpha[t-1, 0] * self.A[0, j] + alpha[t-1, 1] * self.A[1, j]) * emission
            s = alpha[t].sum()
            if s > 0:
                alpha[t] /= s

            max_score = max(max_score, alpha[t, 1])

        return max_score

    def sequence_to_user_score(self, probs_sequence):
        probs_sequence = np.clip(probs_sequence, 1e-6, 1 - 1e-6)
        n = len(probs_sequence)

        if n == 0:
            return 0

        if n <= self.window:
            return self.forward_score(probs_sequence)

        max_score = 0.0
        step = max(1, self.window // 2)

        for start in range(0, n - self.window + 1, step):
            window_probs = probs_sequence[start:start + self.window]
            score = self.forward_score(window_probs)
            max_score = max(max_score, score)

        last_window = probs_sequence[-self.window:]
        max_score = max(max_score, self.forward_score(last_window))

        return max_score


logger.info("ЗАГРУЗКА ДАННЫХ")

train_df = pd.read_pickle("D:/dataset/train_with_probs.pkl")
val_df = pd.read_pickle("D:/dataset/val_with_probs.pkl")
test_df = pd.read_pickle("D:/dataset/test_with_probs.pkl")

logger.info(f"Train: {len(train_df)} сессий")
logger.info(f"Val: {len(val_df)} сессий")
logger.info(f"Test: {len(test_df)} сессий")


def build_user_sequences(df):
    sequences = {}
    df_sorted = df.sort_values(['user', 'start'])

    for user, group in df_sorted.groupby('user'):
        probs = group['prob'].values.tolist()
        label = int(group['has_anomaly'].max())
        sequences[user] = {
            'probs': probs,
            'label': label,
            'n_sessions': len(group),
        }

    return sequences


train_seqs = build_user_sequences(train_df)
val_seqs = build_user_sequences(val_df)
test_seqs = build_user_sequences(test_df)

logger.info(f"Val: {len(val_seqs)} пользователей, {sum(1 for s in val_seqs.values() if s['label'] == 1)} аномальных")
logger.info(f"Test: {len(test_seqs)} пользователей, {sum(1 for s in test_seqs.values() if s['label'] == 1)} аномальных")

val_labels = np.array([s['label'] for s in val_seqs.values()])
test_labels = np.array([s['label'] for s in test_seqs.values()])

val_probs_list = [val_seqs[u]['probs'] for u in val_seqs.keys()]
test_probs_list = [test_seqs[u]['probs'] for u in test_seqs.keys()]

val_max_prob = np.array([max(p) for p in val_probs_list])
test_max_prob = np.array([max(p) for p in test_probs_list])

logger.info("")
logger.info("ПЕРЕБОР КОНФИГУРАЦИЙ HMM")

configs = [
    ("baseline_0.95_0.05_w20", [[0.95, 0.05], [0.05, 0.95]], 20, None),
    ("w10", [[0.95, 0.05], [0.05, 0.95]], 10, None),
    ("w15", [[0.95, 0.05], [0.05, 0.95]], 15, None),
    ("w30", [[0.95, 0.05], [0.05, 0.95]], 30, None),
    ("w50", [[0.95, 0.05], [0.05, 0.95]], 50, None),
    ("w100", [[0.95, 0.05], [0.05, 0.95]], 100, None),
    ("w20_high_inertia", [[0.95, 0.05], [0.01, 0.99]], 20, None),
    ("w20_low_inertia", [[0.90, 0.10], [0.10, 0.90]], 20, None),
    ("w20_asym", [[0.98, 0.02], [0.05, 0.95]], 20, None),
    ("w20_no_prior", [[0.95, 0.05], [0.05, 0.95]], 20, [0.5, 0.5])
]

results = []

for name, matrix, window, pi in configs:
    logger.info(f"")
    logger.info(f"Конфигурация: {name}")
    logger.info(f"Матрица: {matrix}, окно: {window}, pi: {pi}")

    hmm = SimpleHMM(trans_matrix=matrix, window=window, pi=pi)

    val_hmm = np.array([hmm.sequence_to_user_score(p) for p in val_probs_list])

    best_f1_val = 0
    best_thresh = 0.5
    for th in np.arange(0.1, 1.0, 0.01):
        preds = (val_hmm >= th).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1_val:
            best_f1_val = f1
            best_thresh = th

    test_hmm = np.array([hmm.sequence_to_user_score(p) for p in test_probs_list])

    preds_test = (test_hmm >= best_thresh).astype(int)
    p_test = precision_score(test_labels, preds_test, zero_division=0)
    r_test = recall_score(test_labels, preds_test, zero_division=0)
    f1_test = f1_score(test_labels, preds_test, zero_division=0)
    caught = preds_test[test_labels == 1].sum()

    logger.info(f"Val F1 = {best_f1_val:.4f}, порог = {best_thresh:.2f}")
    logger.info(f"Test: P={p_test:.4f}, R={r_test:.4f}, F1={f1_test:.4f}, Поймано={caught}/{test_labels.sum()}")

    results.append({
        'name': name,
        'matrix': str(matrix),
        'window': window,
        'best_thresh': best_thresh,
        'val_f1': best_f1_val,
        'test_precision': p_test,
        'test_recall': r_test,
        'test_f1': f1_test,
        'caught': caught,
        'total': test_labels.sum(),
    })


logger.info("")
logger.info("СВОДНАЯ ТАБЛИЦА (TEST)")

df_results = pd.DataFrame(results).sort_values('val_f1', ascending=False)

logger.info("")
logger.info(f"{'Конфигурация':>25} {'окно':>5} {'th':>5} {'P':>7} {'R':>7} {'F1':>7} {'Поймано':>8}")

for _, row in df_results.iterrows():
    logger.info(f"{row['name']:>25} {row['window']:>5} {row['best_thresh']:>5.2f} {row['test_precision']:>7.4f} {row['test_recall']:>7.4f} {row['test_f1']:>7.4f} {row['caught']:>4}/{row['total']:<4}")

logger.info("")
logger.info("СРАВНЕНИЕ С BiLSTM max prob, порог = 0.83")

preds_base = (test_max_prob >= 0.83).astype(int)
p_base = precision_score(test_labels, preds_base, zero_division=0)
r_base = recall_score(test_labels, preds_base, zero_division=0)
f1_base = f1_score(test_labels, preds_base, zero_division=0)
caught_base = preds_base[test_labels == 1].sum()

logger.info(f"\nBaseline: P={p_base:.4f}, R={r_base:.4f}, F1={f1_base:.4f}, Поймано={caught_base}/{test_labels.sum()}")

best_row = df_results.iloc[0]
logger.info(f"\nЛучшая конфигурация: {best_row['name']}")
logger.info(f"P={best_row['test_precision']:.4f}, R={best_row['test_recall']:.4f}, F1={best_row['test_f1']:.4f}, Поймано={best_row['caught']}/{best_row['total']}")
logger.info(f"Прирост F1: {best_row['test_f1'] - f1_base:+.4f}")

logger.info("")
logger.info(f"АНАЛИЗ ЛУЧШЕЙ КОНФИГУРАЦИИ: {best_row['name']}")

best_config = None
for name, matrix, window, pi in configs:
    if name == best_row['name']:
        best_config = (matrix, window, pi)
        break

hmm_best = SimpleHMM(trans_matrix=best_config[0], window=best_config[1], pi=best_config[2])
test_hmm_best = np.array([hmm_best.sequence_to_user_score(p) for p in test_probs_list])
preds_best = (test_hmm_best > best_row['best_thresh']).astype(int)

users_test = list(test_seqs.keys())
rows = []
for i, user in enumerate(users_test):
    label = test_seqs[user]['label']
    if label == 0:
        continue
    hmm_score = test_hmm_best[i]
    max_p = max(test_seqs[user]['probs'])
    n_sessions = test_seqs[user]['n_sessions']
    caught_hmm = "ДА" if hmm_score >= best_row['best_thresh'] else "нет"
    caught_base = "ДА" if max_p >= 0.83 else "нет"

    rows.append({
        'user': user,
        'n_sessions': n_sessions,
        'max_prob': max_p,
        'hmm_score': hmm_score,
        'base': caught_base,
        'hmm': caught_hmm,
    })

df_users = pd.DataFrame(rows).sort_values('max_prob', ascending=False)

header = f"{'user':>12} {'сессий':>7} {'max_prob':>9} {'hmm_score':>10} {'base':>5} {'hmm':>5}"
logger.info("")
logger.info(header)
for _, row in df_users.iterrows():
    logger.info(f"{row['user']:>12} {row['n_sessions']:>7} {row['max_prob']:>9.4f} {row['hmm_score']:>10.4f} {row['base']:>5} {row['hmm']:>5}")

logger.info("")
logger.info("ГОТОВО")

run_dir = make_run_dir("hmm")

config = {
    "model": "HMM",
    "n_states": 2,
    "trans_matrix": [[0.95, 0.05], [0.05, 0.95]],
    "window": 20,
    "pi": [0.99, 0.01],
    "best_threshold": float(best_thresh_hmm),
    "selection_metric": "val_f1",
}
save_config(run_dir, config)

data_files = {
    "train_with_probs": "D:/dataset/train_with_probs.pkl",
    "val_with_probs": "D:/dataset/val_with_probs.pkl",
    "test_with_probs": "D:/dataset/test_with_probs.pkl",
    "bilstm_model": "D:/dataset/bilstm_best.pt",
}
code_files = {
    "hmm_model.py": "C:/Users/User/Project/hmm_model.py",
    "BiLSTM.py": "C:/Users/User/Project/BiLSTM.py",
}
save_versions(run_dir, data_files, code_files)

train_users = list(train_seqs.keys())
val_users = list(val_seqs.keys())
test_users = list(test_seqs.keys())
save_split(run_dir, train_users, val_users, test_users)

test_pred = pd.DataFrame({
    'user': list(test_scores.keys()),
    'hmm_score': [test_scores[u]['score'] for u in test_scores.keys()],
    'max_prob': [max(test_seqs[u]['probs']) for u in test_scores.keys()],
    'label': [test_scores[u]['label'] for u in test_scores.keys()],
})
save_predictions(run_dir, "test_hmm_scores", test_pred)

val_pred = pd.DataFrame({
    'user': list(val_scores.keys()),
    'hmm_score': [val_scores[u]['score'] for u in val_scores.keys()],
    'max_prob': [max(val_seqs[u]['probs']) for u in val_scores.keys()],
    'label': [val_scores[u]['label'] for u in val_scores.keys()],
})
save_predictions(run_dir, "val_hmm_scores", val_pred)

metrics = {
    "hmm": {
        "precision": float(p_hmm),
        "recall": float(r_hmm),
        "f1": float(f1_hmm),
        "caught": int(preds_hmm[test_labels == 1].sum()),
        "total": int(test_labels.sum()),
    },
    "baseline_max_prob_083": {
        "precision": float(p_base_083),
        "recall": float(r_base_083),
        "f1": float(f1_base_083),
        "caught": int(preds_base_083[test_labels == 1].sum()),
        "total": int(test_labels.sum()),
    },
}
save_metrics(run_dir, metrics)

threshold_info = {
    "best_threshold_hmm": float(best_thresh_hmm),
    "best_threshold_baseline": float(best_thresh_base),
    "selection_metric": "val_f1",
    "f1_at_best_hmm": float(f1_hmm),
    "f1_at_baseline_083": float(f1_base_083),
}
save_threshold(run_dir, threshold_info)

copy_file(run_dir, log_file)