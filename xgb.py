# xgb.py
import os, sys, time, logging
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import *
from save_run import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("C:/Users/user/dataset/log/xgb.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('XGBoost')

BASE_DIR = "C:/Users/user/dataset/"

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
        df.groupby('user')['prob'].transform(lambda s: s.rolling(window, min_periods=1).mean()))
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
    logger.info(f"  TEST {th:.4f}: F1 users = {t_f1:.4f}, P = {t_p:.4f}, R = {t_r:.4f}, caught = {caught}/{total}")

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

logger.info("1. Загрузка данных")

train_df = pd.read_pickle(BASE_DIR + "train_sequences.pkl")
val_df = pd.read_pickle(BASE_DIR + "val_sequences.pkl")
test_df = pd.read_pickle(BASE_DIR + "test_sequences.pkl")
extra_df = pd.read_parquet(BASE_DIR + "extra_day_features.parquet")

for d in (train_df, val_df, test_df, extra_df):
    d['day'] = pd.to_datetime(d['day']).dt.date

extra_feature_cols = [c for c in extra_df.columns if c not in ('user', 'day')]

train_df = pd.merge(train_df.drop(columns=extra_feature_cols, errors='ignore'), extra_df, on=['user', 'day'], how='left')
val_df = pd.merge(val_df.drop(columns=extra_feature_cols, errors='ignore'), extra_df, on=['user', 'day'], how='left')
test_df = pd.merge(test_df.drop(columns=extra_feature_cols, errors='ignore'), extra_df, on=['user', 'day'], how='left')

for d in (train_df, val_df, test_df):
    d[extra_feature_cols] = d[extra_feature_cols].fillna(0)

logger.info("2. Генерация признаков (редкие токены, отношения, популяционные z-score)")
target_tokens = ['http_leak', 'http_hack', 'http_job', 'http_cloud', 'http_social', 'file_sensitive', 'file_archive', 'file_executable', 'email_external', 'email_attach']
for d in (train_df, val_df, test_df):
    for tok in target_tokens:
        d[f'cnt_{tok}'] = d['sequence'].apply(lambda seq: sum(1 for t in seq if t == tok)).astype(np.int32)
    d['usb_per_file'] = np.where(d['n_file_total'] > 0, d['n_usb_connect'] / d['n_file_total'], d['n_usb_connect']).astype(np.float32)
    d['ext_email_share'] = np.where(d['n_email_total'] > 0, d['cnt_email_external'] / d['n_email_total'], 0.0).astype(np.float32)
    d['risk_http_share'] = np.where(d['n_http_total'] > 0, d['n_http_risk'] / d['n_http_total'], 0.0).astype(np.float32)
    d['entropy_x_http'] = (d['domain_entropy'] * np.log1p(d['n_http_total'])).astype(np.float32)
z_cols = ['n_http_total', 'n_file_total', 'n_email_total', 'total_attachments', 'unique_domains', 'domain_entropy', 'distinct_pcs_real', 'n_usb_connect', 'full_length']
for col in z_cols:
    m, s = float(train_df[col].mean()), float(train_df[col].std())
    s = 1.0 if s < 1e-9 else s
    for d in (train_df, val_df, test_df):
        d[f'{col}_zpop'] = ((d[col] - m) / s).astype(np.float32)
FEATURE_COLS = extra_feature_cols + ['full_length', 'usb_per_file', 'ext_email_share', 'risk_http_share', 'entropy_x_http'] + [f'cnt_{t}' for t in target_tokens] + [f'{c}_zpop' for c in z_cols]
logger.info(f"Всего признаков: {len(FEATURE_COLS)}")

logger.info("3. Обучение XGBoost на CPU")
pos_weight = float(np.sqrt(np.load(BASE_DIR + "pos_weight.npy")[0]))
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=pos_weight, eval_metric='aucpr', early_stopping_rounds=50, tree_method='hist', n_jobs=-1, random_state=42)
t0 = time.time()
model.fit(train_df[FEATURE_COLS], train_df['has_anomaly'].values, eval_set=[(train_df[FEATURE_COLS], train_df['has_anomaly'].values), (val_df[FEATURE_COLS], val_df['has_anomaly'].values)], verbose=50)
logger.info(f"Обучение завершено за {time.time() - t0:.1f} сек (best_iteration={model.best_iteration})")
val_probs = model.predict_proba(val_df[FEATURE_COLS])[:, 1]
test_probs = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
test_p, test_r, test_f1, test_auroc, test_auprc = calculate_metrics(test_df['has_anomaly'].values, test_probs)
logger.info(f"\nTEST (дни, порог 0.5): P={test_p:.4f}, R={test_r:.4f}, F1={test_f1:.4f}, AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f}")

logger.info("\n4. Оценка на уровне пользователей (User-level)")
val_df['prob'], test_df['prob'] = val_probs, test_probs
val_df, test_df = add_rolling(val_df, 3), add_rolling(test_df, 3)
vp, vl = user_level_probs(val_df, 'prob')
tp, tl = user_level_probs(test_df, 'prob')
rule_max = report_user_rule("max(prob)", vp, vl, tp, tl, logger)
rules = {'max': rule_max}
for alpha, tag in [(0.4, '@=0.4'), (0.5, '@=0.5')]:
    val_df['hyb'] = val_df['prob'] + alpha * val_df['roll3_mean']
    test_df['hyb'] = test_df['prob'] + alpha * test_df['roll3_mean']
    vp, vl = user_level_probs(val_df, 'hyb')
    tp, tl = user_level_probs(test_df, 'hyb')
    rules[f'hyb_{int(alpha*10)}'] = report_user_rule(f"hybrid {tag}", vp, vl, tp, tl, logger)
iso = IsotonicRegression(out_of_bounds='clip').fit(val_probs, val_df['has_anomaly'].values)
val_df['iso'], test_df['iso'] = iso.predict(val_probs), iso.predict(test_probs)
vp, vl = user_level_probs(val_df, 'iso')
tp, tl = user_level_probs(test_df, 'iso')
rules['isotonic'] = report_user_rule("isotonic(prob)", vp, vl, tp, tl, logger)
best_rule = max(rules.values(), key=lambda r: r['val_f1_users'])
logger.info(f"\nЛУЧШЕЕ ПРАВИЛО: {best_rule['rule']} -> Val F1={best_rule['val_f1_users']:.4f}, TEST F1={best_rule['test_f1_users']:.4f} (caught={best_rule['caught']})")
imp_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)

logger.info("\nТОП-15 ВАЖНЫХ ПРИЗНАКОВ:")
for _, r in imp_df.head(15).iterrows():
    logger.info(f"  {r['feature']:<25} {r['importance']:.4f}")
run_dir = make_run_dir("xgboost_fe")
save_config(run_dir, {"model": "XGBClassifier", "best_iteration": int(model.best_iteration), "features": FEATURE_COLS})
imp_df.to_csv(run_dir + "/feature_importances.csv", index=False)
save_split(run_dir, train_df['user'].unique(), val_df['user'].unique(), test_df['user'].unique())
save_predictions(run_dir, "val_probs", val_df[['user', 'day', 'prob', 'has_anomaly']])
save_predictions(run_dir, "test_probs", test_df[['user', 'day', 'prob', 'has_anomaly']])
save_metrics(run_dir, {"days": {"f1": float(test_f1), "auroc": float(test_auroc), "auprc": float(test_auprc)}, "best_user_rule": best_rule})

logger.info("\n5. Сравнение с классическими бейзлайнами на тех же 42 признаках")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(train_df[FEATURE_COLS])
X_va_s = scaler.transform(val_df[FEATURE_COLS])
X_te_s = scaler.transform(test_df[FEATURE_COLS])

baselines = {
    "LogisticRegression": (LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), True),
    "RandomForest": (RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=-1, random_state=42), False),
    "IsolationForest": (IsolationForest(n_estimators=200, contamination=0.003, n_jobs=-1, random_state=42), False),
}

summary_rows = [{
    "Model": "XGBoost (FE)",
    "Day_AUPRC": round(test_auprc, 4),
    "User_F1": round(best_rule['test_f1_users'], 4),
    "User_P": round(best_rule['test_precision'], 4),
    "User_R": round(best_rule['test_recall'], 4),
    "Caught": best_rule['caught']
}]

for m_name, (clf, scaled) in baselines.items():
    logger.info(f"Обучение {m_name}")
    X_tr_m, X_va_m, X_te_m = (X_tr_s, X_va_s, X_te_s) if scaled else (train_df[FEATURE_COLS], val_df[FEATURE_COLS], test_df[FEATURE_COLS])
    if m_name == "IsolationForest":
        clf.fit(X_tr_m[train_df['has_anomaly'].values == 0])
        va_p, te_p = -clf.score_samples(X_va_m), -clf.score_samples(X_te_m)
    else:
        clf.fit(X_tr_m, train_df['has_anomaly'].values)
        va_p, te_p = clf.predict_proba(X_va_m)[:, 1], clf.predict_proba(X_te_m)[:, 1]
    
    _, _, _, _, m_auprc = calculate_metrics(test_df['has_anomaly'].values, te_p)
    v_tmp = val_df[['user', 'has_anomaly']].assign(p=va_p).groupby('user').max()
    t_tmp = test_df[['user', 'has_anomaly']].assign(p=te_p).groupby('user').max()
    res = report_user_rule(f"{m_name} max(prob)", v_tmp['p'], v_tmp['has_anomaly'], t_tmp['p'], t_tmp['has_anomaly'], logger)
    summary_rows.append({"Model": m_name, "Day_AUPRC": round(m_auprc, 4), "User_F1": round(res['test_f1_users'], 4), "User_P": round(res['test_precision'], 4), "User_R": round(res['test_recall'], 4), "Caught": res['caught']})

summary_df = pd.DataFrame(summary_rows)
logger.info("ИТОГОВАЯ ТАБЛИЦА ДЛЯ (TEST)\n" + summary_df.to_string(index=False))
summary_df.to_csv(run_dir + "/all_models_comparison.csv", index=False)

copy_file(run_dir, BASE_DIR + "log/xgb.log")