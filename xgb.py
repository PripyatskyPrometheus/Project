# xgb.py
import os, sys, time, logging
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import *
from save_run import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("D:/dataset/log/xgb.log", mode='w', encoding='utf-8'),
    logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('XGBoost')

BASE_DIR = "D:/dataset/"

CURRENT = [
    'logon', 'logon_night', 'logoff', 'pc_count',
    'usb', 'usb_night', 'files', 'files_night',
    'email', 'email_night', 'attached', 'external', 'external_night',
    'http', 'http_night', 'night_share', 'external_share', 'attachment_share',
    'weekend', 'usb_per_file', 'risk_http_share',
    'unique_domains', 'domain_entropy',
]

HISTORY = [
    'coverage_30d', 'previous_observation_gap', 'pct_30d_logon', 'delta_30d_logon', 'pct_30d_usb', 'delta_30d_usb', 'pct_30d_files',
    'delta_30d_files', 'pct_30d_http', 'delta_30d_http', 'pct_30d_email', 'delta_30d_email', 'pct_30d_external', 'delta_30d_external',
    'pct_30d_pc_count', 'delta_30d_pc_count', 'pct_30d_night_share', 'delta_30d_night_share', 'pct_30d_unique_domains', 
    'delta_30d_unique_domains', 'pct_30d_domain_entropy', 'delta_30d_domain_entropy'
]

# HTTP-фичи (убираем в no_http)
HTTP_FEATURES = {
    'http', 'http_night', 'n_http_risk', 'risk_http_share', 'domain_entropy', 'unique_domains',
    'night_share', 'pct_30d_night_share', 'delta_30d_night_share', 'pct_30d_http', 'delta_30d_http', 'pct_30d_domain_entropy', 'delta_30d_domain_entropy', 
    'pct_30d_unique_domains', 'delta_30d_unique_domains'
}

def strip_http(cols):
    return [c for c in cols if c not in HTTP_FEATURES]

FEATURE_SETS = {
    'all':            {'with_http': CURRENT + HISTORY,       'no_http': strip_http(CURRENT + HISTORY)},
    'deep_only':      {'with_http': CURRENT,                 'no_http': strip_http(CURRENT)},
    'personal_only':  {'with_http': HISTORY,                 'no_http': strip_http(HISTORY)},
}


def calculate_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return p, r, f1, roc_auc_score(labels, probs), average_precision_score(labels, probs)


def add_rolling(df, window=3):
    df = df.sort_values(['user', 'day']).copy()
    df[f'roll{window}_mean'] = df.groupby('user')['prob'].transform(
        lambda s: s.rolling(window, min_periods=1).mean())
    return df


def user_level_probs(df, prob_col):
    g = df.groupby('user')
    return g[prob_col].max(), g['has_anomaly'].max()


def select_f1_threshold(labels, scores, plateau_delta=0.01):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {'best_threshold': 0.5, 'best_f1': 0.0, 'plateau': (0.5, 0.5), 'plateau_width': 0, 'top5': []}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom != 0)
    best_idx = int(np.argmax(f1))
    best_th = float(thresholds[best_idx]); best_f1 = float(f1[best_idx])
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


def report_user_rule(name, val_probs_series, val_labels_series, test_probs_series, test_labels_series, logger):
    info = select_f1_threshold(val_labels_series, val_probs_series)
    th = info['best_threshold']
    preds = (test_probs_series >= th).astype(int)
    t_p = precision_score(test_labels_series, preds, zero_division=0)
    t_r = recall_score(test_labels_series, preds, zero_division=0)
    t_f1 = f1_score(test_labels_series, preds, zero_division=0)
    caught = int(preds[test_labels_series == 1].sum()); total = int(test_labels_series.sum())
    logger.info(f"[{name}]")
    logger.info(f"Порог (val): {th:.4f} (val F1={info['best_f1']:.4f})")
    logger.info(f"TEST {th:.4f}: F1={t_f1:.4f}, P={t_p:.4f}, R={t_r:.4f}, caught={caught}/{total}")
    return {'rule': name, 'best_threshold': float(th), 'val_f1_users': float(info['best_f1']), 'test_f1_users': float(t_f1), 
    'test_precision': float(t_p), 'test_recall': float(t_r), 'caught': f"{caught}/{total}"}


def run_experiment(train_df, val_df, test_df, feature_cols, name, logger):
    logger.info(f"{name} ({len(feature_cols)} фич)")
    pos_weight = float(np.sqrt(np.load(BASE_DIR + "pos_weight.npy")[0]))
    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=pos_weight, eval_metric='aucpr',
        early_stopping_rounds=50, tree_method='hist', n_jobs=-1, random_state=42)
    t0 = time.time()
    model.fit(train_df[feature_cols], train_df['has_anomaly'].values, eval_set=[(val_df[feature_cols], val_df['has_anomaly'].values)], 
    verbose=False)
    logger.info(f"Обучение: {time.time()-t0:.1f} сек (best_iter={model.best_iteration})")

    val_probs = model.predict_proba(val_df[feature_cols])[:, 1]
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]
    test_p, test_r, test_f1, test_auroc, test_auprc = calculate_metrics(test_df['has_anomaly'].values, test_probs)
    logger.info(f"TEST (дни): P={test_p:.4f} R={test_r:.4f} F1={test_f1:.4f} AUROC={test_auroc:.4f} AUPRC={test_auprc:.4f}")

    # ИСПРАВЛЕННЫЙ ISOTONIC: калибруем колонку той же таблицы
    val_df = val_df.copy(); test_df = test_df.copy()
    val_df['prob'] = val_probs; test_df['prob'] = test_probs
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_df['prob'].values, val_df['has_anomaly'].values)
    val_df['prob_iso'] = iso.predict(val_df['prob'].values)
    test_df['prob_iso'] = iso.predict(test_df['prob'].values)

    # user-level
    val_df = add_rolling(val_df); test_df = add_rolling(test_df)
    vp, vl = user_level_probs(val_df, 'prob'); tp, tl = user_level_probs(test_df, 'prob')
    rules = {'max': report_user_rule('max(prob)', vp, vl, tp, tl, logger)}
    for alpha, tag in [(0.4, 'hybrid_04'), (0.5, 'hybrid_05')]:
        val_df['hyb'] = val_df['prob'] + alpha * val_df['roll3_mean']
        test_df['hyb'] = test_df['prob'] + alpha * test_df['roll3_mean']
        vp, vl = user_level_probs(val_df, 'hyb'); tp, tl = user_level_probs(test_df, 'hyb')
        rules[tag] = report_user_rule(tag, vp, vl, tp, tl, logger)
    vp, vl = user_level_probs(val_df, 'prob_iso'); tp, tl = user_level_probs(test_df, 'prob_iso')
    rules['isotonic'] = report_user_rule('isotonic', vp, vl, tp, tl, logger)

    best_rule_name = max(rules, key=lambda k: rules[k]['val_f1_users'])
    best = rules[best_rule_name]
    logger.info(f"ЛУЧШЕЕ: {best_rule_name} (val={best['val_f1_users']:.4f}, test={best['test_f1_users']:.4f})")

    run_dir = make_run_dir(f"xgb_{name}")
    save_config(run_dir, {'model': 'XGBClassifier', 'features': feature_cols,'best_iteration': int(model.best_iteration), 'name': name})
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    imp_df.to_csv(run_dir + "/feature_importances.csv", index=False)
    save_split(run_dir, train_df['user'].unique(), val_df['user'].unique(), test_df['user'].unique())
    save_predictions(run_dir, "val_probs", val_df[['user', 'day', 'prob', 'has_anomaly']])
    save_predictions(run_dir, "test_probs", test_df[['user', 'day', 'prob', 'has_anomaly']])
    save_metrics(run_dir, {'days': {'f1': float(test_f1), 'auroc': float(test_auroc), 'auprc': float(test_auprc)},
    'best_user_rule': best, 'rules': rules})
    return {'name': name, 'auprc': test_auprc, 'f1_days': test_f1, 'best_rule': best_rule_name, 'best_test_f1': best['test_f1_users'], 
    'best_val_f1': best['val_f1_users'], 'best_test_precision': best['test_precision'], 'best_test_recall': best['test_recall'],
    'caught': best['caught']}


def main():
    logger.info("1. Загрузка данных")
    train_df = pd.read_pickle(BASE_DIR + "train_sequences.pkl")
    val_df = pd.read_pickle(BASE_DIR + "val_sequences.pkl")
    test_df = pd.read_pickle(BASE_DIR + "test_sequences.pkl")
    extra_df = pd.read_parquet(BASE_DIR + "extra_day_features.parquet")

    # защита - extra_df не должен содержать метку
    extra_df = extra_df.drop(columns=['has_anomaly'], errors='ignore')

    for d in (train_df, val_df, test_df, extra_df):
        d['day'] = pd.to_datetime(d['day']).dt.date
    extra_cols = [c for c in extra_df.columns if c not in ('user', 'day')]

    merged = {}
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        d = d.drop(columns=extra_cols, errors='ignore')
        m = d.merge(extra_df, on=['user', 'day'], how='left')
        m[extra_cols] = m[extra_cols].fillna(0)
        merged[name] = m
    train_df, val_df, test_df = merged['train'], merged['val'], merged['test']

    # проверка метки
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        vc = d['has_anomaly'].value_counts(dropna=False).to_dict()
        logger.info(f"{name}: has_anomaly {vc}")
        assert d['has_anomaly'].isin([0, 1]).all(), f"{name}: has_anomaly не 0/1: {vc}"

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    for set_name, sets in FEATURE_SETS.items():
        logger.info(f"{set_name}: with_http={len(sets['with_http'])}, no_http={len(sets['no_http'])}")

    results = []
    for set_name, sets in FEATURE_SETS.items():
        for http_variant, cols in sets.items():
            try:
                r = run_experiment(train_df, val_df, test_df, cols, f"{set_name}_{http_variant}", logger)
                results.append(r)
            except Exception as e:
                logger.error(f"ОШИБКА в {set_name}_{http_variant}: {e}")

    logger.info('ИТОГИ')
    logger.info(f"{'name':<30} {'AUPRC':>8} {'F1_days':>8} {'P_users':>8} {'R_users':>8} {'F1_users':>8} {'caught':>8} {'best_rule':>12}")
    for r in results:
        logger.info(f"{r['name']:<30} {r['auprc']:>8.4f} {r['f1_days']:>8.4f} {r['best_test_precision']:>8.4f} {r['best_test_recall']:>8.4f}"
        f"{r['best_test_f1']:>8.4f} {r['caught']:>8} {r['best_rule']:>12}")


    summary = pd.DataFrame(results)
    summary.to_csv(BASE_DIR + "xgb.csv", index=False)
    logger.info(f"Сводка сохранена: {BASE_DIR}xgb.csv")

if __name__ == '__main__':
    main()