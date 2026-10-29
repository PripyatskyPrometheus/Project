# check_all.py
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

RUN_PATH = "D:/dataset/runs/2026-10-29_16-40-48_bilstm/predictions/"

def load(name):
    return pd.read_parquet(RUN_PATH + name + ".parquet")

def add_rolling(df, window=3):
    df = df.sort_values(['user', 'day']).copy()
    df[f'roll{window}_mean'] = df.groupby('user')['prob'].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return df

def user_level(df, col):
    g = df.groupby('user')
    return g[col].max(), g['has_anomaly'].max().astype(int)

def select_f1_threshold(labels, scores, plateau_delta=0.02):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {'best_threshold': 0.5, 'best_f1': 0.0, 'plateau': (0.5, 0.5), 'plateau_width': 0, 'top5': []}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom != 0)
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
        if t in seen: continue
        seen.add(t); top5.append((t, float(f1[i])))
        if len(top5) == 5: break
    return {'best_threshold': best_th, 'best_f1': best_f1, 'plateau': plateau, 'plateau_width': int(plateau_th.size), 'top5': top5}

def report(name, vp, vl, tp, tl):
    info = select_f1_threshold(vl, vp)  # ← labels first!
    th = info['best_threshold']
    preds = (tp >= th).astype(int)
    t_p = precision_score(tl, preds, zero_division=0)
    t_r = recall_score(tl, preds, zero_division=0)
    t_f1 = f1_score(tl, preds, zero_division=0)
    caught = int(preds[tl == 1].sum())
    total = int(tl.sum())
    print(f"\n[{name}]")
    print(f"  Порог (val): {th:.4f} (val F1={info['best_f1']:.4f})")
    print(f"  Плато: {info['plateau'][0]:.4f}–{info['plateau'][1]:.4f} (ширина {info['plateau_width']})")
    for t, f in info['top5']:
        print(f"    {t:.4f}: {f:.4f}")
    print(f"  TEST @ {th:.4f}: F1={t_f1:.4f}, P={t_p:.4f}, R={t_r:.4f}, caught={caught}/{total}")
    return {'rule': name, 'threshold': th, 'val_f1': info['best_f1'], 'test_f1': t_f1, 'caught': f"{caught}/{total}"}

def main():
    val = load("val_probs"); test = load("test_probs")
    results = []

    # max
    vp, vl = user_level(val, 'prob'); tp, tl = user_level(test, 'prob')
    results.append(report("max(prob)", vp, vl, tp, tl))

    # hybrid α=0.4, 0.5
    val = add_rolling(val); test = add_rolling(test)
    for alpha, tag in [(0.4, 'hybrid_04'), (0.5, 'hybrid_05')]:
        v = val.copy(); t = test.copy()
        v['hybrid'] = v['prob'] + alpha * v['roll3_mean']
        t['hybrid'] = t['prob'] + alpha * t['roll3_mean']
        vp, vl = user_level(v, 'hybrid'); tp, tl = user_level(t, 'hybrid')
        results.append(report(tag, vp, vl, tp, tl))

    # isotonic
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val['prob'].values, val['has_anomaly'].values)
    val['prob_iso'] = iso.predict(val['prob'].values)
    test['prob_iso'] = iso.predict(test['prob'].values)
    vp, vl = user_level(val, 'prob_iso'); tp, tl = user_level(test, 'prob_iso')
    results.append(report("isotonic", vp, vl, tp, tl))

    # лучший по val
    best = max(results, key=lambda r: r['val_f1'])
    print(f"\nЛУЧШЕЕ ПО VAL: {best['rule']} (val={best['val_f1']:.4f}, test={best['test_f1']:.4f})")

    pd.DataFrame(results).to_csv("D:/dataset/recheck_results.csv", index=False)
    print("\nСохранено: D:/dataset/recheck_results.csv")

if __name__ == "__main__":
    main()