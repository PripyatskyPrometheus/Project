import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

df = pd.read_csv("D:/dataset/features_final.csv")

print(f"Размер датасета: {df.shape}")

# Создаем таргет
df['has_anomaly'] = (
    (df['logon_bad'] > 0) |
    (df['device_bad'] > 0) |
    (df['http_bad'] > 0) |
    (df['email_bad'] > 0) |
    (df['file_bad'] > 0)
).astype(int)

print(f"\nРаспределение таргета:")
print(df['has_anomaly'].value_counts())
print(f"Процент аномалий: {df['has_anomaly'].mean()*100}%")

# Убираем колонки, которые не нужны для обучения
exclude_cols = [
    'user', 'day', 'is_insider', 'scenario', 'employee_name', 'top_domain',
    'role', 'business_unit', 'functional_unit', 'department', 'team', 'supervisor',
    
    # Все dummy-переменные отделов и ролей
    'business_unit_1.0',
    'business_unit_2.0',
    'business_unit_3.0',
    'business_unit_4.0',
    'business_unit_5.0',
    
    'functional_unit_2 - ResearchAndEngineering',
    'functional_unit_3 - Manufacturing',
    'functional_unit_4 - Sales',
    'functional_unit_5 - SalesAndMarketing',
    'functional_unit_6 - HR',
    'functional_unit_7 - IT',
    
    'role_ProductionLineWorker',
    'role_Technician',
    'role_Salesman',
    'role_Scientist',
    'role_SoftwareEngineer',
    'role_Manager',
    'role_AdministrativeAssistant',
    
    'department_2 - Sales',
    'department_3 - Assembly',
    'department_3 - Engineering',
    'department_3 - FieldService',
    'department_3 - Operations',
    'department_3 - SoftwareManagement',
    
    # USB признаки
    'device_total',
    'device_unique_pcs',
    'device_connects',
    'device_disconnects',
    'device_night',
    'device_weekend',
    'device_has_bad',
    
    # Другие потенциальные утечки
    'bad_http',
    'logon_bad', 'device_bad', 'http_bad', 'email_bad', 'file_bad',
    'logon_has_bad', 'device_has_bad', 'http_has_bad', 'email_has_bad', 'file_has_bad',
]

# Категориальные колонки кодируем
categorical_cols = ['role', 'business_unit', 'functional_unit', 'department', 'team']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Создаем dummy-переменные
for col in categorical_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)

# Выбираем признаки
feature_cols = [col for col in df.columns if col not in exclude_cols + ['has_anomaly']]

# Находим все колонки отделов, ролей, юнитов
dept_cols = [col for col in feature_cols if col.startswith('department_')]
role_cols = [col for col in feature_cols if col.startswith('role_')]
bu_cols = [col for col in feature_cols if col.startswith('business_unit_')]
func_cols = [col for col in feature_cols if col.startswith('functional_unit_')]

# Формируем полный список утечек
bad_features = [
    # Прямые метки
    'logon_bad', 'device_bad', 'http_bad', 'email_bad', 'file_bad',
    'logon_has_bad', 'device_has_bad', 'http_has_bad', 'email_has_bad', 'file_has_bad',
    'bad_http',
    
    # USB признаки
    'device_total', 'device_unique_pcs', 'device_connects', 'device_disconnects',
    'device_night', 'device_weekend', 'device_has_bad',
] + dept_cols + role_cols + bu_cols + func_cols

# Убираем дубликаты
bad_features = list(set(bad_features))

print(f"Всего утечек: {len(bad_features)}")
print(f"Из них department: {len(dept_cols)}")
print(f"Из них role: {len(role_cols)}")
print(f"Из них business_unit: {len(bu_cols)}")
print(f"Из них functional_unit: {len(func_cols)}")

# Оставляем только безопасные признаки
safe_features = [col for col in feature_cols if col not in bad_features]
X = df[safe_features].fillna(0)
y = df['has_anomaly']

print(f"Признаков после кодирования: {X.shape[1]}")

# Разделяем данные
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nTrain: {len(X_train)} ({y_train.mean()*100}% аномалий)")
print(f"Val: {len(X_val)} ({y_val.mean()*100}% аномалий)")
print(f"Test: {len(X_test)} ({y_test.mean()*100}% аномалий)")

print("\nLOGISTIC REGRESSION")

lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='binary')
auroc_lr = roc_auc_score(y_test, y_proba_lr)
auprc_lr = average_precision_score(y_test, y_proba_lr)

print(f"Precision: {p_lr}")
print(f"Recall: {r_lr}")
print(f"F1: {f1_lr}")
print(f"AUROC: {auroc_lr}")
print(f"AUPRC: {auprc_lr}")

print("\nRANDOM FOREST")

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=10, n_jobs=-1)
rf.fit(X_train, y_train)

y_proba_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict(X_test)

p_rf, r_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='binary')
auroc_rf = roc_auc_score(y_test, y_proba_rf)
auprc_rf = average_precision_score(y_test, y_proba_rf)

print(f"Precision: {p_rf}")
print(f"Recall: {r_rf}")
print(f"F1: {f1_rf}")
print(f"AUROC: {auroc_rf}")
print(f"AUPRC: {auprc_rf}")

# Feature importance
importance = pd.DataFrame({'feature': safe_features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
print("\nТоп-15 важных признаков:")
print(importance.head(15))

print("\nXGBOOST")

xgb = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, scale_pos_weight=330, random_state=42, eval_metric='logloss', 
                    reg_lambda=1.0, reg_alpha=0.5)
xgb.fit(X_train, y_train)

y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = xgb.predict(X_test)

p_xgb, r_xgb, f1_xgb, _ = precision_recall_fscore_support(y_test, y_pred_xgb, average='binary')
auroc_xgb = roc_auc_score(y_test, y_proba_xgb)
auprc_xgb = average_precision_score(y_test, y_proba_xgb)

print(f"Precision: {p_xgb}")
print(f"Recall: {r_xgb}")
print(f"F1: {f1_xgb}")
print(f"AUROC: {auroc_xgb}")
print(f"AUPRC: {auprc_xgb}")

print("\nISOLATION FOREST")
 
X_train_normal = X_train[y_train == 0]

iso = IsolationForest(n_estimators=100, contamination=0.003, random_state=42)
iso.fit(X_train_normal)

# Предсказания: 1 = норма, -1 = аномалия
y_pred_iso = iso.predict(X_test)
y_pred_iso_binary = (y_pred_iso == -1).astype(int)

# Получаем anomaly score (чем выше, тем более аномально)
y_scores_iso = -iso.score_samples(X_test)
# Нормализуем для ROC/PR
y_proba_iso = (y_scores_iso - y_scores_iso.min()) / (y_scores_iso.max() - y_scores_iso.min())

p_iso, r_iso, f1_iso, _ = precision_recall_fscore_support(y_test, y_pred_iso_binary, average='binary')
auroc_iso = roc_auc_score(y_test, y_proba_iso)
auprc_iso = average_precision_score(y_test, y_proba_iso)

print(f"Precision: {p_iso}")
print(f"Recall: {r_iso}")
print(f"F1: {f1_iso}")
print(f"AUROC: {auroc_iso}")
print(f"AUPRC: {auprc_iso}")

print("\nОПТИМИЗАЦИЯ ПОРОГА (Random Forest)")

thresholds = np.linspace(0, 1, 101)
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    pred = (y_proba_rf >= thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Оптимальный порог по F1: {best_thresh}")
y_pred_rf_opt = (y_proba_rf >= best_thresh).astype(int)
p_opt, r_opt, f1_opt, _ = precision_recall_fscore_support(y_test, y_pred_rf_opt, average='binary')
print(f"Precision: {p_opt}, Recall: {r_opt}, F1: {f1_opt}")

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest (def)', 'Random Forest (opt)', 'XGBoost', 'Isolation Forest'],
    'Precision': [p_lr, p_rf, p_opt, p_xgb, p_iso],
    'Recall': [r_lr, r_rf, r_opt, r_xgb, r_iso],
    'F1': [f1_lr, f1_rf, f1_opt, f1_xgb, f1_iso],
    'AUPRC': [auprc_lr, auprc_rf, auprc_rf, auprc_xgb, auprc_iso]
})
print(comparison.to_string())

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba_rf)
axes[0, 0].plot(rec_curve, prec_curve, 'b-', linewidth=2, label=f'RF (AUPRC={auprc_rf})')
axes[0, 0].axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline={y_test.mean()}')
axes[0, 0].set_xlabel('Recall'); axes[0, 0].set_ylabel('Precision')
axes[0, 0].set_title('Precision-Recall Curve'); axes[0, 0].legend(); axes[0, 0].grid(True)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
axes[0, 1].plot(fpr, tpr, 'g-', linewidth=2, label=f'RF (AUROC={auroc_rf})')
axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate'); axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve'); axes[0, 1].legend(); axes[0, 1].grid(True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
axes[0, 2].set_xlabel('Predicted'); axes[0, 2].set_ylabel('Actual')
axes[0, 2].set_title(f'Confusion Matrix (thresh={best_thresh})\nTP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}')

# Feature Importance
top_feat = importance.head(15)
axes[1, 0].barh(range(len(top_feat)), top_feat['importance'].values)
axes[1, 0].set_yticks(range(len(top_feat)))
axes[1, 0].set_yticklabels([f[:30] for f in top_feat['feature'].values])
axes[1, 0].set_xlabel('Importance'); axes[1, 0].set_title('Top 15 Features')
axes[1, 0].invert_yaxis(); axes[1, 0].grid(True, axis='x')

# Distribution of Predictions
axes[1, 1].hist(y_proba_rf[y_test == 0], bins=50, alpha=0.5, label='Normal', density=True)
axes[1, 1].hist(y_proba_rf[y_test == 1], bins=50, alpha=0.5, label='Anomaly', density=True)
axes[1, 1].set_xlabel('Probability'); axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Distribution of Predictions'); axes[1, 1].legend(); axes[1, 1].grid(True)

# Precision/Recall vs Threshold
thresh_range = np.linspace(0, 1, 50)
prec_list, rec_list = [], []
for th in thresh_range:
    pred = (y_proba_rf >= th).astype(int)
    p, r, _, _ = precision_recall_fscore_support(y_test, pred, average='binary')
    prec_list.append(p); rec_list.append(r)
axes[1, 2].plot(thresh_range, prec_list, 'b-', label='Precision')
axes[1, 2].plot(thresh_range, rec_list, 'r-', label='Recall')
axes[1, 2].axvline(x=best_thresh, color='k', linestyle='--', label=f'Optimal ({best_thresh})')
axes[1, 2].set_xlabel('Threshold'); axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Precision/Recall vs Threshold'); axes[1, 2].legend(); axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig("D:/dataset/model_evaluation.png", dpi=150)

print("\nГрафики сохранены в D:/dataset/model_evaluation.png")