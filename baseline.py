# baseline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


def print_metrics(name, y_true, y_pred, y_proba, dataset_type="Test"):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    print(f"\n{name} ({dataset_type}):")
    print(f"  Precision: {p}")
    print(f"  Recall:    {r}")
    print(f"  F1:        {f1}")
    print(f"  AUROC:     {auroc}")
    print(f"  AUPRC:     {auprc}")
    return p, r, f1, auroc, auprc

def find_best_threshold(y_val, y_val_proba):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0, 1, 101):
        pred = (y_val_proba >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, pred, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

def plot_model_evaluation(y_test, y_proba, y_pred, model_name, auprc, auroc, output_path="D:/dataset/"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    axes[0, 0].plot(rec_curve, prec_curve, 'b-', linewidth=2, label=f'{model_name} (AUPRC={auprc:.4f})')
    axes[0, 0].axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline={y_test.mean():.4f}')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title(f'Precision-Recall Curve ({model_name})')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, 'g-', linewidth=2, label=f'{model_name} (AUROC={auroc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'ROC Curve ({model_name})')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].hist(y_proba[y_test == 0], bins=50, alpha=0.5, label='Normal', density=True)
    axes[1, 0].hist(y_proba[y_test == 1], bins=50, alpha=0.5, label='Anomaly', density=True)
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title(f'Distribution of Predictions ({model_name})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title(f'Confusion Matrix ({model_name})\nTP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}{model_name.lower()}_evaluation.png", dpi=150)
    
    print(f"\nМатрица ошибок {model_name}:")
    print(f"TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")
    print(f"Графики сохранены в {output_path}{model_name.lower()}_evaluation.png")
    
    return cm

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
    'business_unit_1.0', 'business_unit_2.0', 'business_unit_3.0', 'business_unit_4.0', 'business_unit_5.0',
    'functional_unit_2 - ResearchAndEngineering', 'functional_unit_3 - Manufacturing',
    'functional_unit_4 - Sales', 'functional_unit_5 - SalesAndMarketing',
    'functional_unit_6 - HR', 'functional_unit_7 - IT',
    'role_ProductionLineWorker', 'role_Technician', 'role_Salesman', 'role_Scientist',
    'role_SoftwareEngineer', 'role_Manager', 'role_AdministrativeAssistant',
    'department_2 - Sales', 'department_3 - Assembly', 'department_3 - Engineering',
    'department_3 - FieldService', 'department_3 - Operations', 'department_3 - SoftwareManagement',
    
    # USB признаки
    'device_total', 'device_unique_pcs', 'device_connects', 'device_disconnects',
    'device_night', 'device_weekend', 'device_has_bad',
    
    # Другие потенциальные утечки
    'bad_http', 'logon_bad', 'device_bad', 'http_bad', 'email_bad', 'file_bad',
    'logon_has_bad', 'device_has_bad', 'http_has_bad', 'email_has_bad', 'file_has_bad', 'total_bad',
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
team_cols = [col for col in feature_cols if col.startswith('team_')]

# Формируем полный список утечек
bad_features = [
    'logon_bad', 'device_bad', 'http_bad', 'email_bad', 'file_bad',
    'logon_has_bad', 'device_has_bad', 'http_has_bad', 'email_has_bad', 'file_has_bad',
    'bad_http', 'device_total', 'device_unique_pcs', 'device_connects', 'device_disconnects',
    'device_night', 'device_weekend', 'device_has_bad',
] + dept_cols + role_cols + bu_cols + func_cols + team_cols

bad_features = list(set(bad_features))

print(f"Всего утечек: {len(bad_features)}")
print(f"Из них department: {len(dept_cols)}")
print(f"Из них role: {len(role_cols)}")
print(f"Из них team: {len(team_cols)}")

# Оставляем только безопасные признаки
safe_features = [col for col in feature_cols if col not in bad_features]

# Нормируем признаки, связанные с активностью
df['logon_night_ratio'] = df['logon_night'] / (df['logon_total'] + 1)
df['device_night_ratio'] = df['device_night'] / (df['device_total'] + 1)
df['http_night_ratio'] = df['http_night'] / (df['http_total'] + 1)
df['email_night_ratio'] = df['email_night'] / (df['email_total'] + 1)
df['sensitive_ratio'] = df['sensitive_files'] / (df['file_total'] + 1)
df['external_ratio'] = df['email_external'] / (df['email_total'] + 1)
df['attachment_ratio'] = df['email_with_attachments'] / (df['email_total'] + 1)

safe_features.extend(['logon_night_ratio', 'device_night_ratio', 'http_night_ratio', 
                      'email_night_ratio', 'sensitive_ratio', 'external_ratio', 'attachment_ratio'])

total_cols = [col for col in safe_features if col.endswith('_total') or col == 'total_http']
print(f"\nУдаляем признаки общей активности: {total_cols}")
safe_features = [col for col in safe_features if col not in total_cols]

# Разделяем по пользователям
unique_users = df['user'].unique()
train_users, test_users = train_test_split(unique_users, test_size=0.3, random_state=42)
train_users, val_users = train_test_split(train_users, test_size=0.2, random_state=42)

X_train = df[df['user'].isin(train_users)]
X_val = df[df['user'].isin(val_users)]
X_test = df[df['user'].isin(test_users)]

y_train = X_train['has_anomaly']
y_val = X_val['has_anomaly']
y_test = X_test['has_anomaly']

X_train = X_train[safe_features].fillna(0)
X_val = X_val[safe_features].fillna(0)
X_test = X_test[safe_features].fillna(0)

print(f"\nTrain: {len(train_users)} пользователей, {len(X_train)} строк, {y_train.mean()*100}% аномалий")
print(f"Val:   {len(val_users)} пользователей, {len(X_val)} строк, {y_val.mean()*100}% аномалий")
print(f"Test:  {len(test_users)} пользователей, {len(X_test)} строк, {y_test.mean()*100}% аномалий")

# Проверка пересечения
print(f"\nПересечение train/val: {len(set(train_users) & set(val_users))}")
print(f"Пересечение train/test: {len(set(train_users) & set(test_users))}")
print(f"Пересечение val/test: {len(set(val_users) & set(test_users))}")

print(f"\nИтоговые признаки для обучения: {safe_features}")
print(f"Итого признаков: {len(safe_features)}")

print("\nLOGISTIC REGRESSION")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Train
y_train_pred = lr.predict(X_train_scaled)
y_train_proba = lr.predict_proba(X_train_scaled)[:, 1]
print_metrics("Logistic Regression", y_train, y_train_pred, y_train_proba, "Train")

# Val
y_val_pred = lr.predict(X_val_scaled)
y_val_proba = lr.predict_proba(X_val_scaled)[:, 1]
print_metrics("Logistic Regression", y_val, y_val_pred, y_val_proba, "Val")

# Test
y_test_pred = lr.predict(X_test_scaled)
y_test_proba = lr.predict_proba(X_test_scaled)[:, 1]
p_lr, r_lr, f1_lr, auroc_lr, auprc_lr = print_metrics("Logistic Regression", y_test, y_test_pred, y_test_proba, "Test")

# Для Logistic Regression
plot_model_evaluation(
    y_test=y_test,
    y_proba=y_test_proba,
    y_pred=y_test_pred,
    model_name="Logistic Regression",
    auprc=auprc_lr,
    auroc=auroc_lr
)

print("\nRANDOM FOREST")

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=6,
    min_samples_split=15,
    min_samples_leaf=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Train
y_train_proba = rf.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_proba >= 0.5).astype(int)
print_metrics("Random Forest", y_train, y_train_pred, y_train_proba, "Train")

# Val (оптимизация порога)
y_val_proba = rf.predict_proba(X_val)[:, 1]
best_thresh_rf = find_best_threshold(y_val, y_val_proba)
y_val_pred = (y_val_proba >= best_thresh_rf).astype(int)
print_metrics("Random Forest", y_val, y_val_pred, y_val_proba, "Val")
print(f"  Оптимальный порог: {best_thresh_rf}")

# Test
y_test_proba = rf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_thresh_rf).astype(int)
p_rf, r_rf, f1_rf, auroc_rf, auprc_rf = print_metrics("Random Forest", y_test, y_test_pred, y_test_proba, "Test")

# Feature importance
importance = pd.DataFrame({'feature': safe_features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
print("\nТоп-15 важных признаков:")
print(importance.head(15))

plot_model_evaluation(
    y_test=y_test,
    y_proba=y_test_proba,
    y_pred=y_test_pred,
    model_name="Random Forest",
    auprc=auprc_rf,
    auroc=auroc_rf
)

print("\nXGBOOST")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=1.0,
    reg_alpha=0.5,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

# Train
y_train_proba = xgb.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_proba >= 0.5).astype(int)
print_metrics("XGBoost", y_train, y_train_pred, y_train_proba, "Train")

# Val 
y_val_proba = xgb.predict_proba(X_val)[:, 1]

best_thresh_xgb = find_best_threshold(y_val, y_val_proba)
print(f"\nОптимальный порог: {best_thresh_xgb} (вычислен автоматически)")

best_thresh_xgb = 0.0
best_f1 = 0
for thresh in np.linspace(0.1, 0.9, 81):
    pred = (y_val_proba >= thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_val, pred, average='binary')
    if r > 0.2 and f1 > best_f1:
        best_f1 = f1
        best_thresh_xgb = thresh

print(f"Оптимальный порог: {best_thresh_xgb} (при ограничении recall > 20%)")

# Test
y_test_proba = xgb.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_thresh_xgb).astype(int)
p_xgb, r_xgb, f1_xgb, auroc_xgb, auprc_xgb = print_metrics("XGBoost", y_test, y_test_pred, y_test_proba, "Test")

# Для XGBoost
plot_model_evaluation(
    y_test=y_test,
    y_proba=y_test_proba,
    y_pred=y_test_pred,
    model_name="XGBoost",
    auprc=auprc_xgb,
    auroc=auroc_xgb
)

print("\nISOLATION FOREST")

X_train_normal = X_train[y_train == 0]

iso = IsolationForest(n_estimators=100, contamination=0.003, random_state=42)
iso.fit(X_train_normal)

y_pred_iso = iso.predict(X_test)
y_pred_iso_binary = (y_pred_iso == -1).astype(int)
y_scores_iso = -iso.score_samples(X_test)
y_proba_iso = (y_scores_iso - y_scores_iso.min()) / (y_scores_iso.max() - y_scores_iso.min())

p_iso, r_iso, f1_iso, _ = precision_recall_fscore_support(y_test, y_pred_iso_binary, average='binary')
auroc_iso = roc_auc_score(y_test, y_proba_iso)
auprc_iso = average_precision_score(y_test, y_proba_iso)

# Для Logistic Regression
plot_model_evaluation(
    y_test=y_test,
    y_proba=y_test_proba,
    y_pred=y_test_pred,
    model_name="Isolation Forest",
    auprc=auprc_iso,
    auroc=auroc_iso
)

print(f"Precision: {p_iso}")
print(f"Recall: {r_iso}")
print(f"F1: {f1_iso}")
print(f"AUROC: {auroc_iso}")
print(f"AUPRC: {auprc_iso}")

print("\nСРАВНЕНИЕ МОДЕЛЕЙ (TEST)")

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Isolation Forest'],
    'Precision': [p_lr, p_rf, p_xgb, p_iso],
    'Recall': [r_lr, r_rf, r_xgb, r_iso],
    'F1': [f1_lr, f1_rf, f1_xgb, f1_iso],
    'AUPRC': [auprc_lr, auprc_rf, auprc_xgb, auprc_iso]
})
print(comparison.to_string())