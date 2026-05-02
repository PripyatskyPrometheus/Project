# Changelog: Исправления модели обнаружения инсайдеров

## 1. Удаление признаков-утечек

**Проблема:** Модель использовала категориальные признаки (отделы, роли, команды) для определения инсайдеров, что давало аномально высокие метрики (AUPRC 95%). Инсайдера нужно определять по его поведению, а не по отделу или роли.

**Было:**
```python
bad_features = [
    # Прямые метки
    'logon_bad', 'device_bad', 'http_bad', 'email_bad', 'file_bad',
    'logon_has_bad', 'device_has_bad', 'http_has_bad', 'email_has_bad', 'file_has_bad',
    'bad_http',
    
    # USB признаки
    'device_total', 'device_unique_pcs', 'device_connects', 'device_disconnects',
    'device_night', 'device_weekend', 'device_has_bad',
] + dept_cols + role_cols + bu_cols + func_cols

**Стало:**
```python
team_cols = [col for col in feature_cols if col.startswith('team_')]
bad_features = [
    'logon_bad', 'device_bad', 'http_bad', 'email_bad', 'file_bad',
    'logon_has_bad', 'device_has_bad', 'http_has_bad', 'email_has_bad', 'file_has_bad',
    'bad_http', 'device_total', 'device_unique_pcs', 'device_connects', 'device_disconnects',
    'device_night', 'device_weekend', 'device_has_bad',
] + dept_cols + role_cols + bu_cols + func_cols + team_cols

**Что изменилось:** Добавлен признак team_cols — команды сотрудников (38 колонок).

## 2. Добавление нормированных признаков

Проблема: Признаки общей активности (logon_total, email_total, file_total, http_total, total_http) создавали сильный перекос в сторону активных пользователей, повышая вероятность определения инсайдера просто по его активности.

Что добавили:

python
df['logon_night_ratio'] = df['logon_night'] / (df['logon_total'] + 1)
df['device_night_ratio'] = df['device_night'] / (df['device_total'] + 1)
df['http_night_ratio'] = df['http_night'] / (df['http_total'] + 1)
df['email_night_ratio'] = df['email_night'] / (df['email_total'] + 1)
df['sensitive_ratio'] = df['sensitive_files'] / (df['file_total'] + 1)
df['external_ratio'] = df['email_external'] / (df['email_total'] + 1)
df['attachment_ratio'] = df['email_with_attachments'] / (df['email_total'] + 1)

safe_features.extend(['logon_night_ratio', 'device_night_ratio', 'http_night_ratio', 
                      'email_night_ratio', 'sensitive_ratio', 'external_ratio', 'attachment_ratio'])
Что изменилось: Модель теперь смотрит на долю подозрительной активности, а не на абсолютное число. Например, sensitive_ratio = 0.5 означает, что половина файлов пользователя — чувствительные, что гораздо более подозрительно, чем просто большое количество файлов.

3. Удаление признаков общей активности
Проблема: Несмотря на добавление нормировки, участие признаков общей активности по-прежнему создавало сильный перекос, а сами признаки не являлись поведенческими.

Что добавили:

python
total_cols = [col for col in safe_features if col.endswith('_total') or col == 'total_http']
print(f"\nУдаляем признаки общей активности: {total_cols}")
safe_features = [col for col in safe_features if col not in total_cols]
Что изменилось: Удалили logon_total, email_total, file_total, http_total, total_http, которые являлись утечкой.

4. Разделение по пользователям
Проблема: При случайном разделении один пользователь мог попасть и в train, и в test, в результате чего модель запоминала его поведение, что приводило к утечке данных.

Было:

python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
Стало:

python
unique_users = df['user'].unique()
train_users, test_users = train_test_split(unique_users, test_size=0.3, random_state=42)
train_users, val_users = train_test_split(train_users, test_size=0.2, random_state=42)

X_train = df[df['user'].isin(train_users)]
X_val = df[df['user'].isin(val_users)]
X_test = df[df['user'].isin(test_users)]

print(f"\nПересечение train/val: {len(set(train_users) & set(val_users))}")
print(f"Пересечение train/test: {len(set(train_users) & set(test_users))}")
print(f"Пересечение val/test: {len(set(val_users) & set(test_users))}")
Что изменилось: Теперь один и тот же сотрудник не окажется в разных выборках, что устраняет утечку данных.

5. Оптимизация порога на валидации
Проблема: Оптимальный порог подбирался на тестовой выборке, что приводило к завышенной оценке метрик и переобучению модели под тест.

Стало:

python
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

best_thresh_rf = find_best_threshold(y_val, y_val_proba)
best_thresh_xgb = find_best_threshold(y_val, y_val_proba)
Что изменилось: Модель не подстраивается под тестовые данные, метрики стали более реалистичными.

6. Стандартизация для Logistic Regression
Проблема: Logistic Regression не сходилась ни за 1000, ни за 2000, ни за 10000 итераций из-за разного масштаба признаков. Несходимость ставила под сомнение её результирующие оценки.

Было:

python
lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
lr.fit(X_train, y_train)
Стало:

python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
Что изменилось: Модель наконец стала сходиться, и её метрики теперь заслуживают больше доверия.

7. Регуляризация Random Forest
Проблема: Random Forest сильно переобучался: разрыв метрики F1 между обучением и валидацией достигал 18%.

Было:

python
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=10, n_jobs=-1)
Стало:

python
rf = RandomForestClassifier(
    n_estimators=50,           # уменьшили количество деревьев
    max_depth=6,               # уменьшили глубину
    min_samples_split=15,      # больше образцов для разделения
    min_samples_leaf=8,        # больше образцов в листе
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
Что изменилось: Несмотря на экстремальный дисбаланс (только 0.3% данных являются аномальными), переобучение удалось снизить с 18% до 12% по метрике F1.