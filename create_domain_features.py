# create_domain_features.py
import pandas as pd


def extract_domain(url):
    try:
        if '://' in url:
            domain = url.split('://')[1].split('/')[0]
        else:
            domain = url.split('/')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        # Проверка: если домен пустой или слишком длинный
        if len(domain) == 0 or len(domain) > 100:
            return 'unknown'
        return domain
    except:
        return 'unknown'

print("\n1. Загрузка http_labeled.csv")
http_df = pd.read_csv("D:/dataset/labels/http_labeled.csv", usecols=['user', 'url', 'is_true_bad'])
print(f"   Всего записей: {len(http_df)}")

print("\n2. Извлечение доменов...")

http_df['domain'] = http_df['url'].apply(extract_domain)

print("\n3. Статистика по доменам:")

unique_domains = http_df['domain'].nunique()
print(f"   Уникальных доменов: {unique_domains}")

top_domains = http_df['domain'].value_counts().head(10)
print("\n   Топ-10 самых посещаемых доменов:")
for domain, count in top_domains.items():
    print(f"     {domain}: {count}")

print("\n4. Проверка на 'unknown':")

unknown_count = (http_df['domain'] == 'unknown').sum()
unknown_percent = unknown_count / len(http_df) * 100
print(f"   Домен 'unknown': {unknown_count} записей ({unknown_percent:.2f}%)")

if unknown_percent > 5:
    print(f"   ВНИМАНИЕ: более 5% доменов не распознаны!")

# 5. ПРОВЕРКА НА ПУСТЫЕ URL
print("\n5. Проверка на пустые URL:")
print("-" * 40)

empty_urls = http_df['url'].isna().sum()
empty_urls_percent = empty_urls / len(http_df) * 100
print(f"   Пустых URL: {empty_urls} записей ({empty_urls_percent:.2f}%)")

print("\n6. Проверка на длинные домены (>50 символов):")

long_domains = http_df[http_df['domain'].str.len() > 50]
print(f"   Длинных доменов (>50 символов): {len(long_domains)}")

# 7. СТАТИСТИКА ПО ПОЛЬЗОВАТЕЛЯМ
print("\n7. Статистика по пользователям:")

# Группируем по пользователю
user_stats = http_df.groupby('user').agg(
    total_requests=('url', 'count'),
    unique_domains=('domain', 'nunique'),
    total_bad=('is_true_bad', 'sum')
).reset_index()

print(f"   Всего пользователей: {len(user_stats)}")
print(f"   Среднее число запросов на пользователя: {user_stats['total_requests'].mean():.1f}")
print(f"   Медиана запросов на пользователя: {user_stats['total_requests'].median():.0f}")
print(f"   Максимум запросов у одного пользователя: {user_stats['total_requests'].max()}")

print("\n8. Пользователи с аномальной активностью:")

top_users = user_stats.nlargest(5, 'total_requests')
print("\n   Топ-5 по числу запросов:")
for _, row in top_users.iterrows():
    print(f"    {row['user']}: {row['total_requests']} запросов, {row['unique_domains']} доменов")

bad_users = user_stats[user_stats['total_bad'] > 0].nlargest(5, 'total_bad')
if len(bad_users) > 0:
    print("\n   Топ-5 по числу вредоносных HTTP-действий:")
    for _, row in bad_users.iterrows():
        print(f"     {row['user']}: {row['total_bad']} плохих действий")
else:
    print("   Нет пользователей с вредоносными HTTP-действиями")

print("\n9. Сохранение результата:")

domain_features = user_stats.copy()
domain_features.to_csv("D:/dataset/domain_features.csv", index=False)

print(f"   Сохранено: D:/dataset/domain_features.csv ({len(domain_features)} записей)")
print(f"   Размер файла: {domain_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("ИТОГИ:")

print(f"""
Обработано записей:   {len(http_df)}
Уникальных доменов:   {unique_domains}
Пользователей:        {len(user_stats)}
'unknown' доменов:    {unknown_count} ({unknown_percent:.2f}%)
Пустых URL:           {empty_urls} ({empty_urls_percent:.2f}%)
Среднее запросов:     {user_stats['total_requests'].mean():.1f}
Вредоносных действий: {user_stats['total_bad'].sum()}
""")

if unknown_percent > 5:
    print("Рекомендация: более 5% доменов не распознаны. Возможно, стоит улучшить функцию extract_domain.")
else:
    print("Все показатели в норме. Можно продолжать.")