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
http_df = pd.read_csv("D:/dataset/labels/http_labeled.csv", usecols=['user', 'date', 'url', 'is_true_bad'])
print(f"   Всего записей: {len(http_df)}")

print("\n2. Извлечение доменов")

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

print("\n5. Проверка на пустые URL:")

empty_urls = http_df['url'].isna().sum()
empty_urls_percent = empty_urls / len(http_df) * 100
print(f"   Пустых URL: {empty_urls} записей ({empty_urls_percent:.2f}%)")

print("\n6. Проверка на длинные домены (>50 символов):")

long_domains = http_df[http_df['domain'].str.len() > 50]
print(f"   Длинных доменов (>50 символов): {len(long_domains)}")

print("\n7. Статистика по пользователям:")

http_df['parsed_date'] = pd.to_datetime(http_df['date'], format='%m/%d/%Y %H:%M:%S')
http_df['day'] = http_df['parsed_date'].dt.date

http_df_sorted = http_df.sort_values(['user', 'parsed_date']).reset_index(drop=True)

result_rows = []

for user, group in http_df_sorted.groupby('user'):
    cumulative_requests = 0
    seen_domains = set()

    for day, day_group in group.groupby('day', sort=True):
        cumulative_requests += len(day_group)
        seen_domains.update(day_group['domain'].unique())

        result_rows.append({
            'user': user,
            'day': day,
            'total_requests_until_t': cumulative_requests,
            'unique_domains_until_t': len(seen_domains),
        })

user_stats = pd.DataFrame(result_rows)

print(f"   Всего записей (пользователь-день): {len(user_stats)}")
print(f"   Уникальных пользователей: {user_stats['user'].nunique()}")
print(f"   Среднее запросов на конец периода: {user_stats.groupby('user')['total_requests_until_t'].max().mean():.1f}")

print("\n8. Пользователи с аномальной активностью:")

last_day_per_user = user_stats.groupby('user').last().reset_index()
top_users = last_day_per_user.nlargest(5, 'total_requests_until_t')

print("\n   Топ-5 по числу запросов (на конец периода):")
for _, row in top_users.iterrows():
    print(f"    {row['user']}: {row['total_requests_until_t']} запросов, {row['unique_domains_until_t']} доменов")

print("\n9. Сохранение результата:")

domain_features = user_stats.copy()
domain_features.to_csv("D:/dataset/domain_features.csv", index=False)

print(f"   Сохранено: D:/dataset/domain_features.csv ({len(domain_features)} записей)")
print(f"   Размер файла: {domain_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("ИТОГИ:")

print(f"""
Обработано записей:   {len(http_df)}
Уникальных доменов:   {unique_domains}
Пользователей:        {user_stats['user'].nunique()}
Записей (user-day):   {len(user_stats)}
'unknown' доменов:    {unknown_count} ({unknown_percent:.2f}%)
Пустых URL:           {empty_urls} ({empty_urls_percent:.2f}%)
""")

if unknown_percent > 5:
    print("Рекомендация: более 5% доменов не распознаны. Возможно, стоит улучшить функцию extract_domain.")
else:
    print("Все показатели в норме. Можно продолжать.")