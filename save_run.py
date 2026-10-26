# save_run.py
import os
import json
import hashlib
import shutil
from datetime import datetime

def make_run_dir(model_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = "D:/dataset/runs/" + timestamp + "_" + model_name

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(run_dir + "/predictions"):
        os.makedirs(run_dir + "/predictions")
    if not os.path.exists(run_dir + "/plots"):
        os.makedirs(run_dir + "/plots")

    print("Папка прогона: " + run_dir)
    return run_dir


def file_hash(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def save_config(run_dir, config):
    with open(run_dir + "/config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False, default=str)
    print("Сохранён: " + run_dir + "/config.json")


def save_versions(run_dir, data_files, code_files):
    lines = []

    lines.append("=== ДАННЫЕ ===")
    for name, filepath in data_files.items():
        if os.path.exists(filepath):
            lines.append(name + ": " + file_hash(filepath))
        else:
            lines.append(name + ": НЕ НАЙДЕН")

    lines.append("")
    lines.append("КОД:")
    for name, filepath in code_files.items():
        if os.path.exists(filepath):
            lines.append(name + ": " + file_hash(filepath))
        else:
            lines.append(name + ": НЕ НАЙДЕН")

    with open(run_dir + "/versions.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("Сохранён: " + run_dir + "/versions.txt")


def save_split(run_dir, train_users, val_users, test_users):
    data = {
        "train_users": list(train_users),
        "val_users": list(val_users),
        "test_users": list(test_users),
    }
    with open(run_dir + "/split.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("Сохранён: " + run_dir + "/split.json")


def save_predictions(run_dir, name, df):
    path = run_dir + "/predictions/" + name + ".parquet"
    df.to_parquet(path, index=False)
    print("Сохранён: " + path)


def save_metrics(run_dir, metrics):
    with open(run_dir + "/metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False, default=str)
    print("Сохранён: " + run_dir + "/metrics.json")


def save_threshold(run_dir, threshold_info):
    with open(run_dir + "/threshold.json", 'w', encoding='utf-8') as f:
        json.dump(threshold_info, f, indent=4, ensure_ascii=False, default=str)
    print("Сохранён: " + run_dir + "/threshold.json")


def copy_file(run_dir, filepath, subfolder=""):
    if not os.path.exists(filepath):
        print("Не найден: " + filepath)
        return

    if subfolder:
        dest = run_dir + "/" + subfolder + "/" + os.path.basename(filepath)
    else:
        dest = run_dir + "/" + os.path.basename(filepath)

    shutil.copy(filepath, dest)
    print("Сохранён: " + dest)