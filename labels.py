# labels.py

import time
import os

def load_all_true_bad_ids(answers_path):
    print("\nВыгружаем точные метки инсайдеров: их конкретные действия")
    
    true_bad_ids = {}
    
    for folder in ['r4.2-1', 'r4.2-2', 'r4.2-3']:
        folder_path = os.path.join(answers_path, folder)
        if not os.path.exists(folder_path):
            continue
        
        print(f"\nОбработка {folder}:")
    
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"  Найдено файлов: {len(files)}")
        
        for file_name in files:
            user = file_name.split('-')[-1].replace('.csv', '')
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if user not in true_bad_ids:
                true_bad_ids[user] = {}
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                
                action_type = parts[0].lower()
                action_id = parts[1].strip('"')
                
                if action_type not in true_bad_ids[user]:
                    true_bad_ids[user][action_type] = set()
                true_bad_ids[user][action_type].add(action_id)
    
    print("\nИтог:")
    
    type_counts = {'logon':0, 'device':0, 'http':0, 'email':0, 'file':0}
    for user, types in true_bad_ids.items():
        for atype, ids in types.items():
            if atype in type_counts:
                type_counts[atype] += len(ids)
    
    for atype, count in type_counts.items():
        if count > 0:
            print(f"  {atype}: {count} точных меток")
    
    print(f"\nВсего пользователей с метками: {len(true_bad_ids)}")
    
    return true_bad_ids

def add_labels_to_logon(input_file, output_file, all_bad_ids):
    print("ДОБАВЛЕНИЕ МЕТОК К LOGON.CSV")
    start_time = time.time()
    
    total_lines = 0
    bad_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        header = f_in.readline().strip()
        new_header = header + ',is_true_bad'
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(new_header + '\n')
            
            for line in f_in:
                total_lines += 1
                
                if total_lines % 100000 == 0:
                    print(f"  Обработано {total_lines} строк...")
                
                parts = line.strip().split(',')
                if len(parts) < 1:
                    f_out.write(line.strip() + ',0\n')
                    continue
                
                action_id = parts[0].strip('"')

                is_bad = 1 if action_id in all_bad_ids else 0
                
                if is_bad:
                    bad_count += 1
                
                f_out.write(line.strip() + f',{is_bad}\n')
    
    print(f"\nОбработано строк: {total_lines}")
    print(f"Найдено плохих действий: {bad_count}")
    print(f"Время: {time.time() - start_time} сек")
    
    return total_lines, bad_count

def add_labels_to_device(input_file, output_file, all_bad_ids):
    print("ДОБАВЛЕНИЕ МЕТОК К DEVICE.CSV (по ID)")
    start_time = time.time()
    
    total_lines = 0
    bad_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        header = f_in.readline().strip()
        new_header = header + ',is_true_bad'
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(new_header + '\n')
            
            for line in f_in:
                total_lines += 1
                
                if total_lines % 50000 == 0:
                    print(f"  Обработано {total_lines:,} строк...")
                
                parts = line.strip().split(',')
                if len(parts) < 1:
                    f_out.write(line.strip() + ',0\n')
                    continue
                
                action_id = parts[0].strip('"')
                
                is_bad = 1 if action_id in all_bad_ids else 0
                
                if is_bad:
                    bad_count += 1
                
                f_out.write(line.strip() + f',{is_bad}\n')
    
    print(f"\nОбработано строк: {total_lines}")
    print(f"Найдено плохих действий: {bad_count}")
    print(f"Время: {time.time() - start_time} сек")
    
    return total_lines, bad_count

def add_labels_to_http(input_file, output_file, all_bad_ids):
    print("ДОБАВЛЕНИЕ МЕТОК К HTTP.CSV (по ID)")
    start_time = time.time()
    
    total_lines = 0
    bad_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        header = f_in.readline().strip()
        new_header = header + ',is_true_bad'
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(new_header + '\n')
            
            for line in f_in:
                total_lines += 1
                
                if total_lines % 1000000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Обработано {total_lines:,} строк... ({elapsed:.1f} сек)")
                
                parts = line.strip().split(',')
                if len(parts) < 1:
                    f_out.write(line.strip() + ',0\n')
                    continue
                
                action_id = parts[0].strip('"')
                
                is_bad = 1 if action_id in all_bad_ids else 0
                
                if is_bad:
                    bad_count += 1
                
                f_out.write(line.strip() + f',{is_bad}\n')
    
    print(f"\nОбработано строк: {total_lines:,}")
    print(f"Найдено плохих действий: {bad_count}")
    print(f"Время: {time.time() - start_time:.1f} сек")
    
    return total_lines, bad_count

def add_labels_to_email(input_file, output_file, all_bad_ids):
    print("ДОБАВЛЕНИЕ МЕТОК К EMAIL.CSV")
    start_time = time.time()
    
    total_lines = 0
    bad_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        header = f_in.readline().strip()
        new_header = header + ',is_true_bad'
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(new_header + '\n')
            
            for line in f_in:
                total_lines += 1
                
                if total_lines % 500000 == 0:
                    print(f"  Обработано {total_lines:,} строк...")
                
                parts = line.strip().split(',')
                if len(parts) < 1:
                    f_out.write(line.strip() + ',0\n')
                    continue
                
                action_id = parts[0].strip('"')
                
                is_bad = 1 if action_id in all_bad_ids else 0
                
                if is_bad:
                    bad_count += 1
                
                f_out.write(line.strip() + f',{is_bad}\n')
    
    print(f"\nОбработано строк: {total_lines:,}")
    print(f"Найдено плохих действий: {bad_count}")
    print(f"Время: {time.time() - start_time:.1f} сек")
    
    return total_lines, bad_count

def add_labels_to_file(input_file, output_file, all_bad_ids):

    print("ДОБАВЛЕНИЕ МЕТОК К FILE.CSV")
    start_time = time.time()
    
    total_lines = 0
    bad_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        header = f_in.readline().strip()
        new_header = header + ',is_true_bad'
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(new_header + '\n')
            
            for line in f_in:
                total_lines += 1
                
                if total_lines % 50000 == 0:
                    print(f"  Обработано {total_lines:,} строк...")
                
                parts = line.strip().split(',')
                if len(parts) < 1:
                    f_out.write(line.strip() + ',0\n')
                    continue
                
                action_id = parts[0].strip('"')

                is_bad = 1 if action_id in all_bad_ids else 0
                
                if is_bad:
                    bad_count += 1
                
                f_out.write(line.strip() + f',{is_bad}\n')
    
    print(f"\nОбработано строк: {total_lines:,}")
    print(f" Найдено плохих действий: {bad_count}")
    print(f" Время: {time.time() - start_time:.1f} сек")
    
    return total_lines, bad_count

if __name__ == '__main__':
    input_path = "D:/НИРС/cert4.2/r4.2"
    answers_path = "D:/НИРС/answers/"
    output_path = "D:/dataset/labels/"

    true_bad_ids = load_all_true_bad_ids(answers_path)

    all_bad_ids = set()

    for user, types in true_bad_ids.items():
        for action_type, ids in types.items():
            all_bad_ids.update(ids)

    print(f"\nВсего уникальных плохих ID: {len(all_bad_ids)}")

    # logon.csv
    total, bad = add_labels_to_logon(os.path.join(input_path, "logon.csv"), os.path.join(output_path, "logon_labeled.csv"), all_bad_ids)
    print(f"\nРезультат: {bad} из {total} действий помечены как плохие")

    # device.csv
    total, bad = add_labels_to_device(os.path.join(input_path, "device.csv"), os.path.join(output_path, "device_labeled.csv"), all_bad_ids)
    print(f"\nРезультат: {bad} из {total} действий помечены как плохие")

    # http.csv
    total, bad = add_labels_to_http(os.path.join(input_path, "http.csv"), os.path.join(output_path, "http_labeled.csv"), all_bad_ids)
    print(f"\nРезультат: {bad} из {total} действий помечены как плохие")

    # email.csv
    total, bad = add_labels_to_email(os.path.join(input_path, "email.csv"), os.path.join(output_path, "email_labeled.csv"), all_bad_ids)
    print(f"\nРезультат: {bad} из {total} действий помечены как плохие")

    # file.csv
    total, bad = add_labels_to_file(os.path.join(input_path, "file.csv"), os.path.join(output_path, "file_labeled.csv"), all_bad_ids)
    print(f"\nРезультат: {bad} из {total} действий помечены как плохие")