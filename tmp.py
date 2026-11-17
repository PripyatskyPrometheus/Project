import re
import sys

def strip_log_format(input_path: str, output_path: str) -> None:
    """
    Убирает префикс формата лога вида:
        "2026-11-16 12:18:26 | INFO | "
    и сохраняет только само сообщение.
    Пустые строки после обрезки сохраняются как пустые.
    """
    # Префикс: дата время | LEVEL | (возможно, с ведущими пробелами)
    pattern = re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*\|\s*[A-Z]+\s*\|\s?')

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(pattern.sub('', line))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: python {sys.argv[0]} <input.txt> <output.txt>')
        sys.exit(1)
    strip_log_format(sys.argv[1], sys.argv[2])