from pathlib import Path

"""
Split the openwebtext dataset into two files, which allow lower memory usage.
"""

data_file = Path('openwebtext_dataset.txt')
output_file1 = Path('openwebtext_dataset.txt')
output_file2 = Path('openwebtext_dataset2.txt')

if data_file.exists():
    with data_file.open('r') as f:
        lines = f.readlines()
    with output_file1.open('w') as f1:
        f1.writelines(lines[:len(lines)//2])
    with output_file2.open('w') as f2:
        f2.writelines(lines[len(lines)//2:])
else:
    print(f"{data_file} does not exist")