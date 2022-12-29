from glob import glob
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool

worker_num = 0
path = Path(f"/mnt/d/yolodatasets/ships2/W{0}_1/*.txt")
txtfiles = glob(str(path))
print(txtfiles)

def openfile(file):
    with open(file, "r") as f:
        return f"{f.read()} {Path(file).stem}.jpg"

print(f"This system has {cpu_count(logical=False)} cores")
with Pool(cpu_count(logical=False)) as p:
    result = p.map(openfile, txtfiles)

with open("/mnt/d/yolodatasets/ships2/filtered.txt", "w") as all:
    all.write("\n".join(result))

uniques = [name.split()[0] for name in result]
items = list(set(uniques))
print(items)
