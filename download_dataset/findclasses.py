from glob import glob
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool
from itertools import chain

def jobs(worker_num):
    """
    Find all txt files from worker folders, which contain image classes
    """
    
    # worker_num = 0
    path = Path(f"/mnt/d/yolodatasets/ships2/W{worker_num}_1/*.txt")
    return glob(str(path))
    # print(txtfiles)

def openfile(file):
    """
    Open txt file which contains image class, return it with image path
    """
    
    with open(file, "r") as f:
        return f"{f.read().replace('/', '_')},{Path(file).with_suffix('.jpg')}"

def generate():
    """
    Collect all txt files containing image path and class, write them to filtered.txt
    Run through all txt files and find unique classes, write them to a classes.txt
    """
    
    print(f"This system has {cpu_count(logical=False)} cores")
    with Pool(cpu_count(logical=False)) as p:
        txtfiles = p.map(jobs, list(range(0, 100)))

    # print(txtfiles)
    txtfiles = chain.from_iterable(txtfiles)

    with Pool(cpu_count(logical=False)) as p:
        result = p.map(openfile, txtfiles)

    with open("/mnt/d/yolodatasets/ships2/filtered.txt", "w") as all:
        all.write("\n".join(result))

    uniques = [name.split(",")[0] for name in result]
    classes = list(set(uniques))
    print(classes)

    with open("/mnt/d/yolodatasets/ships2/all_classes.txt", "w") as all:
        all.write("\n".join(classes))

# generate()

with open("/mnt/d/yolodatasets/ships2/all_classes.txt", "r") as all:
    classes = all.read().split("\n")
    print(classes)
    [Path(f"/mnt/d/yolodatasets/ships2/{folder}").mkdir(parents=True, exist_ok=True) for folder in classes]

with open("/mnt/d/yolodatasets/ships2/filtered.txt", "r") as all:
    items = all.read().split("\n")

folders, photos = zip(*[i.split(",") for i in items])
# print(Path(photos[0]).parent.parent / folders[0])

# root = Path("/path/to/your/root/dir")

def moveall(item):
    photo, folder = item
    try:
        Path(photo).rename(Path(photo).parent.parent / folder / Path(photo).name)
    except FileNotFoundError:
        print(f"File not found, most likely moved to destination {folder} already: {photo}")

with Pool(cpu_count(logical=False)) as p:
    p.map(moveall, zip(photos, folders))
# [Path(photo).rename(Path(photo).parent.parent / folder / Path(photo).name) for (photo, folder) in zip(photos, folders)]

# for file in root.glob("*.pdf"):
#     folder_name = file.stem.rpartition("_")[-1]
#     file.rename(root / folder_name / file.name)