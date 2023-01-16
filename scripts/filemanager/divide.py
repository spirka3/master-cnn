import glob
import shutil

"""
Divide images in file into multiple subfiles
"""

dir_base = '/home/janci/Desktop/DIPLO/2. semester/detection/dataset/Billboards'

paths = sorted(glob.glob(f'{dir_base}/images/*.jpg'))


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]

parts = split_list(paths, wanted_parts=4)

for i, paths in enumerate(parts):
    for path in paths:
        filename = path.split('/')[-1][:-4]
        # copy image
        src = dir_base + '/images/' + filename + '.jpg'
        dst = dir_base + f'/{i}/images/' + filename + '.jpg'
        shutil.copy(src, dst)
        # copy label
        src = dir_base + '/labels/' + filename + '.txt'
        dst = dir_base + f'/{i}/labels/' + filename + '.txt'
        shutil.copy(src, dst)
