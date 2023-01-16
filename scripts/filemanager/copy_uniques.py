import glob
import shutil

# src_dir = 'my_own_dataset/b1/'
dst_dir = 'my_own_dataset/b1/'

for x in range(6):
    src_dir = f'my_own_dataset/b1/{x}/'
    paths = sorted(glob.glob(f'my_own_dataset/b1/{x}/images/*.jpg'))
    for i, path in enumerate(paths):
        filename = path.split('/')[-1].split('.')[0:-1]
        prefix = filename[0]
        filename = '.'.join(filename)
        if i + 1 < len(paths) - 1:
            c_filename = paths[i+1].split('/')[-1].split('.')[0:-1]
            c_prefix = c_filename[0]
            if c_prefix == prefix:
                continue
        # copy image
        src = src_dir + '/images/' + filename + '.jpg'
        dst = dst_dir + '/images/' + filename + '.jpg'
        shutil.copy(src, dst)
        # copy label
        src = src_dir + '/labels/' + filename + '.txt'
        dst = dst_dir + '/labels/' + filename + '.txt'
        shutil.copy(src, dst)
