import glob
import os
import shutil


""""
Create a copy of images with labels to match the pair of image|label
"""

# paths_ref = sorted(glob.glob('/home/janci/Desktop/DIPLO/2. semester/prediction/dataset/BDDA/'))
# src_dir = '/home/janci/Desktop/DIPLO/2. semester/detection/DATASETS/Mapillary'
# dst_dir = '/home/janci/Desktop/DIPLO/2. semester/detection/DATASETS/Mapillary_1%_filter_two_classes'

src_dir = '/home/janci/PycharmProjects/BDD-A/data/inference/camera_images'
dst_dir = '/home/janci/PycharmProjects/BDD-A/data/inference'
paths_ref = sorted(glob.glob(src_dir + '/*.jpg'))


for path in paths_ref:
    filename = path.split('/')[-1]
    vid = filename.split('_')[0]
    print(filename)
    print(vid)
    ouput_file = dst_dir + '/' + vid
    if not os.path.isdir(ouput_file):
        os.makedirs(ouput_file)
    # copy image
    try:
        src = src_dir + '/' + filename
        dst = ouput_file + '/' + filename
        shutil.copy(src, dst)
    except:
        ...

# for path in paths_ref:
#     filename = path.split('/')[-1][:-4]
#     # copy image
#     src = src_dir + '/images/' + filename + '.jpg'
#     dst = dst_dir + '/images/' + filename + '.jpg'
#     shutil.copy(src, dst)
#     # copy label
#     src = src_dir + '/labels/' + filename + '.txt'
#     dst = dst_dir + '/labels/' + filename + '.txt'
#     shutil.copy(src, dst)
