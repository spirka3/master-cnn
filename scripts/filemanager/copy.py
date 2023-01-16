import glob
import shutil


""""
Create a copy of images with labels to match the pair of image|label
"""

# paths_ref = sorted(glob.glob('/home/janci/Desktop/DIPLO/2. semester/prediction/dataset/BDDA/'))
# src_dir = '/home/janci/Desktop/DIPLO/2. semester/detection/DATASETS/Mapillary'
# dst_dir = '/home/janci/Desktop/DIPLO/2. semester/detection/DATASETS/Mapillary_1%_filter_two_classes'

src_dir = '/home/janci/PycharmProjects/BDD-A/data/inference/hd2s/gazemap_images'
dst_dir = '/home/janci/Desktop/DIPLO/2. semester/samples'

folders = [3, 9, 10, 13, 15, 23, 24, 27, 31, 32, 720, 1950]

for folder in folders:
    folder = str(folder)

    paths_ref = sorted(glob.glob(src_dir + '/' + folder + '/*.jpg'))

    for path in paths_ref:
        filename = path.split('/')[-1][:-4]
        try:
            src = src_dir + '/' + folder + '/' + filename + '.jpg'
            dst = src_dir + '/' + filename + '.jpg'
            print(src, dst)
            shutil.copy(src, dst)
        except:
            ...
            # print(filename)

# for filename in range(312, 50):
#     filename = str(filename)
#     # copy image
#     try:
#         src = src_dir + '/camera_videos/' + filename + '.mp4'
#         dst = dst_dir + '/camera_videos/' + filename + '.mp4'
#         shutil.copy(src, dst)
#         # copy label
#         src = src_dir + '/gazemap_videos/' + filename + '.mp4'
#         dst = dst_dir + '/gazemap_videos/' + filename + '.mp4'
#         shutil.copy(src, dst)
#     except:
#         ...

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
