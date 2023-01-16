import glob
from scripts.utils import *

"""
Find intersection of a billboard bb with gaze bb
"""

bdda_dir = '/home/janci/PycharmProjects/BDD-A/data/inference'

c_label_paths = sorted(glob.glob(f'{bdda_dir}/detected/camera_labels/*.txt'))
c_image_paths = sorted(glob.glob(f'{bdda_dir}/camera_images/*.jpg'))

print(len(c_image_paths))
print(len(c_label_paths))

folder = 'sage_net'

# filename = '10_02500'
# img_path = f'{bdda_dir}/{folder}/gazemap_images/{filename}.jpg'
# dst_path = f'{bdda_dir}/{folder}/gazemap_labels/{filename}.txt'
# create_gaze_bb(img_path, dst_path, filename, bdda_dir, folder)
# exit()

# create files with bounding boxes for gaze images
for path in c_label_paths:
    filename = path.split('/')[-1][:-4]
    try:
        img_path = f'{bdda_dir}/{folder}/gazemap_images/{filename}.jpg'
        dst_path = f'{bdda_dir}/{folder}/gazemap_labels/{filename}.txt'
        create_gaze_bb(img_path, dst_path, filename, bdda_dir, folder)
    except:
        print('pass')
print('files created')
# exit()

g_label_paths = sorted(glob.glob(f'{bdda_dir}/{folder}/gazemap_labels/*.txt'))
g_images_dir = f'{bdda_dir}/{folder}/gazemap_images'
img_paths = f'{bdda_dir}/detected/camera_images'
dst_path = f'{bdda_dir}/{folder}/intersection'
draw_interactions_bb(g_label_paths, c_label_paths, img_paths, dst_path, g_images_dir)
