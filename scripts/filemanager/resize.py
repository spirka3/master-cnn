import glob
import shutil
import cv2

""""
Create a copy of images with labels to match the pair of image|label
"""


src_dir = '/home/janci/PycharmProjects/BDD-A/data/inference/gazemap_images'
paths = sorted(glob.glob(src_dir+'/*.jpg'))
dst_dir = '/home/janci/PycharmProjects/BDD-A/data/inference/gazemap_images_640'

progres = 0
batch = len(paths) // 100
print(f'batch size: {batch}')

for path in paths:
    progres += 1
    if progres % batch == 0:
        print(progres // batch)

    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)

    filename = path.split('/')[-1]

    cv2.imwrite(f'{dst_dir}/{filename}', img)
