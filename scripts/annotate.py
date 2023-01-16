"""
Tento skript slúži na vygenerovanie anotácii objektov v obrázkoch v Mapillary Vistas Datasete
do textových súborov vo formáte YOLO
Výstupom tohto scriptu je, že sa do výstupného súboru 'camera_labels' pre každý obrázok vytvorí
textový súbor s anotáciami objektov vo formáte YOLO.
"""
import shutil
import time

import json
import numpy as np
import os
from PIL import Image

dir_base = '/home/janci/Desktop/DIPLO/2. semester/detection/Mapillary2/training'
sub_folder = 'train_dataset'
image_paths = os.listdir(f'{dir_base}/images')

_1 = 0
_5 = 0
progres = 0
batch_start = time.time()
batch = len(image_paths) // 100
print(f'batch size: {batch}')

# read in panoptic file
with open(f'{dir_base}/v1.2/panoptic/panoptic_2018.json') as panoptic_file:
    panoptic = json.load(panoptic_file)

# convert annotation infos to image_id indexed dictionary
panoptic_per_image_id = {}
for annotation in panoptic['annotations']:
    panoptic_per_image_id[annotation['image_id']] = annotation

# convert category infos to category_id indexed dictionary
panoptic_category_per_id = {}
for category in panoptic['categories']:
    panoptic_category_per_id[category['id']] = category

for image_id in image_paths:
    image_id = image_id[:-4]

    progres += 1
    if progres % batch == 0:
        print(f'{progres // batch}% (in {time.time() - batch_start}s)')
        print('1%', _1)
        print('0.5%', _5)
        batch_start = time.time()

    # set up paths for every image
    image_path = f'{dir_base}/images/{image_id}.jpg'

    panoptic_path = f'{dir_base}/v1.2/panoptic/{image_id}.png'
    panoptic_image = Image.open(panoptic_path)

    # convert segment infos to segment id indexed dictionary
    example_panoptic = panoptic_per_image_id[image_id]
    example_segments = {}
    for segment_info in example_panoptic['segments_info']:
        example_segments[segment_info['id']] = segment_info

    img = Image.open(image_path)
    dw, dh = img.size

    panoptic_array = np.array(panoptic_image).astype(np.uint32)
    panoptic_id_array = panoptic_array[:, :, 0] + (2 ** 8) * panoptic_array[:, :, 1] + (2 ** 16) * panoptic_array[:, :, 2]
    panoptic_ids_from_image = np.unique(panoptic_id_array)

    # find suitable images
    _001 = ''
    _0005 = ''
    for panoptic_id in panoptic_ids_from_image:
        if panoptic_id == 0:
            # void image areas don't have segments
            continue
        segment_info = example_segments[panoptic_id]
        category = panoptic_category_per_id[segment_info['category_id']]
        if category['supercategory'] == 'object--banner' or category['supercategory'] == 'object--billboard':
            ##            print('segment {:8d}: label {:<40}, area {:6d}, bbox {}'.format(
            ##            panoptic_id,
            ##            category['supercategory'],
            ##            segment_info['area'],
            ##            segment_info['bbox'],
            ##            ))
            ##            print(0, segment_info['bbox'][0]/dw, segment_info['bbox'][1]/dh, segment_info['bbox'][2]/dw, segment_info['bbox'][3]/dh)
            centerx = (segment_info['bbox'][0] + (segment_info['bbox'][2] / 2)) / dw
            centery = (segment_info['bbox'][1] + (segment_info['bbox'][3] / 2)) / dh
            w = segment_info['bbox'][2] / dw
            h = segment_info['bbox'][3] / dh
            test_w = float(w) * float(dw)
            test_h = float(h) * float(dh)
            if (test_w * test_h) / (dw * dh) > 0.005:
                line = 'Ad ' + str(centerx) + ' ' + str(centery) + ' ' + str(w) + ' ' + str(h) + '\n'
                _0005 += line

                if (test_w * test_h) / (dw * dh) > 0.01:
                    _001 += line

        example_segments.pop(panoptic_id)

    if _001 != '':
        _1 += 1
        # write labels into file
        with open(f'{dir_base}/{sub_folder}/1%/labels/{image_id}.txt', 'w') as file001:
            file001.write(_001)
        src = f'{dir_base}/images/{image_id}.jpg'
        dst = f'{dir_base}/{sub_folder}/1%/images/{image_id}.jpg'
        shutil.copy(src, dst)

    if _0005 != '':
        _5 += 1
        with open(f'{dir_base}/{sub_folder}/0.5%/labels/{image_id}.txt', 'w') as file0005:
            file0005.write(_0005)
            src = f'{dir_base}/images/{image_id}.jpg'
            dst = f'{dir_base}/{sub_folder}/0.5%/images/{image_id}.jpg'
            shutil.copy(src, dst)

    assert len(example_segments) == 0