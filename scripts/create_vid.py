import cv2
import numpy as np
import imageio
import glob

frameSize = (1280, 720)

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 3, frameSize)

src_dir = '/home/janci/PycharmProjects/BDD-A/data/inference'
folder = 'hd2s'
label_paths = sorted(glob.glob(f'{src_dir}/{folder}/sample/*.jpg'))

image_lst = []

for filename in label_paths:
    img = cv2.imread(filename)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_lst.append(frame_rgb)
    out.write(img)

imageio.mimsave('./video.gif', image_lst, fps=10)

out.release()
