import glob
import shutil


""""
Create a copy of images with labels to match the pair of image|label
"""

dir = '/home/janci/Desktop/DIPLO/2. semester/prediction/dataset/'
folder = 'val'

image_paths_ref = sorted(glob.glob(dir + f'BDDA/{folder}/camera_videos/*.mp4'))
image_src_dir = dir + f'BDDA/{folder}/camera_videos/'
gaze_src_dir = dir + f'BDDA/{folder}/gazemap_videos/'

gaze_paths_ref = sorted(glob.glob(dir + f'BDDA/{folder}/gazemap_videos/*.mp4'))
image_dst_dir = dir + f'copy/{folder}/camera_videos/'
gaze_dst_dir = dir + f'copy/{folder}/gazemap_videos/'

print(len(image_paths_ref))
print(len(gaze_paths_ref))

if len(image_paths_ref) < len(gaze_paths_ref):
    for path in gaze_paths_ref:
        g_filename = path.split('/')[-1].split('.')[0]
        c_filename = g_filename.split('_')[0]
        if f'{image_src_dir}{c_filename}.mp4' in image_paths_ref:
            # copy image
            src = image_src_dir + c_filename + '.mp4'
            dst = image_dst_dir + c_filename + '.mp4'
            shutil.copy(src, dst)
            # copy label
            src = gaze_src_dir + g_filename + '.mp4'
            dst = gaze_dst_dir + c_filename + '.mp4'
            shutil.copy(src, dst)
else:
    for path in image_paths_ref:
        c_filename = path.split('/')[-1].split('.')[0]
        g_filename = c_filename + '_pure_hm'
        if f'{gaze_src_dir}{g_filename}.mp4' in gaze_paths_ref:
            # copy image
            src = image_src_dir + c_filename + '.mp4'
            dst = image_dst_dir + c_filename + '.mp4'
            shutil.copy(src, dst)
            # copy label
            src = gaze_src_dir + g_filename + '.mp4'
            dst = gaze_dst_dir + c_filename + '.mp4'
            shutil.copy(src, dst)

