from scripts.metrics import *
from scripts.utils import *
from os import path
from glob import glob
from tqdm import tqdm

src_dir = '/home/janci/PycharmProjects/BDD-A/data/inference'
folder = 'tased_net'

label_paths = sorted(glob(f'{src_dir}/detected/camera_labels/*.txt'))

count = auc_judd_score = nss_score = sim_score = cc_score = 0

visualize = False
benchmark = True  # 1.75s/it

for label_path in tqdm(label_paths, desc=f"progres"):
    filename = label_path.split('/')[-1][:-4]
    img_path = f'{src_dir}/detected/camera_images/{filename}.jpg'
    ggt_path = f'{src_dir}/bdda_gt/gazemap_images/{filename}.jpg'

    gze_path = f'{src_dir}/{folder}/gazemap_images/{filename}.jpg'
    dst_path = f'{src_dir}/{folder}/intersection/{filename}.jpg'

    if not path.exists(gze_path):
        continue

    # has_iou, iou_data = get_iou(img_path, label_path, gze_path)
    has_iou, iou_data = path.exists(dst_path), []

    if not has_iou:
        continue

    # if visualize:
    #     visualizer(iou_data, dst_path)

    if benchmark:
        gt = cv2.imread(ggt_path, 0)

        s_map = cv2.imread(gze_path, 0)
        s_map = cv2.resize(s_map, (gt.shape[1], gt.shape[0]))

        sal_norm = normalize_map(s_map)

        auc_judd_score += auc_judd(sal_norm, gt)
        nss_score += nss(s_map, gt)
        sim_score += similarity(s_map, gt)
        cc_score += cc(s_map, gt)
        count += 1

auc_judd_score = round(auc_judd_score / count, 4)
nss_score = round(nss_score / count, 4)
sim_score = round(sim_score / count, 4)
cc_score = round(cc_score / count, 4)

with open(f'{src_dir}/{folder}/metrics.txt', 'w') as f:
    print('auc judd :', auc_judd_score, file=f)
    print('auc judd :', auc_judd_score)
    print('nss :', nss_score, file=f)
    print('nss :', nss_score)
    print('sim score :', sim_score, file=f)
    print('sim score :', sim_score)
    print('cc score :', cc_score, file=f)
    print('cc score :', cc_score)
    print('count :', count, file=f)
    print('count :', count)


