import numpy as np
import cv2
import torch

W_DIMENSION = 1280
H_DIMENSION = 720


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def calc_iou(mask1, mask2):
    mask2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    return intersection / mask2_area


def get_iou(img_path, lbl_path, gze_path, dst_path):
    write = False
    img = cv2.imread(img_path)
    gze_img = cv2.imread(gze_path)
    gze_img = cv2.resize(gze_img, (1280, 720))

    thresh = cv2.threshold(gze_img, 30, 255, cv2.THRESH_BINARY)[1]
    mask1 = np.asarray(thresh)

    with open(lbl_path, 'r') as file:
        lines = file.read().split('\n')
        for line in lines:
            data = line.split(' ')
            if len(data) > 1:
                c, x, y, w, h, conf = data
                w = float(w) * W_DIMENSION
                h = float(h) * H_DIMENSION
                x = float(x) * W_DIMENSION - w / 2
                y = float(y) * H_DIMENSION - h / 2
                w = int(w)
                h = int(h)
                x = int(x)
                y = int(y)
                conf = conf[:4]

                # create mask from label
                black_img = cv2.imread('./bg.jpg')
                cv2.rectangle(black_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                black_img = cv2.threshold(black_img, 0, 255, cv2.THRESH_BINARY)[1]
                # cv2.imwrite(f'./{filename}_gz.jpg', thresh)

                # find iou with gze mask
                mask2 = np.asarray(black_img)

                iou = calc_iou(mask1, mask2)
                if iou > 0.1:
                    write = True
                    draw_label(img, (x, y, w, h, conf))
        if write:
            src2 = gze_img

            heatmap_img = cv2.applyColorMap(src2, cv2.COLORMAP_JET)
            hsv = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2HSV)

            # Define lower and uppper limits of what we call "brown"
            brown_lo = np.array([120, 120, 0])
            brown_hi = np.array([255, 255, 240])

            # Mask image to only select browns
            mask = cv2.inRange(hsv, brown_lo, brown_hi)

            # Change image to red where we found brown
            heatmap_img[mask > 0] = (0, 0, 0)
            heatmap_img = cv2.blur(heatmap_img, (25, 25))

            super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.9, 0.0)

            cv2.imwrite(dst_path, super_imposed_img)

    return write


def create_gaze_bb(img_path, dst_path, filename, bdda_dir, folder):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1280, 720))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    f = open(dst_path, "w")

    gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
        xyxy = torch.Tensor([x, y, x + w, y + h])
        xywh = (xyxy2xywh(xyxy.clone().detach().view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        x, y, w, h = xywh
        f.write(' '.join(['1', str(x), str(y), str(w), str(h), '1\n']))
        # cropped_image = img[80:280, 150:330]

    # cv2.imwrite(f'./output_images/ads/{filename}.jpg', img)
    f.close()


def compute_iou(bb_1, bb_2):
    x1, y1, w1, h1, conf = bb_1
    x2, y2, w2, h2, conf = bb_2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (x1 + w1 - x1) * (y1 + h1 - y1)
    bb2_area = (x2 + w2 - x2) * (y2 + h2 - y2)

    return intersection_area / bb2_area


def get_bb_map(paths):
    result_map = dict()
    for path in paths:
        filename = path.split('/')[-1][:-4]
        result_map[filename] = []
        with open(path, 'r') as file:
            lines = file.read().split('\n')
            for line in lines:
                data = line.split(' ')
                if len(data) > 1:
                    cls, x, y, w, h, conf = data
                    w = float(w) * W_DIMENSION
                    h = float(h) * H_DIMENSION
                    x = float(x) * W_DIMENSION - w / 2
                    y = float(y) * H_DIMENSION - h / 2
                    conf = conf[:4]
                    result_map[filename].append((int(x), int(y), int(w), int(h), conf))
    return result_map

def draw_text(img, text,
          font=cv2.FONT_ITALIC,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 255, 0)
          ):

    padding = 4
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    x, y = pos

    if y < text_h:
        y = y + text_h + padding

    cv2.rectangle(img, (x, y), (x + text_w + padding, y - text_h - 2*padding), text_color_bg, -1)
    cv2.putText(img, text, (x, y + font_scale - 1), font, font_scale, text_color, font_thickness)

    # return text_size


def draw_label(img, bb, color=(36, 255, 12), border=4):
    x1, y1, w, h, conf = bb
    x2 = x1 + w
    y2 = y1 + h
    cv2.rectangle(img, (x1, y1), (x2, y2), color, border)
    draw_text(img, f'Ad {conf}', pos=(x1 - border, y1))



# def draw_label(img, bb, color, border, text=False):
#     x1, y1, w, h, conf = bb
#     x2 = x1 + w
#     y2 = y1 + h
#     cv2.rectangle(img, (x1, y1), (x2, y2), color, border)
#     if text:
#         draw_text(img, f'Ad {conf}', pos=(x1 - border, y1))


def draw_interactions_bb(g_label_paths, c_label_paths, c_src_dir, dst_dir, g_images_dir):
    g_map = get_bb_map(g_label_paths)
    c_map = get_bb_map(c_label_paths)
    i = 0
    for filename in g_map.keys():
        target = f'{dst_dir}/{filename}.jpg'
        img = cv2.imread(f'{c_src_dir}/{filename}.jpg')
        write = False
        for g_bb in g_map[filename]:
            for c_bb in c_map[filename]:
                iou = compute_iou(g_bb, c_bb)
                if iou > 0.05:
                    write = True
                    draw_label(img, c_bb, (35, 255, 12), 3, True)
                    draw_label(img, g_bb, (35, 255, 12), 1)
        if write:
            i += 1

            src2 = cv2.imread(f'{g_images_dir}/{filename}.jpg')
            src2 = cv2.resize(src2, (1280, 720))

            heatmap_img = cv2.applyColorMap(src2, cv2.COLORMAP_JET)
            hsv = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2HSV)

            # Define lower and uppper limits of what we call "brown"
            brown_lo = np.array([120, 120, 0])
            brown_hi = np.array([255, 255, 240])

            # Mask image to only select browns
            mask = cv2.inRange(hsv, brown_lo, brown_hi)

            # Change image to red where we found brown
            heatmap_img[mask > 0] = (0, 0, 0)
            heatmap_img = cv2.blur(heatmap_img, (25, 25))

            super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.9, 0.0)

            cv2.imwrite(target, super_imposed_img)

        # if i == 5:
        #     return


def draw_advertisements_bb(labels, image_paths):
    for i in range(len(labels)):
        bb = labels[i]
        if bb == 0.0:
            continue

        image_path = image_paths[i]
        image_name = image_path.split('/')[-1]
        target = 'draw/advertisements/' + image_name

        img = cv2.imread(image_path)

        x1, y1, x2, y2 = bb
        cv2.rectangle(img, (x1, y1), (x2, y2), (12, 35, 255), 2)
        # TODO confidence

        cv2.imwrite(target, img)
