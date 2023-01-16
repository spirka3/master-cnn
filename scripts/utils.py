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


def visualizer(iou_results, dst_path):
    image, gazemap, bbs = iou_results

    heatmap = cv2.applyColorMap(gazemap, cv2.COLORMAP_JET)
    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([120, 120, 0])
    brown_hi = np.array([255, 255, 240])

    # Mask image to only select browns
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    # Change image to red where we found brown
    heatmap[mask > 0] = (0, 0, 0)
    heatmap = cv2.blur(heatmap, (25, 25))

    # Recolor object bounding box
    for bb in bbs:
        draw_label(image, *bb)

    composed_img = cv2.addWeighted(heatmap, 0.5, image, 0.9, 0.0)
    cv2.imwrite(dst_path, composed_img)


def calc_iou(mask1, mask2):
    mask2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    return intersection / mask2_area


def prep_data(data):
    c, x, y, w, h, conf = data
    w = float(w) * W_DIMENSION
    h = float(h) * H_DIMENSION
    x = float(x) * W_DIMENSION - w / 2
    y = float(y) * H_DIMENSION - h / 2
    return int(x), int(y), int(w), int(h), conf[:4]


def get_iou(img_path, lbl_path, gze_path):
    img = cv2.imread(img_path)
    gze_img = cv2.imread(gze_path)
    gze_img = cv2.resize(gze_img, (1280, 720))
    thresh = cv2.threshold(gze_img, 30, 255, cv2.THRESH_BINARY)[1]

    mask1 = np.asarray(thresh)
    results = [img, gze_img, []]

    with open(lbl_path, 'r') as file:
        lines = file.read().split('\n')
        for line in lines:
            data = line.split(' ')
            if len(data) > 1:
                x, y, w, h, conf = prep_data(data)

                # create mask from label
                black_img = cv2.imread('./bg.jpg')
                cv2.rectangle(black_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                # black_img = cv2.threshold(black_img, 0, 255, cv2.THRESH_BINARY)[1]

                mask2 = np.asarray(black_img)

                if calc_iou(mask1, mask2) > 0.1:
                    results[2].append((x, y, w, h, conf))

    return results, len(results[2]) > 0


def draw_text(img,
              text,
              font=cv2.FONT_ITALIC,
              pos=(0, 0),
              font_scale=1,
              font_thickness=2,
              text_color=(255, 255, 255),
              text_color_bg=(0, 255, 0),
              padding=4
              ):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    x, y = pos
    if y < text_h:
        y = y + text_h + padding

    cv2.rectangle(img, (x, y), (x + text_w + padding, y - text_h - 2 * padding), text_color_bg, -1)
    cv2.putText(img, text, (x, y + font_scale - 1), font, font_scale, text_color, font_thickness)

    # return text_size


def draw_label(img, bb, color=(36, 255, 12), border=4):
    x1, y1, w, h, conf = bb
    x2 = x1 + w
    y2 = y1 + h
    cv2.rectangle(img, (x1, y1), (x2, y2), color, border)
    draw_text(img, f'Ad {conf}', pos=(x1 - border, y1), text_color_bg=color, padding=border)
