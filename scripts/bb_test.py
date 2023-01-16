import cv2

img = cv2.imread('./1.jpg')
img = cv2.resize(img, (1280, 720))
dimensions = img.shape
dh, dw, ch = dimensions
with open('./1.txt', 'r') as file:
    lines = file.read().split('\n')
    for line in lines[:-1]:
        cls, x, y, w, h, c = line.split(' ')
        w = float(w) * float(dw)
        h = float(h) * float(dh)
        x1 = float(x) * float(dw) - w / 2
        y1 = float(y) * float(dh) - h / 2
        x2 = x1 + w
        y2 = y1 + h
        color = (12, 35, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    cv2.imwrite(f'./bb_test.jpg', img)
