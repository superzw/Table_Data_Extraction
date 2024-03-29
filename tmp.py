import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

"""
在空白图片上画box
"""


def draw_boxes(img, boxes):
    img_empty = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    color = (255, 0, 0)  # 红色
    thickness = 2
    # 绘制矩形

    # 显示图像
    for box in boxes:
        # 定义矩形的参数
        x, y, w, h = box
        cv2.rectangle(img_empty, (x, y), (x + w, y + h), color, thickness)
    plt.imshow(img_empty)
    plt.show()


"""
在空白图片上画box
"""


def draw_box(img, box):
    draw_boxes(img, [box])


"""
显示图片
"""


def show_img(img):
    plt.imshow(img)
    plt.show()


def show_img_gray(img):
    plt.imshow(img, cmap="gray")
    plt.show()


"""
显示图片的一部分
done
"""


def show_part_img(img, box):
    y, x, w, h = box
    # roi = img[y : y + w, x : x + h]
    roi = img[x : x + h, y : y + w]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    border = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
    plt.imshow(border)
    plt.show()


"""
done
"""


def print_boxes_content(img, boxes):
    for box in boxes:
        y, x, w, h = box
        roi = img[x : x + h, y : y + w]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        border = cv2.copyMakeBorder(
            roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255]
        )
        resizing = cv2.resize(border, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        dilation = cv2.dilate(resizing, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=2)

        print("----", erosion.shape)
        plt.imshow(erosion, cmap="gray")
        plt.show()
        out = pytesseract.image_to_string(erosion)
        if len(out) == 0:
            out = pytesseract.image_to_string(erosion)

        print(f"box: {box}, content: {out}")

    pass
