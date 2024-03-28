import cv2
import matplotlib.pyplot as plt


def show_img(bitnot, box):
    x, y, w, h = box
    roi = bitnot[y : y + h, x : x + w]
    border = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
    plt.imshow(border, cmap="gray")
    plt.show()
