# %% [markdown]
# <a href="https://colab.research.google.com/github/Soumi7/Table_Data_Extraction/blob/main/medium_table.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Loading original image to display

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


"""
从图像中提取表格数组
"""


class Image:
    def __init__(self, file, line_length=100):
        img = cv2.imread(file, 0)
        self.img = img
        self.line_length = line_length

        self.get_lines()
        self.get_boundingBoxes()
        self.get_boxes()

    def get_lines(self):
        img = self.img
        thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img_bin = 255 - img_bin

        img_bin1 = 255 - img
        thresh1, img_bin1_otsu = cv2.threshold(img_bin1, 128, 255, cv2.THRESH_OTSU)

        img_bin2 = 255 - img
        thresh1, img_bin_otsu = cv2.threshold(
            img_bin2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, np.array(img).shape[1] // self.line_length)
        )
        eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)

        vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)

        hor_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (np.array(img).shape[1] // self.line_length, 1)
        )
        horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=5)

        horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)

        vertical_horizontal_lines = cv2.addWeighted(
            vertical_lines, 0.5, horizontal_lines, 0.5, 0.0
        )
        vertical_horizontal_lines = cv2.erode(
            ~vertical_horizontal_lines, kernel, iterations=3
        )

        thresh, vertical_horizontal_lines = cv2.threshold(
            vertical_horizontal_lines, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        bitxor = cv2.bitwise_xor(img, vertical_horizontal_lines)

        bitnot = cv2.bitwise_not(bitxor)

        self.lines = vertical_horizontal_lines

        self.bitnot = bitnot

        return vertical_horizontal_lines

    def get_boundingBoxes(self):
        contours, hierarchy = cv2.findContours(
            self.lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
        (contours, boundingBoxes) = zip(
            *sorted(zip(contours, boundingBoxes), key=lambda x: x[1][1])
        )

        self.boundingBoxes = boundingBoxes
        return boundingBoxes

    def get_boxes(self):
        boxes = []
        mh, mw = self.img.shape
        for x, y, w, h in self.boundingBoxes:
            if w < mw and h < mh:
                boxes.append([x, y, w, h])

        self.boxes = boxes
        return boxes

    def print_csv(self):

        rows = []
        columns = []

        heights = [self.boundingBoxes[i][3] for i in range(len(self.boundingBoxes))]

        mean = np.mean(heights)
        print(mean)

        columns.append(self.boxes[0])
        previous = self.boxes[0]
        for i in range(1, len(self.boxes)):
            if self.boxes[i][1] <= previous[1] + mean / 2:
                columns.append(self.boxes[i])
                previous = self.boxes[i]
                if i == len(self.boxes) - 1:
                    rows.append(columns)
            else:
                rows.append(columns)
                columns = []
                previous = self.boxes[i]
                columns.append(self.boxes[i])
        print("Rows")
        # print(rows)
        for row in rows:
            print(row)

        print("Columns")
        for col in columns:
            print(col)

        total_cells = 0
        for i in range(len(row)):
            if len(row[i]) > total_cells:
                total_cells = len(row[i])
        print(total_cells)

        center = [
            int(rows[i][j][0] + rows[i][j][2] / 2)
            for j in range(len(rows[i]))
            if rows[0]
        ]
        print(center)

        center = np.array(center)
        center.sort()
        print(center)

        boxes_list = []
        for i in range(len(rows)):
            l = []
            for k in range(total_cells):
                l.append([])
            for j in range(len(rows[i])):
                diff = abs(center - (rows[i][j][0] + rows[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                l[indexing].append(rows[i][j])
            boxes_list.append(l)

        self.boxes_list = boxes_list

        for box in boxes_list:
            print(box)

        dataframe_final = []
        for i in range(len(boxes_list)):
            for j in range(len(boxes_list[i])):
                s = ""
                if len(boxes_list[i][j]) == 0:
                    dataframe_final.append(" ")
                else:
                    for k in range(len(boxes_list[i][j])):
                        y, x, w, h = (
                            boxes_list[i][j][k][0],
                            boxes_list[i][j][k][1],
                            boxes_list[i][j][k][2],
                            boxes_list[i][j][k][3],
                        )
                        roi = self.bitnot[x : x + h, y : y + w]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(
                            roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255]
                        )
                        resizing = cv2.resize(
                            border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                        )
                        dilation = cv2.dilate(resizing, kernel, iterations=1)
                        erosion = cv2.erode(dilation, kernel, iterations=2)
                        out = pytesseract.image_to_string(erosion)
                        if len(out) == 0:
                            out = pytesseract.image_to_string(erosion)
                        s = s + " " + out
                    dataframe_final.append(s)
        print(dataframe_final)
        pass
