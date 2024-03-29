task.md

> 先跑起来

使用例子可以跑出来，换一张图片就不行了，会报错。

# 排错

## 错误 1: 因为识别表格多出来的噪点，报错数组越界

```py
vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, np.array(img).shape[1] // 100)
    )
```

手动修改这里可以调整。

或者在这里设置：
img=Image('./test6.png',line_length=100)

## 错误 2 ：前面 OK，还是报错数组越界

# 记录

通过 cv2.findContours 返回的轮廓数组，为什么比单元格数量多？

比如我有个 2x2 的表格，
会返回 4 个单元格，还会整个表格也是一个轮廓，+ 整张图片也是一个轮廓（0，0 开始的）

# 测试，我用一个最简单的 2x2 的图片测试

发现
没有报错，但是很多单元格识别的文字是空字符串。

调试发现，
imge to string 是，传的裁切图片有问题，没有正确的裁切到的。

代码里面有错啊吧，改成：

```py
roi = bitnot[y : y + h, x : x + w]
```

为什么原先的能跑呢？

第二处错误：

```
print(dataframe[i][j],end=" ")
```

应该改成：

```
print(dataframe.iloc[i][j],end=" ")
```

否则会报错的

# 卧槽不行啊，改了之后，原先的图片识别有问题了。。。。

# 源代码解析：

没看懂 centre 的意义。

# 逻辑解析：

1. 图片识别，找到表格（这一步要清晰，避免噪点）
2.

# 问题：位置没有对上啊。

位置对上了。

# 位置没有对上的问题 😂，源自于 box 中 x,y 顺序不一样的问题

```py
y,x,w,h = boxes_list[i][j][k][0],boxes_list[i][j][k][1], boxes_list[i][j][k][2],boxes_list[i][j][k][3]
roi = bitnot[x:x+h, y:y+w]
```

有的地方是这么写的。。。。
坑爹啊。

img 数组是 y,x 排列的。。。。

```
img = cv2.imread(file)
```

OpenCV 中图像的 shape 属性返回的元组是 (height, width)，而不是 (width, height)，主要有以下几个原因：

1. 历史原因

OpenCV 最初是在 C++ 中开发的，而 C++ 中的矩阵通常是按行存储的。因此，OpenCV 中图像的 shape 属性也遵循了这种行存储的格式。

2. 与其他库兼容

许多其他图像处理库，例如 NumPy 和 Matplotlib，也都使用 (height, width) 格式来表示图像的尺寸。为了与这些库兼容，OpenCV 也采用了相同的格式。

3. 符合直觉

对于大多数图像来说，高度比宽度更重要。因此，将高度放在元组的第一个位置可以更直观地表示图像的尺寸。

4. 便于计算

在许多图像处理操作中，例如图像缩放和旋转，都需要使用图像的高度和宽度。将高度放在元组的第一个位置可以方便地进行这些计算。

总而言之，OpenCV 中图像的 shape 属性返回的元组是 (height, width)，主要有历史原因、兼容性、直觉和计算方便等方面的考虑。

以下是一些其他相关的信息：

在 OpenCV 中，图像的宽度可以通过 image.cols 属性获取，图像的高度可以通过 image.rows 属性获取。
在 NumPy 中，图像的形状可以通过 image.shape 属性获取，返回的元组也是 (height, width) 格式。
在 Matplotlib 中，可以使用 plt.imshow 函数显示图像，该函数的第一个参数是图像数据，第二个参数是可选的 cmap 参数，用于设置图像的颜色映射。

# test6 单元格位置正确了，识别还是有问题。

奇怪了。。。

```
import pytesseract
img = cv2.imread('./output_name.png', 0)
pytesseract.image_to_string(img)
```

这种，能识别出来啊。

妹哟识别出来
我把 resizing 调大了。 250 宽度，识别出来了。

可见，文字要大一点，才能识别啊。

# baocuo

```
center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
print(center)
```

i 是上一步

```
total_cells=0
for i in range(len(row)):
  if len(row[i]) > total_cells:
    total_cells = len(row[i])
print(total_cells)
```

生成的。

这 tmd 明显是 bug 啊。
注意上一步是对 row 循环，然后去 rows[i]这就扯淡了。。。。
哎，bug 真多啊。

这段代码应该是：

```
total_cells=0
for i in range(len(rows)):
    if len(rows[i]) > total_cells:
        total_cells = len(rows[i])
print(total_cells)
```

# 重新梳理这个逻辑：

1. 图片中的表格提取。

```
boundingBoxes = [cv2.boundingRect(c) for c in cnts]

(contours, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
key=lambda x:x[1][1]))
```

这段代码用 gemini 看懂了。
原来是排序啊/

终于跑通了。。。

# 终于搞定了。

不同的图片，需要修改 line_size，这样会影响识别格子时的噪声。

修改 fxy（图片放大倍数），这样会影响识别效果。
