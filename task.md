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

# 用原版代码测试看看

roi = bitnot[y : y + h, x : x + w]
