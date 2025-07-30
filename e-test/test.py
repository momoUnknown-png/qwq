import cv2
import numpy as np

# 读取图像并转换为灰度图
image = cv2.imread(r"C:\Users\chen_\Desktop\e-test\target_simulation.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 HoughCircles 检测圆形
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=50)

# 如果检测到圆形
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # 绘制圆形
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # 绘制圆心

# 显示结果
cv2.imshow("output", image)
cv2.waitKey(0)
