import os
import sys

from randomizedHoughEllipseDetection import FindEllipseRHT

import cv2
import numpy as np

# 1. 读取原始图并转灰度
img = cv2.imread(r"C:\Users\chen_\Desktop\e-test\target_simulation.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 用 Canny 得到二值边缘图，作为 RHT 的输入
edges = cv2.Canny(gray, 100, 200)
mask = edges.astype(bool)   # FindEllipseRHT 需要一个布尔型掩码

# 3. 调用随机化 Hough 算子
detector = FindEllipseRHT(gray, mask)
# plot_mode=True 会弹 matplotlib 窗口显示中间结果；debug_mode=False 关掉额外日志
result_img = detector.run(debug_mode=False, plot_mode=False)

# 4. result_img 是在原始灰度图上画了找到的椭圆线条和中心点的图
#    如果你要叠加回彩色原图，可以先转回 BGR，再 overlay：
result_bgr = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

# 5. （可选）继续用 HoughCircles 在增强后图上找最小圆心
circles = cv2.HoughCircles(
    result_img, 
    cv2.HOUGH_GRADIENT, 
    dp=1.2, 
    minDist=50, 
    param1=50, 
    param2=30, 
    minRadius=5, 
    maxRadius=100
)
if circles is not None:
    for x, y, r in np.round(circles[0]).astype(int):
        cv2.circle(result_bgr, (x,y), r, (0,255,0), 2)
        cv2.circle(result_bgr, (x,y), 2, (0,0,255), 3)

# 6. 显示最终结果
cv2.imshow("Enhanced Ellipse Detection", result_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
