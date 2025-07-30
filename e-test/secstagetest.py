from maix import image, camera, display, app, uart, time
import cv2
import numpy as np
from struct import pack
 
# 初始化串口
serial = uart.UART("/dev/ttyS0", 115200)
 
# 初始化摄像头和显示器
cam = camera.Camera(320, 240, fps=80)
disp = display.Display()
 
# 红色在 RGB 空间容易受光照干扰，而在 LAB 色彩空间的 A 通道 中，红色数值范围更集中。
# 红点检测阈值（LAB 色彩空间 A 通道）
A_MIN = 150
A_MAX = 255
kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# 矩形轮廓面积范围
MIN_AREA = 2000
MAX_AREA = 100000
 
# 靶心(红点)轮廓面积范围 (需要根据实际情况微调)
RED_DOT_MIN_AREA = 2
RED_DOT_MAX_AREA = 50

# 中心点融合权重 (总和建议为1)
W_RECT_OUTER = 0.35  # 外框中心权重
W_RECT_INNER = 0.35  # 内框中心权重
W_RED_DOT = 0.1    # 靶心权重
W_CIRCLES = 0.2    # 圆心平均值权重
 
pause_rect_detect = False  # 是否暂停矩形检测

def get_rect_center(rect_points):
    """计算矩形/四边形的中心点（四个顶点的平均值）"""
    if rect_points is None or len(rect_points) < 4:
        return None
    sum_x = sum(p[0] for p in rect_points)
    sum_y = sum(p[1] for p in rect_points)
    return (int(sum_x / len(rect_points)), int(sum_y / len(rect_points)))

def is_rectangle(approx):
    """判断轮廓是否近似为矩形"""
    if approx is None or len(approx) != 4 or not cv2.isContourConvex(approx):
        return False
    pts = [point[0] for point in approx]
    def angle(p1, p2, p3):
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi
    angles = [angle(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return all(80 < ang < 100 for ang in angles)
 
def calculate_weighted_center(points_info):
    """
    根据权重计算最终中心点
    points_info: 一个列表，每个元素是元组 (point, weight)，例如 [((160, 120), 0.5), ((162, 122), 0.5)]
    """
    if not points_info:
        return (-1, -1)

    total_weight = sum(info[1] for info in points_info)
    if total_weight == 0:
        return (-1, -1)

    weighted_x = sum(info[0][0] * info[1] for info in points_info)
    weighted_y = sum(info[0][1] * info[1] for info in points_info)

    final_x = int(weighted_x / total_weight)
    final_y = int(weighted_y / total_weight)

    return (final_x, final_y) 

# 主循环
 
while not app.need_exit():
    try:
        # 检查串口是否接收到 0x66
        try:
            data = serial.read()
            if data and b'\x66' in data:
                pause_rect_detect = True
        except Exception as e:
            print("串口读取异常:", e)
        img = cam.read()
        if img is None:
            continue
        try:
            img_raw = image.image2cv(img, copy=True)
        except Exception as e:
            print("图像转换失败:", e)
            continue
 
        # 初始化本帧检测结果
        detected_centers_info = []
        final_center = (-1, -1)

        # --- 1. 矩形内外边框检测 ---
        if not pause_rect_detect:
            try:
                gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                # 使用自适应阈值对黑色胶带进行二值化
                bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 21, 5) # INV用于检测黑色
                closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

                # 使用 RETR_TREE 来获取轮廓间的层级关系
                contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if hierarchy is not None:
                    # 遍历所有轮廓，寻找 "父-子" 轮廓对 (即外框和内框)
                    for i in range(len(contours)):
                        # 如果当前轮廓i有一个子轮廓 (hierarchy[0][i][2] != -1)
                        # 并且它自己没有父轮廓 (hierarchy[0][i][3] == -1)
                        if hierarchy[0][i][2] != -1 and hierarchy[0][i][3] == -1:
                            outer_contour = contours[i]
                            inner_contour_index = hierarchy[0][i][2]
                            inner_contour = contours[inner_contour_index]

                            outer_area = cv2.contourArea(outer_contour)
                            inner_area = cv2.contourArea(inner_contour)

                            # 面积筛选，确保是目标
                            if RECT_MIN_AREA < outer_area < RECT_MAX_AREA:
                                # 简化轮廓为四边形
                                outer_approx = cv2.approxPolyDP(outer_contour, 0.02 * cv2.arcLength(outer_contour, True), True)
                                inner_approx = cv2.approxPolyDP(inner_contour, 0.02 * cv2.arcLength(inner_contour, True), True)

                                if is_rectangle(outer_approx, outer_area) and is_rectangle(inner_approx, inner_area):
                                    outer_rect_pts = [tuple(p[0]) for p in outer_approx]
                                    inner_rect_pts = [tuple(p[0]) for p in inner_approx]

                                    # 计算中心点
                                    outer_center = get_rect_center(outer_rect_pts)
                                    inner_center = get_rect_center(inner_rect_pts)

                                    if outer_center and inner_center:
                                        # 验证内外框中心是否重合
                                        dist = np.linalg.norm(np.array(outer_center) - np.array(inner_center))
                                        if dist < 10: # 允许10个像素的误差
                                            detected_centers_info.append((outer_center, W_RECT_OUTER))
                                            detected_centers_info.append((inner_center, W_RECT_INNER))

                                            # 可视化
                                            cv2.drawContours(img_raw, [outer_approx], -1, (0, 255, 0), 2)
                                            cv2.drawContours(img_raw, [inner_approx], -1, (0, 255, 255), 2)
                                            cv2.circle(img_raw, outer_center, 5, (0, 255, 0), -1)
                                            cv2.circle(img_raw, inner_center, 5, (0, 255, 255), -1)
                                    break # 找到一对就跳出循环
            except Exception as e:
                print("矩形检测异常:", e)


        # --- 2. 红色特征检测 (靶心与圆环) ---
        try:
            lab = cv2.cvtColor(img_raw, cv2.COLOR_BGR2Lab)
            _, A, _ = cv2.split(lab)
            A = A.astype(np.uint8)
            red_mask = cv2.inRange(A, np.array(A_MIN, dtype=np.uint8), np.array(A_MAX, dtype=np.uint8))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_red)

            # 2.1 靶心检测 (小面积轮廓)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 寻找面积在设定范围内的最小轮廓
                possible_dots = [cnt for cnt in contours if RED_DOT_MIN_AREA < cv2.contourArea(cnt) < RED_DOT_MAX_AREA]
                if possible_dots:
                    red_dot_contour = min(possible_dots, key=cv2.contourArea)
                    M = cv2.moments(red_dot_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        red_dot_center = (cx, cy)
                        detected_centers_info.append((red_dot_center, W_RED_DOT))
                        cv2.circle(img_raw, red_dot_center, 5, (0, 0, 255), -1) # 红色标记靶心

            # 2.2 圆环检测 (HoughCircles)
            # 对红色掩码图进行模糊以连接断线，有助于霍夫圆检测
            blurred_mask = cv2.GaussianBlur(red_mask, (5, 5), 1.5)
            circles = cv2.HoughCircles(blurred_mask, cv2.HOUGH_GRADIENT,
                                    dp=1.2, minDist=20, # minDist可以小一些，因为圆是同心的
                                    param1=50, param2=25, # param2可以调低，因为线很细
                                    minRadius=10, maxRadius=150) # 半径范围根据实际情况调整

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                circle_centers = []
                for (x, y, r) in circles:
                    circle_centers.append((x, y))
                    cv2.circle(img_raw, (x, y), r, (255, 0, 0), 2) # 蓝色标记圆环

                if circle_centers:
                    # 计算所有检测到圆心的平均位置
                    avg_circle_x = int(sum(p[0] for p in circle_centers) / len(circle_centers))
                    avg_circle_y = int(sum(p[1] for p in circle_centers) / len(circle_centers))
                    avg_circles_center = (avg_circle_x, avg_circle_y)
                    detected_centers_info.append((avg_circles_center, W_CIRCLES))
                    cv2.circle(img_raw, avg_circles_center, 5, (255, 0, 0), -1)

        except Exception as e:
            print("红色特征检测异常:", e)


        # --- 3. 中心点融合计算 ---
        if detected_centers_info:
            final_center = calculate_weighted_center(detected_centers_info)
            # 在图像上用一个醒目的标记画出最终的中心点
            cv2.drawMarker(img_raw, final_center, (255, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)


        # --- 4. 数据打包并发送 ---
        try:
            # 发送帧头和最终的中心点坐标
            payload = b'\xAA\x55'
            payload += pack("<hh", *final_center) # h表示16位短整型
            # 你也可以添加一个状态字节来表示检测到了哪些特征
            # payload += pack("<b", status_byte)
            payload += b'\x55\xAA' # 帧尾
            serial.write(payload)

        except Exception as e:
            print("串口发送异常:", e)

        # ... (图像显示和延时部分保持不变) ...
        try:
            img_show = image.cv2image(img_raw, copy=False)
            disp.show(img_show)
        except Exception as e:
            print("图像显示失败:", e)

        time.sleep_ms(1)

    except Exception as e:
        print("主循环异常:", e)
