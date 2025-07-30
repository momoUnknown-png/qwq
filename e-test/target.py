from PIL import Image, ImageDraw

# 1. 参数设定
dpi = 100
mm_to_px = lambda mm: int(mm / 25.4 * dpi)
width_px, height_px = mm_to_px(210+18*2), mm_to_px(297+18*2)

# 2. 创建白底画布
img = Image.new("RGB", (width_px, height_px), "white")
draw = ImageDraw.Draw(img)

# 3. 黑色边框（1.8cm厚）
border = mm_to_px(18)  # 1.8 cm → 18 mm → px
draw.rectangle(
    [(0, 0), (width_px, height_px)],
    fill="black"
)
draw.rectangle(
    [(border, border), (width_px-border, height_px-border)],
    fill="white"
)

# 4. 红色同心圆（半径 2,4,6,8,10 cm）
center = (width_px//2, height_px//2)
for r_cm in [2,4,6,8,10]:
    r_px = mm_to_px(r_cm*10)  # cm→mm→px
    bbox = [
        (center[0]-r_px, center[1]-r_px),
        (center[0]+r_px, center[1]+r_px)
    ]
    draw.ellipse(bbox, outline="red", width= max(1, int(mm_to_px(1)/10)) )
    # 0.1cm = 1mm → px → line width

# 5. 红色靶心（直径 0.1cm）
r_center = mm_to_px(1) // 2  # 0.1cm = 1mm, 半径 = 0.5mm
bbox_center = [
    (center[0]-r_center, center[1]-r_center),
    (center[0]+r_center, center[1]+r_center)
]
draw.ellipse(bbox_center, fill="red")

# 6. 保存结果
img.save("target_simulation.png")
