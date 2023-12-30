import cv2
from PIL import Image, ImageDraw
import numpy as np
import json
import pycocotools.mask as coco_mask
import json

import cv2
import numpy as np
import pycocotools.mask as coco_mask
from PIL import Image, ImageDraw


def encode_mask(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle_dict = {
        'counts': rle['counts'].decode('utf-8'),
        'size': rle['size']
    }
    return rle_dict


def decode_mask(encoded_mask):
    rle = {
        'counts': encoded_mask['counts'].encode('utf-8'),
        'size': encoded_mask['size']
    }
    mask = coco_mask.decode(rle).astype(bool)
    return mask


img = f'../upload/9f3b73f2-1b54-49ca-a4cb-8eed0cb13e75.png'
res = f'../upload/9f3b73f2-1b54-49ca-a4cb-8eed0cb13e75.txt'

image = cv2.imread(img)
masks = []
with open(res, 'r') as file:
    for line in file:
        # 将字符串读取为python对象
        line_data = json.loads(line)
        line_data['segmentation'] = decode_mask(line_data['segmentation'])
        masks.append(line_data)

print(masks)


def show_anns(image, anns):
    if not anns:
        return image

    # 转换图像为 RGBA 模式
    image = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')

    for ann in anns:
        m = ann['segmentation']
        color_mask = tuple(np.random.randint(0, 256, 3).tolist() + [int(0.35 * 255)])

        # 将掩码转换为正确格式用于 findContours
        m = (m.astype(np.uint8) * 255)

        # 找到轮廓
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 转换轮廓为 PIL 可用的格式
            contour = [tuple(point) for point in contour.reshape(-1, 2)]

            # 绘制轮廓和填充
            draw.polygon(contour, outline=(0, 0, 0, 255), fill=color_mask)

    # 结合原图与覆盖层
    image = Image.alpha_composite(image, overlay)

    return image


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 转换成PIL格式
image = Image.fromarray(image)
segment_image = show_anns(image, masks)
# 显示
segment_image.show()
# 保存
output_path = "../upload/out.png"
segment_image.save(output_path)
