import gc
import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


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


# 生成掩码
# <model_type> 参数如下：
# default or vit_h: ViT-H SAM model.
# vit_l: ViT-L SAM model.
# vit_b: ViT-B SAM model.
def generate_masks(input_image_path, checkpoint, masks_result_path, device="cuda", model_type="default"):
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    with open(masks_result_path, 'w') as f:
        for mask in masks:
            rle = encode_mask(mask['segmentation'])
            data = {
                'segmentation': rle,
                'area': mask['area'],
                'bbox': mask['bbox'],
                'stability_score': mask['stability_score'],
                'point_coords': mask['point_coords']
            }
            json.dump(data, f)
            f.write('\n')

    gc.collect()
    torch.cuda.empty_cache()
    return len(masks)


# 提供原图和该图所有掩码，生成分割图片返回。
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

            # print(contour)
            if len(contour) >= 3:  # 确保至少有三个坐标点
                draw.polygon(contour, outline=(0, 0, 0, 255), fill=color_mask)

    # 结合原图与覆盖层
    image = Image.alpha_composite(image, overlay)

    return image


def draw_with_masks(origin_path, masks_txt, output_path):
    # 用opencv打开
    img = cv2.imread(origin_path)
    # 从BGR转成RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转成PIL
    img = Image.fromarray(img)
    # 读取txt中的掩码并使用cocorle解码
    masks = []
    with open(masks_txt, 'r') as file:
        for line in file:
            # 将字符串读取为python对象
            line_data = json.loads(line)
            line_data['segmentation'] = decode_mask(line_data['segmentation'])
            masks.append(line_data)

    # print(masks)
    segment_image = show_anns(img, masks)
    segment_image.save(output_path)
    del img, masks, segment_image
    gc.collect()
